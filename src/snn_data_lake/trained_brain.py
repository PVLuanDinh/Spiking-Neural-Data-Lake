"""General trained-brain inference (zero-dep) — serve GPU-trained spiking weights.

The [train] GPU tier (snn_data_lake.training) trains a multi-layer spiking classifier
with surrogate-gradient BPTT and exports its per-layer weight matrices here as a plain
JSON "brain". This module loads that brain and runs it with NO torch, so a clone-and-run
install serves a real trained model that the rest of the stack (lakehouse queries, the
matcher, the arm loop) can call.

Forward semantics (the trainer matches these EXACTLY, so weights transfer faithfully --
this is the fix for the v2.74 caveat where the trainer's dynamics diverged from the core's):

  * input x in [0, 1] is RATE-CODED over `steps` timesteps: input i spikes with prob x_i
    each step (deterministic: the per-sample RNG is seeded from crc32(x), never the global RNG).
  * HIDDEN layers are leaky integrate-and-fire: membrane m += W . spikes_in each step, decays
    by `beta`, emits a spike and resets when m >= `threshold`. Their spike trains feed the next layer.
  * the OUTPUT layer accumulates weighted input-spike currents over all steps (no leak, no
    threshold): out[c] = sum_t sum_i W[c][i] * spike_i(t); the prediction is argmax(out).

A single-layer brain (no hidden layers) reduces EXACTLY to snn_classifier.forward -- it is the
generalisation of that readout to arbitrary input/output sizes and optional hidden layers.

  python -m snn_data_lake.trained_brain        # (also run by run_selfchecks.py)
"""
import json
import zlib

BRAIN_FORMAT = "snn_brain/1"
DEFAULT_BETA = 0.9
DEFAULT_THRESHOLD = 1.0


def _seed_of(x):
    """Stable per-input seed from the rate vector (crc32 of rounded bytes; never the global RNG)."""
    b = bytes(bytearray(int(min(1.0, max(0.0, v)) * 255) & 0xFF for v in x))
    return zlib.crc32(b) & 0xFFFFFFFF


class _Rng:
    """Tiny deterministic LCG (self-contained, so encoding never touches random's global state)."""

    def __init__(self, seed):
        self.s = (seed ^ 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def random(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.s >> 11) / float(1 << 53)


def predict_scores(brain, x):
    """Run the trained brain on input vector x (len == brain['in']); return the output scores."""
    layers = brain["layers"]
    steps = int(brain["steps"])
    beta = float(brain.get("beta", DEFAULT_BETA))
    thr = float(brain.get("threshold", DEFAULT_THRESHOLD))
    if len(x) != brain["in"]:
        raise ValueError(f"input length {len(x)} != brain in={brain['in']}")
    rng = _Rng(_seed_of(x))
    n_last = len(layers) - 1
    mems = [[0.0] * len(W) for W in layers]      # membrane per neuron, per layer
    out_acc = [0.0] * len(layers[-1])
    for _ in range(steps):
        spikes = [1 if rng.random() < xi else 0 for xi in x]
        active = [i for i, s in enumerate(spikes) if s]
        for li, W in enumerate(layers):
            cur = [0.0] * len(W)
            for o in range(len(W)):
                row = W[o]
                cur[o] = sum(row[i] for i in active)     # weighted incoming spikes (sparse)
            if li == n_last:
                for o in range(len(W)):
                    out_acc[o] += cur[o]                  # output: accumulate currents (linear readout)
                break
            m = mems[li]
            fired = []
            for o in range(len(W)):
                m[o] = m[o] * beta + cur[o]               # leaky integrate
                if m[o] >= thr:
                    fired.append(o)
                    m[o] = 0.0                            # spike + reset
            active = fired                                # hidden spikes feed the next layer
    return out_acc


def predict(brain, x):
    """Predicted class = argmax of the output scores."""
    scores = predict_scores(brain, x)
    return max(range(len(scores)), key=lambda c: scores[c])


def accuracy(brain, data):
    """data: list of (x, label). Returns fraction correct."""
    correct = sum(1 for x, y in data if predict(brain, x) == y)
    return correct / len(data)


def make_brain(layers, steps, beta=DEFAULT_BETA, threshold=DEFAULT_THRESHOLD, meta=None):
    """Assemble a brain dict from per-layer weight matrices (each W is [out][in] floats)."""
    layers = [[[float(w) for w in row] for row in W] for W in layers]
    return {"format": BRAIN_FORMAT, "in": len(layers[0][0]), "out": len(layers[-1]),
            "steps": int(steps), "beta": float(beta), "threshold": float(threshold),
            "layers": layers, "meta": dict(meta or {})}


def save_brain(brain, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(brain, f)
    return path


def load_brain(path):
    """Load + validate a brain at this trust boundary (format, layer shapes chain in->out)."""
    with open(path, encoding="utf-8") as f:
        brain = json.load(f)
    if not isinstance(brain, dict) or str(brain.get("format", "")) != BRAIN_FORMAT:
        raise ValueError(f"{path}: not a {BRAIN_FORMAT} file (format={brain.get('format')!r})")
    layers = brain.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"{path}: 'layers' must be a non-empty list")
    prev_in = brain["in"]
    for k, W in enumerate(layers):
        if not W or not all(len(row) == len(W[0]) for row in W):
            raise ValueError(f"{path}: layer {k} is ragged")
        if len(W[0]) != prev_in:
            raise ValueError(f"{path}: layer {k} expects in={len(W[0])} but got {prev_in}")
        prev_in = len(W)                                  # this layer's out feeds the next
    if prev_in != brain["out"]:
        raise ValueError(f"{path}: final layer out={prev_in} != brain out={brain['out']}")
    return brain


def main():
    # Build a single-layer brain from the core classifier's delta-rule weights and confirm it
    # reproduces snn_classifier.forward EXACTLY (single layer == that readout), then a 2-layer smoke.
    import os
    import random
    import tempfile

    from snn_data_lake import snn_classifier as scl

    random.seed(0)
    temps = scl.templates()
    C = len(temps)
    train = scl.dataset(scl.TRAIN_PER_CLASS, temps)
    test = scl.dataset(scl.TEST_PER_CLASS, temps)
    W, _ = scl.train(train, C, quiet=True)

    brain = make_brain([W], steps=scl.STEPS, meta={"source": "delta-rule 1-layer"})
    # accuracy of the single-layer brain on the held-out set
    acc = accuracy(brain, test)

    # faithfulness: single-layer trained_brain == snn_classifier argmax on the same input
    agree = 0
    for x, _ in test:
        c_scl = scl.forward(W, x, C)[0]
        pred_scl = max(range(C), key=lambda c: c_scl[c])
        agree += predict(brain, x) == pred_scl
    faithful = agree / len(test)

    # round-trip + a 2-layer brain runs deterministically
    bp = os.path.join(tempfile.gettempdir(), "snn_brain_selfcheck.json")
    save_brain(brain, bp)
    brain2 = load_brain(bp)
    os.remove(bp)
    two = make_brain([[[0.05] * scl.N for _ in range(8)], [[0.1] * 8 for _ in range(C)]], steps=8)
    x0 = test[0][0]
    det = predict_scores(two, x0) == predict_scores(two, x0)

    print("=" * 60)
    print("TRAINED BRAIN -- zero-dep inference over GPU-trained spiking weights")
    print("=" * 60)
    print(f"single-layer brain (from delta-rule W): test accuracy {acc:.0%}")
    print(f"faithful to snn_classifier.forward     : {faithful:.0%} identical predictions")
    print(f"round-trip save/load                   : {brain2['layers'] == brain['layers']}")
    print(f"2-layer brain (784->8->C) deterministic: {det}")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert acc >= 0.85, f"single-layer trained brain must classify ({acc:.2f})"
    assert faithful == 1.0, f"single-layer brain must equal snn_classifier.forward ({faithful:.2f})"
    assert brain2["layers"] == brain["layers"], "brain must round-trip through save/load exactly"
    assert det, "predict must be deterministic for identical input"
    assert predict(brain, test[0][0]) in range(C), "prediction must be a valid class"
    print(f"self-check OK: trained-brain serves GPU-trainable weights zero-dep; {acc:.0%} acc, "
          f"faithful to the core readout, round-trips, deterministic")


if __name__ == "__main__":
    main()

"""
IS-SNN forward pass — intrinsically-stable multi-layer LIF forward dynamics (pure Python, no tensors).

A custom spiking forward pass over a stack of LIF layers (reused `lif_neuron.LIFNeuron`) with a hard-coded
INTRINSICALLY-STABLE init: sparse excitatory weights scaled to the LIF's self-sustaining rate fixed point
(mean synaptic current ~ v_threshold at the operating rate) so the per-layer firing rate is DEPTH-INVARIANT
-- it neither dies nor saturates as signal propagates, WITHOUT any dynamic BatchNorm. A deliberately-weak
init is the control that suffers catastrophic firing-rate decay. The connectivity is SPARSE (FlyHash-style,
like flyhash_index / sdm in this repo): a dense positive projection would low-pass every neuron toward the
same mean drive and collapse the code, so each unit samples a random subset of inputs -> the first hidden
layer is a discriminative random projection whose spike-rate vector is a class-separable embedding a
Van Rossum / Victor-Purpura index can store. Pure standard library, deterministic. The zero-dep counterpart
to the torch IS-SNN in research/neuromorphic_torch.

  python -m snn_data_lake.is_snn_forward        # (also run by run_selfchecks.py)
"""
import random

from snn_data_lake.lif_neuron import LIFNeuron

STABLE_GAIN = 2.5   # excitatory scale = STABLE_GAIN*v_threshold/fan_in_sparse -> rate flat across depth
WEAK_GAIN = 1.0     # too weak -> firing rate decays layer to layer (the bad-init control)
SPARSITY = 0.4      # fraction of inputs each unit samples (sparse random projection keeps the code separable)


class ISSNNForward:
    """Stacked LIF layers scaled to the firing-rate fixed point; forward() -> (embedding, per-layer rates)."""

    def __init__(self, sizes, tau_m=8.0, v_threshold=1.0, seed=0, stable=True):
        self.sizes = sizes
        self.tau_m = tau_m
        self.v_threshold = v_threshold
        rng = random.Random(seed)
        gain = STABLE_GAIN if stable else WEAK_GAIN
        self.w = []
        for fan_in, fan_out in zip(sizes, sizes[1:]):
            k = max(1, int(SPARSITY * fan_in))
            scale = gain * v_threshold / k                     # scale sparse fan-in to the LIF rate fixed point
            layer = []
            for _ in range(fan_out):
                row = [0.0] * fan_in
                for i in rng.sample(range(fan_in), k):         # sparse random projection (FlyHash-style)
                    row[i] = rng.uniform(0.5, 1.0) * scale
                layer.append(row)
            self.w.append(layer)

    def forward(self, spike_input):
        """spike_input: (T, n_in) list of 0/1 vectors. Returns (embedding rates[hidden1], per-layer mean rate)."""
        t_steps = len(spike_input)
        layers = [[LIFNeuron(self.tau_m, self.v_threshold) for _ in range(b)] for b in self.sizes[1:]]
        counts = [[0] * b for b in self.sizes[1:]]
        for t in range(t_steps):
            x = spike_input[t]
            for li, (wl, neurons) in enumerate(zip(self.w, layers)):
                y = []
                for j, nrn in enumerate(neurons):
                    current = sum(wl[j][i] * x[i] for i in range(len(x)))
                    s = nrn.step(current)
                    y.append(s)
                    counts[li][j] += s
                x = y
        layer_rates = [sum(c) / (len(c) * t_steps) for c in counts]
        embedding = [c / t_steps for c in counts[0]]           # first hidden = discriminative sparse projection
        return embedding, layer_rates


def _spike_batch(rng, rates, t_steps):
    return [[1 if rng.random() < r else 0 for r in rates] for _ in range(t_steps)]


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def main():
    rng = random.Random(1)
    n_in, K, T = 16, 3, 30
    class_rates = [[rng.random() * 0.5 + 0.1 for _ in range(n_in)] for _ in range(K)]

    net = ISSNNForward([n_in, 32, 32, 16], seed=0, stable=True)
    bad = ISSNNForward([n_in, 32, 32, 16], seed=0, stable=False)

    # per-class centroid embeddings (train) then classify fresh jittered samples (nearest centroid)
    centroids = [net.forward(_spike_batch(rng, class_rates[k], T))[0] for k in range(K)]
    hits, trials = 0, 60
    stable_rates = None
    for i in range(trials):
        k = i % K
        emb, stable_rates = net.forward(_spike_batch(rng, class_rates[k], T))
        pred = min(range(K), key=lambda c: _dist2(emb, centroids[c]))
        hits += pred == k
    acc = hits / trials
    _, bad_rates = bad.forward(_spike_batch(rng, class_rates[0], T))

    print("=" * 60)
    print("IS-SNN FORWARD — intrinsically-stable multi-layer LIF forward pass")
    print("=" * 60)
    print(f"net {net.sizes}; sparse init gain {STABLE_GAIN}/fan_in @ p={SPARSITY} (rate fixed point), no BatchNorm")
    print(f"stable layer rates : {[round(r, 3) for r in stable_rates]} (alive + bounded across depth)")
    print(f"bad-init rates     : {[round(r, 3) for r in bad_rates]} (catastrophic decay)")
    print(f"embedding classification: {acc:.0%} nearest-centroid over {K} classes")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert min(stable_rates) > 0.2 and max(stable_rates) < 0.75, \
        f"stable init must keep the rate ALIVE and BOUNDED across depth (no death, no saturation): {stable_rates}"
    assert stable_rates[-1] > 0.7 * stable_rates[0], f"stable rate must not decay across depth: {stable_rates}"
    assert bad_rates[-1] < 0.5 * stable_rates[-1], f"weak init must decay vs stable ({bad_rates} vs {stable_rates})"
    assert acc >= 0.8, f"the sparse forward embedding must separate classes ({acc:.2f})"
    emb1 = net.forward(_spike_batch(random.Random(9), class_rates[0], T))[0]
    emb2 = net.forward(_spike_batch(random.Random(9), class_rates[0], T))[0]
    assert emb1 == emb2, "same seed+input -> identical forward pass (deterministic)"
    print(f"self-check OK: stable rates {[round(r,2) for r in stable_rates]} (alive+bounded) vs bad "
          f"{[round(r,2) for r in bad_rates]} (decays); embedding acc {acc:.0%}; deterministic")


if __name__ == "__main__":
    main()

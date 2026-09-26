"""
v0.32 — stronger matching: a LEARNED spiking classifier vs the template baseline.

signal_loop.py matches a live window to the single nearest enrolled reference
(Van Rossum distance). That is brittle: one template per command, a generative
distance. This trains a supervised spiking classifier (reuses snn_classifier.py,
stdlib) on MANY noisy exemplars per command -> discriminative weights, the data
written into W by a local delta rule. Enrollment gives labels, so SUPERVISED
beats the unsupervised STDP/D&C nets here (and needs no torch/GPU).

Honesty: "stronger" is a claim, so this BENCHMARKS both matchers on the SAME
windows across a rising-noise sweep and reports who wins where. Low noise: a tie
(both ~100%). The learned model earns its keep as noise grows.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.cortex.learned_matcher
"""

import random

from snn_data_lake import snn_classifier as C  # train/forward/softmax + globals N, STEPS, EPOCHS
from snn_data_lake.brainstem.signal_loop import build_default_library
from snn_data_lake.cortex.spike_preprocessing import encode_latency, van_rossum_distance

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "train_learned": (
        "LOCAL",
        "builds the training set and delegates to snn_classifier.train, a single-layer spiking delta "
        "rule. A harness carrying the claim of the rule it drives",
    ),
    "calibrate_gate": (
        "NOT_A_RULE",
        "computes a THRESHOLD separating known from novel, set at a midpoint. A decision boundary "
        "derived once from measurements; it trains nothing",
    ),
    "learned_label": (
        "NOT_A_RULE",
        "returns the raw learned prediction with no rejection. A read-out of an already-trained "
        "matcher",
    ),
}

N = C.N  # 64 features — matches the signal window
THR = 0.5  # binarize a [0,1] window for the rate-coder
K_PER = 30  # exemplars enrolled per command (multi-shot)
TRAIN_NOISE = 0.20  # noise the learner is trained under


def binarize(vec, thr=THR):
    return [1 if v >= thr else 0 for v in vec]


def jitter(vec, n, rng):
    return [min(1.0, max(0.0, v + rng.uniform(-n, n))) for v in vec]


def make_windows(library, n_each, noise, seed):
    """Labeled noisy windows for every command + the same count of novel windows."""
    rng = random.Random(seed)
    labels = list(library)
    known, novel = [], []
    for li, lab in enumerate(labels):
        for _ in range(n_each):
            known.append((jitter(library[lab], noise, rng), li, lab))
    for _ in range(n_each * len(labels)):
        novel.append([rng.random() for _ in range(N)])
    return labels, known, novel


# ---- learned matcher: train once, classify with confidence-gated rejection --
def train_learned(library, seed=0):
    rng = random.Random(seed)
    labels = list(library)
    data = []
    for li, lab in enumerate(labels):
        for _ in range(K_PER):
            data.append((binarize(jitter(library[lab], TRAIN_NOISE, rng)), li))
    rng.shuffle(data)
    W, _ = C.train(data, len(labels), quiet=True)
    return W, labels


def learned_label(W, n_classes, window):
    """Raw learned prediction (no rejection) — strong accuracy, but overconfident
    on out-of-distribution garbage, so it must be gated for novelty (see hybrid)."""
    currents, _, _ = C.forward(W, binarize(window), n_classes)
    return max(range(n_classes), key=lambda c: currents[c])


# ---- novelty gate: Van Rossum distance to the nearest enrolled reference -----
def nearest_dist_enc(refs_enc, window):
    """Nearest Van Rossum distance to any enrolled reference AND the encoded window.
    Callers reuse both instead of re-encoding + re-distancing; since encode_latency is
    deterministic the returned distance equals nearest_dist(refs_enc, window)."""
    q = encode_latency(window)
    return min(van_rossum_distance(q, ev) for ev in refs_enc), q


def nearest_dist(refs_enc, window):
    return nearest_dist_enc(refs_enc, window)[0]


def calibrate_gate(library, refs_enc, seed=7):
    """Threshold separating known (noisy, up to a high operating noise) from novel.
    Set at the midpoint of the two distance distributions — anything farther than a
    real signal ever lands is treated as garbage and the loop stays silent."""
    rng = random.Random(seed)
    labels = list(library)
    known = [
        nearest_dist(refs_enc, jitter(library[l], 0.35, rng)) for l in labels for _ in range(20)
    ]
    novel = [nearest_dist(refs_enc, [rng.random() for _ in range(N)]) for _ in range(100)]
    return (max(known) + min(novel)) / 2.0


def hybrid_match(W, nC, refs_enc, gate, window):
    """Learned classifier picks the command; distance gate vetoes novel signals.

    v2.4 perf (decision-identical): returns (idx_or_None, nearest_dist, encoded_window).
    The distance and the latency encoding are computed ONCE here so callers (signal_loop)
    reuse them instead of recomputing nearest_dist + encode_latency per live window
    (was ~2 extra Van Rossum sweeps + ~3 re-encodes). The gate/label decision is unchanged."""
    dist, q = nearest_dist_enc(refs_enc, window)
    if dist > gate:
        return None, dist, q
    return learned_label(W, nC, window), dist, q


# ---- score the three matchers: accuracy on known, rejection on novel --------
def acc_rej(label_fn, labels, known, novel):
    corr = sum(1 for w, li, _ in known if label_fn(w) == li)
    rej = sum(1 for w in novel if label_fn(w) is None)
    return corr / len(known), rej / len(novel)


def main():
    # (was `C.random.seed(0)` here) DELETED 2026-08-04. It reached into snn_classifier's imported
    # `random` module and seeded the PROCESS-GLOBAL Mersenne Twister -- the exact side effect
    # snn_classifier.py:28 warns against, moved from import time to call time. It was also DEAD:
    # every RNG this function reaches is per-instance (`train_learned`/`calibrate_gate`/
    # `make_windows` each build their own `random.Random(seed)`, and `C.forward` seeds itself from
    # the sample), so nothing here consumed the global stream. Removing it leaves this module's
    # output byte-identical (verified) and stops it perturbing every module that runs after it in
    # the same process -- run_selfchecks has an in-process runpy path, and pytest imports are
    # in-process by construction.
    library = build_default_library()
    nC = len(library)
    refs_enc = [encode_latency(v) for v in library.values()]

    # baseline: nearest template + Lowe ratio reject (signal_loop's matcher logic)
    def template_label(w):
        q = encode_latency(w)
        s = sorted((van_rossum_distance(q, ev), i) for i, ev in enumerate(refs_enc))
        return s[0][1] if s[0][0] < 0.85 * s[1][0] else None

    W, labels = train_learned(library)  # learned (supervised spiking)
    gate = calibrate_gate(library, refs_enc)  # novelty threshold

    print("=" * 78)
    print("STRONGER MATCHING — template vs learned vs hybrid (learned + novelty gate)")
    print("=" * 78)
    print(
        f"commands={nC}  exemplars/cmd={K_PER}  train_noise={TRAIN_NOISE:.0%}  "
        f"gate={gate:.2f}  (chance {1 / nC:.0%})"
    )
    print(f"{'noise':>6} | {'TEMPLATE':>14} | {'LEARNED':>14} | {'HYBRID':>14}")
    print(f"{'':>6} | {'acc':>7}{'rej':>7} | {'acc':>7}{'rej':>7} | {'acc':>7}{'rej':>7}")
    print("-" * 78)

    rows = []
    for noise in (0.10, 0.20, 0.30, 0.40, 0.50):
        labs, known, novel = make_windows(library, 40, noise, seed=100 + int(noise * 100))
        t_acc, t_rej = acc_rej(template_label, labs, known, novel)
        l_acc, l_rej = acc_rej(lambda w: learned_label(W, nC, w), labs, known, novel)
        h_acc, h_rej = acc_rej(
            lambda w: hybrid_match(W, nC, refs_enc, gate, w)[0], labs, known, novel
        )
        rows.append((noise, t_acc, t_rej, l_acc, l_rej, h_acc, h_rej))
        print(
            f"{noise:>5.0%} | {t_acc:>6.0%}{t_rej:>7.0%} | {l_acc:>6.0%}{l_rej:>7.0%} "
            f"| {h_acc:>6.0%}{h_rej:>7.0%}"
        )
    print("=" * 78)

    hi = [r for r in rows if r[0] >= 0.40]
    t_hi = sum(r[1] for r in hi) / len(hi)
    h_hi = sum(r[5] for r in hi) / len(hi)
    print(
        f"high-noise (>=40%) acc:  template {t_hi:.0%}  ->  hybrid {h_hi:.0%}  "
        f"({h_hi - t_hi:+.0%})   |   hybrid novelty rejection holds where learned-alone fails"
    )

    # ---- self-checks --------------------------------------------------------
    assert rows[0][1] >= 0.85 and rows[0][5] >= 0.85, "a matcher fails at low noise"
    # hybrid keeps the learned model's high-noise accuracy edge over the template
    assert h_hi >= t_hi + 0.20, f"hybrid not stronger at high noise: {h_hi:.2f} vs {t_hi:.2f}"
    # hybrid rejects novel signals at every noise level (learned-alone does not)
    assert all(r[6] >= 0.5 for r in rows), "hybrid novelty rejection broke"
    # documents the learned-alone weakness this fixes
    assert rows[-1][4] < 0.3, "expected learned-alone to over-fire on novel signals"
    print(
        "self-check OK: hybrid keeps learned's high-noise accuracy AND template's "
        "novelty rejection; learned-alone over-fires on garbage (documented)"
    )


if __name__ == "__main__":
    main()

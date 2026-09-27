"""
Tempotron gesture matcher — supervised spike-TIMING recognition for the command library.

PROMOTED from research/ into the product (v0.82). The matchers so far decide by MAGNITUDE/DISTANCE
(learned_matcher's delta-rule readout, signal_loop's Van Rossum template) or firing ORDER
(rank_order_coding). The tempotron (Gutig & Sompolinsky, Nat. Neurosci. 2006) adds the third,
biologically-grounded option: a leaky neuron trained to FIRE for one command's spatiotemporal spike
pattern and stay silent for the others, learning from the actual spike TIMES via the gradient of its
peak sub-threshold voltage. `TempotronMatcher` enrolls one tempotron per command on the same
`encode_latency` codes the rest of the pipeline uses and classifies a window by which command
neuron peaks highest. On-thesis ("compute in spike timing"); a self-contained matcher option, not
the live default. Refs: Gutig & Sompolinsky, Nature Neuroscience 9 (2006).

  python -m snn_data_lake.cortex.tempotron        # (also run by run_selfchecks.py)
"""

import math
import random

from snn_data_lake.sensorimotor.signal_loop import build_default_library
from snn_data_lake.sensorimotor.spike_preprocessing import N, T, encode_latency

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "Tempotron.train": (
        "LOCAL",
        "the tempotron rule: on an error, move weights by the pre-synaptic contribution at the "
        "membrane PEAK time. Spike-timing-local, driven by a binary teacher, never by a propagated "
        "gradient",
    ),
}

TAU_M = 8.0  # membrane time constant (steps), tuned to the T=32 latency window
TAU_S = 2.0
DT = 1.0
_T0 = (TAU_M * TAU_S / (TAU_M - TAU_S)) * math.log(TAU_M / TAU_S)
_V0 = 1.0 / (math.exp(-_T0 / TAU_M) - math.exp(-_T0 / TAU_S))


def _psp(dt_):
    if dt_ <= 0:
        return 0.0
    return _V0 * (math.exp(-dt_ / TAU_M) - math.exp(-dt_ / TAU_S))


def _pattern(window):
    """encode_latency(window) -> per-afferent spike-time lists over N afferents."""
    pat = [[] for _ in range(N)]
    for t, i in encode_latency(window):
        pat[i].append(float(t))
    return pat


class Tempotron:
    """One leaky neuron; fires if its peak free-membrane voltage crosses threshold."""

    def __init__(self, n, threshold=1.0, seed=0):
        rng = random.Random(seed)
        self.w = [rng.uniform(0.0, 0.01) for _ in range(n)]
        self.thr = threshold

    def vmax(self, pattern):
        best_v, best_t, t = -1e9, 0.0, 0.0
        while t <= T:
            v = 0.0
            for wi, spikes in zip(self.w, pattern):
                if wi and spikes:
                    v += wi * sum(_psp(t - tf) for tf in spikes if tf < t)
            if v > best_v:
                best_v, best_t = v, t
            t += DT
        return best_v, best_t

    def train(self, patterns, targets, epochs=15, lr=0.01):
        for _ in range(epochs):
            for pat, tgt in zip(patterns, targets):
                vm, tm = self.vmax(pat)
                if (vm >= self.thr) == tgt:
                    continue
                sign = 1.0 if tgt else -1.0
                for i, spikes in enumerate(pat):
                    self.w[i] += lr * sign * sum(_psp(tm - tf) for tf in spikes if tf < tm)


class TempotronMatcher:
    """One tempotron per command (one-vs-rest); classify a window by the highest peak voltage."""

    def __init__(self, library, per_class=12, epochs=15, seed=0):
        rng = random.Random(seed)
        self.labels = list(library)
        pats, labs = [], []
        for li, lab in enumerate(self.labels):
            for _ in range(per_class):
                jit = [min(1.0, max(0.0, v + rng.uniform(-0.08, 0.08))) for v in library[lab]]
                pats.append(_pattern(jit))
                labs.append(li)
        self.tempo = []
        for c in range(len(self.labels)):
            tp = Tempotron(N, seed=c)
            tp.train(pats, [l == c for l in labs], epochs=epochs)
            self.tempo.append(tp)

    def label(self, window):
        pat = _pattern(window)
        return self.labels[max(range(len(self.tempo)), key=lambda c: self.tempo[c].vmax(pat)[0])]


def main():
    rng = random.Random(1)
    library = build_default_library()
    labels = list(library)
    matcher = TempotronMatcher(library)

    def jitter(v, n):
        return [min(1.0, max(0.0, x + rng.uniform(-n, n))) for x in v]

    trials = 150
    ok = 0
    for i in range(trials):
        lab = labels[i % len(labels)]
        ok += matcher.label(jitter(library[lab], 0.10)) == lab
    acc = ok / trials

    print("=" * 60)
    print("TEMPOTRON MATCHER — supervised spike-timing gesture recognition")
    print("=" * 60)
    print(f"{len(labels)} commands, one tempotron/command, {trials} jittered windows")
    print(f"recognition accuracy: {acc:.0%}  (chance {1 / len(labels):.0%})")
    print("=" * 60)

    # ---- self-check: deterministic asserts ----
    assert acc >= 0.9, f"tempotron matcher must recognize the commands ({acc:.2f})"
    clean = sum(matcher.label(library[lab]) == lab for lab in labels)
    assert clean == len(labels), "every clean command must be recognized exactly"
    assert TempotronMatcher(library).label(library[labels[0]]) == TempotronMatcher(library).label(
        library[labels[0]]
    ), "deterministic"
    print(
        f"self-check OK: tempotron matcher {acc:.0%} on jittered windows, {clean}/{len(labels)} clean; "
        f"spike-timing recognition wired to the command library; deterministic"
    )


if __name__ == "__main__":
    main()

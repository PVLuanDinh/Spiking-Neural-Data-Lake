"""
Tempotron — supervised learning of spike-TIMING patterns in a single leaky neuron
(fresh sweep, temporal-compute/learning axis). Pure-stdlib.

The repo learns by STDP (valence_stdp, unsupervised), by synaptic delays (delay_learning), or by
surrogate gradient (torch, own venv). None is the canonical *supervised temporal* rule. The
tempotron (Gutig & Sompolinsky, Nat. Neurosci. 2006) trains one leaky neuron to fire for one class
of spatiotemporal spike patterns and stay silent for another by nudging weights along the gradient
of its PEAK sub-threshold voltage at the instant it peaks -- a decision made on spike timing, not
rate. On-thesis ("compute in spike timing") and a clean supervised counterpart to the unsupervised
STDP models. Refs: Gutig & Sompolinsky, "The tempotron: a neuron that learns spike timing-based
decisions", Nature Neuroscience 9 (2006).

  Run: python tempotron.py        # (also run by run_selfchecks.py)
"""
import math
import random

TAU_M = 15.0                  # membrane time constant (ms)
TAU_S = TAU_M / 4.0           # synaptic time constant
T = 50.0                      # pattern window (ms)
DT = 0.5                      # voltage-grid step (ms)
_T0 = (TAU_M * TAU_S / (TAU_M - TAU_S)) * math.log(TAU_M / TAU_S)    # PSP peak time
_V0 = 1.0 / (math.exp(-_T0 / TAU_M) - math.exp(-_T0 / TAU_S))        # normalize PSP peak to 1


def _psp(dt_):
    """Normalized double-exponential PSP kernel; 0 before the presynaptic spike."""
    if dt_ <= 0:
        return 0.0
    return _V0 * (math.exp(-dt_ / TAU_M) - math.exp(-dt_ / TAU_S))


class Tempotron:
    """One leaky integrate neuron; fires if its peak free-membrane voltage crosses threshold."""

    def __init__(self, n, threshold=1.0, seed=0):
        self.n = n
        self.thr = threshold
        rng = random.Random(seed)
        self.w = [rng.uniform(0.0, 0.01) for _ in range(n)]

    def vmax(self, pattern):
        """Peak (v_max, t_max) of the free membrane potential over the time grid."""
        best_v, best_t, t = -1e9, 0.0, 0.0
        while t <= T:
            v = 0.0
            for wi, spikes in zip(self.w, pattern):
                if wi:
                    v += wi * sum(_psp(t - tf) for tf in spikes if tf < t)
            if v > best_v:
                best_v, best_t = v, t
            t += DT
        return best_v, best_t

    def predict(self, pattern):
        return self.vmax(pattern)[0] >= self.thr

    def train(self, patterns, targets, epochs=100, lr=0.01):
        for _ in range(epochs):
            for pat, tgt in zip(patterns, targets):
                vm, tm = self.vmax(pat)
                fired = vm >= self.thr
                if fired == tgt:
                    continue
                sign = 1.0 if tgt else -1.0        # should-fire but silent -> potentiate; vice versa -> depress
                for i, spikes in enumerate(pat):
                    self.w[i] += lr * sign * sum(_psp(tm - tf) for tf in spikes if tf < tm)


def _pattern(rng, n, max_spikes=3):
    return [sorted(rng.uniform(0, T) for _ in range(rng.randint(1, max_spikes))) for _ in range(n)]


def _accuracy(tp, patterns, targets):
    return sum(tp.predict(p) == t for p, t in zip(patterns, targets)) / len(patterns)


def main():
    rng = random.Random(1)
    n, k = 20, 5
    patterns = [_pattern(rng, n) for _ in range(2 * k)]
    targets = [True] * k + [False] * k          # first k must fire, last k stay silent
    tp = Tempotron(n, threshold=1.0, seed=0)

    acc0 = _accuracy(tp, patterns, targets)
    tp.train(patterns, targets, epochs=100, lr=0.01)
    acc1 = _accuracy(tp, patterns, targets)

    vpos = [tp.vmax(p)[0] for p, t in zip(patterns, targets) if t]
    vneg = [tp.vmax(p)[0] for p, t in zip(patterns, targets) if not t]

    print("=" * 62)
    print("TEMPOTRON — supervised spike-TIMING classification (one leaky neuron)")
    print("=" * 62)
    print(f"{n} afferents, {2 * k} spatiotemporal patterns ({k} fire / {k} silent), T={T:.0f}ms")
    print(f"accuracy: {acc0:.0%} (before) -> {acc1:.0%} (after 100 epochs)")
    print(f"peak voltage: fire class avg {sum(vpos) / len(vpos):.2f} vs silent {sum(vneg) / len(vneg):.2f} "
          f"(threshold {tp.thr:.2f})")
    print("=" * 62)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert acc1 >= 0.9, f"tempotron must learn to separate the two timing classes ({acc1:.2f})"
    assert acc1 > acc0, f"training must improve over the initial guess ({acc0:.2f} -> {acc1:.2f})"
    assert sum(vpos) / len(vpos) > tp.thr > sum(vneg) / len(vneg), \
        "learned peak voltage must separate fire vs silent classes across threshold"
    assert Tempotron(n, seed=0).w == Tempotron(n, seed=0).w, "same seed -> identical init (deterministic)"
    print(f"self-check OK: learned {acc0:.0%}->{acc1:.0%} on spike-timing patterns; "
          f"voltage margin {sum(vpos) / len(vpos):.2f}>{tp.thr:.2f}>{sum(vneg) / len(vneg):.2f}; deterministic")


if __name__ == "__main__":
    main()

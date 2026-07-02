"""
Homeostatic synaptic scaling (Turrigiano) — firing-rate stability for the learning stores
(plasticity/stability axis). Pure-stdlib.

Every learning rule in the repo is Hebbian-flavoured — valence_stdp, tempotron, delay_learning all
push weights UP on correlated activity, and nothing pushes back. Pure Hebbian potentiation is
unstable: weights (and firing rate) run away or fall silent. Biology's fix is homeostatic synaptic
scaling (Turrigiano 1998): the neuron measures its own average rate and MULTIPLICATIVELY rescales
all its synapses toward a set-point. Multiplicative is the crucial part -- it drags the rate back to
target while preserving the RELATIVE synaptic pattern Hebbian learning carved (unlike subtractive
normalization, which distorts it). That is the missing stabilizer that keeps the STDP stores usable
over long runs. Refs: Turrigiano et al., "Activity-dependent scaling of quantal amplitude", Nature
391 (1998); Turrigiano, "The self-tuning neuron", Cell 135 (2008).

  Run: python homeostatic_scaling.py        # (also run by run_selfchecks.py)
"""
import random


class HomeostaticNeuron:
    """Hebbian potentiation + multiplicative synaptic scaling toward a rate set-point."""

    def __init__(self, n, target=1.0, hebb_lr=0.02, scale_rate=0.5, seed=0):
        rng = random.Random(seed)
        self.w = [rng.uniform(0.4, 0.6) for _ in range(n)]
        self.target, self.hebb_lr, self.scale_rate = target, hebb_lr, scale_rate

    def output(self, x):
        return sum(w * xi for w, xi in zip(self.w, x))

    def hebb(self, x):
        y = self.output(x)
        for i in range(len(self.w)):
            self.w[i] += self.hebb_lr * x[i] * y      # correlated activity -> potentiation

    def scale(self, x):
        y = self.output(x)
        if y > 0:
            f = (self.target / y) ** self.scale_rate   # multiplicative: same factor on every synapse
            self.w = [w * f for w in self.w]

    def step(self, x, homeostasis=True):
        self.hebb(x)
        if homeostasis:
            self.scale(x)
        return self.output(x)


def main():
    rng = random.Random(1)
    n = 8
    x = [rng.uniform(0.2, 1.0) for _ in range(n)]      # fixed input-rate pattern (varied strengths)

    hn = HomeostaticNeuron(n, target=1.0, seed=0)
    for _ in range(300):
        hn.step(x, homeostasis=True)
    conv, wmax = hn.output(x), max(hn.w)

    free = HomeostaticNeuron(n, target=1.0, seed=0)    # same start, Hebbian only (no stabilizer)
    for _ in range(300):
        free.step(x, homeostasis=False)
    runaway, wmax_free = free.output(x), max(free.w)

    # multiplicative scaling preserves EVERY pairwise weight ratio (the learned pattern)
    ratios_before = [w / hn.w[0] for w in hn.w]
    hn.scale(x)
    ratios_after = [w / hn.w[0] for w in hn.w]
    max_ratio_drift = max(abs(a - b) for a, b in zip(ratios_before, ratios_after))

    print("=" * 62)
    print("HOMEOSTATIC SYNAPTIC SCALING (Turrigiano) — rate-stable learning")
    print("=" * 62)
    print(f"{n} synapses, target rate {hn.target:.1f}, 300 steps")
    print(f"with scaling    : rate {conv:.2f} (-> target), max weight {wmax:.2f} (bounded)")
    print(f"Hebbian only    : rate {runaway:.1f} (runaway), max weight {wmax_free:.1f}")
    print(f"ratio drift under a scale step: {max_ratio_drift:.1e} (multiplicative = pattern preserved)")
    print("=" * 62)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert abs(conv - hn.target) < 0.15, f"scaling must hold the rate near its set-point ({conv:.3f})"
    assert runaway > 5 * hn.target, f"Hebbian alone must run away without homeostasis ({runaway:.1f})"
    assert wmax < wmax_free, f"homeostatic weights must stay bounded, far below runaway ({wmax:.2f} vs {wmax_free:.1f})"
    assert max_ratio_drift < 1e-9, "multiplicative scaling must preserve all relative weight ratios"
    assert argmax(hn.w) == argmax(x), "learned selectivity tracks the strongest input (Hebbian shaping)"
    assert HomeostaticNeuron(n, seed=0).w == HomeostaticNeuron(n, seed=0).w, "same seed -> identical init"
    print(f"self-check OK: rate {conv:.2f}~target vs runaway {runaway:.0f}; ratios preserved "
          f"({max_ratio_drift:.0e}); selectivity kept; deterministic")


def argmax(seq):
    return max(range(len(seq)), key=lambda i: seq[i])


if __name__ == "__main__":
    main()

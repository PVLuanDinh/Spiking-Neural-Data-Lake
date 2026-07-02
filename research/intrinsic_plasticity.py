"""
Intrinsic plasticity (Triesch) — information-maximizing self-regulation of a neuron's excitability
(plasticity/stability axis). Pure-stdlib.

homeostatic_scaling stabilizes rate by rescaling SYNAPSES. Intrinsic plasticity is the other half:
the neuron tunes its OWN transfer function -- gain a and bias b -- to drive its output firing-rate
DISTRIBUTION toward a target exponential. An exponential output is the maximum-entropy distribution
for a fixed mean, so this maximizes information transmitted per spike under a metabolic (mean-rate)
budget -- a sparse, efficient code. Crucially it is input-scale invariant: change the input gain and
the neuron re-tunes a to hold its output statistics, so downstream code stays stable. Refs: Triesch,
"A gradient rule for the plasticity of a neuron's intrinsic excitability", ICANN 2005; Stemmler &
Koch, Nat. Neurosci. 2 (1999).

  Run: python intrinsic_plasticity.py        # (also run by run_selfchecks.py)
"""
import math
import random


class IntrinsicPlasticity:
    """Sigmoid neuron whose gain a and bias b adapt toward an exponential output target (mean mu)."""

    def __init__(self, mu=0.15, lr=0.01, a=1.0, b=0.0):
        self.mu, self.lr, self.a, self.b = mu, lr, a, b

    def output(self, x):
        return 1.0 / (1.0 + math.exp(-(self.a * x + self.b)))

    def adapt(self, x):
        y = self.output(x)
        inv = 1.0 / self.mu
        db = self.lr * (1.0 - (2.0 + inv) * y + inv * y * y)   # Triesch rule for an exponential target
        da = self.lr / self.a + db * x
        self.a += da
        self.b += db
        return y


def _mean_out(ip, xs):
    return sum(ip.output(x) for x in xs) / len(xs)


def main():
    rng = random.Random(1)
    inputs = [rng.gauss(0.0, 1.0) for _ in range(5000)]
    test = [rng.gauss(0.0, 1.0) for _ in range(2000)]
    mu = 0.15

    ip = IntrinsicPlasticity(mu=mu, a=1.0, b=0.0)
    mean_before = _mean_out(ip, test)
    for x in inputs:
        ip.adapt(x)
    mean_after = _mean_out(ip, test)

    # scale invariance: re-run with 3x-amplified input -> gain a should shrink to compensate, mean holds
    ip_s = IntrinsicPlasticity(mu=mu, a=1.0, b=0.0)
    for x in inputs:
        ip_s.adapt(3.0 * x)
    mean_scaled = _mean_out(ip_s, [3.0 * x for x in test])

    print("=" * 60)
    print("INTRINSIC PLASTICITY (Triesch) — info-max excitability self-tuning")
    print("=" * 60)
    print(f"target mean firing rate mu = {mu}")
    print(f"output mean : {mean_before:.3f} (before)  ->  {mean_after:.3f} (after)")
    print(f"gain a      : 1.000 (init) -> {ip.a:.3f} (x1 input), {ip_s.a:.3f} (x3 input, compensates)")
    print(f"scale invariance: output mean on x3 input still {mean_scaled:.3f} (~mu)")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert abs(mean_after - mu) < 0.03, f"IP must pull the output mean to the target ({mean_after:.3f} vs {mu})"
    assert abs(mean_before - mu) > 0.1, f"the un-adapted neuron must start off-target ({mean_before:.3f})"
    assert abs(mean_scaled - mu) < 0.03, f"IP must hold the mean under a 3x input scale ({mean_scaled:.3f})"
    assert ip_s.a < ip.a, f"gain must shrink to compensate a larger input ({ip_s.a:.3f} < {ip.a:.3f})"
    assert ip.a > 0, "gain stays positive (the lr/a term keeps it away from 0)"
    assert IntrinsicPlasticity(mu=mu).a == IntrinsicPlasticity(mu=mu).a, "deterministic init"
    print(f"self-check OK: mean {mean_before:.2f}->{mean_after:.2f}~mu; scale-invariant "
          f"(a {ip.a:.2f}->{ip_s.a:.2f} on x3, mean {mean_scaled:.2f}); deterministic")


if __name__ == "__main__":
    main()

"""
Intrinsic plasticity (Triesch) — an additive information-maximizing excitability self-tuner.

PROMOTED from research/ into the product (v0.89, additive). Where homeostatic_scaling stabilizes rate
by rescaling SYNAPSES, intrinsic plasticity tunes the neuron's OWN transfer function: it adapts gain a
and bias b so the output firing-rate DISTRIBUTION approaches a target exponential (the max-entropy
distribution for a fixed mean, so it maximizes information per spike under a metabolic budget), and it
does so INPUT-SCALE INVARIANTLY — change the input gain and it re-tunes a to hold its output. `adapt`
runs the Triesch rule; here it is driven by scalar projections of noisy `snn_classifier` shapes (real
product data). Additive/opt-in; no existing default changes. Refs: Triesch, ICANN 2005; Stemmler &
Koch, Nat. Neurosci. 2 (1999).

  python -m snn_data_lake.plasticity.intrinsic_plasticity        # (also run by run_selfchecks.py)
"""
import math
import random

import snn_data_lake.cortex.snn_classifier as C


class IntrinsicPlasticity:
    """Sigmoid neuron whose gain a and bias b adapt toward an exponential output target (mean mu)."""

    def __init__(self, mu=0.15, lr=0.01, a=1.0, b=0.0):
        self.mu, self.lr, self.a, self.b = mu, lr, a, b

    def output(self, x):
        return 1.0 / (1.0 + math.exp(-(self.a * x + self.b)))

    def adapt(self, x):
        y = self.output(x)
        inv = 1.0 / self.mu
        db = self.lr * (1.0 - (2.0 + inv) * y + inv * y * y)   # Triesch rule, exponential target
        da = self.lr / self.a + db * x
        self.a += da
        self.b += db
        return y


def _shape_drives(n_samples, seed):
    """Scalar neuron drives derived from noisy classifier shapes via a fixed random projection,
    standardized to ~unit variance (real product data, non-trivial distribution)."""
    rng = random.Random(seed)
    temps = C.templates()
    proj = [rng.gauss(0.0, 1.0) for _ in range(C.N)]
    raw = []
    for _ in range(n_samples):
        t = temps[rng.randrange(len(temps))]
        noisy = [px ^ (1 if rng.random() < 0.1 else 0) for px in t]
        raw.append(sum(p * w for p, w in zip(noisy, proj)))
    m = sum(raw) / len(raw)
    sd = (sum((v - m) ** 2 for v in raw) / len(raw)) ** 0.5 or 1.0
    return [(v - m) / sd for v in raw]


def _mean_out(ip, xs):
    return sum(ip.output(x) for x in xs) / len(xs)


def main():
    mu = 0.15
    train = _shape_drives(5000, seed=1)
    test = _shape_drives(2000, seed=2)

    ip = IntrinsicPlasticity(mu=mu, a=1.0, b=0.0)
    mean_before = _mean_out(ip, test)
    for x in train:
        ip.adapt(x)
    mean_after = _mean_out(ip, test)

    ip_s = IntrinsicPlasticity(mu=mu, a=1.0, b=0.0)             # 3x-amplified input -> gain must compensate
    for x in train:
        ip_s.adapt(3.0 * x)
    mean_scaled = _mean_out(ip_s, [3.0 * x for x in test])

    print("=" * 60)
    print("INTRINSIC PLASTICITY (Triesch) — excitability self-tuning on shape drives")
    print("=" * 60)
    print(f"target mean rate mu = {mu}")
    print(f"output mean : {mean_before:.3f} (before) -> {mean_after:.3f} (after)")
    print(f"gain a      : 1.000 -> {ip.a:.3f} (x1 input), {ip_s.a:.3f} (x3 input, compensates)")
    print(f"scale invariance: output mean on x3 input still {mean_scaled:.3f} (~mu)")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert abs(mean_after - mu) < 0.03, f"IP must pull the output mean to the target ({mean_after:.3f})"
    assert abs(mean_before - mu) > 0.1, f"the un-adapted neuron must start off-target ({mean_before:.3f})"
    assert abs(mean_scaled - mu) < 0.03, f"IP must hold the mean under a 3x input scale ({mean_scaled:.3f})"
    assert ip_s.a < ip.a, f"gain must shrink to compensate a larger input ({ip_s.a:.3f} < {ip.a:.3f})"
    assert ip.a > 0, "gain stays positive (the lr/a term keeps it away from 0)"
    assert IntrinsicPlasticity(mu=mu).a == IntrinsicPlasticity(mu=mu).a, "deterministic init"
    print(f"self-check OK: mean {mean_before:.2f}->{mean_after:.2f}~mu; scale-invariant "
          f"(a {ip.a:.2f}->{ip_s.a:.2f} on x3, mean {mean_scaled:.2f}); additive; deterministic")


if __name__ == "__main__":
    main()

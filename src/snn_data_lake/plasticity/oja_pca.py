"""
Oja's rule -- unsupervised principal-component extraction in one neuron (numpy-backed).

A plain Hebbian synapse (w += lr * y * x, y = w.x) grows without bound; Oja (1982) added a decay term that
normalizes the weight on the fly -- w += lr * y * (x - y * w) -- so a single linear neuron's weight vector
converges to the TOP PRINCIPAL COMPONENT of its input stream (the max-variance direction), staying unit-norm
without any explicit renormalization. Stacked with Gram-Schmidt deflation (Sanger's rule) the same local update
extracts the top-k components in order. This is the neuromorphic route to unsupervised feature learning /
whitening: a self-organizing dimensionality reducer built only from local Hebbian plasticity, distinct from
bcm_plasticity (which learns selectivity, not variance directions). Deterministic.
On-thesis with bcm_plasticity / intrinsic_plasticity / population_encoding.

  python -m snn_data_lake.plasticity.oja_pca    # self-check; also run by run_selfchecks.py
"""

import math
import random

import numpy as np

from snn_data_lake.substrate._mathutils import dot, unit

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "OjaNeuron.learn": (
        "LOCAL",
        "Oja's rule: Hebbian product minus a decay proportional to the squared output, which "
        "normalises the weight vector locally. No error is propagated from anywhere",
    ),
}


class OjaNeuron:
    """Linear neuron whose weight converges to the top principal component via Oja's local rule.

    NUMPY-BACKED since 2026-08-04 (P7 redesign, applied BEFORE promotion into `plasticity/`).

    WHAT THE ZERO-DEP RULE COST HERE, and why this one is a real compromise unlike the scalar
    neuron models: `self.w` was a Python list and the update ran as a list comprehension over
    `zip(self.w, x)`. The live consumer is numpy -- `cortex/percept_surprise.py` holds a numpy
    residual and had to call `.tolist()` purely to DEMOTE it so this class could eat it, then
    re-promote afterwards. That is a conversion per sample, at `in_dim=2312`, in both directions,
    existing only to satisfy a rule abolished 2026-07-20. numpy is an unconditional core dependency
    and 2312 elements is squarely a vector op, so the dependency genuinely buys something -- which
    is the P7 test, not whether the docstring says "stdlib".

    NUMERICS ARE NOT BIT-IDENTICAL, and that is disclosed rather than discovered later. The old path
    accumulated strictly left-to-right via `_mathutils.dot`; numpy sums pairwise. The measured effect
    on the consumer's four reported statistics is recorded in the promotion note in
    `src/snn_data_lake/__init__.py`. `unit()` is deliberately KEPT for the seeded init so the
    starting weight vector is byte-identical to the old one -- only the update arithmetic moves.
    """

    def __init__(self, dim, lr=0.01, seed=0):
        rng = random.Random(seed)
        # `unit()` (stdlib, left-to-right) retained on purpose: identical seeded start vector.
        self.w = np.asarray(unit([rng.uniform(-1, 1) for _ in range(dim)]), dtype=float)
        self.lr = lr

    def learn(self, x):
        xv = np.asarray(x, dtype=float)
        y = float(self.w @ xv)
        self.w = self.w + self.lr * y * (xv - y * self.w)  # Oja: Hebb + self-normalizing decay
        return y

    def run(self, data, epochs=1):
        for _ in range(epochs):
            for x in data:
                self.learn(x)
        return self.w


def sanger(data, k, dim, lr=0.01, epochs=6, seed=0):
    """Sanger's rule (Generalized Hebbian): k Oja neurons + deflation -> the top-k components in order."""
    rng = random.Random(seed)
    w = [unit([rng.uniform(-1, 1) for _ in range(dim)]) for _ in range(k)]
    for _ in range(epochs):
        for x in data:
            y = [dot(w[j], x) for j in range(k)]
            for j in range(k):
                deflate = [
                    sum(y[m] * w[m][i] for m in range(j + 1)) for i in range(dim)
                ]  # subtract higher PCs
                w[j] = [w[j][i] + lr * y[j] * (x[i] - deflate[i]) for i in range(dim)]
    return w


def _abs_cos(a, b):
    return abs(dot(unit(a), unit(b)))


def main():
    rng = random.Random(1)
    dim = 8
    e1 = unit([1.0, 0.6, -0.3, 0.2, -0.8, 0.1, 0.4, -0.5])  # planted dominant variance direction
    e2 = unit([-0.5, 0.4, 0.7, -0.1, 0.2, 0.3, -0.6, 0.1])
    e2 = unit([e2[i] - dot(e2, e1) * e1[i] for i in range(dim)])  # orthogonalize the 2nd direction

    # samples: big variance along e1, medium along e2, small isotropic noise
    data = []
    for _ in range(400):
        a, b = rng.gauss(0, 3.0), rng.gauss(0, 1.2)
        data.append([a * e1[i] + b * e2[i] + rng.gauss(0, 0.25) for i in range(dim)])

    oja = OjaNeuron(dim, lr=0.01, seed=0)
    w = oja.run(data, epochs=8)
    cos1 = _abs_cos(w, e1)
    norm = math.sqrt(dot(w, w))

    comps = sanger(data, k=2, dim=dim, lr=0.01, epochs=8, seed=0)
    cos_top = _abs_cos(comps[0], e1)
    cos_2nd = _abs_cos(comps[1], e2)

    print("=" * 60)
    print("OJA'S RULE -- unsupervised top principal component in one neuron")
    print("=" * 60)
    print(f"data: {len(data)} samples in R^{dim}; planted variance e1 >> e2 >> noise")
    print(f"Oja neuron : |cos(w, e1)| = {cos1:.3f}, ||w|| = {norm:.3f} (self-normalized)")
    print(f"Sanger k=2 : PC1.e1 = {cos_top:.3f}, PC2.e2 = {cos_2nd:.3f} (top-2 in order)")
    print("=" * 60)

    # ---- self-check: deterministic asserts ----
    assert cos1 > 0.95, f"Oja neuron must converge to the top principal component ({cos1:.3f})"
    assert 0.9 < norm < 1.1, f"Oja must self-normalize the weight to ~unit norm ({norm:.3f})"
    assert cos_top > 0.95 and cos_2nd > 0.9, (
        f"Sanger must extract PC1,PC2 in order ({cos_top:.3f}, {cos_2nd:.3f})"
    )
    # `==` on two ndarrays is ELEMENTWISE and raises "truth value is ambiguous" under assert.
    # This was a list comparison before the 2026-08-04 numpy redesign; np.array_equal is the
    # same assertion, and still bit-exact rather than approximate.
    assert np.array_equal(OjaNeuron(dim, seed=0).run(data, epochs=8), w), (
        "same data -> identical weight (deterministic)"
    )
    print(
        f"self-check OK: Oja neuron -> top PC (|cos| {cos1:.2f}, ||w|| {norm:.2f}); Sanger -> top-2 in order "
        f"({cos_top:.2f}, {cos_2nd:.2f}); deterministic"
    )


if __name__ == "__main__":
    main()

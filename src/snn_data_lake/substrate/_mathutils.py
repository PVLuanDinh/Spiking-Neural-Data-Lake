"""
Shared math helpers, extracted verbatim from duplicated copies that
lived across the core package and the research modules. Every function here is
byte-for-byte equivalent (same formula, same numeric behavior) to the canonical
implementation it was lifted from, so importing it in place of a local copy
changes no result.

Canonical sources:
  softmax  <- snn_classifier.softmax           (max-shift, numerically stable)
  pearson  <- adaptive_spiking.correlation     (Pearson correlation, no guard)
  mean     <- forward_model._mean              (arithmetic mean, sum / len)
  sigmoid  <- neural_sampling._sigmoid         (numerically stable logistic)
  dot      <- oja_pca._dot / lca_sparse_coding._dot
  unit     <- oja_pca._unit                    (L2-normalize, math.sqrt form)

Copies whose formula DIFFERS were deliberately left in place (not merged):
  - adaptive_spiking keeps the public name `correlation` (thin wrapper here).
  - synfire_chain._mean has an empty-guard (`... if xs else 0.0`).
  - signal_loop._mean is a column-wise VECTOR mean (different shape).
  - wilson_cowan._sigmoid is a 3-arg parameterized logistic.
  - burst_multiplexing._pearson / stdp_wta_bayes.pearson use zero-denominator
    guards, so they are NOT byte-equivalent to `pearson` here.
  - lca_sparse_coding._unit uses `** 0.5` rather than math.sqrt.

ASCII-only, deterministic.
"""

import math


def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def pearson(a, b):
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((v - ma) ** 2 for v in a)
    vb = sum((v - mb) ** 2 for v in b)
    return cov / math.sqrt(va * vb)


def mean(v):
    return sum(v) / len(v)


def sigmoid(x):
    """Numerically stable logistic 1 / (1 + exp(-x))."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(v):
    n = math.sqrt(sum(c * c for c in v)) or 1.0
    return [c / n for c in v]


def main():
    # softmax: sums to 1, order-preserving, shift-invariant (max-shift stability)
    p = softmax([1.0, 2.0, 3.0])
    assert abs(sum(p) - 1.0) < 1e-12
    assert p[2] > p[1] > p[0]
    assert all(abs(x - y) < 1e-12 for x, y in zip(softmax([0.0, 1.0]), softmax([10.0, 11.0])))

    # pearson: identical -> 1, exact anti-correlation -> -1
    assert abs(pearson([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-12
    assert abs(pearson([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) + 1.0) < 1e-12

    # mean: arithmetic mean
    assert mean([2.0, 4.0, 6.0]) == 4.0

    # sigmoid: 0 -> 0.5, symmetric, stable (no OverflowError) at extremes
    assert sigmoid(0.0) == 0.5
    assert abs(sigmoid(2.0) + sigmoid(-2.0) - 1.0) < 1e-12
    assert 0.0 < sigmoid(-40.0) < 1e-16
    assert sigmoid(-1000.0) == 0.0 and sigmoid(1000.0) == 1.0

    # dot: 1*4 + 2*5 + 3*6 = 32
    assert dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 32.0

    # unit: L2 norm == 1; zero vector -> zero (guarded denominator)
    u = unit([3.0, 4.0])
    assert abs(math.sqrt(sum(c * c for c in u)) - 1.0) < 1e-12
    assert unit([0.0, 0.0]) == [0.0, 0.0]

    print("_mathutils OK: softmax pearson mean sigmoid dot unit")


if __name__ == "__main__":
    main()

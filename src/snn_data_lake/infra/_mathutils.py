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
  Rng      <- trained_brain._Rng / spiking_lm_serve._Rng   (64-bit LCG; the two were byte-equivalent
              in formula and differed only in docstring, so this is the first case above rather than
              the second -- merged 2026-08-15, both originals now import it)
  q95      <- attention_schema._q95            (the 95th-percentile ORDER STATISTIC; lifted
              2026-09-15 so a LIVE module can import it without making that EXEMPT instrument
              reachable -- the dormancy-gate argument `instruments/paired_contrast` records too)

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


# consolidates: src/snn_data_lake/instruments/attention_schema.py (merge) -- its `_q95` becomes an
#   alias of this one in the same change, exactly as `clamp01` below did for eight copies.
def q95(draws):
    """The 95th percentile of `draws` as an ORDER STATISTIC, never an interpolation: the
    element at round(0.95 * (len - 1)) of the sorted sample. Sorts `draws` IN PLACE, which
    every caller in this tree already relies on.

    LIFTED 2026-09-15 (specs/022 T079) from `instruments/attention_schema._q95`, itself T078's
    restatement of `faculties/metacognitive_monitor`'s `track_error` convention. Moved rather
    than copied for a MEASURED reason: `agency_monitor` is a LIVE module (autonomous_loop
    imports it) and attention_schema is an EXEMPT instrument, so a live module importing it
    -- even lazily, inside a function -- makes `gates/dormancy_gate` name that exemption
    stale and the gate exits 1 (run 2026-09-15: "1 EXEMPT entries are now reachable
    (retire them): ['attention_schema']"). The statistic therefore lives where any module
    may import it, and the instruments keep their exemptions -- the same argument, and the
    same fix, as `instruments/paired_contrast`'s own lift.

    ONE convention for every floor and every band that uses it; the formula is reproduced
    exactly, so no consumer's number moves."""
    draws.sort()
    return draws[min(len(draws) - 1, int(round(0.95 * (len(draws) - 1))))]


# consolidates: src/snn_data_lake/faculties/self_model.py (merge), src/snn_data_lake/faculties/wholistic_loop.py (merge), src/snn_data_lake/faculties/affect_integrator.py (merge), src/snn_data_lake/faculties/agency_monitor.py (merge), src/snn_data_lake/faculties/metacognitive_monitor.py (merge), src/snn_data_lake/sensorimotor/recognizer_bank.py (merge), src/snn_data_lake/substrate/stream_sketch_panel.py (merge), src/snn_data_lake/faculties/second_order_access.py (merge)
def clamp01(x):
    """Clamp a scalar into [0,1], always returning a float.

    MERGED 2026-08-15 from EIGHT copies across cortex/, neuromodulation/ and substrate/. A static
    AST comparison called them four different formulas; running them showed only TWO behavioural
    mechanisms -- a type-preserving pass-through (5 copies) and this float-coercing one (3) --
    identical on every float input and differing ONLY when handed an int or a bool.

    Which is why the merge needed no aesthetic tiebreak: instrumented over the eight modules' own
    self-checks plus 40 brain ticks, `_clamp01` took **160,460 arguments and every one was already
    a float** (0 int, 0 bool). The two mechanisms are indistinguishable in this tree, so exactly one
    survives and nothing changes. The coercing form is the survivor because it is strictly the more
    defined of the two for a caller that does not exist yet.

    (An AST pass can only ever prove "certainly the same"; its "differs" is a hypothesis to read.
    `min(1,max(0,x))` and `max(0,min(1,x))` have different trees and agree on every input.)"""
    x = float(x)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


# consolidates: src/snn_data_lake/sensorimotor/trained_brain.py (merge), src/snn_data_lake/sensorimotor/spiking_lm_serve.py (merge)
# Both originals lose their local `_Rng` in this same change and import this one instead. A merge
# that leaves both paths reachable is additive with extra steps -- the failure P2 exists to prevent.
class Rng:
    """Deterministic 64-bit LCG. Self-contained, so a caller never touches `random`'s global state.

    Merged 2026-08-15 from two byte-equivalent `_Rng` copies in CORE cortex modules. Proven
    equivalent BEFORE either was deleted, per P2's honesty guard: 5 seeds x 500 draws identical
    between the two copies, with a negative control confirming a different seed moves the stream.
    The XOR constant, multiplier, increment and 53-bit shift are reproduced exactly -- any edit to
    them changes every consumer's stream, which is a determinism change, not a refactor."""

    def __init__(self, seed):
        self.s = (seed ^ 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF

    def random(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (self.s >> 11) / float(1 << 53)


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

    # q95: the ORDER STATISTIC at round(0.95 * (n - 1)), not an interpolation. On 0..99 that
    # index is 94, so the answer is the ELEMENT 94.0 -- an interpolating percentile would say
    # 94.05. The red: 1..100 (no zero) must read 95.0, so a fencepost slip is visible.
    assert q95(list(range(100))) == 94.0
    assert q95([float(v) for v in range(1, 101)]) == 95.0
    assert q95([3.0]) == 3.0 and q95([2.0, 1.0]) == 2.0  # short samples, and it sorts
    assert q95([0.0] * 50) == 0.0

    print("_mathutils OK: softmax pearson mean sigmoid dot unit q95")


if __name__ == "__main__":
    main()

"""
Streaming quantile -- the P-square (P2) estimator: any percentile of a stream in O(1) memory (pure stdlib).

The lake keeps several fixed-memory questions running over a flood of values; this one estimates an ORDER
STATISTIC -- the p-quantile, the value below which a fraction p of the data falls (the p50 median, the p99 tail).
The naive way stores every observation and sorts (O(N) memory, exact). P2 estimates the SAME quantile while
NEVER storing the observations: it keeps exactly FIVE markers -- five (height q_i, position n_i) pairs that ride
the empirical CDF at 0, p/2, p, (1+p)/2, 1 -- and slides them with a parabolic prediction as each value arrives.
Memory is CONSTANT (5 markers) whether the stream is 5 values or 100 million; the trade is a small transient
error before the markers converge and an approximation thereafter, in exchange for never holding the data.

The P2 algorithm (Jain & Chlamtac, "The P-square Algorithm for Dynamic Calculation of Quantiles and Histograms
Without Storing Observations", CACM 28:1076, 1985):
  * initialise from the first 5 observations sorted ascending -> heights q_1..q_5; positions n = [1,2,3,4,5];
    desired positions n' = [1, 1+2p, 1+4p, 3+2p, 5] with per-observation increments dn' = [0, p/2, p, (1+p)/2, 1];
  * for each later value x: find the cell k it lands in (stretch q_1 down or q_5 up on a new extreme); bump n_i
    for every marker to the right of the cell; add dn' to every desired position; then for each interior marker
    i in {2,3,4}, if it has drifted a full step from its desired position (d = n'_i - n_i, |d| >= 1) and there is
    room, move it one step using the PARABOLIC (P2) prediction -- falling back to a LINEAR step if the parabola
    would break marker order;
  * the estimate is the middle marker height q_3.
The p-quantile estimate is read straight off the middle marker; no sort, no stored sample.

DISTINCTNESS from the sibling fixed-memory sketches / the exact latency percentile (made explicit):
  * research/loop_latency.py computes EXACT p50/p95/p99 by STORING every latency sample and sorting it -- O(N)
    memory, exact, and quarantined to the latency-instrumentation experiment. THIS module is the opposite trade:
    an O(1)-memory (5 markers) APPROXIMATE estimator that NEVER stores the observations, and a CORE, general
    p-quantile estimator for any stream -- not the latency-specific exact computation. Constant memory is the
    whole point of P2.
  * frequency_moment.py (AMS F2) estimates an aggregate MOMENT sum_i f_i^2 (the distribution's skew / self-join
    size). spike_cardinality.py (HyperLogLog) estimates a distinct COUNT F0. spike_stream_sketch.py (Count-Min)
    estimates a point FREQUENCY f_i. sliding_window_count.py (DGIM) estimates a windowed COUNT. A quantile is an
    ORDER STATISTIC -- the value at rank p through the sorted data -- a different kind of statistic none of the
    sketches computes. Different question, different machinery: five CDF-riding markers with a parabolic update,
    not signed counters, leading-zero ranks, bucket histograms, or timestamp bins.

ponytail: the classic P2 5-marker single-quantile estimator (parabolic prediction with a linear fallback) on
synthetic streams -- the streaming-quantile level; a small transient error before the markers converge, one
quantile per estimator (StreamingQuantiles just wraps several independent P2 estimators), not a spiking
implementation, not cognition.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.streaming_quantile   # (also run by selfchecks)
"""
import math
import random


def _exact_quantile(sorted_vals, p):
    """Exact p-quantile of an already-sorted list (linear interpolation between ranks). Oracle / <5 fallback only."""
    n = len(sorted_vals)
    if n == 0:
        raise ValueError("no observations")
    if n == 1:
        return float(sorted_vals[0])
    rank = p * (n - 1)
    lo = int(math.floor(rank))
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def _parabolic(q, n, i, d):
    """P2 parabolic prediction of the new height of marker i moved one step in direction d (+1/-1)."""
    return q[i] + d * (
        (n[i] - n[i - 1] + d) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
        + (n[i + 1] - n[i] - d) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
    ) / (n[i + 1] - n[i - 1])


def _linear(q, n, i, d):
    """Linear fallback prediction when the parabola would push marker i out of order."""
    return q[i] + d * (q[i + d] - q[i]) / (n[i + d] - n[i])


class P2Quantile:
    """P-square single-quantile estimator: 5 markers, O(1) memory, never stores the observations.

    add(x) offers one value (returns the current estimate once >= 5 have been seen, else None); add_all(xs) folds
    in an iterable; quantile() returns the current p-quantile estimate (the exact quantile of what has been seen
    while fewer than 5 observations exist). n_seen counts observations; n_markers is always 5. Deterministic:
    identical p and identical stream -> identical estimate; no RNG inside."""

    n_markers = 5

    def __init__(self, p):
        if not 0.0 < p < 1.0:
            raise ValueError("p must be strictly between 0 and 1")
        self.p = float(p)
        self.reset()

    def reset(self):
        """Forget everything -- start a fresh estimate over a new stream."""
        self._buf = []          # holds the first <5 observations until the markers are initialised
        self._q = None          # marker heights q_1..q_5 (observed values)
        self._np = None         # marker positions n_i (integer counts)
        self._npd = None        # desired marker positions n'_i (floats)
        self._dnp = None        # per-observation increments of the desired positions
        self._n_seen = 0

    def _initialize(self):
        """Set the 5 markers from the first 5 observations sorted ascending."""
        s = sorted(self._buf)
        p = self.p
        self._q = [float(v) for v in s]
        self._np = [1, 2, 3, 4, 5]
        self._npd = [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0]
        self._dnp = [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0]
        self._buf = None

    def _update(self, x):
        """Fold one observation into the 5 markers (the P2 bookkeeping)."""
        q, n, npd, dnp = self._q, self._np, self._npd, self._dnp
        # B1: find the cell k that x lands in; stretch the end markers on a new extreme.
        if x < q[0]:
            q[0] = x
            k = 0
        elif x >= q[4]:
            q[4] = x
            k = 3
        else:
            k = 3
            for i in range(4):
                if q[i] <= x < q[i + 1]:
                    k = i
                    break
        # B2: bump the actual positions of every marker to the right of the cell; advance all desired positions.
        for i in range(k + 1, 5):
            n[i] += 1
        for i in range(5):
            npd[i] += dnp[i]
        # B3: move each interior marker at most one step toward its desired position.
        for i in (1, 2, 3):
            d = npd[i] - n[i]
            if (d >= 1 and n[i + 1] - n[i] > 1) or (d <= -1 and n[i - 1] - n[i] < -1):
                sd = 1 if d >= 0 else -1
                cand = _parabolic(q, n, i, sd)
                if q[i - 1] < cand < q[i + 1]:
                    q[i] = cand
                else:
                    q[i] = _linear(q, n, i, sd)
                n[i] += sd

    def add(self, x):
        """Offer one observation. Returns the current estimate once >= 5 have been seen, else None."""
        x = float(x)
        self._n_seen += 1
        if self._q is None:
            self._buf.append(x)
            if len(self._buf) >= self.n_markers:
                self._initialize()
                return self._q[2]
            return None
        self._update(x)
        return self._q[2]

    def add_all(self, xs):
        """Fold every value in `xs` into the estimator. Returns self for chaining."""
        for x in xs:
            self.add(x)
        return self

    def quantile(self):
        """Current p-quantile estimate (the exact quantile of the buffer while fewer than 5 have been seen)."""
        if self._q is not None:
            return self._q[2]
        if not self._buf:
            raise ValueError("no observations yet")
        return _exact_quantile(sorted(self._buf), self.p)

    @property
    def n_seen(self):
        """Number of observations offered so far."""
        return self._n_seen


class StreamingQuantiles:
    """Track several quantiles at once -- one independent P2Quantile per p, sharing the streamed values."""

    def __init__(self, ps):
        self.estimators = {}
        for p in ps:
            self.estimators[float(p)] = P2Quantile(p)

    def add(self, x):
        """Offer one observation to every tracked quantile. Returns self for chaining."""
        for est in self.estimators.values():
            est.add(x)
        return self

    def add_all(self, xs):
        """Fold every value in `xs` into every tracked quantile. Returns self for chaining."""
        for x in xs:
            self.add(x)
        return self

    def quantile(self, p):
        """Current estimate of the p-quantile (p must have been registered at construction)."""
        return self.estimators[float(p)].quantile()

    def quantiles(self):
        """Dict of every tracked p -> its current estimate."""
        return {p: est.quantile() for p, est in self.estimators.items()}

    @property
    def n_seen(self):
        """Observations offered so far (shared across all tracked quantiles)."""
        if not self.estimators:
            return 0
        return next(iter(self.estimators.values())).n_seen


def main():
    n = 100000
    acc_bound = 0.03          # central/upper quantiles of a light-tailed stream converge tight
    tail_bound = 0.05         # a heavy-tailed p99 sits on a sparse tail -> a slightly looser but honest bound

    # ---- 1. accuracy vs the EXACT quantile: a seeded normal-ish positive stream (mean 100, sd 15 -> all values
    # positive so relative error is well-defined), several p, P2 estimate vs the sorted-sample oracle. ----
    rng = random.Random(20260706)
    sample = [rng.gauss(100.0, 15.0) for _ in range(n)]
    oracle = sorted(sample)                                  # O(N) oracle: allowed for the TEST, NOT the algorithm
    ps = [0.5, 0.9, 0.95, 0.99]
    mq = StreamingQuantiles(ps).add_all(sample)
    rows = []
    max_rel = 0.0
    for p in ps:
        exact = _exact_quantile(oracle, p)
        est = mq.quantile(p)
        rel = abs(est - exact) / abs(exact)
        max_rel = max(max_rel, rel)
        rows.append((p, exact, est, rel))

    # ---- 2. O(1) memory: estimate the p99 of N observations with exactly 5 markers (the oracle stores all N). ----
    p99_est = P2Quantile(0.99)
    p99_est.add_all(sample)
    n_markers = p99_est.n_markers

    # ---- 3. skewed / heavy-tailed stream: an exponential (mean 1.0, median ~0.693 -- mean and median differ a
    # lot, long right tail); P2 p50/p90/p99 must all track their exact values (not fooled by skew). ----
    rng2 = random.Random(777)
    exp_sample = [rng2.expovariate(1.0) for _ in range(n)]
    exp_oracle = sorted(exp_sample)
    exp_mean = sum(exp_sample) / n
    exp_ps = [0.5, 0.9, 0.99]
    emq = StreamingQuantiles(exp_ps).add_all(exp_sample)
    exp_rows = []
    for p in exp_ps:
        exact = _exact_quantile(exp_oracle, p)
        est = emq.quantile(p)
        rel = abs(est - exact) / abs(exact)
        exp_rows.append((p, exact, est, rel))

    # ---- 4. determinism: same p / seeded stream -> identical estimate across two fresh instances. ----
    rngd = random.Random(4242)
    det_stream = [rngd.gauss(50.0, 5.0) for _ in range(20000)]
    det_a = P2Quantile(0.95).add_all(det_stream).quantile()
    det_b = P2Quantile(0.95).add_all(det_stream).quantile()

    # exact quantile of a tiny (<5) buffer, for the fewer-than-5 path
    small = P2Quantile(0.5)
    small.add_all([3.0, 1.0, 2.0])
    small_med = small.quantile()

    print("=" * 72)
    print("STREAMING QUANTILE -- P-square (P2) estimator: 5 markers, O(1) memory")
    print("=" * 72)
    print(f"stream: N = {n} seeded observations; estimator holds n_markers = {n_markers} markers (never stored)")
    print("accuracy vs exact (normal mean=100 sd=15):")
    for p, exact, est, rel in rows:
        tag = "p" + str(int(round(p * 100)))
        print(f"  {tag:>4} : exact {exact:10.4f}  ->  P2 estimate {est:10.4f}   (rel err {rel * 100:5.2f}%)")
    print(f"  worst-case relative error = {max_rel * 100:.2f}%  (bound {acc_bound * 100:.0f}%)")
    print(f"O(1) memory: p99 of N = {n} observations kept in {n_markers} markers "
          f"(exact oracle stores all {n})")
    print(f"skewed exponential (mean {exp_mean:.3f} vs median ~0.693 -- mean >> median, long tail):")
    for p, exact, est, rel in exp_rows:
        tag = "p" + str(int(round(p * 100)))
        print(f"  {tag:>4} : exact {exact:8.4f}  ->  P2 estimate {est:8.4f}   (rel err {rel * 100:5.2f}%)")
    print(f"determinism: two fresh P2Quantile(0.95) on the same seeded stream -> identical: {det_a == det_b}")
    print(f"fewer-than-5 path: exact median of [3,1,2] = {small_med:.1f}")
    print("=" * 72)

    # ---- asserts (deterministic, zero-dep; genuine estimates vs a real sorted oracle, nothing rigged) ----
    for p, _exact, _est, rel in rows:
        assert rel < acc_bound, f"P2 p{int(round(p*100))} off by {rel*100:.2f}% (> {acc_bound*100:.0f}% bound)"
    assert n_markers == 5, f"P2 must hold exactly 5 markers, not {n_markers}"
    assert p99_est.n_seen == n, "estimator must have consumed the whole stream"
    for p, _exact, _est, rel in exp_rows:
        assert rel < tail_bound, f"skewed p{int(round(p*100))} off by {rel*100:.2f}% (> {tail_bound*100:.0f}%)"
    assert exp_mean > 1.4 * exp_oracle[n // 2], "sanity: exponential mean must sit well above its own median"
    assert det_a == det_b, "same p / seeded stream must give an identical estimate"
    assert small_med == 2.0, f"exact median of [3,1,2] must be 2.0, got {small_med}"
    print(f"self-check OK: {len(ps)} quantiles within {max_rel*100:.2f}% of exact using {n_markers} markers over "
          f"N={n}; skewed p50/p90/p99 all track; deterministic")


if __name__ == "__main__":
    main()

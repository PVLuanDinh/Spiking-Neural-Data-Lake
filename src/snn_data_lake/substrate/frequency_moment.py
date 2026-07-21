"""
Frequency-moment sketch -- AMS "tug-of-war" estimate of F2 = sum_i f_i^2 in sublinear memory.

The streaming lake asks several fixed-memory questions about a flood of spike motifs; each has its own sketch,
and this one estimates a DIFFERENT statistic from all of them: the SECOND FREQUENCY MOMENT
  F2 = sum_i f_i^2                                          (sum over distinct motifs i of its count squared)
the "surprise number" -- a single number for the whole frequency distribution's SKEW / self-join size. F2 is
SMALL when the stream is uniform (many motifs each seen a little) and LARGE when it is skewed (a few heavy hitters
dominate); it equals the size of a self-join of the stream with itself. It is neither a point frequency, nor a
distinct count, nor a windowed count.

The AMS sketch (Alon, Matias & Szegedy, "The Space Complexity of Approximating the Frequency Moments", STOC 1996 /
JCSS 58:137, 1999 -- Godel Prize 2005) estimates F2 with signed "tug-of-war" counters:
  * hash each motif deterministically to a sign s(motif) in {+1, -1};
  * keep a running counter X = sum over the stream of s(motif) = sum_i f_i * s_i;
  * then X^2 is an UNBIASED estimator of F2 -- expanding X^2 = sum_i f_i^2 s_i^2 + sum_{i!=j} f_i f_j s_i s_j, the
    diagonal is sum_i f_i^2 = F2 (since s_i^2 = 1) and every cross term has mean 0 (the signs are ~independent and
    zero-mean), so E[X^2] = F2. A single counter is unbiased but high-variance, so reduce variance the standard
    way: keep `d` independent counters (different sign hashes) and AVERAGE their X^2 -> one group mean; keep `g`
    such groups and take the MEDIAN of the g means (classic median-of-means: the mean cuts variance, the median
    gives the (epsilon, delta) tail). Total memory = d*g counters, INDEPENDENT of the number of distinct motifs --
    an exact F2 needs one counter per distinct motif; this needs a fixed d*g. Deterministic: the sign is a hashlib
    digest bit with the seed folded into the bytes (NEVER the builtin hash(), which is per-process salted).

DISTINCTNESS from the sibling fixed-memory sketches (made explicit):
  * spike_stream_sketch.py (Count-Min) estimates a POINT frequency f_i of one individual heavy-hitter motif
    (an F_infinity-ish per-item query). spike_cardinality.py (HyperLogLog) estimates F0 = the number of DISTINCT
    motifs. sliding_window_count.py (DGIM) estimates a windowed COUNT of events in the last W ticks. THIS module
    estimates the AGGREGATE second moment F2 = sum_i f_i^2 -- one number summarizing the whole distribution's skew
    / self-join size ("surprise number"), which none of the others computes. Different statistic, different sketch:
    signed tug-of-war counters + median-of-means, not per-item counters, leading-zero ranks, or bucket histograms.

The AMS tug-of-war F2 estimator with median-of-means, using hashlib pseudo-random signs (a demonstration
-- NOT a provably 4-wise-independent hash family), F2 only (no general F_k) -- the frequency-moment-sketch level;
not a spiking implementation, not cognition.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.substrate.frequency_moment   # (also run by selfchecks)
"""

import hashlib

_MASK64 = (1 << 64) - 1


def _payload(item):
    """Stable UTF-8 bytes for a motif. Sets are order-independent; no builtin hash() (per-process salted)."""
    if isinstance(item, (set, frozenset)):
        return ",".join(str(x) for x in sorted(item)).encode("utf-8")
    if isinstance(item, (list, tuple)):
        return ",".join(str(x) for x in item).encode("utf-8")
    return str(item).encode("utf-8")


def _median(vals):
    """Median of a list (average of the two middle values when the length is even)."""
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return 0.5 * (s[mid - 1] + s[mid])


class AMSSketch:
    """AMS tug-of-war F2 sketch: d*g signed counters -> median-of-means of X^2 estimates F2 = sum_i f_i^2.

    Counters are laid out flat as g groups of d counters; counter k uses sign bit k of a deterministic hashlib
    digest of the motif (seed folded in). estimate() averages X^2 within each group (variance reduction) and takes
    the median across groups (the (epsilon, delta) tail). Memory = d*g counters, independent of distinct-motif count."""

    def __init__(self, d=16, g=9, seed=0):
        if d < 1 or g < 1:
            raise ValueError("d and g must both be >= 1")
        self.d = int(d)
        self.g = int(g)
        self.seed = int(seed)
        self._n = self.d * self.g  # total counters = the memory footprint
        self.reset()

    def reset(self):
        """Zero every tug-of-war counter -- start a fresh estimate over a new stream."""
        self.X = [0] * self._n

    def _signs(self, item):
        """d*g signs in {+1,-1} for one motif: successive bits of blake2b(seed || block || payload)."""
        payload = _payload(item)
        seed = self.seed & _MASK64
        n = self._n
        bits = []
        block = 0
        while len(bits) < n:
            data = seed.to_bytes(8, "big") + block.to_bytes(4, "big") + payload
            h = int.from_bytes(hashlib.blake2b(data, digest_size=32).digest(), "big")
            for k in range(256):
                bits.append(1 if (h >> k) & 1 else -1)
                if len(bits) >= n:
                    break
            block += 1
        return bits

    def add(self, item):
        """Fold one motif into every tug-of-war counter: X[k] += s_k(item)."""
        signs = self._signs(item)
        x = self.X
        for k in range(self._n):
            x[k] += signs[k]

    def add_all(self, iterable):
        """Add every motif in `iterable`. Returns self for chaining."""
        for item in iterable:
            self.add(item)
        return self

    def estimate(self):
        """Median-of-means of X^2: average X^2 within each group of d, then take the median of the g group means."""
        d, g, x = self.d, self.g, self.X
        means = []
        for j in range(g):
            base = j * d
            s = 0
            for k in range(base, base + d):
                s += x[k] * x[k]
            means.append(s / d)
        return _median(means)

    @property
    def n_counters(self):
        """Total counters held (= d*g) -- the memory, independent of the number of DISTINCT motifs."""
        return self._n


def exact_f2(items):
    """Ground-truth F2 = sum_i f_i^2 from a real frequency table (self-check oracle only, NOT part of the sketch)."""
    freq = {}
    for item in items:
        freq[item] = freq.get(item, 0) + 1
    return sum(f * f for f in freq.values())


def _lcg(state):
    """Tiny deterministic LCG (Numerical Recipes constants) -> next state; avoids importing random for the demo."""
    return (state * 1664525 + 1013904223) & _MASK64


def main():
    d, g = 16, 15  # 240 counters; large enough that median-of-means lands cleanly
    seed = 7
    acc_bound = 0.18  # stated relative-error bound for the accuracy test

    # ---- 1. accuracy vs EXACT F2: a skewed stream (heavy hitters + a long light tail), exact F2 from a real
    # frequency table, AMS estimate within acc_bound. ----
    stream = []
    for item, reps in ((101, 900), (102, 700), (103, 500), (104, 300)):
        stream += [item] * reps  # four heavy hitters
    st = 12345
    for _ in range(30000):  # ...buried in a light tail of ~4000 distinct rare motifs
        st = _lcg(st)
        stream.append(1000 + (st >> 17) % 4000)
    st = 999
    for i in range(len(stream) - 1, 0, -1):  # deterministic Fisher-Yates shuffle (no random import)
        st = _lcg(st)
        j = st % (i + 1)
        stream[i], stream[j] = stream[j], stream[i]

    f2_true = exact_f2(stream)
    sk = AMSSketch(d=d, g=g, seed=seed).add_all(stream)
    f2_est = sk.estimate()
    rel = abs(f2_est - f2_true) / f2_true
    n_distinct = len({x for x in stream})

    # ---- 2. sublinear memory: many thousands of distinct motifs estimated with a few hundred counters (an exact
    # F2 needs one counter per distinct motif). ----
    mem_stream = []
    st = 555
    for _ in range(30000):
        st = _lcg(st)
        mem_stream.append(20000 + (st >> 13) % 9000)  # ~9000 distinct motifs
    mem_distinct = len({x for x in mem_stream})
    mem_f2_true = exact_f2(mem_stream)
    mem_sk = AMSSketch(d=d, g=g, seed=seed).add_all(mem_stream)
    mem_f2_est = mem_sk.estimate()
    mem_rel = abs(mem_f2_est - mem_f2_true) / mem_f2_true

    # ---- 3. skew sensitivity (the MEANING of F2): two streams of the SAME length -- one near-uniform, one heavily
    # skewed -- AMS F2 must be much LARGER for the skewed one (tracking the exact F2s). ----
    length = 30000
    uni = []  # near-uniform: 6000 motifs each seen ~5 times -> small F2
    st = 4242
    for _ in range(length):
        st = _lcg(st)
        uni.append(50000 + (st >> 11) % 6000)
    skew = [70001] * 9000 + [70002] * 8000 + [70003] * 7000  # three heavy hitters -> large F2
    st = 313
    for _ in range(length - len(skew)):
        st = _lcg(st)
        skew.append(80000 + (st >> 11) % 6000)  # small light tail to reach the same total length
    uni_true, skew_true = exact_f2(uni), exact_f2(skew)
    uni_est = AMSSketch(d=d, g=g, seed=seed).add_all(uni).estimate()
    skew_est = AMSSketch(d=d, g=g, seed=seed).add_all(skew).estimate()

    # ---- 4. determinism: same d/g/seed/stream -> identical estimate across two fresh instances. ----
    det_stream = stream[:4000]
    det_a = AMSSketch(d=d, g=g, seed=seed).add_all(det_stream).estimate()
    det_b = AMSSketch(d=d, g=g, seed=seed).add_all(det_stream).estimate()

    print("=" * 70)
    print("FREQUENCY-MOMENT SKETCH -- AMS tug-of-war F2 = sum f_i^2 (median-of-means)")
    print("=" * 70)
    print(
        f"sketch: d={d} counters/group, g={g} groups, n_counters = d*g = {sk.n_counters}; seed={seed}"
    )
    print(
        f"accuracy : exact F2 = {f2_true} vs estimate {f2_est:.0f}  (rel err {rel * 100:.2f}%, "
        f"bound {acc_bound * 100:.0f}%) over {n_distinct} distinct in {sk.n_counters} counters"
    )
    print(
        f"sublinear: {mem_sk.n_counters} counters estimate F2 over {mem_distinct} distinct motifs "
        f"(exact F2 needs {mem_distinct} counters)"
    )
    print(
        f"           exact F2 = {mem_f2_true} vs estimate {mem_f2_est:.0f} (rel err {mem_rel * 100:.2f}%)"
    )
    print(f"skew     : uniform  exact F2 = {uni_true:>10} -> estimate {uni_est:>12.0f}")
    print(
        f"           skewed   exact F2 = {skew_true:>10} -> estimate {skew_est:>12.0f}  "
        f"(skewed est / uniform est = {skew_est / uni_est:.0f}x)"
    )
    print(
        f"determinism: two fresh instances, same d/g/seed/stream -> identical estimate: {det_a == det_b}"
    )
    print("=" * 70)

    # ---- asserts: deterministic; genuine estimates vs a real F2 oracle, nothing rigged ----
    assert rel < acc_bound, (
        f"AMS F2 estimate off by {rel * 100:.2f}% (> {acc_bound * 100:.0f}% bound)"
    )
    assert sk.n_counters < n_distinct, (
        f"sketch must be sublinear in distinct motifs ({sk.n_counters} vs {n_distinct})"
    )
    assert mem_sk.n_counters < mem_distinct, (
        f"memory {mem_sk.n_counters} must be below distinct {mem_distinct}"
    )
    assert mem_rel < acc_bound, (
        f"sublinear-memory F2 estimate off by {mem_rel * 100:.2f}% (> bound)"
    )
    assert skew_true > 20 * uni_true, (
        "exact F2 must itself be far larger for the skewed stream (sanity on the oracle)"
    )
    assert skew_est > 10 * uni_est, (
        f"AMS F2 must be much larger for the skewed stream ({skew_est:.0f} vs {uni_est:.0f})"
    )
    assert abs(uni_est - uni_true) / uni_true < 0.30, (
        f"uniform F2 estimate must track its exact ({uni_est:.0f})"
    )
    assert abs(skew_est - skew_true) / skew_true < acc_bound, (
        f"skewed F2 estimate must track its exact ({skew_est:.0f})"
    )
    assert det_a == det_b, "same d/g/seed/stream must give an identical estimate"
    print(
        f"self-check OK: F2 within {rel * 100:.2f}% of exact ({f2_est:.0f} vs {f2_true}) using {sk.n_counters} "
        f"counters over {n_distinct} distinct; skewed F2 {skew_est / uni_est:.0f}x the uniform; deterministic"
    )


if __name__ == "__main__":
    main()

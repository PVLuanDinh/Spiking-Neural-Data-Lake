"""
Spike-cardinality sketch -- HyperLogLog distinct-motif counting in fixed tiny memory (pure stdlib).

The streaming lake ingests an unbounded flood of spike motifs. Three DIFFERENT questions recur, each answered
by its own fixed-memory sketch:
  * "how OFTEN does each motif occur?"  -> FREQUENCY  -> spike_stream_sketch.py (Count-Min + top-K heavy hitters)
  * "have I seen THIS exact motif before?" -> MEMBERSHIP -> research/spike_bloom.py (Bloom filter)
  * "how MANY DISTINCT motifs has the stream contained?" -> CARDINALITY -> THIS module (HyperLogLog).
Cardinality is neither frequency nor membership: it is the size of the distinct set. HyperLogLog (Flajolet, Fusy,
Gandouet & Meunier 2007) estimates it in memory that is FIXED regardless of the count -- m = 2^p one-byte
registers -- with a relative standard error of ~1.04/sqrt(m). Each item is hashed to a wide integer; the first p
bits pick a register j, the position of the leftmost 1-bit in the rest is a rank rho, and register[j] keeps the
running max rho. Rare long runs of leading zeros are a proxy for how many distinct items were hashed, so the
harmonic-mean estimate E = alpha_m * m^2 / sum(2^-register) recovers the cardinality; a linear-counting
correction E = m * ln(m / V) takes over at low load (V = empty registers). Deterministic (hashlib, seed folded
into the hashed bytes; NEVER the builtin hash(), which is per-process salted).

Contrast with the sibling sketches, made explicit:
  * spike_stream_sketch is a Count-Min sketch of COUNTERS -- it is duplicate-DRIVEN (every occurrence bumps a
    counter) and reports how often the heavy hitters fire. THIS sketch is duplicate-INSENSITIVE: re-adding a motif
    a million times leaves the registers (and the estimate) unchanged, because rho is a property of the item's
    hash, not of how many times it arrived. It counts the distinct set, not the traffic.
  * research/spike_bloom answers a yes/no membership query about ONE motif; it cannot tell you the set's size.
    HyperLogLog answers only the set's size and cannot test membership of any single motif. Together the three
    (frequency / membership / cardinality) cover the standard streaming-analytics questions in fixed memory.

Fixed-memory analytics family: like spike_stream_sketch and research/spike_bloom this is a zero-dep, read-only
primitive that fits streaming_hub's live log -- "how many distinct motifs have streamed?" in O(m) memory that is
independent of the count and carries to the documented scale-out path, not a saving on the local index.

A standard HyperLogLog with the linear-counting small-range correction only (no 64-bit/large-range
bias correction, no sparse representation) over synthetic motif ids -- the cardinality-sketch level; not a
spiking implementation, not cognition. (A 64-bit hash puts the large-range regime near ~10^8 distinct items,
far above anything the lake counts here, so that correction is deliberately omitted.)

  python -m snn_data_lake.substrate.spike_cardinality        # (also run by run_selfchecks.py)
"""
import hashlib
import math

_BITS = 64                                   # width of the hash used for index + rank extraction
_MASK64 = (1 << _BITS) - 1


def _hash64(item, seed):
    """Deterministic 64-bit hash of a motif. Uses hashlib (NOT builtin hash(), which is per-process salted)."""
    if isinstance(item, (set, frozenset)):
        payload = ",".join(str(x) for x in sorted(item))     # order-independent for a set of active features
    elif isinstance(item, (list, tuple)):
        payload = ",".join(str(x) for x in item)
    else:
        payload = str(item)
    data = (seed & _MASK64).to_bytes(8, "big") + payload.encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def _rho(w, width):
    """Rank = leading zeros + 1 = 1-based position of the leftmost set bit in the low `width` bits; width+1 if 0."""
    if w == 0:
        return width + 1
    return width - w.bit_length() + 1


def _alpha_m(m):
    """Bias-correction constant for m registers (small-m table, else the m>=128 approximation)."""
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


class HyperLogLog:
    """Cardinality sketch: m = 2^p one-byte registers of leading-zero ranks. Estimates DISTINCT count in O(m)."""

    def __init__(self, p=11, seed=0):
        if not 4 <= p <= 16:
            raise ValueError("p must be in 4..16 (m = 2^p registers)")
        self.p = p
        self.m = 1 << p
        self.seed = seed
        self.registers = bytearray(self.m)             # each byte holds a rank rho (max (64-p)+1 <= 61)
        self._tail = _BITS - p                          # bits left for the rank after the p index bits
        self._alpha = _alpha_m(self.m)

    def add(self, item):
        """Fold one motif into the sketch: top p bits pick the register, keep the max leading-zero rank there."""
        h = _hash64(item, self.seed)
        j = h >> self._tail                             # top p bits = register index
        w = h & ((1 << self._tail) - 1)                 # remaining bits carry the rank
        rho = _rho(w, self._tail)
        if rho > self.registers[j]:
            self.registers[j] = rho

    def add_all(self, iterable):
        """Add every item in `iterable`. Returns self for chaining."""
        for item in iterable:
            self.add(item)
        return self

    def count(self):
        """Estimated number of DISTINCT items added (harmonic-mean estimate + small-range linear counting)."""
        m = self.m
        sum_inv = 0.0
        zeros = 0
        for r in self.registers:
            sum_inv += 2.0 ** (-r)
            if r == 0:
                zeros += 1
        est = self._alpha * m * m / sum_inv
        if est <= 2.5 * m and zeros > 0:                # low load: harmonic estimate is biased -> linear counting
            est = m * math.log(m / zeros)
        return est

    def __len__(self):
        return int(round(self.count()))

    def merge(self, other):
        """Register-wise max -> a new HLL over the UNION of both streams. Requires identical p and seed."""
        if self.p != other.p:
            raise ValueError("cannot merge HLLs with different precision p")
        if self.seed != other.seed:
            raise ValueError("cannot merge HLLs with different seed (registers use different hashes)")
        out = HyperLogLog(p=self.p, seed=self.seed)
        for i in range(self.m):
            out.registers[i] = max(self.registers[i], other.registers[i])
        return out

    @property
    def relative_error(self):
        """Theoretical relative standard error of the estimate: 1.04 / sqrt(m)."""
        return 1.04 / math.sqrt(self.m)


def main():
    p = 14
    m = 1 << p
    se = 1.04 / math.sqrt(m)                             # theoretical relative std error
    bound = 3.0 * se

    # ---- 1. accuracy across scales: distinct N with duplicates mixed in, estimate within 3 sigma ----
    scales = (100, 1000, 10000, 100000)
    hlls, rows = {}, []
    for n in scales:
        hll = HyperLogLog(p=p, seed=0)
        for i in range(n):
            hll.add(("motif", i))
            if i % 4 == 0:
                hll.add(("motif", i))                   # duplicates in the stream (must not change the count)
        hlls[n] = hll
        est = hll.count()
        rows.append((n, est, abs(est - n) / n))

    # ---- 2. duplicate-insensitive: re-adding the same items many more times must not move the estimate ----
    n_dup = 5000
    dup = HyperLogLog(p=p, seed=0)
    dup.add_all(("dup", i) for i in range(n_dup))
    before = dup.count()
    for _round in range(20):
        dup.add_all(("dup", i) for i in range(n_dup))   # same distinct items, 20x more occurrences
    after = dup.count()

    # ---- 3. fixed tiny memory: same m registers for N=100 and N=100000 (a real id-set would grow with N) ----
    small = HyperLogLog(p=p, seed=0).add_all(("motif", i) for i in range(100))
    big = hlls[100000]
    max_n = max(scales)

    # ---- 4. merge = union: two overlapping sets, merged estimate approximates the true union cardinality ----
    a = HyperLogLog(p=p, seed=0).add_all(("u", i) for i in range(0, 30000))
    b = HyperLogLog(p=p, seed=0).add_all(("u", i) for i in range(20000, 50000))
    union = a.merge(b)
    union_truth = 50000                                 # |[0,30000) U [20000,50000)| = |[0,50000)|
    union_est = union.count()
    union_rel = abs(union_est - union_truth) / union_truth

    # ---- 5. determinism: same p/seed/items -> identical registers + identical estimate, fresh instances ----
    x1 = HyperLogLog(p=p, seed=0).add_all(("d", i) for i in range(2000))
    x2 = HyperLogLog(p=p, seed=0).add_all(("d", i) for i in range(2000))
    deterministic = (x1.registers == x2.registers) and (x1.count() == x2.count())

    print("=" * 64)
    print("SPIKE-CARDINALITY SKETCH -- HyperLogLog distinct-motif counting")
    print("=" * 64)
    print(f"registers m = 2^{p} = {m} bytes; std error 1.04/sqrt(m) = {se * 100:.2f}%; "
          f"3 sigma bound {bound * 100:.2f}%")
    print(f"{'true N':>9}  ->  {'estimate':>12}   {'rel err':>8}")
    for n, est, rel in rows:
        print(f"{n:>9}  ->  {est:>12.1f}   {rel * 100:>7.2f}%")
    print(f"duplicate-insensitive: {n_dup} distinct, est before {before:.1f} -> after {after:.1f} "
          f"(each item then added 20x more)")
    print(f"fixed memory: m = {len(small.registers)} registers for N=100 AND N={max_n} "
          f"(a real id-set would need {max_n} entries)")
    print(f"merge = union: est {union_est:.1f} vs true {union_truth} (rel err {union_rel * 100:.2f}%)")
    print(f"determinism: two fresh instances -> identical registers + estimate: {deterministic}")
    print("=" * 64)

    # ---- asserts: deterministic, zero-dep; genuine estimates, not rigged ----
    for n, _est, rel in rows:
        assert rel < bound, f"N={n}: rel err {rel * 100:.2f}% exceeds 3 sigma bound {bound * 100:.2f}%"
    assert after == before, f"cardinality must ignore duplicates ({before} -> {after})"
    assert len(small.registers) == m == len(big.registers), "register count must be exactly m regardless of N"
    assert m < max_n, f"memory m={m} must be independent of (and below) the largest cardinality {max_n}"
    assert union_rel < bound, f"merged union estimate {union_est:.1f} off by {union_rel * 100:.2f}% (> bound)"
    assert deterministic, "same p/seed/items must give identical registers and estimate"
    print(f"self-check OK: distinct counted within {bound * 100:.2f}% across N in {list(scales)} using {m} bytes; "
          f"duplicate-insensitive; merge=union; deterministic")


if __name__ == "__main__":
    main()

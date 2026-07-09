"""
Morris approximate counter -- count n events in a ~log2 log2 n -bit register (pure stdlib).

The lake often needs ONE running total -- "how many events so far?" -- kept in almost no memory
(per-channel tallies, firmware-grade counters). An exact counter for n events needs ~log2(n)
bits; the Morris counter (Morris, "Counting Large Numbers of Events in Small Registers", CACM
21:840, 1978 -- the FIRST probabilistic counting algorithm, the seed of the whole streaming-
sketch field) needs only ~log2 log2 n: a byte counts past a billion. Keep a small log-register
c; on each event increment c only with probability a^-c (base a > 1, classically a = 2), so
increments get exponentially rarer as c grows and c only ever reaches ~log_a(n). The estimate
(a^c - 1)/(a - 1) is exactly unbiased -- E[a^c] = n*(a - 1) + 1; for a = 2 that is the classic
2^c - 1 with E[2^c] = n + 1. A single register is high-variance (relative std error
~ sqrt((a-1)/2), i.e. ~71% at a = 2); BOTH standard reductions are implemented: average m
independent registers (rel std error ~ sqrt((a-1)/(2m))) and/or a base a closer to 1 (rel std
error ~ sqrt((a-1)/2) shrinks, register grows slightly). The self-check uses m = 64 averaged
base-2 registers. add(k) fast-forwards k events per register by sampling the geometric gap to
the next probabilistic increment (memoryless -> exactly the same distribution as k coin flips,
in O(increments) draws instead of O(k)). Deterministic: one random.Random(seed) drives all
registers; same m/a/seed and the same call sequence -> identical estimate.

Contrast with the sibling counting sketches, made explicit -- four DIFFERENT questions:
  * spike_cardinality (HyperLogLog) counts the number of DISTINCT items, F0: "how many UNIQUE
    motifs?" -- duplicate-insensitive leading-zero-rank registers.
  * spike_stream_sketch (Count-Min) counts the per-item FREQUENCY of each of many items: "how
    often is EACH motif?" -- a d x w table of hashed counters.
  * sliding_window_count (DGIM) counts events in a sliding WINDOW: "how many in the last W
    ticks?" -- an exponential bucket histogram, recency-scoped.
  * THIS counts ONE total -- how many times a single event has happened, over all time -- in a
    register EXPONENTIALLY smaller than the count itself, via a probabilistic log-counter: not
    distinct-rank registers, not per-item counters, not bucket histograms. It is the historical
    ROOT of the whole sketch family (1978): the different question is "what is this one big
    count, approximately, in almost no memory?".

The classic Morris probabilistic counter (base-a increment probability a^-c, m
averaged registers) counting synthetic events -- the approximate-counting level; no Morris+ /
Csuros bias-corrected variants, not a spiking implementation, not cognition.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.substrate.morris_counter   # selfchecks
"""
import math
import random


class MorrisCounter:
    """Approximate event counter: m log-registers; bump c with prob a^-c; estimate (a^c-1)/(a-1).

    Averaging the m independent registers cuts the relative std error to ~sqrt((a-1)/(2m)); a
    base a closer to 1 cuts it further at the cost of a slightly larger register. No RNG state
    outside the instance -- fully deterministic given (m, a, seed) and the call sequence."""

    def __init__(self, m=64, a=2.0, seed=0):
        if m < 1:
            raise ValueError("m must be >= 1 (number of averaged registers)")
        if not a > 1.0:
            raise ValueError("base a must be > 1 (a -> 1 degenerates to an exact unit counter)")
        self.m = int(m)
        self.a = float(a)
        self.seed = seed
        self.reset()

    def reset(self):
        """Zero every register and rewind the RNG -- a reset instance replays like a fresh one."""
        self.registers = [0] * self.m
        self._rng = random.Random(self.seed)

    def increment(self):
        """Offer ONE event to every register (register at c increments with probability a^-c)."""
        self.add(1)

    def add(self, k=1):
        """Offer k events to every register.

        Fast-forward: instead of k coin flips per register, sample the geometric gap to the next
        probabilistic increment (inverse CDF). The geometric is memoryless, so discarding the
        final overshooting gap is exact -- same distribution as k Bernoulli(a^-c) trials, in
        O(number of increments) draws instead of O(k)."""
        if k < 0:
            raise ValueError("k must be >= 0")
        a = self.a
        rng = self._rng
        for i in range(self.m):
            c = self.registers[i]
            remaining = k
            while remaining > 0:
                p = a ** (-c)
                if p >= 1.0:
                    gap = 1                    # c == 0: the first event always increments
                else:
                    u = rng.random()           # gap ~ Geometric(p) on {1, 2, ...}
                    gap = max(1, math.ceil(math.log1p(-u) / math.log1p(-p)))
                if gap > remaining:
                    break                      # no increment within these k events (memoryless)
                remaining -= gap
                c += 1
            self.registers[i] = c

    def count(self):
        """Unbiased estimate of the events offered: mean over registers of (a^c - 1)/(a - 1)."""
        a = self.a
        total = sum((a ** c - 1.0) / (a - 1.0) for c in self.registers)
        return total / self.m

    def register_bits(self):
        """Bits the LARGEST register currently needs (max c stored in c.bit_length() bits).

        This is the whole point: c ~ log_a(n), so the register needs ~log2 log2 n bits while an
        exact counter needs ~log2 n -- exponentially smaller than the count it tracks."""
        return max(self.registers).bit_length()

    @property
    def relative_error(self):
        """Theoretical relative standard error of the averaged estimate: sqrt((a-1)/(2m))."""
        return math.sqrt((self.a - 1.0) / (2.0 * self.m))


def main():
    m, a, seed = 64, 2.0, 0
    se = math.sqrt((a - 1.0) / (2.0 * m))    # rel std error of the m-averaged estimate (8.84%)
    bound = 0.15                             # ~1.7 sigma at m=64 -- must hold at the fixed seed

    # ---- 1. accuracy across orders of magnitude: count a KNOWN n, estimate within the bound.
    # n=1000 goes through increment() one event at a time (exercises the per-event path);
    # the larger n use add(n) (exercises the geometric fast-forward -- same distribution). ----
    scales = (1000, 100000, 1000000)
    rows, counters = [], {}
    for n in scales:
        mc = MorrisCounter(m=m, a=a, seed=seed)
        if n <= 1000:
            for _ in range(n):
                mc.increment()
            path = "increment() x n"
        else:
            mc.add(n)
            path = "add(n) fast-forward"
        est = mc.count()
        rows.append((n, est, abs(est - n) / n, path))
        counters[n] = mc

    # ---- 2. tiny memory (the point): after n = 1,000,000 events each register sits at
    # c ~ log2(n) ~ 20, so it needs ~log2 log2 n ~ 5 bits -- vs ~20 bits for an exact counter.
    n_big = max(scales)
    big = counters[n_big]
    max_c = max(big.registers)
    bits = big.register_bits()
    exact_bits = n_big.bit_length()          # bits an exact counter needs for the same n
    loglog = math.log2(math.log2(n_big))

    # ---- 3. unbiasedness / no systematic drift: mean estimate over R independent runs (fixed
    # seeds 1..R) of the same n must be close to n -- unbiased, not lucky on one seed. ----
    n_unbias, runs = 100000, 200
    total = 0.0
    for s in range(1, runs + 1):
        rmc = MorrisCounter(m=m, a=a, seed=s)
        rmc.add(n_unbias)
        total += rmc.count()
    mean_est = total / runs
    mean_rel = abs(mean_est - n_unbias) / n_unbias
    mean_se = se / math.sqrt(runs)           # rel std error of the R-run mean (~0.63%)

    # ---- 4. determinism: same m/a/seed/n -> identical registers + estimate, fresh instances ----
    d1 = MorrisCounter(m=m, a=a, seed=7)
    d1.add(31337)
    d2 = MorrisCounter(m=m, a=a, seed=7)
    d2.add(31337)
    deterministic = (d1.registers == d2.registers) and (d1.count() == d2.count())

    print("=" * 68)
    print("MORRIS APPROXIMATE COUNTER -- one big total in ~log2 log2 n bits")
    print("=" * 68)
    print(f"config: m = {m} averaged base-a registers, a = {a}; rel std error "
          f"sqrt((a-1)/(2m)) = {se * 100:.2f}%; bound {bound * 100:.0f}%")
    print(f"{'true n':>9}  ->  {'estimate':>12}   {'rel err':>8}   path")
    for n, est, rel, path in rows:
        print(f"{n:>9}  ->  {est:>12.1f}   {rel * 100:>7.2f}%   {path}")
    print(f"memory     : n = {n_big} -> largest register c = {max_c} (~log2 n = "
          f"{math.log2(n_big):.1f}) -> register_bits() = {bits} bits "
          f"(~log2 log2 n = {loglog:.1f}) vs exact counter {exact_bits} bits")
    print(f"unbiased   : mean estimate over {runs} runs (seeds 1..{runs}) of n = {n_unbias} "
          f"-> {mean_est:.1f} (rel {mean_rel * 100:.2f}%; mean's own std ~ {mean_se * 100:.2f}%)")
    print(f"determinism: two fresh instances, same m/a/seed/n -> identical estimate: "
          f"{deterministic}")
    print("=" * 68)

    # ---- asserts: deterministic, zero-dep; genuine estimates, nothing rigged ----
    for n, _est, rel, _path in rows:
        assert rel < bound, f"n={n}: rel err {rel * 100:.2f}% exceeds bound {bound * 100:.0f}%"
    assert 15 <= max_c <= 28, f"register c={max_c} should sit near log2(n)~{math.log2(n_big):.0f}"
    assert bits <= 6, f"register needs {bits} bits, expected ~log2 log2 n ~ {loglog:.1f} (<= 6)"
    assert 3 * bits < exact_bits, (
        f"register ({bits} bits) must be far below an exact counter ({exact_bits} bits)")
    assert mean_rel < 0.03, (
        f"mean over {runs} runs {mean_est:.1f} drifts {mean_rel * 100:.2f}% from n={n_unbias} "
        f"(estimator must be unbiased)")
    assert deterministic, "same m/a/seed/n must give identical registers and estimate"
    print(f"self-check OK: n in {list(scales)} each within {bound * 100:.0f}% using {m} "
          f"registers of <= {bits} bits (exact needs {exact_bits}); mean over {runs} runs "
          f"within {mean_rel * 100:.2f}% of n; deterministic")


if __name__ == "__main__":
    main()

"""
Sliding-window spike counting -- DGIM approximate "1s in the last W ticks" in O(log^2 W) memory (pure stdlib).

The streaming lake sees a binary event stream: each tick either a spike fired (1) or it did not (0). A recurring
question is recency-scoped -- "how many spikes in the last W ticks?" -- and it must be answered continuously while
the stream runs, without ever storing W bits. A full ring buffer answers it EXACTLY but costs O(W) memory; DGIM
(Datar, Gionis, Indyk & Motwani, "Maintaining Stream Statistics over Sliding Windows", SIAM J Comput 31:1794,
2002) answers it APPROXIMATELY in O(log^2 W) memory -- exponentially less -- and NEVER stores the events at all.

The trick is an exponential histogram of buckets. Each bucket carries a timestamp and a size that is a power of 2
(the number of 1s it aggregates); sizes are non-decreasing into the past, with at most k buckets of any one size.
On each tick: advance time t, then EXPIRE the oldest bucket if its timestamp fell out of the window (<= t - W). If
the bit is 1, add a fresh size-1 bucket at t and CASCADE-MERGE -- whenever more than k buckets share a size, merge
the two OLDEST of that size into one bucket of double size (timestamp = the more-recent of the two), which may in
turn overflow the next size up. The query estimate = (sum of the sizes of every bucket except the oldest still in
the window) + (half the size of that oldest bucket): the oldest bucket straddles the window boundary, so half of
it is its expected in-window contribution. Keeping at least k-1 buckets of every size below the oldest bounds the
only source of error (that one straddling bucket) to a guaranteed relative <= 1/(k-1) -- k=2 recovers the classic
<=50% DGIM bound, larger k is tighter -- while total memory stays O(k * log^2 W) bits. Deterministic: the counter
holds no RNG; any demo stream is seeded OUTSIDE via random.Random(seed).

Contrast with the sibling streaming primitives, made explicit:
  * streaming_hub answers windowed queries by STORING every event in a per-channel index and binary-searching the
    window boundaries -- O(N) memory, EXACT. THIS module answers "how many events in the last W?" in O(log^2 W)
    memory that NEVER stores the events -- an approximate, sublinear-memory windowed count (memory-vs-exactness
    trade). Different tool for a different budget.
  * spike_cardinality (HyperLogLog) counts DISTINCT motifs over the WHOLE stream; spike_stream_sketch (Count-Min)
    counts per-item FREQUENCY over the whole stream; reservoir_sampling keeps a uniform SAMPLE of the whole stream.
    None of those is windowed. DGIM is the SLIDING-WINDOW count -- a recency-scoped statistic, the new axis: not
    "how many distinct / how often / which samples over all time" but "how many in the last W ticks, right now".

The classic DGIM exponential-histogram for a BINARY (event / no-event) stream -- the sliding-window-COUNT
level, count queries only (no sum-of-values or other windowed aggregate), timestamps kept as plain ints rather
than mod-2W. Not a spiking implementation, not cognition.

  set PYTHONUTF8=1 & set PYTHONPATH=src & python -m snn_data_lake.substrate.sliding_window_count   # (also run by selfchecks)
"""
import math
import random
from collections import deque


class SlidingWindowCounter:
    """DGIM sliding-window 1s-counter: estimate the number of 1s in the last `window` ticks in O(log^2 W) memory.

    Buckets are kept newest-first (index 0 is the most recent, smallest size); sizes are powers of 2, non-decreasing
    into the past, at most `max_buckets` (k) of each size. No RNG inside -- fully deterministic."""

    def __init__(self, window, max_buckets=2):
        if window < 1:
            raise ValueError("window must be >= 1")
        if max_buckets < 2:
            raise ValueError("max_buckets (k) must be >= 2 (the error bound is 1/(k-1))")
        self.window = int(window)
        self.k = int(max_buckets)
        self.reset()

    def reset(self):
        """Drop all buckets and rewind time -- start a fresh count over a new stream."""
        self.buckets = []          # list of [timestamp, size]; index 0 = newest (largest t, smallest size)
        self.t = 0                 # current tick (advances by 1 on every add)

    def add(self, bit):
        """Advance one tick with a 0/1 event; expire, then insert+cascade if it is a 1. Returns the estimate."""
        self.t += 1
        horizon = self.t - self.window
        while self.buckets and self.buckets[-1][0] <= horizon:   # expire buckets that slid out of the window
            self.buckets.pop()
        if bit:
            self.buckets.insert(0, [self.t, 1])                  # newest size-1 bucket at the front
            self._cascade()
        return self.count()

    def add_all(self, bits):
        """Feed every bit in `bits` (each 0 or 1). Returns self for chaining."""
        for bit in bits:
            self.add(bit)
        return self

    def _cascade(self):
        """Restore the invariant: while >k buckets share a size, merge the two OLDEST into one of double size."""
        size = 1
        while True:
            idxs = [i for i, b in enumerate(self.buckets) if b[1] == size]
            if len(idxs) <= self.k:
                break                                            # this size is legal -> nothing above it changed
            older, newer = idxs[-1], idxs[-2]                    # two oldest of this size (largest indices)
            ts = self.buckets[newer][0]                          # merged timestamp = the more-recent of the two
            del self.buckets[older]                              # delete the higher index first (keeps `newer` valid)
            del self.buckets[newer]
            self.buckets.insert(newer, [ts, size * 2])           # merged bucket is the newest of the doubled size
            size *= 2                                            # the double size may now overflow -> keep cascading

    def count(self):
        """Estimate of 1s in the last `window` ticks: sum of all in-window buckets minus half the oldest one."""
        if not self.buckets:
            return 0.0
        total = sum(b[1] for b in self.buckets)                 # all stored buckets are in-window (expired above)
        return total - self.buckets[-1][1] / 2.0                # oldest straddles the boundary -> count half of it

    @property
    def n_buckets(self):
        """Number of buckets currently held -- the memory footprint (O(k log W) buckets = O(k log^2 W) bits)."""
        return len(self.buckets)


def main():
    W = 4096
    k = 8
    bound = 1.0 / (k - 1)                # conservative DGIM guarantee; empirical worst case is ~1/(2*(k-1))
    p = 0.4                              # P(spike) per tick
    n_ticks = 20000                      # warmup W ticks to fill the window, then query every tick after
    L = int(math.log2(W))                # 12 for W=4096

    # ---- 1 & 2. accuracy within the guarantee + sublinear memory: drive a seeded binary stream, compare the
    # DGIM estimate to the EXACT last-W count from a full ring buffer (the oracle -- allowed here, NOT part of
    # the algorithm) at every tick once the window is full. ----
    rng = random.Random(12345)
    counter = SlidingWindowCounter(W, max_buckets=k)
    oracle = deque(maxlen=W)             # full W-bit buffer = ground truth (what DGIM avoids storing)
    exact_sum = 0
    rel_errors = []
    peak_buckets = 0
    for tick in range(1, n_ticks + 1):
        bit = 1 if rng.random() < p else 0
        if len(oracle) == W:
            exact_sum -= oracle[0]       # the bit about to be evicted by the maxlen deque
        oracle.append(bit)
        exact_sum += bit
        est = counter.add(bit)
        peak_buckets = max(peak_buckets, counter.n_buckets)
        if tick >= W and exact_sum > 0:                          # measure only once the window is genuinely full
            rel_errors.append(abs(est - exact_sum) / exact_sum)

    mean_rel = sum(rel_errors) / len(rel_errors)
    max_rel = max(rel_errors)
    mem_bound = k * L * L                                        # small multiple of k*log2(W)^2 (O(log^2 W))

    # ---- 3. sliding / expiry: a burst of 1s must lift the estimate, then a run of 0s longer than W slides the
    # whole burst out of the window and the estimate must decay back to ~0. ----
    burst = SlidingWindowCounter(W, max_buckets=k)
    b_ones = 1000
    burst.add_all([1] * b_ones)
    burst_est = burst.count()                                   # ~b_ones (all still in the window)
    burst.add_all([0] * W)                                      # W zeros -> every burst 1 has slid out
    after_est = burst.count()                                   # -> 0 (no in-window buckets left)

    # ---- 4. determinism: same window / k / seeded stream -> identical estimates across two fresh instances. ----
    def run(n):
        r = random.Random(999)
        c = SlidingWindowCounter(W, max_buckets=k)
        out = []
        for i in range(n):
            e = c.add(1 if r.random() < p else 0)
            if i % 500 == 0:
                out.append(round(e, 9))
        return out
    det_a, det_b = run(8000), run(8000)
    deterministic = det_a == det_b

    # ---- invariant sanity on the final histogram: sizes are powers of 2, non-decreasing into the past ----
    sizes = [b[1] for b in counter.buckets]
    powers_of_2 = all(s & (s - 1) == 0 for s in sizes)
    non_decreasing = all(sizes[i] <= sizes[i + 1] for i in range(len(sizes) - 1))

    print("=" * 68)
    print("DGIM SLIDING-WINDOW COUNT -- 1s in the last W ticks in O(log^2 W) memory")
    print("=" * 68)
    print(f"stream: {n_ticks} ticks, P(spike)={p}, W={W}, k(max buckets/size)={k}")
    print(f"error bound 1/(k-1) = {bound * 100:.2f}%  (empirical worst case ~1/(2(k-1)))")
    print(f"accuracy   : mean rel err {mean_rel * 100:.3f}%, max rel err {max_rel * 100:.3f}% "
          f"over {len(rel_errors)} queries")
    print(f"memory     : peak {peak_buckets} buckets vs W={W} bits "
          f"(~k*log2(W)={k * L}; bound k*log2(W)^2={mem_bound})")
    print(f"expiry     : burst estimate {burst_est:.1f} (~{b_ones}) -> after W zeros {after_est:.1f}")
    print(f"determinism: two fresh instances, same W/k/seeded stream -> identical estimates: {deterministic}")
    print("=" * 68)

    # ---- asserts: deterministic, zero-dep; genuine estimates vs a real oracle, nothing rigged ----
    assert max_rel <= bound, f"a query broke the DGIM guarantee: max rel err {max_rel:.4f} > 1/(k-1) {bound:.4f}"
    assert mean_rel < bound / 3, f"mean rel err {mean_rel:.4f} should be far under the bound {bound:.4f}"
    assert peak_buckets <= mem_bound, f"buckets {peak_buckets} exceed O(log^2 W) bound k*log2(W)^2={mem_bound}"
    assert peak_buckets < W // 4, f"buckets {peak_buckets} not dramatically below W={W}"
    assert powers_of_2, f"every bucket size must be a power of 2 ({sizes})"
    assert non_decreasing, f"bucket sizes must be non-decreasing into the past ({sizes})"
    assert burst_est >= 0.8 * b_ones, f"burst must lift the estimate ({burst_est:.1f} vs {b_ones})"
    assert after_est == 0.0, f"the burst must decay to 0 once slid out of the window ({after_est:.1f})"
    assert deterministic, "same W/k/seeded stream must give identical estimates"
    print(f"self-check OK: {len(rel_errors)} queries all within 1/(k-1)={bound * 100:.2f}% "
          f"(mean {mean_rel * 100:.3f}%, max {max_rel * 100:.3f}%); peak {peak_buckets} buckets << W={W}; "
          f"burst {burst_est:.0f}->{after_est:.0f} on expiry; deterministic")


if __name__ == "__main__":
    main()

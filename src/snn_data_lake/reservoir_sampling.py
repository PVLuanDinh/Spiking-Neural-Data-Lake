"""
Reservoir sampling -- a uniform random sample of k items from an UNBOUNDED stream in ONE pass (pure stdlib).

The streaming lake ingests an endless flood of spike motifs; sometimes you do not want a STATISTIC of the stream
but a fair HANDFUL of the items THEMSELVES -- k real motifs to inspect, log, or feed downstream -- chosen so that
every item the stream ever produced was equally likely to be in your hand, even though you never knew how long the
stream would be and could never hold it all. That is reservoir sampling (Knuth's Algorithm R; Vitter, "Random
Sampling with a Reservoir", ACM TOMS 11:37, 1985). Keep a reservoir of size k: the first k items fill it, then the
i-th item (1-indexed, i > k) is kept with probability k/i and, if kept, overwrites a uniformly-random reservoir
slot. After one pass over N items in O(k) memory -- never O(N) -- the reservoir is a UNIFORM sample of k of the N:
every item, early or late, ends up present with probability exactly k/N, with N never known in advance. The k/i
acceptance is the whole trick: it exactly cancels the extra chances a late item has to survive later draws, so
there is no position bias (the naive "keep the first k" or "keep the last k" both fail this). Deterministic given
a seed via random.Random(seed) -- NEVER module-level random.seed() and NEVER the per-process-salted builtin hash().

Contrast with the sibling streaming primitives, made explicit:
  * spike_cardinality (HyperLogLog) and spike_stream_sketch (Count-Min) are SKETCHES: they estimate a STATISTIC of
    the stream -- distinct COUNT / per-item FREQUENCY -- in fixed memory and retain NO items at all; you cannot pull
    a single actual motif back out of them. THIS module retains an actual UNIFORM SAMPLE of the items themselves:
    k real stream elements you can read and pass on. Sampling, not sketching.
  * experience_replay is a BOUNDED ring buffer that samples uniformly from what it has already STORED; it never
    solves the one-pass problem. Reservoir sampling is the ONE-PASS algorithm for sampling uniformly from an
    UNBOUNDED stream of unknown length WITHOUT storing it all -- the non-trivial part is precisely the k/i
    acceptance probability that keeps early and late items equally likely without ever counting to the end first.

Fixed-memory streaming family: like the sketches this is a zero-dep, read-only primitive sized in k, not in N --
"give me k fair samples of whatever streamed" in O(k) memory independent of the (unknown, unbounded) stream length.

Classic Algorithm-R reservoir sampling -- a SINGLE reservoir, uniform, UNWEIGHTED -- over synthetic ids;
the stream-sampling level. No weighted (A-Res / A-ExpJ), distributed, or time-decayed variants; not a spiking
implementation, not cognition.

  PYTHONUTF8=1 PYTHONPATH=src python -m snn_data_lake.reservoir_sampling        # (also run by run_selfchecks.py)
"""
import math
import random


class Reservoir:
    """One-pass uniform sampler: holds k items; each stream item ends up present with probability k/n_seen."""

    def __init__(self, k, seed=0):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.seed = seed
        self.reset()

    def reset(self):
        """Empty the reservoir and re-seed -- start a fresh one-pass sample of a new stream."""
        self._reservoir = []
        self.n_seen = 0                                  # items OFFERED so far (this is i, the stream position)
        self._rng = random.Random(self.seed)

    def add(self, item):
        """Offer one stream item. Returns True if it is now in the reservoir, False if it was passed over."""
        self.n_seen += 1
        i = self.n_seen                                  # 1-indexed position of THIS item
        if len(self._reservoir) < self.k:
            self._reservoir.append(item)                 # fill phase: the first k items are always kept
            return True
        j = self._rng.randrange(i)                       # j uniform in 0..i-1; P(j < k) = k/i (Algorithm R)
        if j < self.k:
            self._reservoir[j] = item                    # kept: overwrite a uniformly-random reservoir slot
            return True
        return False                                     # passed over: reservoir unchanged

    def add_all(self, iterable):
        """Offer every item in `iterable`. Returns self for chaining."""
        for item in iterable:
            self.add(item)
        return self

    def sample(self):
        """The current reservoir as a fresh list copy (a uniform sample of min(k, n_seen) items seen so far)."""
        return list(self._reservoir)

    def __len__(self):
        return len(self._reservoir)


def reservoir_sample(stream, k, seed=0):
    """Convenience: one-pass uniform sample of k items from `stream` (an iterable of unknown length)."""
    return Reservoir(k, seed=seed).add_all(stream).sample()


def main():
    N, k, runs = 50, 10, 5000
    target = k / N                                       # uniform inclusion probability every id must hit
    se = math.sqrt(target * (1.0 - target) / runs)       # std error of an inclusion frequency over `runs` runs
    band = 0.03                                          # tolerance (~5 std errors); comfortably holds at seeds 0..

    # ---- 1 & 4. UNIFORMITY (no position bias): stream ids 0..N-1 through a size-k reservoir over many seeded
    # runs; each id's empirical inclusion frequency must sit within `band` of k/N -- early and late alike. ----
    counts = [0] * N
    for seed in range(runs):
        res = Reservoir(k, seed=seed)
        res.add_all(range(N))
        for item in res.sample():
            counts[item] += 1
    freqs = [c / runs for c in counts]
    mean_freq = sum(freqs) / N                           # equals k/N exactly (exactly k ids chosen every run)
    deviations = [abs(f - target) for f in freqs]
    max_dev = max(deviations)
    worst_id = deviations.index(max_dev)
    first_freq = freqs[0]                                # FIRST item offered
    last_freq = freqs[N - 1]                             # LAST item offered

    # ---- 2. ONE-PASS, FIXED MEMORY: a huge stream leaves the reservoir at exactly k (O(k), not O(N)). ----
    n_big = 100000
    big = Reservoir(k, seed=0).add_all(range(n_big))
    big_len = len(big.sample())

    # ---- 3. EXACT SAMPLE SIZE + REAL ITEMS: N>=k -> exactly k items, all from the stream; N<k -> whole stream. ----
    stream20 = list(range(20))
    r20 = Reservoir(k, seed=0).add_all(stream20)
    s20 = r20.sample()
    subset_ok = set(s20) <= set(stream20)                # every sampled item is an actual input (real items)
    tiny = [7, 3, 9]                                     # N < k
    tiny_res = Reservoir(k, seed=0).add_all(tiny)

    # ---- 5. DETERMINISM: same k/seed/stream -> identical sample across two fresh instances. ----
    det_a = Reservoir(k, seed=42).add_all(range(N)).sample()
    det_b = Reservoir(k, seed=42).add_all(range(N)).sample()

    print("=" * 64)
    print("RESERVOIR SAMPLING -- one-pass uniform sample of k from an unbounded stream")
    print("=" * 64)
    print(f"k/N = {k}/{N} = {target:.4f}; runs = {runs}; std error {se:.5f}; band {band:.3f} "
          f"(~{band / se:.1f} std errors)")
    print(f"uniformity : mean inclusion freq {mean_freq:.5f} (== k/N); max deviation {max_dev:.5f} "
          f"at id {worst_id}")
    print(f"no bias    : FIRST-item freq {first_freq:.4f} vs LAST-item freq {last_freq:.4f} (both ~ {target:.4f})")
    print(f"fixed mem  : reservoir holds {big_len} items for N={n_big} (a full copy would hold {n_big})")
    print(f"exact size : N=20 sample size {len(s20)} (all real inputs: {subset_ok}); "
          f"N=3<k sample {tiny_res.sample()}")
    print(f"determinism: two fresh instances, same k/seed/stream -> identical sample: {det_a == det_b}")
    print("=" * 64)

    # ---- asserts: deterministic, zero-dep; genuine uniform result, nothing rigged ----
    assert abs(mean_freq - target) < 1e-9, f"mean inclusion freq {mean_freq} must equal k/N {target}"
    for _id, dev in enumerate(deviations):
        assert dev < band, f"id {_id}: inclusion freq off by {dev:.5f} (> band {band}) -- position bias?"
    assert max_dev < band, f"max deviation {max_dev:.5f} exceeds band {band}"
    assert abs(first_freq - target) < band, f"first item biased: {first_freq:.4f} vs k/N {target:.4f}"
    assert abs(last_freq - target) < band, f"last item biased: {last_freq:.4f} vs k/N {target:.4f}"
    assert big_len == k, f"reservoir must stay at k={k} regardless of N ({big_len})"
    assert big.n_seen == n_big, f"n_seen must count every offered item ({big.n_seen} != {n_big})"
    assert len(s20) == k and subset_ok, "N>=k must give exactly k items, all drawn from the actual stream"
    assert tiny_res.sample() == tiny, f"N<k must return the whole stream in order ({tiny_res.sample()})"
    assert tiny_res.n_seen == len(tiny), "n_seen must equal the number offered even when N<k"
    assert det_a == det_b, "same k/seed/stream must give identical samples"
    assert reservoir_sample(range(N), k, seed=7) == reservoir_sample(range(N), k, seed=7), "helper deterministic"
    fresh = Reservoir(k, seed=0)
    assert all(fresh.add(x) for x in range(k)) and fresh.sample() == list(range(k)), "first k items always kept"
    fresh.reset()
    assert fresh.n_seen == 0 and fresh.sample() == [], "reset must empty the reservoir"
    print(f"self-check OK: uniform inclusion k/N={target:.4f} (mean {mean_freq:.4f}, max dev {max_dev:.5f} < "
          f"{band}); no position bias (first {first_freq:.4f} ~ last {last_freq:.4f}); reservoir fixed at k={k} "
          f"for N={n_big}; deterministic")


if __name__ == "__main__":
    main()

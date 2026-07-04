"""
Spike-stream sketch — Count-Min heavy-hitter motif analytics in fixed memory (pure stdlib).

A common streaming-lake question is "which spike motifs recur most?" (frequent gestures, hot channel
co-activations). The Count-Min Sketch (Cormode-Muthukrishnan 2005) answers frequency queries in memory that is
FIXED regardless of how many DISTINCT motifs stream through: d hash rows x w counters; add() bumps one counter
per row, estimate() takes the MIN across rows (collisions only ever inflate -> one-sided error <= eps*N). Paired
with a small top-K tracker it is a streaming HEAVY-HITTER detector that keeps neither the rare tail nor the full
key set. Deterministic (seeded Carter-Wegman hashes).

PRODUCT integration (additive, opt-in, read-only): `StreamingHub.hot_motifs()` uses this to report the most
frequent channel-coactivation motifs over the live log without a per-motif table. ponytail ceiling: the
single-machine hub keeps every event, so it COULD exact-count motifs in a dict; the sketch's win is fixed memory
independent of distinct-motif count -- it is the analytics primitive that carries to the documented scale-out
path (Pub/Sub -> Dataflow -> GCS), not a saving on the local index. On-thesis with streaming_hub /
spike_telemetry_hub / drift_monitor.

  python -m snn_data_lake.spike_stream_sketch        # (also run by run_selfchecks.py)
"""
import random

_P = (1 << 61) - 1                          # Mersenne prime: modulus for Carter-Wegman universal hashing


def _digest(pattern):
    """Stable integer digest of a spike pattern (frozenset of active features) -- no builtin hash (randomized)."""
    h = 1469598103934665603
    for idx in sorted(pattern):
        h = (h * 1000003 + (idx + 1)) % _P
    return h


class CountMinSketch:
    """d x w counters; frequency estimate = min across rows (one-sided over-estimate only)."""

    def __init__(self, depth=4, width=256, seed=0):
        self.depth = depth
        self.width = width
        self.table = [[0] * width for _ in range(depth)]
        rng = random.Random(seed)
        self.ab = [(rng.randrange(1, _P), rng.randrange(0, _P)) for _ in range(depth)]  # per-row hash (a,b)

    def _cols(self, key):
        h = _digest(key)
        return [((a * h + b) % _P) % self.width for a, b in self.ab]     # (a*h+b) mod prime mod w (universal)

    def add(self, key, count=1):
        for r, c in enumerate(self._cols(key)):
            self.table[r][c] += count

    def estimate(self, key):
        return min(self.table[r][c] for r, c in enumerate(self._cols(key)))

    def counters(self):
        return self.depth * self.width


class HeavyHitters:
    """Count-Min sketch + a fixed top-K set -> streaming heavy-hitter detector in fixed memory."""

    def __init__(self, k=3, depth=4, width=256, seed=0):
        self.k = k
        self.cms = CountMinSketch(depth, width, seed)
        self.top = {}                       # key -> current estimate (at most k+1 entries)

    def add(self, key):
        self.cms.add(key)
        for tracked in self.top:                          # refresh tracked estimates (avoid stale-eviction)
            self.top[tracked] = self.cms.estimate(tracked)
        est = self.cms.estimate(key)
        if key in self.top or len(self.top) < self.k:
            self.top[key] = est
        elif est > min(self.top.values()):
            del self.top[min(self.top, key=self.top.get)]
            self.top[key] = est

    def heavy(self):
        return sorted(self.top, key=self.top.get, reverse=True)


def _rand_pattern(rng, dim=64, w=6):
    return frozenset(rng.sample(range(dim), w))


def main():
    rng = random.Random(1)
    heavy_motifs = [_rand_pattern(rng) for _ in range(3)]      # 3 frequent motifs...
    reps = [170, 140, 110]
    stream, truth = [], {}
    for motif, r in zip(heavy_motifs, reps):
        stream += [motif] * r
        truth[motif] = truth.get(motif, 0) + r
    for _ in range(1500):                                      # ...buried in 1500 distinct rare patterns
        p = _rand_pattern(rng)
        stream.append(p)
        truth[p] = truth.get(p, 0) + 1
    rng.shuffle(stream)

    hh = HeavyHitters(k=3, depth=4, width=256)
    for p in stream:
        hh.add(p)

    n = len(stream)
    distinct = len(truth)
    recovered = hh.heavy()
    true_top3 = sorted(truth, key=truth.get, reverse=True)[:3]
    eps_bound = 2.718281828 / hh.cms.width * n                # Count-Min error bound eps*N (eps = e/w)
    max_err = max(hh.cms.estimate(m) - truth[m] for m in heavy_motifs)
    one_sided = all(hh.cms.estimate(k) >= truth[k] for k in truth)

    print("=" * 60)
    print("SPIKE-STREAM SKETCH — Count-Min heavy-hitter motif detection")
    print("=" * 60)
    print(f"stream: {n} events, {distinct} distinct patterns; sketch {hh.cms.counters()} counters + top-3")
    print(f"true heavy counts   : {[truth[m] for m in true_top3]}")
    print(f"sketch estimates    : {[hh.cms.estimate(m) for m in true_top3]} (>= true, one-sided)")
    print(f"heavy-hitters match : {set(recovered) == set(true_top3)}; max over-est {max_err} (bound {eps_bound:.0f})")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert set(recovered) == set(true_top3), f"must recover the true heavy motifs ({recovered} vs {true_top3})"
    assert one_sided, "Count-Min error is one-sided: every estimate must be >= the true count"
    assert max_err <= eps_bound, f"over-estimation must respect the eps*N bound ({max_err} > {eps_bound:.0f})"
    assert hh.cms.counters() < distinct, f"sketch must be sublinear in distinct patterns ({hh.cms.counters()} vs {distinct})"
    hh2 = HeavyHitters(k=3, depth=4, width=256)
    for p in stream:
        hh2.add(p)
    assert set(hh2.heavy()) == set(recovered), "same stream -> identical heavy hitters (deterministic)"
    print(f"self-check OK: recovered top-3 motifs from {distinct} distinct in {hh.cms.counters()} counters; "
          f"one-sided error <= {eps_bound:.0f}; deterministic")


if __name__ == "__main__":
    main()

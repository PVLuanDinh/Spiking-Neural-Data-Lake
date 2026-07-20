"""
Spike Bloom filter — fixed-memory set membership over spike patterns (pure Python, no tensors).

The streaming lake ingests an unbounded flood of spike patterns; a second common question (beside "how often?",
answered by the Count-Min sketch in spike_stream_sketch) is "have I EVER seen this exact pattern before?". A
Bloom filter (Bloom 1970) answers set-membership in memory that is FIXED regardless of how many distinct patterns
arrive: m bits, k independent hashes; add() sets k bits, contains() checks them. It NEVER gives a false negative
(a stored pattern always tests present) and gives a false positive only at a small, tunable, analytically-known
rate (~(1 - e^(-kn/m))^k). Pure standard library, deterministic (seeded Carter-Wegman hashes). The membership
counterpart to the Count-Min FREQUENCY sketch; together they are the lake's fixed-memory analytics pair.
On-thesis with spike_stream_sketch / streaming_hub / drift_monitor.

  python -m research.spike_bloom        # (also run by run_selfchecks.py, root on PYTHONPATH)
"""
import math
import random

_P = (1 << 61) - 1                          # Mersenne prime: modulus for Carter-Wegman universal hashing


def _digest(pattern):
    """Stable integer digest of a spike pattern (frozenset of active features) -- no builtin hash (randomized)."""
    h = 1469598103934665603
    for idx in sorted(pattern):
        h = (h * 1000003 + (idx + 1)) % _P
    return h


class BloomFilter:
    """m bits x k hashes; add() sets bits, contains() tests them. No false negatives; tunable false-positive rate."""

    def __init__(self, m=4096, k=4, seed=0):
        self.m = m
        self.k = k
        self.bits = bytearray(m)
        rng = random.Random(seed)
        self.ab = [(rng.randrange(1, _P), rng.randrange(0, _P)) for _ in range(k)]   # per-hash (a,b)

    def _idx(self, key):
        h = _digest(key)
        return [((a * h + b) % _P) % self.m for a, b in self.ab]                      # universal hashes

    def add(self, key):
        for i in self._idx(key):
            self.bits[i] = 1

    def contains(self, key):
        return all(self.bits[i] for i in self._idx(key))

    def fill_ratio(self):
        return sum(self.bits) / self.m


def _rand_pattern(rng, dim=64, w=6):
    return frozenset(rng.sample(range(dim), w))


def main():
    rng = random.Random(1)
    m, k, n_in = 4096, 4, 500
    bloom = BloomFilter(m=m, k=k)

    inserted = [_rand_pattern(rng) for _ in range(n_in)]
    for p in inserted:
        bloom.add(p)

    # no false negatives: every inserted pattern must test present
    false_neg = sum(not bloom.contains(p) for p in inserted)

    # false-positive rate on fresh patterns vs the analytic bound (1 - e^(-k n / m))^k
    seen = set(inserted)
    trials, fp = 3000, 0
    while trials:
        q = _rand_pattern(rng)
        if q in seen:
            continue
        trials -= 1
        fp += bloom.contains(q)
    fp_rate = fp / 3000
    fp_theory = (1.0 - math.exp(-k * n_in / m)) ** k

    print("=" * 60)
    print("SPIKE BLOOM FILTER — fixed-memory pattern membership")
    print("=" * 60)
    print(f"filter: {m} bits, {k} hashes; inserted {n_in} patterns; fill {bloom.fill_ratio():.2f}")
    print(f"false negatives : {false_neg} (must be 0 -- membership is exact for stored keys)")
    print(f"false positives : {fp_rate:.3f} observed vs {fp_theory:.3f} analytic bound")
    print("=" * 60)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert false_neg == 0, f"a Bloom filter must never give a false negative ({false_neg})"
    assert fp_rate < 3 * fp_theory + 0.01, f"false-positive rate must track the analytic bound ({fp_rate:.3f} vs {fp_theory:.3f})"
    assert bloom.contains(inserted[0]) and not BloomFilter(m=m, k=k).contains(inserted[0]), "empty filter contains nothing; add works"
    b2 = BloomFilter(m=m, k=k)
    for p in inserted:
        b2.add(p)
    assert b2.bits == bloom.bits, "same inserts -> identical filter (deterministic)"
    print(f"self-check OK: {n_in} patterns, 0 false negatives, FP {fp_rate:.3f} ~ bound {fp_theory:.3f} "
          f"in {m} bits; deterministic")


if __name__ == "__main__":
    main()

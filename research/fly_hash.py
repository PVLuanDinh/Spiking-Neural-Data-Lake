"""
FlyHash — bio-inspired locality-sensitive hashing for the spike store's similarity search
(fresh research sweep). Pure-stdlib.

The store's content-addressable query (Paradigm A/B) ranks stored spike windows by brute-force
distance — O(N) per query. The fruit-fly olfactory circuit solves the same approximate-nearest-
neighbour problem with two steps this repo already uses elsewhere (random expansion + winner-take-
all): project the d-dim input UP into a much larger sparse space through a sparse *binary* random
matrix (each hash unit sums a few random input coords), then keep only the top-k activations as a
sparse binary tag. Similar inputs land on overlapping tags, so tag-overlap is a cheap LSH proxy for
true similarity — turning the linear scan into a candidate-set lookup, using the fly's own
primitives. Unlike classical LSH this EXPANDS dimensionality and uses sparse binary projections.
Refs: Dasgupta, Stevens & Navlakha, "A neural algorithm for a fundamental computing problem",
Science 358 (2017) 10.1126/science.aam9868; BioHash, Ryali et al., ICML 2020 (arXiv:2001.04907).

  Run: python fly_hash.py        # (also run by run_selfchecks.py)
"""
import math
import random


class FlyHash:
    """Sparse random expansion + WTA -> a sparse binary tag (the set of winner unit indices)."""

    def __init__(self, d, expansion=20, wta=0.05, samples=6, seed=0):
        self.d = d
        self.m = d * expansion                       # Kenyon-cell layer: expand, do not reduce
        self.k = max(1, round(self.m * wta))         # WTA winners kept per tag (~5%)
        rng = random.Random(seed)
        s = min(samples, d)                          # each unit sums a few random inputs
        self.proj = [rng.sample(range(d), s) for _ in range(self.m)]   # sparse binary projection

    def tag(self, x):
        """Return the sparse binary hash: a frozenset of the top-k activated unit indices."""
        act = [sum(x[i] for i in idx) for idx in self.proj]
        winners = sorted(range(self.m), key=lambda j: (act[j], -j), reverse=True)[:self.k]
        return frozenset(winners)


def overlap(a, b):
    """Tag similarity = shared winners; the LSH proxy for input similarity (bigger = closer)."""
    return len(a & b)


def _vec(rng, d):
    return [rng.random() for _ in range(d)]


def _jitter(rng, x, noise):
    return [max(0.0, v + rng.uniform(-noise, noise)) for v in x]


def main():
    rng = random.Random(1)
    d = 50
    fh = FlyHash(d, expansion=20, wta=0.05, samples=6, seed=0)

    # (1) LSH property: near-duplicates share far more winners than random pairs do
    near, far = [], []
    for _ in range(200):
        x = _vec(rng, d)
        near.append(overlap(fh.tag(x), fh.tag(_jitter(rng, x, 0.05))))
        far.append(overlap(fh.tag(x), fh.tag(_vec(rng, d))))
    mean_near = sum(near) / len(near)
    mean_far = sum(far) / len(far)

    # (2) approximate NN: does tag-overlap recover the true item behind a corrupted query?
    db = [_vec(rng, d) for _ in range(300)]
    tags = [fh.tag(v) for v in db]
    hits = 0
    for _ in range(100):
        qi = rng.randrange(len(db))
        qt = fh.tag(_jitter(rng, db[qi], 0.05))          # a 5%-jittered copy of item qi
        best = max(range(len(db)), key=lambda j: overlap(qt, tags[j]))
        hits += (best == qi)
    recall = hits / 100

    print("=" * 60)
    print("FLYHASH — bio-inspired LSH (sparse expansion + WTA) for the store")
    print("=" * 60)
    print(f"d={d} -> m={fh.m} units, top-k={fh.k} winners/tag ({fh.k / fh.m:.0%} sparse)")
    print(f"tag overlap: near-duplicate {mean_near:.1f} vs random pair {mean_far:.1f} winners")
    print(f"approx-NN recall@1 (recover a 5%-jittered item): {recall:.0%} over 100 queries")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert fh.m > d, "FlyHash must EXPAND dimensionality (m > d), unlike classical LSH"
    assert all(len(fh.tag(_vec(rng, d))) == fh.k for _ in range(5)), "every tag has exactly k winners"
    assert mean_near > 3 * mean_far, f"near-dup tags must overlap >> random ({mean_near:.1f} vs {mean_far:.1f})"
    assert recall >= 0.85, f"approx-NN must recover the jittered item (recall@1={recall:.2f})"
    x = _vec(rng, d)
    assert fh.tag(x) == FlyHash(d, seed=0).tag(x), "same seed -> identical projection -> identical tag"
    print(f"self-check OK: expansion+WTA LSH; near {mean_near:.1f} >> far {mean_far:.1f}; "
          f"recall@1 {recall:.0%}; deterministic")


if __name__ == "__main__":
    main()

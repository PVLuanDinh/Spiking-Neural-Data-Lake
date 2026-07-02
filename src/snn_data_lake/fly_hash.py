"""
FlyHash — bio-inspired LSH index for fast candidate filtering over stored windows.

PROMOTED from research/ into the product (v0.85, additive). The matchers and cluster_unknowns scan
stored/enrolled windows linearly. The fruit-fly olfactory hash (Dasgupta, Stevens & Navlakha,
Science 2017) turns each window into a sparse binary tag via random EXPANSION + winner-take-all, so
similar windows share tag winners — tag-overlap is a cheap locality-sensitive proxy for similarity.
`FlyHashIndex` uses that to return the most similar stored entries for a query in a candidate-set
lookup (the exact matcher then rescoring only the candidates). Additive and opt-in: nothing in the
default matcher/cluster path changes. Refs: Dasgupta, Stevens & Navlakha, "A neural algorithm for a
fundamental computing problem", Science 358 (2017); BioHash, Ryali et al., ICML 2020 (arXiv:2001.04907).

  python -m snn_data_lake.fly_hash        # (also run by run_selfchecks.py)
"""
import random

from snn_data_lake.signal_loop import build_default_library
from snn_data_lake.spike_preprocessing import N


class FlyHash:
    """Sparse random expansion + WTA -> a sparse binary tag (the set of winner unit indices)."""

    def __init__(self, d, expansion=20, wta=0.05, samples=6, seed=0):
        self.d = d
        self.m = d * expansion
        self.k = max(1, round(self.m * wta))
        rng = random.Random(seed)
        s = min(samples, d)
        self.proj = [rng.sample(range(d), s) for _ in range(self.m)]

    def tag(self, x):
        act = [sum(x[i] for i in idx) for idx in self.proj]
        winners = sorted(range(self.m), key=lambda j: (act[j], -j), reverse=True)[:self.k]
        return frozenset(winners)


def overlap(a, b):
    return len(a & b)


class FlyHashIndex:
    """Add (vector, payload) entries; query returns payloads of the most tag-similar entries."""

    def __init__(self, d=N, **kw):
        self.fh = FlyHash(d, **kw)
        self.entries = []                                # [(tag, payload)]

    def add(self, vec, payload):
        self.entries.append((self.fh.tag(vec), payload))

    def query(self, vec, top=5):
        qt = self.fh.tag(vec)
        ranked = sorted(self.entries, key=lambda e: -overlap(qt, e[0]))
        return [p for _, p in ranked[:top]]


def main():
    rng = random.Random(1)
    library = build_default_library()
    labels = list(library)

    def jitter(v, n):
        return [min(1.0, max(0.0, x + rng.uniform(-n, n))) for x in v]

    index = FlyHashIndex(d=N)
    for lab in labels:                                   # enroll many jittered windows per command
        for _ in range(20):
            index.add(jitter(library[lab], 0.08), lab)

    # approx-NN recall: does the top tag-overlap candidate share the query's command?
    trials, hits = 150, 0
    for i in range(trials):
        lab = labels[i % len(labels)]
        hits += index.query(jitter(library[lab], 0.10), top=1)[0] == lab
    recall = hits / trials

    # LSH property: near-duplicate tags overlap far more than random-pair tags
    fh = index.fh
    near = sum(overlap(fh.tag(jitter(library[labels[0]], 0.05)), fh.tag(library[labels[0]]))
               for _ in range(50)) / 50
    far = sum(overlap(fh.tag(library[labels[0]]), fh.tag(library[labels[1 + i % (len(labels) - 1)]]))
              for i in range(50)) / 50

    print("=" * 60)
    print("FLYHASH INDEX — bio-inspired LSH candidate filter over stored windows")
    print("=" * 60)
    print(f"{len(index.entries)} enrolled windows, tag = {fh.k}/{fh.m} winners")
    print(f"approx-NN recall@1 (top candidate shares the command): {recall:.0%}")
    print(f"tag overlap: same-command {near:.1f} vs different-command {far:.1f} winners")
    print("=" * 60)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert recall >= 0.9, f"FlyHash index must return the right command as top candidate ({recall:.2f})"
    assert near > 2 * far, f"same-command tags must overlap >> different-command ({near:.1f} vs {far:.1f})"
    assert FlyHash(N, seed=0).tag(library[labels[0]]) == FlyHash(N, seed=0).tag(library[labels[0]]), \
        "same seed -> identical tag (deterministic)"
    print(f"self-check OK: recall@1 {recall:.0%}, LSH separation {near:.1f}>>{far:.1f}; additive "
          f"candidate filter, default matcher/cluster path unchanged; deterministic")


if __name__ == "__main__":
    main()

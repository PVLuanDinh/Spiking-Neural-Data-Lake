"""
Rank-order coding (Thorpe) — gain-invariant classification by spike ORDER, not rate/latency
(temporal-compute axis). Pure-stdlib.

temporal_coding_storage encodes time-to-first-spike, where latency tracks magnitude. Thorpe's
rank-order code (Thorpe, Delorme & VanRullen, 2001) discards magnitude entirely and keeps only the
ORDER in which afferents first fire (strongest input spikes first) — one spike per afferent,
ultra-sparse. The point: a rank code is INVARIANT to any monotonic change of input gain/contrast,
which latency and rate codes are not. The retina recognizes the same object under any lighting
because the firing ORDER is unchanged. On-thesis (compute in spike timing), and a distinct code from
the latency store. Refs: Thorpe, Delorme & VanRullen, "Spike-based strategies for rapid processing",
Neural Networks 14 (2001).

  Run: python rank_order_coding.py        # (also run by run_selfchecks.py)
"""
import math
import random
from collections import defaultdict


def rank_code(x):
    """Descending rank per afferent: 0 = fires first (largest input). Depends only on ORDER."""
    order = sorted(range(len(x)), key=lambda i: x[i], reverse=True)
    rank = [0] * len(x)
    for pos, i in enumerate(order):
        rank[i] = pos
    return rank


def _rank_agree(rx, template):
    """Rank agreement (higher = closer): negative L1 distance between two rank vectors."""
    return -sum(abs(a - b) for a, b in zip(rx, template))


def _dist2(a, b):
    return sum((p - q) ** 2 for p, q in zip(a, b))


def _sample(rng, centers, per=40, spread=0.08):
    pts, lab = [], []
    for c, ctr in enumerate(centers):
        for _ in range(per):
            pts.append([min(1.0, max(0.0, v + rng.uniform(-spread, spread))) for v in ctr])
            lab.append(c)
    return pts, lab


def main():
    rng = random.Random(1)
    d = 10
    centers = [[rng.random() for _ in range(d)] for _ in range(3)]   # shared class centers
    train, ytr = _sample(rng, centers)
    test, yte = _sample(rng, centers)

    # train: per-class centroid + its rank-order template
    sums, cnt = defaultdict(lambda: [0.0] * d), defaultdict(int)
    for x, y in zip(train, ytr):
        cnt[y] += 1
        for j in range(d):
            sums[y][j] += x[j]
    cls = sorted(sums)
    centroid = {c: [sums[c][j] / cnt[c] for j in range(d)] for c in cls}
    template = {c: rank_code(centroid[c]) for c in cls}

    def rank_clf(x):
        rx = rank_code(x)
        return max(cls, key=lambda c: _rank_agree(rx, template[c]))

    def cent_clf(x):                                    # magnitude baseline (nearest centroid)
        return min(cls, key=lambda c: _dist2(x, centroid[c]))

    f = lambda v: math.exp(1.5 * v)                     # monotonic gain/contrast change (order-preserving)
    dist_test = [[f(v) for v in x] for x in test]

    def acc(clf, xs):
        return sum(clf(x) == y for x, y in zip(xs, yte)) / len(xs)

    rank_clean, rank_dist = acc(rank_clf, test), acc(rank_clf, dist_test)
    cent_clean, cent_dist = acc(cent_clf, test), acc(cent_clf, dist_test)

    print("=" * 64)
    print("RANK-ORDER CODING (Thorpe) — gain-invariant classification by spike ORDER")
    print("=" * 64)
    print(f"d={d} afferents, {len(cls)} classes, {len(test)} test points")
    print(f"                     clean   after monotonic gain change f(v)=exp(1.5v)")
    print(f"  rank-order code : {rank_clean:.0%}     {rank_dist:.0%}   (order unchanged -> invariant)")
    print(f"  magnitude base  : {cent_clean:.0%}     {cent_dist:.0%}   (distances distorted -> collapses)")
    print("=" * 64)

    # ---- self-check (ponytail: deterministic asserts, zero-dep) ----
    assert rank_clean >= 0.9, f"rank-order classifier must work on separable clusters ({rank_clean:.2f})"
    assert rank_dist == rank_clean, "rank code must be EXACTLY invariant to a monotonic gain change"
    assert cent_dist < cent_clean, f"magnitude baseline must degrade under the same change ({cent_clean:.2f}->{cent_dist:.2f})"
    assert rank_code([0.2, 0.9, 0.5]) == rank_code([f(0.2), f(0.9), f(0.5)]), "order preserved under f"
    assert rank_code([3.0, 1.0, 2.0]) == [0, 2, 1], "descending rank: largest -> 0"
    print(f"self-check OK: rank {rank_clean:.0%} invariant under gain change ({rank_clean:.0%}=={rank_dist:.0%}); "
          f"magnitude base {cent_clean:.0%}->{cent_dist:.0%}; deterministic")


if __name__ == "__main__":
    main()

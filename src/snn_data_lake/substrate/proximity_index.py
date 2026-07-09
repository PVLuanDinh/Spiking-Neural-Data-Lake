"""
Hierarchical proximity index over the Victor-Purpura spike-train distance (backend indexing).

The store's content-addressable query ranks spike patterns by a spike distance (Van Rossum, or the new
Victor-Purpura edit distance in spike_preprocessing) -- an O(N) brute-force scan. This builds a small
TWO-LEVEL hierarchical index over the VP distance so a query only rescans a candidate set: farthest-point
sampling picks ~sqrt(N) medoids, every pattern is assigned to its nearest medoid, and a query probes only
the nearest few medoids' members, then does an EXACT VP rescoring of just those candidates. No Euclidean /
cosine / HNSW / IVF -- the geometry is the spike-train edit distance. Pure stdlib, deterministic.

Two query paths, with different guarantees:
  - nearest_exact(query, k)  -- TRUSTWORTHY exact k-NN. VP is a true metric (the triangle inequality
    holds), so LAESA pivot pruning (triangle lower bounds over the medoids) returns EXACTLY the same
    items as brute_force while computing VP for fewer than N candidates. Use when you need the right item.
  - nearest(query, k, probe) -- FAST APPROXIMATE. Probes only the nearest `probe` medoids' members, so it
    can miss the true nearest item. Its real exact-NN recall is MEASURED and disclosed by the self-check
    (not hidden behind a same-class-counts-as-a-hit metric). Use when approximate-but-cheap is acceptable.

  python -m snn_data_lake.substrate.proximity_index        # (also run by run_selfchecks.py)
"""
import random

from snn_data_lake.cortex.spike_preprocessing import N, encode_latency, victor_purpura_distance


class ProximityIndex:
    """Medoid-partitioned hierarchical index over VP distance. build() then nearest(query, k)."""

    def __init__(self, q=1.0, n_medoids=None):
        self.q = q
        self.n_medoids = n_medoids
        self.patterns, self.payloads, self.medoids, self.clusters = [], [], [], {}
        self.pd = []      # N x m pivot-distance table: pd[i][j] = d(pattern_i, medoid_j) (LAESA bounds)

    def _d(self, ev_a, ev_b):
        return victor_purpura_distance(ev_a, ev_b, q=self.q)

    def build(self, patterns, payloads=None):
        """patterns: list of encoded event lists ([(t, feature), ...]). payloads: parallel labels."""
        self.patterns = list(patterns)
        self.payloads = list(payloads) if payloads is not None else list(range(len(self.patterns)))
        n = len(self.patterns)
        m = self.n_medoids or max(1, int(n ** 0.5))
        # farthest-point sampling: seed at 0, then repeatedly add the pattern farthest from all medoids
        medoids = [0]
        while len(medoids) < min(m, n):
            best, best_d = None, -1.0
            for i in range(n):
                if i in medoids:
                    continue
                d = min(self._d(self.patterns[i], self.patterns[mi]) for mi in medoids)
                if d > best_d:
                    best_d, best = d, i
            medoids.append(best)
        self.medoids = medoids
        self.clusters = {mi: [] for mi in medoids}
        # store the pivot-distance table while assigning clusters: the row d(pattern_i, each medoid) is
        # needed for BOTH the nearest-medoid assignment and LAESA's triangle bounds -- compute it once.
        self.pd = []
        for i in range(n):
            row = [self._d(self.patterns[i], self.patterns[mi]) for mi in medoids]
            self.pd.append(row)
            nearest_j = min(range(len(medoids)), key=lambda j: row[j])
            self.clusters[medoids[nearest_j]].append(i)
        return self

    def nearest(self, query, k=1, probe=2):
        """Return (payloads[:k], n_candidates_scanned): probe the nearest `probe` medoids, then exact VP
        rank of only their members."""
        near_medoids = sorted(self.medoids, key=lambda mi: self._d(query, self.patterns[mi]))[:probe]
        cand = sorted({i for mi in near_medoids for i in self.clusters[mi]},
                      key=lambda i: self._d(query, self.patterns[i]))
        return [self.payloads[i] for i in cand[:k]], len(cand)

    def nearest_exact(self, query, k=1):
        """Return (payloads[:k], n_exact_computed): the EXACT k nearest by LAESA metric pruning.

        VP is a true metric, so for candidate i the triangle inequality lower-bounds d(query, pattern_i)
        by lb_i = max_j abs(dq[j] - pd[i][j]) over the medoid pivots (max = tightest bound). Candidates
        are visited in ascending lb_i; a running k-best radius PRUNES (skips the exact VP compute for)
        any candidate whose lb_i already reaches the k-th best distance, and the ascending order lets us
        stop entirely at that point. The result is guaranteed identical to brute_force while computing
        exact VP for fewer than N candidates on clustered data."""
        dq = [self._d(query, self.patterns[mi]) for mi in self.medoids]
        m = len(dq)
        order = []
        for i in range(len(self.patterns)):
            row = self.pd[i]
            lb = 0.0
            for j in range(m):
                b = dq[j] - row[j]
                if b < 0.0:
                    b = -b
                if b > lb:
                    lb = b
            order.append((lb, i))
        order.sort()
        best = []          # k smallest (dist, index) so far, sorted ascending (ties -> lower index)
        n_exact = 0
        for lb, i in order:
            if len(best) == k and lb > best[-1][0]:
                break      # ascending lb: no remaining candidate can beat the current k-th best
            d = self._d(query, self.patterns[i])
            n_exact += 1
            best.append((d, i))
            best.sort()
            if len(best) > k:
                best.pop()
        return [self.payloads[i] for _, i in best], n_exact

    def brute_force(self, query, k=1):
        ranked = sorted(range(len(self.patterns)), key=lambda i: self._d(query, self.patterns[i]))
        return [self.payloads[i] for i in ranked[:k]]


def _counts_diff(ev_a, ev_b, n_features=N):
    """Per-channel spike-count difference = the q->0 limit of the VP distance."""
    ca = [0] * n_features
    cb = [0] * n_features
    for _, i in ev_a:
        ca[i] += 1
    for _, i in ev_b:
        cb[i] += 1
    return sum(abs(ca[i] - cb[i]) for i in range(n_features))


def main():
    rng = random.Random(1)
    protos = [[rng.random() if rng.random() < 0.4 else 0.0 for _ in range(N)] for _ in range(6)]

    def jitter(p):
        return [min(1.0, max(0.0, v + rng.uniform(-0.06, 0.06))) for v in p]

    patterns = [encode_latency(jitter(protos[i % 6])) for i in range(180)]
    labels = [i % 6 for i in range(180)]
    a, b = patterns[0], patterns[7]

    # ---- Victor-Purpura metric axioms + q-limits (the new metric) ----
    identity = victor_purpura_distance(a, a, q=1.0)
    symmetric = victor_purpura_distance(a, b, q=1.0) == victor_purpura_distance(b, a, q=1.0)
    vp_q0 = victor_purpura_distance(a, b, q=0.0)
    vp_lo = victor_purpura_distance(a, b, q=0.5)
    vp_hi = victor_purpura_distance(a, b, q=2.0)

    # payloads = item INDICES so recall is scored against the exact nearest ITEM, not just its class
    # (same-class items are interchangeable, which would inflate any class-label recall toward 100%).
    idx = ProximityIndex(q=1.0).build(patterns, list(range(len(patterns))))
    trials = 60
    exact_matches_bf = 0     # nearest_exact vs brute_force        (must agree on every query)
    approx_exact_hits = 0    # nearest(probe=2) returns the EXACT nearest item (the honest recall)
    exact_scanned = 0
    for _ in range(trials):
        qi = rng.randrange(len(patterns))
        query = encode_latency(jitter(protos[labels[qi]]))
        bf = idx.brute_force(query, k=1)[0]
        (ex,), n_exact = idx.nearest_exact(query, k=1)
        (ap,), _ = idx.nearest(query, k=1, probe=2)
        exact_matches_bf += ex == bf
        approx_exact_hits += ap == bf
        exact_scanned += n_exact
    exact_agree = exact_matches_bf / trials
    approx_recall = approx_exact_hits / trials
    avg_exact_scanned = exact_scanned / trials
    scan_frac = avg_exact_scanned / len(patterns)

    print("=" * 62)
    print("PROXIMITY INDEX - exact metric-pruned NN + honest approximate recall")
    print("=" * 62)
    print(f"{len(patterns)} patterns, {len(idx.medoids)} medoids; Victor-Purpura backend")
    print(f"VP axioms: identity={identity:.1f}, symmetric={symmetric}; "
          f"q-monotone {vp_q0:.0f}(q0) <= {vp_lo:.1f}(q.5) <= {vp_hi:.1f}(q2)")
    print(f"q->0 equals spike-count difference: {vp_q0:.1f} == {_counts_diff(a, b):.1f}")
    print(f"nearest_exact vs brute_force : {exact_agree:.0%} identical (exact, trustworthy default)")
    print(f"nearest_exact scan fraction  : {avg_exact_scanned:.0f}/{len(patterns)} candidates "
          f"({scan_frac:.0%}) via LAESA pruning -- sublinear yet still EXACT")
    print(f"nearest(probe=2) exact-NN recall: {approx_recall:.0%}  "
          f"(HONEST -- fraction returning the true nearest ITEM, not just same class)")
    print("=" * 62)

    # ---- self-check: deterministic asserts, zero-dep, HONEST numbers ----
    assert identity == 0.0, f"VP(a,a) must be 0 (identity), got {identity}"
    assert symmetric, "VP must be symmetric"
    assert vp_q0 <= vp_lo <= vp_hi, f"VP must be non-decreasing in q ({vp_q0} {vp_lo} {vp_hi})"
    assert vp_q0 == _counts_diff(a, b), f"q->0 VP must equal the spike-count difference ({vp_q0})"
    # nearest_exact is the trustworthy path: it must reproduce brute_force exactly...
    assert exact_agree == 1.0, f"nearest_exact must equal brute_force on every query ({exact_agree:.2f})"
    # ...while actually pruning (computing exact VP for fewer than all patterns).
    assert avg_exact_scanned < len(patterns), "nearest_exact must scan fewer than all patterns (pruning)"
    # the approximate path's REAL exact-NN recall -- assert only an honest floor, report the true value.
    assert approx_recall >= 0.5, f"approximate exact-NN recall below honest floor (real value {approx_recall:.2f})"
    # existing guarantee: a full probe of the approximate path degenerates to brute force.
    bf_a = idx.brute_force(a, k=1)[0]
    assert idx.nearest(a, k=1, probe=len(idx.medoids))[0][0] == bf_a, "full-probe index must match brute force"
    print(f"self-check OK: VP axioms; nearest_exact == brute_force scanning {scan_frac:.0%} of patterns; "
          f"nearest(probe=2) honest exact-NN recall {approx_recall:.0%}; deterministic")


if __name__ == "__main__":
    main()

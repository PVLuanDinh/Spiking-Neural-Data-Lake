"""
Hierarchical proximity index over the Victor-Purpura spike-train distance (backend indexing).

The store's content-addressable query ranks spike patterns by a spike distance (Van Rossum, or the new
Victor-Purpura edit distance in spike_preprocessing) -- an O(N) brute-force scan. This builds a small
TWO-LEVEL hierarchical index over the VP distance so a query only rescans a candidate set: farthest-point
sampling picks ~sqrt(N) medoids, every pattern is assigned to its nearest medoid, and a query probes only
the nearest few medoids' members, then does an EXACT VP rescoring of just those candidates. No Euclidean /
cosine / HNSW / IVF -- the geometry is the spike-train edit distance. Pure stdlib, deterministic.

  python -m snn_data_lake.proximity_index        # (also run by run_selfchecks.py)
"""
import random

from snn_data_lake.spike_preprocessing import N, encode_latency, victor_purpura_distance


class ProximityIndex:
    """Medoid-partitioned hierarchical index over VP distance. build() then nearest(query, k)."""

    def __init__(self, q=1.0, n_medoids=None):
        self.q = q
        self.n_medoids = n_medoids
        self.patterns, self.payloads, self.medoids, self.clusters = [], [], [], {}

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
        for i in range(n):
            nearest_m = min(medoids, key=lambda mi: self._d(self.patterns[i], self.patterns[mi]))
            self.clusters[nearest_m].append(i)
        return self

    def nearest(self, query, k=1, probe=2):
        """Return (payloads[:k], n_candidates_scanned): probe the nearest `probe` medoids, then exact VP
        rank of only their members."""
        near_medoids = sorted(self.medoids, key=lambda mi: self._d(query, self.patterns[mi]))[:probe]
        cand = sorted({i for mi in near_medoids for i in self.clusters[mi]},
                      key=lambda i: self._d(query, self.patterns[i]))
        return [self.payloads[i] for i in cand[:k]], len(cand)

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

    idx = ProximityIndex(q=1.0).build(patterns, labels)
    hits, scanned = 0, 0
    trials = 60
    for _ in range(trials):
        qi = rng.randrange(len(patterns))
        query = encode_latency(jitter(protos[labels[qi]]))
        (pred,), c = idx.nearest(query, k=1, probe=2)
        scanned += c
        hits += pred == labels[qi]
    recall = hits / trials
    avg_scanned = scanned / trials

    print("=" * 62)
    print("PROXIMITY INDEX — hierarchical VP-distance index for spike queries")
    print("=" * 62)
    print(f"{len(patterns)} patterns, {len(idx.medoids)} medoids; Victor-Purpura backend")
    print(f"VP axioms: identity={identity:.1f}, symmetric={symmetric}; "
          f"q-monotone {vp_q0:.0f}(q0) <= {vp_lo:.1f}(q.5) <= {vp_hi:.1f}(q2)")
    print(f"q->0 equals spike-count difference: {vp_q0:.1f} == {_counts_diff(a, b):.1f}")
    print(f"index recall@1: {recall:.0%}; scans {avg_scanned:.0f}/{len(patterns)} patterns "
          f"({avg_scanned / len(patterns):.0%}, sublinear)")
    print("=" * 62)

    # ---- self-check: deterministic asserts, zero-dep ----
    assert identity == 0.0, f"VP(a,a) must be 0 (identity), got {identity}"
    assert symmetric, "VP must be symmetric"
    assert vp_q0 <= vp_lo <= vp_hi, f"VP must be non-decreasing in q ({vp_q0} {vp_lo} {vp_hi})"
    assert vp_q0 == _counts_diff(a, b), f"q->0 VP must equal the spike-count difference ({vp_q0})"
    assert recall >= 0.9, f"index recall@1 must track the metric ({recall:.2f})"
    assert avg_scanned < len(patterns), "index must scan fewer than all patterns (sublinear candidate set)"
    bf = idx.brute_force(a, k=1)[0]
    assert idx.nearest(a, k=1, probe=len(idx.medoids))[0][0] == bf, "full-probe index must match brute force"
    print(f"self-check OK: VP metric (identity/symmetry/q-monotone/count-limit); index recall {recall:.0%} "
          f"scanning {avg_scanned / len(patterns):.0%} of patterns; deterministic")


if __name__ == "__main__":
    main()

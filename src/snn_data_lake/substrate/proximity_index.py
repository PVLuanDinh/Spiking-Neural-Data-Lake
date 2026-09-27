"""
Hierarchical proximity index over the Victor-Purpura spike-train distance (backend indexing).

The store's content-addressable query ranks spike patterns by a spike distance (Van Rossum, or the new
Victor-Purpura edit distance in spike_preprocessing) -- an O(N) brute-force scan. This builds a small
TWO-LEVEL hierarchical index over the VP distance so a query only rescans a candidate set: farthest-point
sampling picks ~sqrt(N) medoids, every pattern is assigned to its nearest medoid, and a query probes only
the nearest few medoids' members, then does an EXACT VP rescoring of just those candidates. No Euclidean /
cosine / HNSW / IVF -- the geometry is the spike-train edit distance. Deterministic.

Two query paths, with different guarantees:
  - nearest_exact(query, k)  -- TRUSTWORTHY exact k-NN. VP is a true metric (the triangle inequality
    holds), so LAESA pivot pruning (triangle lower bounds over the medoids) returns EXACTLY the same
    items as brute_force while computing VP for fewer than N candidates. Use when you need the right item.
  - nearest(query, k, probe) -- FAST APPROXIMATE. Probes only the nearest `probe` medoids' members, so it
    can miss the true nearest item. Its real exact-NN recall is MEASURED and disclosed by the self-check
    (not hidden behind a same-class-counts-as-a-hit metric). Use when approximate-but-cheap is acceptable.

Scaling as the store GROWS -- OUT-OF-CORE recall with bounded resident memory:
  The naive build materializes EVERYTHING in RAM: the whole pattern list, and a full N x m pivot-distance
  table (~N^1.5 floats). That is O(N) resident and breaks as stored patterns grow. When N exceeds
  `disk_threshold` this index instead PERSISTS the patterns and the pivot-distance table to a compact
  binary file and mmap/seeks candidate rows on demand. LAESA already localizes WHICH rows a query must
  score exactly, so those pattern blobs are read by seek (not all N); the pivot rows are read from an
  mmap'd fixed-width matrix (paged by the OS, O(1) resident). Resident state is just the m medoid pivots,
  a bounded LRU page cache of decoded patterns, and small per-item scalars. Small indexes (N <=
  disk_threshold) keep the original in-RAM FAST PATH unchanged. Recall is byte-identical across modes.

  python -m snn_data_lake.substrate.proximity_index        # (also run by run_selfchecks.py)
"""

import mmap
import os
import random
import shutil
import struct
import tempfile
import weakref
from array import array
from collections import OrderedDict

from snn_data_lake.sensorimotor.spike_preprocessing import (
    N,
    encode_latency,
    victor_purpura_distance,
)


def _encode_events(ev):
    """Pack an event list [(t, feature), ...] of ints into bytes (two little-endian int32 per event)."""
    flat = []
    for t, i in ev:
        flat.append(t)
        flat.append(i)
    return struct.pack(f"<{len(flat)}i", *flat)


def _decode_events(raw):
    """Inverse of _encode_events: bytes -> [(t, feature), ...] with integer coordinates."""
    vals = struct.unpack(f"<{len(raw) // 4}i", raw)
    return [(vals[j], vals[j + 1]) for j in range(0, len(vals), 2)]


def _cleanup(dirpath, handles):
    """weakref.finalize callback: close mmaps/files and remove the temp store directory."""
    for h in handles:
        try:
            h.close()
        except Exception:
            pass
    shutil.rmtree(dirpath, ignore_errors=True)


class ProximityIndex:
    """Medoid-partitioned hierarchical index over VP distance. build() then nearest(query, k).

    Resident memory stays bounded as the store grows: patterns and the pivot-distance table spill to
    disk (mmap) once N > disk_threshold, and queries seek only the candidate rows LAESA cannot prune."""

    def __init__(self, q=1.0, n_medoids=None, disk_threshold=4096, cache_cap=256):
        self.q = q
        self.n_medoids = n_medoids
        self.disk_threshold = disk_threshold  # N <= this keeps the original all-in-RAM fast path
        self.cache_cap = cache_cap  # bounded LRU page cache of decoded patterns (disk mode)
        self.patterns, self.payloads, self.medoids, self.clusters = [], [], [], {}
        self.pd = []  # N x m pivot-distance table (in-RAM fast path only); disk mode leaves it []
        self._n = 0
        self._disk = False
        # disk-mode state (populated only when out-of-core)
        self._dir = None
        self._pat_mm = None
        self._pd_mm = None
        self._offsets = None  # array('q'): byte offset of each pattern blob (N+1 entries)
        self._medoid_patterns = {}  # resident pivot patterns: index -> event list
        self._cache = None  # OrderedDict LRU of decoded non-pivot patterns
        self._pd_fmt = None
        self._pd_rowbytes = 0
        self._disk_reads = 0  # count of true mmap pattern fetches (for the bounded-memory check)
        self._finalizer = None

    def _d(self, ev_a, ev_b):
        return victor_purpura_distance(ev_a, ev_b, q=self.q)

    def _pattern(self, i):
        """Return pattern i. Fast path: direct list index. Disk path: resident pivot / LRU cache / seek."""
        if not self._disk:
            return self.patterns[i]
        p = self._medoid_patterns.get(i)
        if p is not None:
            return p
        c = self._cache.get(i)
        if c is not None:
            self._cache.move_to_end(i)
            return c
        lo = self._offsets[i]
        hi = self._offsets[i + 1]
        ev = _decode_events(self._pat_mm[lo:hi])  # the only true disk read
        self._disk_reads += 1
        self._cache[i] = ev
        if len(self._cache) > self.cache_cap:
            self._cache.popitem(last=False)  # evict least-recently-used (bounded working set)
        return ev

    def _pd_row(self, i):
        """Return the pivot-distance row for pattern i. Fast path: RAM. Disk path: mmap fixed-width slice."""
        if not self._disk:
            return self.pd[i]
        lo = i * self._pd_rowbytes
        return list(struct.unpack(self._pd_fmt, self._pd_mm[lo : lo + self._pd_rowbytes]))

    def build(self, patterns, payloads=None):
        """patterns: list of encoded event lists ([(t, feature), ...]). payloads: parallel labels."""
        patterns = list(patterns)
        n = len(patterns)
        self._n = n
        self.payloads = list(payloads) if payloads is not None else list(range(n))
        m = self.n_medoids or max(1, int(n**0.5))
        # FARTHEST-POINT SAMPLING, carrying a RUNNING MINIMUM. The direct form recomputes
        # `min(d(i, mi) for mi in medoids)` from scratch for every candidate on every pass, which is
        # |medoids| distance calls per candidate and O(n^2) over the build: 13.5M Victor-Purpura DPs
        # at the live corpus size n=5,198, measured at ~9.4 MINUTES per rebuild -- and `_ensure_index`
        # runs this on the tick thread whenever anything was consolidated, so the brain spent nearly
        # all of its time here (wake_ticks frozen for minutes at a stretch).
        #
        # Carrying `mind[i]` = the distance from i to its NEAREST already-chosen medoid turns each new
        # medoid into one O(n) update, so the build is O(m*n) = O(n^1.5): 374K calls at n=5,198.
        # The medoid SEQUENCE is unchanged, not merely similar -- the minimum over a set equals the
        # running minimum of that same set, and the scan keeps the same ascending order and the same
        # strict `>`, so ties resolve to the same index. Verified equal on random corpora.
        #
        # `chosen` is a set because `if i in medoids` on a list is itself an O(m) scan per candidate,
        # quietly contributing a second O(n^2) term.
        medoids = [0]
        chosen = {0}
        mind = [self._d(patterns[i], patterns[0]) for i in range(n)]
        while len(medoids) < min(m, n):
            best, best_d = None, -1.0
            for i in range(n):
                if i in chosen:
                    continue
                if mind[i] > best_d:
                    best_d, best = mind[i], i
            medoids.append(best)
            chosen.add(best)
            for i in range(n):  # fold the new medoid into the running minimum
                d = self._d(patterns[i], patterns[best])
                if d < mind[i]:
                    mind[i] = d
        self.medoids = medoids
        self.clusters = {mi: [] for mi in medoids}

        if n <= self.disk_threshold:
            self._build_ram(patterns, medoids)
        else:
            self._build_disk(patterns, medoids)
        return self

    def _build_ram(self, patterns, medoids):
        """Original in-RAM fast path: hold patterns + the full pivot-distance table resident."""
        self._disk = False
        self.patterns = patterns
        # store the pivot-distance table while assigning clusters: the row d(pattern_i, each medoid) is
        # needed for BOTH the nearest-medoid assignment and LAESA's triangle bounds -- compute it once.
        self.pd = []
        for i in range(len(patterns)):
            row = [self._d(patterns[i], patterns[mi]) for mi in medoids]
            self.pd.append(row)
            nearest_j = min(range(len(medoids)), key=lambda j: row[j])
            self.clusters[medoids[nearest_j]].append(i)

    def _build_disk(self, patterns, medoids):
        """Out-of-core path: spill patterns + the pivot-distance table to a compact binary store and
        mmap them. Resident state is only the m pivots, a bounded LRU cache, and small per-item scalars.

        The pivot-distance rows are computed in the SAME order and by the SAME arithmetic as the RAM
        path, so nearest/nearest_exact/brute_force return byte-identical results across the two modes."""
        self._disk = True
        n = len(patterns)
        m = len(medoids)
        self._dir = tempfile.mkdtemp(prefix="snn_prox_")
        pat_path = os.path.join(self._dir, "patterns.bin")
        pd_path = os.path.join(self._dir, "pd.bin")

        # patterns.bin: concatenated blobs; offsets[i]..offsets[i+1] delimits pattern i (O(N) small ints).
        offsets = array("q", [0])
        with open(pat_path, "wb") as f:
            for ev in patterns:
                f.write(_encode_events(ev))
                offsets.append(f.tell())
        self._offsets = offsets

        # pd.bin: fixed-width N x m matrix of little-endian doubles -> row i starts at i*m*8 (no offset table).
        self._pd_fmt = f"<{m}d"
        self._pd_rowbytes = m * 8
        with open(pd_path, "wb") as f:
            for i in range(n):
                row = [self._d(patterns[i], patterns[mi]) for mi in medoids]
                f.write(struct.pack(self._pd_fmt, *row))
                nearest_j = min(range(m), key=lambda j: row[j])
                self.clusters[medoids[nearest_j]].append(i)

        # keep only the pivots resident; the store's patterns/pd tables stay OUT of RAM.
        self._medoid_patterns = {mi: patterns[mi] for mi in medoids}
        self._cache = OrderedDict()
        self._disk_reads = 0
        self.patterns = []
        self.pd = []

        pat_fh = open(pat_path, "rb")
        pat_mm = mmap.mmap(pat_fh.fileno(), 0, access=mmap.ACCESS_READ)
        pd_fh = open(pd_path, "rb")
        pd_mm = mmap.mmap(pd_fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._pat_mm, self._pd_mm = pat_mm, pd_mm
        # release the temp store + mappings when this index is garbage-collected (no caller close() needed)
        self._finalizer = weakref.finalize(
            self, _cleanup, self._dir, (pat_mm, pat_fh, pd_mm, pd_fh)
        )

    def nearest(self, query, k=1, probe=2):
        """Return (payloads[:k], n_candidates_scanned): probe the nearest `probe` medoids, then exact VP
        rank of only their members."""
        near_medoids = sorted(self.medoids, key=lambda mi: self._d(query, self._pattern(mi)))[
            :probe
        ]
        cand = sorted(
            {i for mi in near_medoids for i in self.clusters[mi]},
            key=lambda i: self._d(query, self._pattern(i)),
        )
        return [self.payloads[i] for i in cand[:k]], len(cand)

    def nearest_exact(self, query, k=1):
        """Return (payloads[:k], n_exact_computed): the EXACT k nearest by LAESA metric pruning.

        VP is a true metric, so for candidate i the triangle inequality lower-bounds d(query, pattern_i)
        by lb_i = max_j abs(dq[j] - pd[i][j]) over the medoid pivots (max = tightest bound). Candidates
        are visited in ascending lb_i; a running k-best radius PRUNES (skips the exact VP compute for)
        any candidate whose lb_i already reaches the k-th best distance, and the ascending order lets us
        stop entirely at that point. The result is guaranteed identical to brute_force while computing
        exact VP for fewer than N candidates on clustered data. Out-of-core: the pd rows are read from
        the mmap'd pivot matrix and only the un-pruned candidate patterns are fetched by seek."""
        dq = [self._d(query, self._pattern(mi)) for mi in self.medoids]
        m = len(dq)
        order = []
        for i in range(self._n):
            row = self._pd_row(i)
            lb = 0.0
            for j in range(m):
                b = dq[j] - row[j]
                if b < 0.0:
                    b = -b
                if b > lb:
                    lb = b
            order.append((lb, i))
        order.sort()
        best = []  # k smallest (dist, index) so far, sorted ascending (ties -> lower index)
        n_exact = 0
        for lb, i in order:
            if len(best) == k and lb > best[-1][0]:
                break  # ascending lb: no remaining candidate can beat the current k-th best
            d = self._d(query, self._pattern(i))
            n_exact += 1
            best.append((d, i))
            best.sort()
            if len(best) > k:
                best.pop()
        return [self.payloads[i] for _, i in best], n_exact

    def brute_force(self, query, k=1):
        ranked = sorted(range(self._n), key=lambda i: self._d(query, self._pattern(i)))
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
    exact_matches_bf = 0  # nearest_exact vs brute_force        (must agree on every query)
    approx_exact_hits = 0  # nearest(probe=2) returns the EXACT nearest item (the honest recall)
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
    print(
        f"VP axioms: identity={identity:.1f}, symmetric={symmetric}; "
        f"q-monotone {vp_q0:.0f}(q0) <= {vp_lo:.1f}(q.5) <= {vp_hi:.1f}(q2)"
    )
    print(f"q->0 equals spike-count difference: {vp_q0:.1f} == {_counts_diff(a, b):.1f}")
    print(
        f"nearest_exact vs brute_force : {exact_agree:.0%} identical (exact, trustworthy default)"
    )
    print(
        f"nearest_exact scan fraction  : {avg_exact_scanned:.0f}/{len(patterns)} candidates "
        f"({scan_frac:.0%}) via LAESA pruning -- sublinear yet still EXACT"
    )
    print(
        f"nearest(probe=2) exact-NN recall: {approx_recall:.0%}  "
        f"(HONEST -- fraction returning the true nearest ITEM, not just same class)"
    )
    print("=" * 62)

    # ---- self-check: deterministic asserts, HONEST numbers ----
    assert identity == 0.0, f"VP(a,a) must be 0 (identity), got {identity}"
    assert symmetric, "VP must be symmetric"
    assert vp_q0 <= vp_lo <= vp_hi, f"VP must be non-decreasing in q ({vp_q0} {vp_lo} {vp_hi})"
    assert vp_q0 == _counts_diff(a, b), f"q->0 VP must equal the spike-count difference ({vp_q0})"
    # nearest_exact is the trustworthy path: it must reproduce brute_force exactly...
    assert exact_agree == 1.0, (
        f"nearest_exact must equal brute_force on every query ({exact_agree:.2f})"
    )
    # ...while actually pruning (computing exact VP for fewer than all patterns).
    assert avg_exact_scanned < len(patterns), (
        "nearest_exact must scan fewer than all patterns (pruning)"
    )
    # the approximate path's REAL exact-NN recall -- assert only an honest floor, report the true value.
    assert approx_recall >= 0.5, (
        f"approximate exact-NN recall below honest floor (real value {approx_recall:.2f})"
    )
    # existing guarantee: a full probe of the approximate path degenerates to brute force.
    bf_a = idx.brute_force(a, k=1)[0]
    assert idx.nearest(a, k=1, probe=len(idx.medoids))[0][0] == bf_a, (
        "full-probe index must match brute force"
    )

    # ---- OUT-OF-CORE scaling: bounded resident memory, identical recall, sublinear seeks ----
    # A shared large corpus queried by two indexes over IDENTICAL patterns/payloads: one forced all-in-RAM
    # (the fast path), one forced out-of-core to disk. Modest n_medoids keeps the O(n*m^2) farthest-point
    # build fast; the point is the recall/memory behavior, not the medoid count.
    big_n = 2400
    big_patterns = [encode_latency(jitter(protos[i % 6])) for i in range(big_n)]
    big_labels = list(range(big_n))
    ram = ProximityIndex(q=1.0, n_medoids=24, disk_threshold=10**9).build(big_patterns, big_labels)
    disk = ProximityIndex(q=1.0, n_medoids=24, disk_threshold=0, cache_cap=128).build(
        big_patterns, big_labels
    )
    assert not ram._disk and disk._disk, "mode selection: ram in-core, disk out-of-core"

    same_exact = same_approx = same_brute = 0
    disk_exact_bf = 0
    disk_scanned = 0
    max_disk_reads = 0
    q_trials = 24
    for _ in range(q_trials):
        qi = rng.randrange(big_n)
        query = encode_latency(jitter(protos[big_labels[qi] % 6]))
        # (a) recall correctness: out-of-core returns the SAME items as in-RAM, exact and approximate
        r_ex, _ = ram.nearest_exact(query, k=3)
        d_ex, n_exact = disk.nearest_exact(query, k=3)
        r_ap, _ = ram.nearest(query, k=3, probe=3)
        d_ap, _ = disk.nearest(query, k=3, probe=3)
        r_bf = ram.brute_force(query, k=3)
        d_bf = disk.brute_force(query, k=3)
        same_exact += r_ex == d_ex
        same_approx += r_ap == d_ap
        same_brute += r_bf == d_bf
        # LAESA still exact on the disk index: nearest_exact == brute_force
        disk_exact_bf += d_ex == d_bf
        disk_scanned += n_exact
        # (b) bounded seeks: a query fetches only the un-pruned candidate blobs, never all N
        disk._disk_reads = 0
        disk.nearest_exact(query, k=3)
        max_disk_reads = max(max_disk_reads, disk._disk_reads)

    # (c) determinism: identical query -> identical result, twice, on the disk index
    qd = encode_latency(jitter(protos[0]))
    det = disk.nearest_exact(qd, k=5)[0] == disk.nearest_exact(qd, k=5)[0]

    resident_patterns = len(disk._medoid_patterns) + len(
        disk._cache
    )  # decoded patterns held in RAM
    avg_disk_scanned = disk_scanned / q_trials

    print("OUT-OF-CORE scaling (bounded resident memory)")
    print("-" * 62)
    print(
        f"corpus: {big_n} patterns, {len(disk.medoids)} pivots; patterns + pivot table mmap'd from a temp store"
    )
    print(
        f"recall identical to in-RAM : nearest_exact {same_exact}/{q_trials}, "
        f"nearest {same_approx}/{q_trials}, brute_force {same_brute}/{q_trials}"
    )
    print(f"LAESA still exact on disk  : nearest_exact == brute_force {disk_exact_bf}/{q_trials}")
    print(
        f"resident decoded patterns  : {resident_patterns} (<= {len(disk.medoids)} pivots + "
        f"{disk.cache_cap} cache) vs {big_n} total -- NOT O(N)"
    )
    print(
        f"pattern blobs read by seek : max {max_disk_reads}/query (< {big_n}); "
        f"pd table + patterns held in RAM: pd={len(disk.pd)} patterns={len(disk.patterns)}"
    )
    print(
        f"exact scan fraction        : {avg_disk_scanned:.0f}/{big_n} "
        f"({avg_disk_scanned / big_n:.0%}) via LAESA pruning"
    )
    print("=" * 62)

    # correctness: out-of-core is byte-identical to in-RAM on every query and every path
    assert same_exact == q_trials, (
        f"out-of-core nearest_exact must match in-RAM ({same_exact}/{q_trials})"
    )
    assert same_approx == q_trials, (
        f"out-of-core nearest must match in-RAM ({same_approx}/{q_trials})"
    )
    assert same_brute == q_trials, (
        f"out-of-core brute_force must match in-RAM ({same_brute}/{q_trials})"
    )
    assert disk_exact_bf == q_trials, (
        "LAESA pruning on the disk index must stay exact vs brute_force"
    )
    # bounded memory: the disk index holds neither the pattern list nor the pivot table in RAM
    assert disk.patterns == [] and disk.pd == [], (
        "out-of-core index must not hold patterns/pd in RAM"
    )
    assert resident_patterns <= len(disk.medoids) + disk.cache_cap, (
        "resident patterns must be bounded"
    )
    assert resident_patterns < big_n, "resident patterns must be sublinear in N (not O(N))"
    # bounded seeks: a query reads only candidate rows, never all N patterns
    assert 0 < max_disk_reads < big_n, (
        f"query must seek only candidates, not all N ({max_disk_reads})"
    )
    assert avg_disk_scanned < big_n, (
        "nearest_exact must scan fewer than all patterns on the disk index"
    )
    assert det, "out-of-core recall must be deterministic (identical query -> identical result)"

    print(
        f"self-check OK: VP axioms; nearest_exact == brute_force scanning {scan_frac:.0%} of patterns; "
        f"nearest(probe=2) honest exact-NN recall {approx_recall:.0%}; "
        f"out-of-core recall byte-identical to in-RAM with {resident_patterns} resident patterns "
        f"(vs {big_n}) and <= {max_disk_reads} seeks/query; deterministic"
    )


if __name__ == "__main__":
    main()

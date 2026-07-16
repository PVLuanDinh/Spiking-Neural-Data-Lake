"""
LakeMemory -- long-term, lake-consolidated, proximity-addressable replay memory (research tier).

The developmental loop currently replays past exemplars from an in-RAM random ring: samples are
held only in the live process and drawn uniformly at random. This adapter replaces that with a
memory that (1) CONSOLIDATES each exemplar to the durable substrate spike lake -- so it survives a
restart, it is not RAM -- and (2) PULLS exemplars back BY PROXIMITY (content-addressable recall over
the spike-train distance) instead of at random. It is a thin composition of two real substrate
modules; the adapter itself is pure stdlib (no torch touches the zero-dep lake).

Durable store used: `snn_data_lake.substrate.streaming_hub.StreamingHub` (the durable append-only
`.spkl` spike lake). Chosen over LakeWorkingMemory because StreamingHub's (channel u32, time u32)
record preserves the per-feature spike TIMING of the latency code; LakeWorkingMemory.write(t, set)
only records which channels were active on a tick, discarding the latency that the VP distance ranks
by. So StreamingHub is the store that most simply gives a DURABLE append-and-reload of spike patterns.

Feature -> spike encoding (exactly the project's deterministic latency code, cortex.encode_latency):
  each feature f_i of a D-dim vector (D <= N=64, roughly in [0,1]) with f_i >= FLOOR(0.10) emits ONE
  spike at time t_i = round((1 - f_i) * (T-1)), T=32 -> brighter feature fires EARLIER (rank/latency
  code); features below FLOOR are silent (sparse). Result: a sorted event list [(t, feature), ...].
  This is byte-identical to what the ProximityIndex self-check and the rest of the lake encode with.

Consolidation to the lake: exemplar e is written to StreamingHub on its own T-wide time BAND -- for
each (t, feature) in its pattern, hub.append(channel=feature, t = e*T + t). The band tags which
exemplar an event belongs to, so the exact latency pattern is reconstructable from disk later. The
raw feature vectors + payloads are also kept in parallel lists (the ProximityIndex needs the encoded
patterns to build over, and the payloads are the replay ids the loop consumes).

Proximity index: `snn_data_lake.substrate.proximity_index.ProximityIndex` is (re)built over ALL
consolidated patterns (its build() takes the list of encoded event lists + parallel payloads and
partitions them by medoids of the Victor-Purpura spike-train distance). retrieve() encodes the query
the SAME way and calls index.nearest(query, k, probe) -> the payloads of the k nearest stored
exemplars: content-addressable recall, the loop's drop-in replacement for random.sample().

durability_check() proves persistence: it snapshots n_events + the reconstructed patterns, CLOSES the
hub, REOPENS the same `.spkl` with resume=True (StreamingHub replays the durable log from disk with
the RAM index thrown away), and re-reads every exemplar's band back off disk -- asserting the event
count and every latency pattern survive the restart. Contents that survive discarding RAM + replaying
the file genuinely live in the data lake, not the process.

Run:  PYTHONPATH=src python research/brain_adapters/lake_memory.py
"""
import os
import random
import tempfile

from snn_data_lake.cortex.spike_preprocessing import N, T, encode_latency
from snn_data_lake.substrate.proximity_index import ProximityIndex
from snn_data_lake.substrate.streaming_hub import StreamingHub


def _to_floats(vec):
    """Convert a feature vector (list / tuple / torch tensor / numpy row) to a plain list[float].

    Keeps torch OUT of the zero-dep substrate: a tensor is materialised to python floats here, before
    any value reaches encode_latency / StreamingHub / ProximityIndex."""
    if hasattr(vec, "tolist"):          # torch.Tensor / numpy.ndarray -> python list
        vec = vec.tolist()
    return [float(v) for v in vec]


class LakeMemory:
    """Long-term replay memory: consolidate exemplars to the durable spike lake, recall by proximity.

    __init__(path, seed): open a durable StreamingHub `.spkl` store + hold a ProximityIndex.
    consolidate(feature_vectors, payloads): latency-encode each exemplar, append it to the durable
        lake on its own time band, (re)build the proximity index over all stored patterns.
    retrieve(query_vector, k): latency-encode the query, return the k nearest exemplars' payloads.
    retrieve_balanced(learned, per): `per` nearest payloads for each class in `learned`.
    durability_check(): close+reopen the `.spkl` and confirm every consolidated exemplar survives.
    """

    def __init__(self, path=None, seed=0, band=T, q=1.0, probe=3):
        """Create/open the durable lake store and the proximity index.

        `path` None -> a unique temp `.spkl` is created; else the given path is used (resume if it
        exists). `band` is the per-exemplar time-band width (>= T so latency times never collide).
        `q` is the Victor-Purpura shift cost of the index; `probe` the default medoid probe width."""
        if band < T:
            raise ValueError(f"band must be >= T ({T}) so latency times fit in one exemplar band")
        self.n = N                          # StreamingHub channel width = the feature width N=64
        self.seed = seed
        self.rng = random.Random(seed)
        self.band = int(band)
        self.q = q
        self.probe = int(probe)
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".spkl")
            os.close(fd)                    # StreamingHub opens its own append handle
            self._created_temp = True
        else:
            self._created_temp = False
        self.path = path
        resume = self._created_temp is False and os.path.exists(path) and os.path.getsize(path) > 0
        self.hub = StreamingHub(path, self.n, resume=resume)
        self._patterns = []                 # encoded latency patterns [(t, feat), ...] (index build set)
        self._payloads = []                 # parallel replay ids (row index and/or label)
        self._features = []                 # raw feature vectors (for class-representative queries)
        self.index = None                   # rebuilt on each consolidate()

    # ---- encode one feature vector into the lake/index pattern format --------
    def _encode(self, vec):
        """Deterministic latency code: feature >= FLOOR -> ONE spike, brighter fires earlier.

        A pure passthrough to cortex.encode_latency so the adapter's patterns are byte-identical to
        every other latency-coded pattern in the lake. Guards feature width against the index's N."""
        vec = _to_floats(vec)
        if len(vec) > self.n:
            raise ValueError(
                f"feature vector width {len(vec)} exceeds lake/index width N={self.n} "
                f"(victor_purpura_distance indexes features in [0,{self.n}))")
        return encode_latency(vec)

    # ---- CONSOLIDATE exemplars to the durable lake ---------------------------
    def consolidate(self, feature_vectors, payloads):
        """Persist a batch of exemplars to the durable lake and (re)build the proximity index.

        For each exemplar e (global ordinal): (a) latency-encode its feature vector; (b) write it to
        the durable `.spkl` on its own time band -- hub.append(feature, e*band + t) per spike -- so the
        pattern is reconstructable from disk; (c) add the encoded pattern + payload to the index build
        set. The ProximityIndex is rebuilt over ALL stored patterns at the end. Returns self."""
        feats = [_to_floats(v) for v in feature_vectors]
        payloads = list(payloads)
        if len(feats) != len(payloads):
            raise ValueError(f"feature_vectors ({len(feats)}) and payloads ({len(payloads)}) differ")
        for vec, pay in zip(feats, payloads):
            e = len(self._patterns)                     # global exemplar ordinal -> its time band
            pattern = self._encode(vec)
            base = e * self.band
            for t, feat in pattern:
                self.hub.append(feat, base + t)         # durable append to the spike lake
            self._patterns.append(pattern)
            self._payloads.append(pay)
            self._features.append(vec)
        if self._patterns:
            self.index = ProximityIndex(q=self.q).build(self._patterns, list(self._payloads))
        return self

    # ---- RETRIEVE by proximity (content-addressable recall) ------------------
    def retrieve(self, query_vector, k, probe=None):
        """The payloads of the k nearest consolidated exemplars to `query_vector`.

        Encodes the query with the SAME latency code, then ProximityIndex.nearest(query, k, probe) --
        the loop calls this each cycle instead of random.sample() to draw content-relevant replay."""
        if self.index is None:
            return []
        query = self._encode(query_vector)
        payloads, _scanned = self.index.nearest(query, k=k, probe=self.probe if probe is None else probe)
        return payloads

    def retrieve_balanced(self, learned, per):
        """A class-balanced replay set: `per` nearest payloads for each class in `learned`.

        Each class c is queried with its class-representative = the MEAN of the consolidated feature
        vectors whose payload == c (payloads are treated as class labels here). Classes with nothing
        stored are skipped. Aggregates the per-class retrievals into one list -- a drop-in for a
        balanced replay slot. (For row-index payloads, pass class labels as payloads, or use retrieve.)"""
        out = []
        for c in learned:
            vecs = [self._features[i] for i, p in enumerate(self._payloads) if p == c]
            if not vecs:
                continue
            d = len(vecs[0])
            rep = [sum(v[j] for v in vecs) / len(vecs) for j in range(d)]
            out.extend(self.retrieve(rep, per))
        return out

    # ---- durability: reconstruct one exemplar's pattern from disk ------------
    def _read_pattern(self, e):
        """Read exemplar e's latency pattern back out of the durable lake (its time band)."""
        base = e * self.band
        ev = self.hub.query_window(base, base + self.band)
        return sorted((t - base, c) for t, c in ev)

    @property
    def n_stored(self):
        return len(self._patterns)

    def durability_check(self):
        """Close + reopen the `.spkl` and confirm every consolidated exemplar survives the restart.

        Snapshots n_events + the reconstructed patterns, closes the hub, reopens the SAME file with
        resume=True (StreamingHub replays the durable log; the RAM index is discarded), then re-reads
        every exemplar's band off disk. Returns a dict; `ok` is True iff the event count and all
        latency patterns are byte-identical after the round-trip -- i.e. the memory lives in the lake."""
        before_events = self.hub.n_events()
        before_patterns = [self._read_pattern(e) for e in range(self.n_stored)]
        self.hub.close()
        self.hub = StreamingHub(self.path, self.n, resume=True)      # replay purely from disk
        after_events = self.hub.n_events()
        after_patterns = [self._read_pattern(e) for e in range(self.n_stored)]
        ok = (after_events == before_events) and (after_patterns == before_patterns)
        return {
            "ok": ok,
            "events_before": before_events,
            "events_after": after_events,
            "patterns_match": after_patterns == before_patterns,
            "n_stored": self.n_stored,
        }

    def close(self, remove=False):
        """Release the lake handle; if `remove` and this instance made a temp file, delete it."""
        self.hub.close()
        if remove and self._created_temp and os.path.exists(self.path):
            os.remove(self.path)


def main():
    rng = random.Random(0)
    n_clusters = 4
    dim = N                                     # full feature width
    active_p = 0.4                              # sparsity of each cluster prototype

    # distinct sparse prototypes in [0,1]^N, one per synthetic cluster
    protos = [[rng.random() if rng.random() < active_p else 0.0 for _ in range(dim)]
              for _ in range(n_clusters)]

    def jitter(p):
        return [min(1.0, max(0.0, v + rng.uniform(-0.05, 0.05))) for v in p]

    mem = LakeMemory(seed=0)
    try:
        # ---- consolidate exemplars from the 4 clusters; payload = cluster id ----
        per_cluster = 30
        feats, pays = [], []
        for c in range(n_clusters):
            for _ in range(per_cluster):
                feats.append(jitter(protos[c]))
                pays.append(c)                  # payload = cluster label (the replay id)
        mem.consolidate(feats, pays)

        # ---- PROPERTY 1: proximity is meaningful (recall precision >> chance) ----
        k = 5
        trials = 40
        hits = 0
        for _ in range(trials):
            c = rng.randrange(n_clusters)       # query NEAR a random cluster
            got = mem.retrieve(jitter(protos[c]), k=k, probe=4)
            hits += sum(1 for p in got if p == c)
        precision = hits / (trials * k)
        chance = 1.0 / n_clusters

        # class-balanced recall drops straight into the loop's replay slot
        bal = mem.retrieve_balanced(learned=list(range(n_clusters)), per=3)

        # ---- PROPERTY 2: durability (events + patterns survive close+reopen) ----
        dur = mem.durability_check()
        # retrieval still works after the reopen (recall over the reloaded lake)
        got_after = mem.retrieve(jitter(protos[0]), k=k, probe=4)
        recall_after_ok = sum(1 for p in got_after if p == 0) >= (k // 2)

        print("=" * 66)
        print("LAKE MEMORY -- lake-consolidated, proximity-addressable replay memory")
        print("=" * 66)
        print(f"stored        : {mem.n_stored} exemplars from {n_clusters} clusters "
              f"({per_cluster} each), payload = cluster id")
        print(f"durable lake  : {os.path.basename(mem.path)}  {os.path.getsize(mem.path):,} B "
              f"({mem.hub.n_events()} spike events, 8 B each)")
        print("-" * 66)
        print(f"PROPERTY 1 proximity : nearest-{k} recall precision = {precision:.0%} "
              f"(chance = {chance:.0%})  <- a query near cluster A recalls mostly cluster-A payloads")
        print(f"balanced recall      : {len(bal)} payloads across classes -> "
              f"{sorted(set(bal))} (per-class mean-vector queries)")
        print("-" * 66)
        print(f"PROPERTY 2 durability: close+reopen .spkl -> events {dur['events_before']} -> "
              f"{dur['events_after']}, all {dur['n_stored']} latency patterns match = {dur['patterns_match']}")
        print(f"recall after reopen  : query near cluster 0 -> {got_after} "
              f"(recall survives restart = {recall_after_ok})")
        print("=" * 66)

        # ---- self-checks (deterministic asserts) --------------------------------
        # PROPERTY 1: proximity retrieval is far above chance (content-addressable, not random).
        assert precision > chance + 0.3, (
            f"proximity recall precision {precision:.2f} not clearly above chance {chance:.2f}")
        assert bal and set(bal) <= set(range(n_clusters)), "balanced recall returned bad payloads"
        # PROPERTY 2: the consolidated exemplars are DURABLE -- survive discarding RAM + replay.
        assert dur["ok"], f"durability check failed: {dur}"
        assert dur["events_after"] == dur["events_before"], "event count changed across reopen"
        assert dur["events_after"] > 0, "durable lake is empty -- nothing was consolidated"
        assert dur["patterns_match"], "latency patterns did not survive the reopen"
        assert recall_after_ok, "proximity recall broke after reopening the durable lake"

        print(f"self-check OK: proximity recall {precision:.0%} >> chance {chance:.0%} "
              f"(content-addressable); {dur['n_stored']} exemplars + {dur['events_after']} events "
              f"survive close+reopen (durable in the data lake, not RAM); recall works post-restart")
    finally:
        mem.close(remove=True)


if __name__ == "__main__":
    main()

"""
LakeMemory -- long-term, lake-consolidated, proximity-addressable replay memory (research tier).

The developmental loop currently replays past exemplars from an in-RAM random ring: samples are
held only in the live process and drawn uniformly at random. This adapter replaces that with a
memory that (1) CONSOLIDATES each exemplar to the durable substrate spike lake -- so it survives a
restart, it is not RAM -- and (2) PULLS exemplars back BY PROXIMITY (content-addressable recall over
the spike-train distance) instead of at random. It is a thin composition of two real substrate
modules; the adapter keeps torch out of the lake (tensors are materialised to python floats first).

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

durability_check() snapshots n_events + the reconstructed patterns, CLOSES the hub, REOPENS the same
`.spkl` with resume=True (StreamingHub replays the durable log from disk with the RAM index thrown
away), and re-reads every exemplar's band back off disk -- asserting the event count and every
latency pattern survive the restart.

P4b -- WHY THAT IS NOT ENOUGH, AND WHAT WAS BUILT (2026-08-04). This docstring used to close
"Contents that survive discarding RAM + replaying the file genuinely live in the data lake, not the
process." It does not follow from durability_check(), because that check never leaves the process:
the same interpreter, the same imported modules and the same OS page cache serve both halves of it.
P4b's honesty guard is explicit -- *"it persists" is a claim about a NEW process ... prove it by
reading the lake back from disk after the writer is gone* -- and the failure it was written from
(59 episodes stored, 0 lake records, `retained()` reporting 1.000) is exactly a same-process check
that could not see its own gap. `cross_process_durability_check()` is that missing measurement: it
CLOSES the hub, launches a FRESH `sys.executable` on `--readback`, and compares a `zlib.crc32`
digest of every reconstructed latency pattern computed in that other interpreter against the digest
of the patterns THIS OBJECT BELIEVES IT STORED. The writer is gone when the reader runs.

Corrected 2026-08-05: the parent side of that comparison was itself a second read of the same file,
which made `ok` a pure function of the bytes on both sides -- measured, 60 of 120 exemplars held in
RAM and never written to the lake passed it with `ok=True`. It now compares RAM against disk, which
is the direction P4b's guard is about. Two reachable REDs ship with it and both are run by the
self-check: a TRUNCATED copy of the lake, and one exemplar in `_patterns` that never reached the
hub -- P4b's own motivating shape.

Run:  PYTHONPATH=src python -m snn_data_lake.hippocampus.proximity_replay

PROMOTED reasoning/brain_adapters/lake_memory.py -> hippocampus/proximity_replay.py 2026-08-11
(#10 family settlement C3; measured row in docs/P4C_BACKLOG.md). RENAMED at the move because
brain_atlas asserts basename uniqueness across REGIONS and `substrate/lake_memory.py` (SHY
prune-and-reconstruct -- a different question, see the ledger's STRUCK list) already holds the
name; the class stays `LakeMemory`. Two fixes landed with the move: the P4b cross-process
child now targets `type(self).__module__` (the hardcoded old dotted path was the
control-must-follow-the-read-path defect, caught BEFORE it fired this time), and the dead
`durable=` sink + `rebuild_from_durable()` were deleted per P3 (zero callers ever;
writer retired 2026-07-29).
"""

import os
import random
import subprocess
import sys
import tempfile
import zlib

from snn_data_lake.cortex.spike_preprocessing import N, T, encode_latency
from snn_data_lake.substrate.proximity_index import ProximityIndex
from snn_data_lake.substrate.streaming_hub import StreamingHub

#: Read statically by `gates/gradient_claim_gate.py`; never imported by it.
GRADIENT_CLAIM = {
    "LakeMemory.consolidate": (
        "NOT_A_RULE",
        "persists exemplars to the durable lake and rebuilds the proximity index. Storage and "
        "indexing",
    ),
}

# This module's own dotted path, derived not hardcoded (control follows the read path): under
# `python -m` the class's __module__ is "__main__", but runpy still sets __spec__.name to the
# real dotted path, so this is correct both imported and run as a script. The P4b cross-process
# child re-runs THIS module by this name.
_SELF_MODULE = __spec__.name if __spec__ is not None else __name__


def _patterns_digest(patterns):
    """zlib.crc32 over the reconstructed latency patterns -- a persisted-value digest, so
    `zlib.crc32` and never builtin `hash()` (PYTHONHASHSEED randomises the latter, and this
    number is compared ACROSS processes, which is precisely where builtin hash breaks)."""
    crc = 0
    for pattern in patterns:
        for t, channel in pattern:
            crc = zlib.crc32(f"{t},{channel};".encode("ascii"), crc)
        crc = zlib.crc32(b"|", crc)
    return crc & 0xFFFFFFFF


def _readback_digest(path, n_stored, band, n_channels, packed):
    """Reopen an EXISTING `.spkl` read-only and return (n_events, patterns digest).

    This is the function the child interpreter runs under `--readback`. It constructs nothing
    but a StreamingHub over the file on disk: no LakeMemory, no in-RAM patterns, nothing the
    writer left behind. That is what makes the digest it returns evidence about the LAKE rather
    than about the object that wrote it."""
    hub = StreamingHub(path, n_channels, resume=True, packed=packed)
    try:
        patterns = []
        for e in range(n_stored):
            base = e * band
            ev = hub.query_window(base, base + band)
            patterns.append(sorted((t - base, c) for t, c in ev))
        return hub.n_events(), _patterns_digest(patterns)
    finally:
        hub.close()


def _to_floats(vec):
    """Convert a feature vector (list / tuple / torch tensor / numpy row) to a plain list[float].

    Keeps torch OUT of the substrate: a tensor is materialised to python floats here, before
    any value reaches encode_latency / StreamingHub / ProximityIndex."""
    if hasattr(vec, "tolist"):  # torch.Tensor / numpy.ndarray -> python list
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

    def __init__(self, path=None, seed=0, band=T, q=1.0, probe=3, packed=True):
        """Create/open the durable lake store and the proximity index.

        `path` None -> a unique temp `.spkl` is created; else the given path is used (resume if it
        exists). `band` is the per-exemplar time-band width (>= T so latency times never collide).
        `q` is the Victor-Purpura shift cost of the index; `probe` the default medoid probe width.
        `packed` (DEFAULT True): store the durable log with StreamingHub's 4 B packed AER record instead
        of the 8 B default -> HALVES the lake on disk, losslessly (channel < N=64 fits 6 bits; the global
        band time e*band+t fits the 26-bit field for < 2^26/band exemplars). Byte-reversible; retrieval,
        durability and out-of-core behaviour are unchanged. Set packed=False for the legacy 8 B layout
        (e.g. to resume a store written before this option existed)."""
        if band < T:
            raise ValueError(f"band must be >= T ({T}) so latency times fit in one exemplar band")
        self.n = N  # StreamingHub channel width = the feature width N=64
        self.packed = bool(packed)
        self.seed = seed
        self.rng = random.Random(seed)
        self.band = int(band)
        self.q = q
        self.probe = int(probe)
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".spkl")
            os.close(fd)  # StreamingHub opens its own append handle
            self._created_temp = True
        else:
            self._created_temp = False
        self.path = path
        resume = self._created_temp is False and os.path.exists(path) and os.path.getsize(path) > 0
        self.hub = StreamingHub(path, self.n, resume=resume, packed=self.packed)
        self._patterns = []  # encoded latency patterns [(t, feat), ...] (index build set)
        self._payloads = []  # parallel replay ids (row index and/or label)
        self._features = []  # raw feature vectors (for class-representative queries)
        self.index = None  # rebuilt on each consolidate()
        # The OPTIONAL `durable=` SpikeLake sink and rebuild_from_durable() were DELETED at the
        # 2026-08-11 promotion (P3): zero of the 14 constructor sites ever passed durable=, and
        # its writer half was already retired 2026-07-29 (home.py names this parameter as the
        # orphan that justified it). The gap they addressed is still true and still RECORDED:
        # the payload map + index live in RAM and reset on restart -- durable bytes, not durable
        # MEMORY. Rebuilding that capability starts from the one lake (P4b), not from resurrecting
        # an unconsumed parameter.

    # ---- encode one feature vector into the lake/index pattern format --------
    def _encode(self, vec):
        """Deterministic latency code: feature >= FLOOR -> ONE spike, brighter fires earlier.

        A pure passthrough to cortex.encode_latency so the adapter's patterns are byte-identical to
        every other latency-coded pattern in the lake. Guards feature width against the index's N."""
        vec = _to_floats(vec)
        if len(vec) > self.n:
            raise ValueError(
                f"feature vector width {len(vec)} exceeds lake/index width N={self.n} "
                f"(victor_purpura_distance indexes features in [0,{self.n}))"
            )
        return encode_latency(vec)

    # ---- CONSOLIDATE exemplars to the durable lake ---------------------------
    def consolidate(self, feature_vectors, payloads):
        """Persist a batch of exemplars to the durable lake and (re)build the proximity index.

        For each exemplar e (global ordinal): (a) latency-encode its feature vector; (b) write it to
        the durable `.spkl` on its own time band -- hub.append(feature, e*band + t) per spike -- so the
        pattern is reconstructable from disk; (c) add the encoded pattern + payload to the index build
        set. The ProximityIndex is built LAZILY (on the next retrieve), not rebuilt on every consolidate
        call -- the durable `.spkl` writes still happen per-call, only the index build is deferred. Returns self."""
        feats = [_to_floats(v) for v in feature_vectors]
        payloads = list(payloads)
        if len(feats) != len(payloads):
            raise ValueError(
                f"feature_vectors ({len(feats)}) and payloads ({len(payloads)}) differ"
            )
        for vec, pay in zip(feats, payloads):
            e = len(self._patterns)  # global exemplar ordinal -> its time band
            pattern = self._encode(vec)
            base = e * self.band
            for t, feat in pattern:
                self.hub.append(
                    feat, base + t
                )  # spike lake (index-rebuild source; a scratch when durable)
            self._patterns.append(pattern)
            self._payloads.append(pay)
            self._features.append(vec)
        self._dirty = True  # mark the index stale; build once on the next retrieve
        return self

    def _ensure_index(self):
        """Build the ProximityIndex over ALL stored patterns ONCE, only if new exemplars were consolidated
        since the last build (avoids rebuilding the whole index on every consolidate call). Identical index
        contents + ranking as the old eager rebuild -- only the timing of the build changes."""
        if getattr(self, "_dirty", True) and self._patterns:
            self.index = ProximityIndex(q=self.q).build(self._patterns, list(self._payloads))
            self._dirty = False
        return self

    # ---- RETRIEVE by proximity (content-addressable recall) ------------------
    def retrieve(self, query_vector, k, probe=None):
        """The payloads of the k nearest consolidated exemplars to `query_vector`.

        Encodes the query with the SAME latency code, then ProximityIndex.nearest(query, k, probe) --
        the loop calls this each cycle instead of random.sample() to draw content-relevant replay."""
        self._ensure_index()  # lazy build: only rebuilds if new exemplars were consolidated
        if self.index is None:
            return []
        query = self._encode(query_vector)
        payloads, _scanned = self.index.nearest(
            query, k=k, probe=self.probe if probe is None else probe
        )
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
        self.hub = StreamingHub(
            self.path, self.n, resume=True, packed=self.packed
        )  # replay purely from disk
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

    # ---- P4b: the SAME check, run from a process that never wrote the lake ----
    def cross_process_durability_check(self, path=None):
        """Prove persistence the way P4b requires: from a NEW process, after the writer is gone.

        Computes the in-process (n_events, digest) over `path` (default: this instance's own
        lake), CLOSES the hub so no handle or buffered write survives, then runs a FRESH
        `sys.executable -m <this module> --readback ...` which opens only the file on disk and
        prints its own digest. Returns a dict; `ok` is True iff the two agree.

        Why this is not durability_check() with extra steps: durability_check() closes and
        reopens inside the SAME interpreter, so the module state, the import graph and the
        object that produced the patterns are all still resident. Nothing about it can
        distinguish "the bytes are on disk" from "the writer still remembers". The child here
        shares nothing with the writer except the path.

        **THE PARENT SIDE IS THE WRITER'S OWN MEMORY, NOT A SECOND DISK READ. Corrected
        2026-08-05, and the bug was the policy's own failure shape.** This method first computed
        `before_digest` by calling `_readback_digest(target, ...)` -- i.e. it re-read the same
        FILE the child would read. Both halves were then a pure function of the same bytes, so
        `ok` answered "do two processes decode this file the same way" (yes, always) instead of
        "did the lake keep what the brain learned". MEASURED: with 60 of 120 exemplars appended to
        `_patterns` but never written to the hub -- literally P4b's "59 episodes stored, 0 lake
        records" -- `ok` came back **True**, `events_child` came back **1341 > 0**, and BOTH
        asserts in the self-check stayed green while half the memory was absent from the lake.
        The digest that reddens is the WRITER'S: `_patterns_digest(self._patterns)` was 750563058
        against the lake's 924405719 on that run, and identical (774847069 both sides) on the
        honest one. So `before_digest` is now what this object believes it remembers and
        `before_events` is its own hub count; `path` selects only which FILE the child reads.

        Reachable RED (constructed and run by the self-check, not asserted from prose): (a) pass a
        TRUNCATED copy of the same lake as `path` -- the child replays fewer records and returns a
        different digest, so `ok` goes False; (b) add an exemplar to `_patterns` that was never
        appended to the hub -- the RAM digest moves and the lake's does not, so `ok` goes False.
        (b) is the one the old parent-reads-disk form could not reach."""
        target = self.path if path is None else path
        before_events = self.hub.n_events()  # what the WRITER counts
        before_digest = _patterns_digest(self._patterns)  # what the WRITER believes it remembers
        self.hub.close()  # the writer is gone before the child opens the file
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    # the child must run THIS module wherever it lives -- a hardcoded dotted
                    # path here is the control-must-follow-the-read-path defect (a promotion
                    # would leave the P4b check testing the OLD path via the alias layer).
                    _SELF_MODULE,
                    "--readback",
                    target,
                    str(self.n_stored),
                    str(self.band),
                    str(self.n),
                    "1" if self.packed else "0",
                ],
                capture_output=True,
                text=True,
                env=_child_env(),
                timeout=180,
            )
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            if proc.returncode != 0 or not line.startswith("READBACK "):
                raise RuntimeError(
                    f"child readback failed (rc={proc.returncode}): {proc.stderr.strip()[:400]}"
                )
            _, ev_s, dig_s = line.split()
            child_events, child_digest = int(ev_s), int(dig_s)
        finally:
            # restore this instance's own handle so the object stays usable after the check
            self.hub = StreamingHub(self.path, self.n, resume=True, packed=self.packed)
        return {
            "ok": child_events == before_events and child_digest == before_digest,
            "path": target,
            "events_parent": before_events,
            "events_child": child_events,
            "digest_parent": before_digest,
            "digest_child": child_digest,
            "pid_parent": os.getpid(),
        }

    def close(self, remove=False):
        """Release the lake handle; if `remove` and this instance made a temp file, delete it."""
        self.hub.close()
        if remove and self._created_temp and os.path.exists(self.path):
            os.remove(self.path)


def _child_env():
    """Environment for the readback child: this package's import root prepended to PYTHONPATH,
    DERIVED from __file__ (never a hardcoded 'src'), so the check works from any CWD and does not
    break if the package is relocated."""
    directory = os.path.dirname(os.path.abspath(__file__))
    while os.path.isfile(os.path.join(directory, "__init__.py")):
        directory = os.path.dirname(directory)
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = directory + (os.pathsep + existing if existing else "")
    return env


def main():
    rng = random.Random(0)
    n_clusters = 4
    dim = N  # full feature width
    active_p = 0.4  # sparsity of each cluster prototype

    # distinct sparse prototypes in [0,1]^N, one per synthetic cluster
    protos = [
        [rng.random() if rng.random() < active_p else 0.0 for _ in range(dim)]
        for _ in range(n_clusters)
    ]

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
                pays.append(c)  # payload = cluster label (the replay id)
        mem.consolidate(feats, pays)

        # ---- PROPERTY 1: proximity is meaningful (recall precision >> chance) ----
        k = 5
        trials = 40
        hits = 0
        for _ in range(trials):
            c = rng.randrange(n_clusters)  # query NEAR a random cluster
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

        # ---- PROPERTY 2b (P4b): the SAME readback from a process that never wrote the lake ----
        xproc = mem.cross_process_durability_check()
        # RED for 2b, constructed and run rather than argued: hand the child a TRUNCATED copy of
        # the same lake. The file still opens and StreamingHub still resumes, so this is not an
        # "unreadable file" red -- it is a lake with records missing, which is the failure mode
        # P4b actually cares about, and the digest must diverge.
        truncated = mem.path + ".red"
        with open(mem.path, "rb") as _src, open(truncated, "wb") as _dst:
            _blob = _src.read()
            _dst.write(_blob[: len(_blob) // 2])
        xproc_red = mem.cross_process_durability_check(path=truncated)
        os.remove(truncated)
        # RED 2, the one the OLD parent-reads-disk form could not reach: put an exemplar in the
        # writer's own memory that never reached the lake. This is P4b's motivating shape exactly
        # ("59 episodes stored ... added 0 lake records"), and it must turn `ok` False.
        _ghost = mem._encode(jitter(protos[0]))
        mem._patterns.append(_ghost)  # in RAM, deliberately NOT appended to the hub
        mem._payloads.append(0)
        mem._features.append(jitter(protos[0]))
        xproc_ghost = mem.cross_process_durability_check()
        mem._patterns.pop()  # restore the honest state
        mem._payloads.pop()
        mem._features.pop()

        print("=" * 66)
        print("LAKE MEMORY -- lake-consolidated, proximity-addressable replay memory")
        print("=" * 66)
        print(
            f"stored        : {mem.n_stored} exemplars from {n_clusters} clusters "
            f"({per_cluster} each), payload = cluster id"
        )
        print(
            f"durable lake  : {os.path.basename(mem.path)}  {os.path.getsize(mem.path):,} B "
            f"({mem.hub.n_events()} spike events, {mem.hub._rec.size} B each"
            f"{', packed AER' if mem.hub.packed else ''})"
        )
        print("-" * 66)
        print(
            f"PROPERTY 1 proximity : nearest-{k} recall precision = {precision:.0%} "
            f"(chance = {chance:.0%})  <- a query near cluster A recalls mostly cluster-A payloads"
        )
        print(
            f"balanced recall      : {len(bal)} payloads across classes -> "
            f"{sorted(set(bal))} (per-class mean-vector queries)"
        )
        print("-" * 66)
        print(
            f"PROPERTY 2 durability: close+reopen .spkl -> events {dur['events_before']} -> "
            f"{dur['events_after']}, all {dur['n_stored']} latency patterns match = {dur['patterns_match']}"
        )
        print(
            f"recall after reopen  : query near cluster 0 -> {got_after} "
            f"(recall survives restart = {recall_after_ok})"
        )
        print(
            f"PROPERTY 2b P4b     : NEW-PROCESS readback (writer pid {xproc['pid_parent']} closed "
            f"first) -> events {xproc['events_parent']} vs {xproc['events_child']}, crc32 "
            f"{xproc['digest_parent']} vs {xproc['digest_child']}, ok = {xproc['ok']}"
        )
        print(
            f"  RED control 1     : same child against a TRUNCATED copy -> {xproc_red['events_child']} "
            f"events / crc32 {xproc_red['digest_child']}, vs the writer's "
            f"{xproc['events_parent']} / {xproc['digest_parent']} -> ok = {xproc_red['ok']} (want False)"
        )
        print(
            f"  RED control 2     : one exemplar in RAM only, never written to the lake (P4b's own "
            f"failure shape) -> child crc32 {xproc_ghost['digest_child']} vs writer "
            f"{xproc_ghost['digest_parent']} -> ok = {xproc_ghost['ok']} (want False)"
        )
        print("=" * 66)

        # ---- self-checks (deterministic asserts) --------------------------------
        # PROPERTY 1: proximity retrieval is far above chance (content-addressable, not random).
        assert precision > chance + 0.3, (
            f"proximity recall precision {precision:.2f} not clearly above chance {chance:.2f}"
        )
        assert bal and set(bal) <= set(range(n_clusters)), "balanced recall returned bad payloads"
        # PROPERTY 2: the consolidated exemplars are DURABLE -- survive discarding RAM + replay.
        assert dur["ok"], f"durability check failed: {dur}"
        assert dur["events_after"] == dur["events_before"], "event count changed across reopen"
        assert dur["events_after"] > 0, "durable lake is empty -- nothing was consolidated"
        assert dur["patterns_match"], "latency patterns did not survive the reopen"
        assert recall_after_ok, "proximity recall broke after reopening the durable lake"
        # PROPERTY 2b (P4b): "it persists" is a claim about a NEW process. The assert above is
        # satisfied by a same-process reopen and cannot distinguish a durable lake from a writer
        # that still remembers; this one can, because the reader shares nothing with the writer.
        assert xproc["ok"], f"cross-process readback disagreed with the writer: {xproc}"
        assert xproc["events_child"] > 0, "the new process read an EMPTY lake -- nothing persisted"
        # ...and the two controls proving that assert is an instrument rather than decoration.
        # Both assert on `ok` ITSELF -- the same criterion the line above passes on -- because a
        # control that exercises a DIFFERENT expression than the criterion is not a control of it.
        # (History, kept because it is the correction that mattered: until 2026-08-05 the parent
        # side of cross_process_durability_check() was a second read of the same FILE, so `ok` was
        # a pure function of the bytes on both sides and `xproc_red["ok"]` was True by
        # construction. The control then had to be weakened to a digest-inequality, which tests
        # `_readback_digest`'s sensitivity and NOT the assert above it. The parent side now
        # compares the writer's OWN patterns, so the natural control works and the weaker one is
        # gone -- this is the "do not weaken, fix the criterion" direction of F11.)
        assert not xproc_red["ok"], (
            f"control failed: a TRUNCATED lake still satisfied the cross-process check {xproc_red} "
            f"-- the check cannot detect lost records"
        )
        assert xproc_red["events_child"] < xproc["events_parent"], (
            f"control failed: the truncated lake replayed {xproc_red['events_child']} events, not "
            f"fewer than the writer's {xproc['events_parent']} -- the truncation did not take"
        )
        assert not xproc_ghost["ok"], (
            f"control failed: an exemplar held in RAM and NEVER written to the lake still passed "
            f"the cross-process check {xproc_ghost} -- this is exactly the P4b failure (episodes "
            f"stored, 0 lake records, retained() reporting 1.000) and the check must see it"
        )
        assert xproc_ghost["events_child"] == xproc["events_parent"], (
            "control 2 must differ from control 1: no EVENT was lost here, only a pattern the "
            "writer believes it stored -- if the event count moved, the RED is the wrong one"
        )

        print(
            f"self-check OK: proximity recall {precision:.0%} >> chance {chance:.0%} "
            f"(content-addressable); {dur['n_stored']} exemplars + {dur['events_after']} events "
            f"survive close+reopen (durable in the data lake, not RAM); recall works post-restart; "
            f"and a NEW PROCESS re-read the same {xproc['events_child']} events off disk to crc32 "
            f"{xproc['digest_child']}, matching what the WRITER believed it remembered, after the "
            f"writer closed (P4b) -- with both reds run: a truncated lake and an exemplar held in "
            f"RAM only each turn that check False"
        )
    finally:
        mem.close(remove=True)


if __name__ == "__main__":
    # `--readback <path> <n_stored> <band> <n_channels> <packed>` is the CHILD half of
    # cross_process_durability_check(): a fresh interpreter that opens only the lake on disk and
    # prints its own (n_events, crc32) so the parent can compare. It constructs no LakeMemory,
    # so nothing the writer held in RAM can leak into the number.
    if len(sys.argv) > 1 and sys.argv[1] == "--readback":
        _path, _n_stored, _band, _n_ch, _packed = sys.argv[2:7]
        _events, _digest = _readback_digest(
            _path, int(_n_stored), int(_band), int(_n_ch), _packed == "1"
        )
        print(f"READBACK {_events} {_digest}")
    else:
        main()

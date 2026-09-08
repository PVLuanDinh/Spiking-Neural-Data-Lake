"""Retention classes + the IMMORTAL append-budget instrument (specs/022 T010-i).

The retention rule, operational (lands in docs/POLICIES.md P4b in this same commit, per the
gate-before-rule coordination of T004): every record class in the ONE lake is labeled
IMMORTAL or COMPACTABLE at introduction, here, in RETENTION. "Remember everything" means an
appended record is either retained forever or provably reconstructible before any compaction
reclaims it -- COMPACTABLE is a claim about reconstruction, never permission to forget.

IMMORTAL classes carry an append BUDGET: bytes/tick measured against a recorded baseline
(tools/lake_append_baseline.json, the perf_baseline pattern -- record on the pinned host,
ratchet thereafter). The red is a constructed FLOOD: an append rate far above baseline must
trip the assert BEFORE disk pressure becomes the detector. The measured pathology this
instrument exists to prevent: 97.64% of the live lake is repeated whole-substrate _T_GROWTH
snapshots (83,405 records) that nothing ever reclaims -- MEASURED 2026-08-05, and NO LONGER THE
SHAPE OF THE LAKE. Re-censused read-only 2026-09-06 (specs/022 L6): 4,753,532 records, of which
_T_GROWTH is 86,095 (1.81%) and _T_COALITION 3,812,834 (80.21%). Both numbers were true when
taken; the 2026-08-05 one is kept because it is why this instrument exists, and dated because a
ratio without its date reads as current. Four declared classes -- 0x02, 0x06, 0x09, 0x0b -- held
ZERO records at the 2026-09-06 census.

Tag labels (keyed to home.py's _T_* constants BY SYMBOL, same census as
lake_migrate.TAG_DISPOSITIONS):
  IMMORTAL     _T_FACT, _T_EPISODE, _T_SELFBUILD, _T_SPIKE, _T_CALIB, _T_STATE, and
               _T_SYNDELTA (0x09, SynapseDelta below -- every record load-bearing by
               construction, no FULL kind to compact against yet; T010a)
  COMPACTABLE  _T_GROWTH (legacy snapshots, closed to npz writes since T010-ii) and
               _T_GROWTH_DELTA (the chain) -- both `reclaimable=True` since the T010-ii
               reconstruction proof landed (tests/test_growth_delta_chain.py: chain replay
               reconstructs the substrate and REFUSES tampered links); the flag records
               that proof, and each row's reason records what a reclaimer still owes.
               _T_BLOCKDELTA (0x0D, BlockDeltaRecord below, specs/022 T067) joins the class:
               a delta the sleep-night merge has folded is superseded by the merged 0x0C
               record, which is proven equal to base + deltas re-applied from the lake
               before its pointer moves.

Bare invocation (run_selfchecks): registry completeness vs lake_migrate's disposition table
(every dispositioned tag owns a retention label -- one census, two views, never divergent),
budget wiring on a constructed store, and the FLOOD red with its calm positive control.
"""

from __future__ import annotations

import json
import struct
import sys
import threading
from pathlib import Path

from snn_data_lake.substrate.lake_migrate import TAG_DISPOSITIONS

REPO = Path(__file__).resolve().parents[3]
APPEND_BASELINE = REPO / "tools" / "lake_append_baseline.json"
FLOOD_FACTOR = 8.0  # a class appending >8x its recorded per-tick p95 is flooding, not learning

#: class label -> (retention, reclaimable_today, reason)
RETENTION: dict[int, tuple[str, bool, str]] = {
    0x01: (
        "IMMORTAL",
        False,
        "_T_FACT: learned knowledge is never reclaimed. THE PER-RECORD EXEMPTION, stated here because "
        "the label is per TAG while the one reclaimer decides per RECORD, and that gap is how the two "
        "drifted apart (specs/022 L6, 2026-09-06): a fact whose `src` names a bulk re-derivable ingest "
        "(lake_consolidation.RE_DERIVABLE_SRC -- WordNet/ConceptNet and the like) MAY be reclaimed, "
        "because it reconstructs at boot from that same source, which is the COMPACTABLE contract "
        "applied per record. Everything else under this tag is LEARNED and is never dropped: the "
        "sleep-tick cap that used to drop it is refused by `LakeConsolidation.consolidate`, and "
        "`KnowledgeLog.consolidate` refuses a rewrite that would lose one. Measured when the exemption "
        "landed: 26,037 facts in the live lake, ALL learned, ZERO re-derivable -- so the cap had never "
        "bound and no learned record is known to have been lost. Append budget: UNBASELINED (reported, "
        "never trips) -- this tag has no row in tools/lake_append_baseline.json, so removing the cap "
        "leaves the learned class with NO live bound, which L4's first AppendBudget caller does not "
        "supply either (it is scoped to the _T_STATE family). Disclosed rather than implied, after the "
        "2026-09-06 panel measured the claim that the budget replaced the cap and refuted it",
    ),
    0x02: ("IMMORTAL", False, "_T_SPIKE: AER exemplars are evidence"),
    0x03: (
        "COMPACTABLE",
        True,
        "_T_GROWTH whole-substrate snapshots: 97.64% of the lake, the measured pathology. The "
        "LEGACY class -- closed to npz writes since T010-ii (new saves land under 0x08). "
        "reclaimable flipped True by the landed reconstruction proof "
        "(tests/test_growth_delta_chain.py); a reclaimer still owes per-record proof of "
        "state-identity with a retained neighbor (the chain writer's member-hash census is "
        "that instrument) before dropping anything -- the flag licenses reclaim, it never "
        "performs it",
    ),
    0x04: ("IMMORTAL", False, "_T_EPISODE: lived experience is never reclaimed"),
    0x05: ("IMMORTAL", False, "_T_SELFBUILD: the self-modification archive is the rollback"),
    0x06: ("IMMORTAL", False, "_T_CALIB: calibration provenance"),
    0x07: (
        "IMMORTAL",
        False,
        "_T_STATE: name-keyed state frames (upserted by replay; small, except the coverage ledger, which "
        "scales with the body's workspace -- T038). THE PER-RECORD EXEMPTION, stated here on the same "
        "grounds 0x01's is (specs/022 L6 fold, 2026-09-06, after the adversarial panel measured this class "
        "being reclaimed on the live lake while the label said otherwise): a SUPERSEDED frame -- any record "
        "under a name that a later record replaces -- MAY be dropped, because no read path can ever return "
        "it again (`load_named_state` and `_state_index` both take the LAST record per name), so dropping "
        "it loses no state. The CURRENT frame for every name is never dropped, and "
        "`home._prune_state_history` reads this label and refuses if the exemption is ever removed",
    ),
    0x08: (
        "COMPACTABLE",
        True,
        "_T_GROWTH_DELTA chain records (specs/022 T010-ii): a prefix ending BEFORE the newest "
        "FULL is reconstructible from that FULL (replay resets there), so whole prefixes may be "
        "reclaimed -- never a link inside a chain, which the reader would REFUSE by hash. "
        "Append budget: baselined at T012 as promised (2,902 B/tick p95, scratch-brain method "
        "in tools/lake_append_baseline.json; 31% under the legacy 4,205 -- live refits "
        "genuinely rewrite W/b/beta, so member-granularity keeps them; per-synapse granularity "
        "is 0x09's job at the kernel block)",
    ),
    0x09: (
        "IMMORTAL",
        False,
        "_T_SYNDELTA spike-derived synapse deltas (SynapseDelta, specs/022 T010a): compare-drop "
        "means every record is load-bearing -- there are no redundant snapshots to reclaim, by "
        "construction -- and no FULL-snapshot kind exists yet, so there is no base to compact a "
        "prefix against. The label revisits with the kernel block's own proof (T013+). Append "
        "budget: unbaselined (reported, never trips) until a live writer exists to measure",
    ),
    0x0A: (
        "IMMORTAL",
        False,
        "_T_COALITION settled workspace coalitions (global_workspace.Coalition, specs/022 "
        "T031): the recall contract operates here -- Episode consolidation consumes sequences "
        "of these, so every record is lived experience (same class as _T_EPISODE). Append "
        "budget: BASELINED 2026-08-21 (p95 434 B/tick, the documented instrument over "
        "make_brain x120; sleep ticks emit nothing) -- the flood assert is live for this tag",
    ),
    0x0B: (
        "IMMORTAL",
        False,
        "_T_GOAL goal event rows (learning_progress.GoalStore, specs/022 T046a): the brain's "
        "own intentions and their LP/outcome history are never reclaimed -- the store's "
        "materialized view folds every row, so each is load-bearing. Append budget: "
        "unbaselined (goals are LOW-frequency, created deliberately, never per-tick; a "
        "baseline lands with the first live per-tick writer if one ever exists)",
    ),
    0x0C: (
        "IMMORTAL",
        False,
        "_T_UNITBLOCK per-block out-edge records (UnitBlockRecord below, specs/022 T066): the "
        "block IS the synaptic identity of its units and there is no snapshot to reconstruct it "
        "from, so nothing here is reclaimable. The pager's PAGE-OUT destroys no record -- it "
        "changes residency in RAM only, which its lake-digest identity arm measures (the store's "
        "segment bytes before and after a full page-out/page-in cycle are equal). Append budget: "
        "unbaselined (reported, never trips) -- the store's writes are a ONE-TIME BUILD, not "
        "per-tick appends, so there is no per-tick rate to bound; the append-budget class for "
        "per-tick growth deltas at the block grain is T067's",
    ),
    0x0D: (
        "COMPACTABLE",
        False,
        "_T_BLOCKDELTA per-block out-edge deltas (BlockDeltaRecord below, specs/022 T067): the "
        "per-tick class at the block grain that RETENTION[0x0C] named as this row's. COMPACTABLE "
        "because a delta the sleep-night merge has FOLDED is superseded by the merged 0x0C record "
        "it was folded into, and that record is proven equal to base + deltas re-applied from the "
        "lake BEFORE the pointer moves (lake_compactor.LakeCompactor's relocation proof, whose "
        "reachable red is a merge whose apply skips one delta) -- the same state-identity claim "
        "0x08 makes for a prefix ending before its newest FULL, at the block grain. reclaimable "
        "is FALSE AT INTRODUCTION and flips only when that reconstruction proof has LANDED and "
        "been COUNTED (the L6 rule: the flag licenses reclaim, it never performs it); the "
        "per-record exemption a reclaimer must read is SUPERSEDED -- only a delta whose index "
        "meta row records it as folded may ever be touched, never an unmerged one, which the "
        "fresh-tier read still applies. Nothing in the segment family reclaims anything today. "
        "Append budget: unbaselined (reported, never trips) -- the writer calls home.observe_append "
        "at its own write site so the class is OBSERVED from day one, and a bound follows a "
        "recorded row in tools/lake_append_baseline.json or 32 writing ticks of its own",
    ),
}


def retention_of(tag: int) -> tuple[str, bool, str]:
    if tag not in RETENTION:
        raise KeyError(f"tag {tag:#04x} has no retention label -- label it at introduction (P4b)")
    return RETENTION[tag]


def registry_consistent() -> tuple[bool, str]:
    """One census, two views: every dispositioned tag owns a retention label and vice versa."""
    disp, ret = set(TAG_DISPOSITIONS), set(RETENTION)
    if disp != ret:
        return False, f"registry divergence: dispositions {sorted(disp)} vs retention {sorted(ret)}"
    return True, f"{len(ret)} classes labeled, dispositions and retention agree"


# distinct-from: substrate/basic_components.py -- its Rule protocol COMPUTES a weight change dw
#   from local activity; this class PERSISTS already-computed weights as lake records.
# distinct-from: substrate/home.py -- the _T_GROWTH_DELTA chain diffs whole npz MEMBERS of the
#   grown-substrate blob; this diffs individual synapses inside a weight matrix, keyed by spikes.
# distinct-from: claustrum/claustrum.py -- detects cross-area spike coincidence windows.
# distinct-from: thalamus/global_workspace.py -- formats an ASCII trace of one tick.
# distinct-from: basal_ganglia/self_build_loop.py -- snapshots safety-envelope FILES for restore.
# All sites read.
class SynapseDelta:
    """Spike-derived synapse-weight delta records (specs/022 T010a) -- tag 0x09.

    THE DIRTY SET IS SPIKE-DERIVED: only synapses with a fired PRE or fired POST endpoint are
    candidates this tick, which is the locality every plasticity rule in this tree already obeys
    (e-prop/STDP move weights only where activity was). Candidates are QUANTIZED to `q_step` and an
    entry is emitted ONLY when its quantized value differs from what the lake last saw
    (compare-drop). Two consequences, both deliberate:
      * a byte-frozen substrate whose endpoints keep firing emits NOTHING -- that is the measured
        97.64%-repeated-snapshot pathology (83,405 _T_GROWTH records) refused at the door, and the
        constructed red in main() reproduces exactly that shape;
      * a synapse that changes while its endpoints are silent is captured LATER, on the first tick
        an endpoint fires -- deferred, never lost, because the compare is against what was WRITTEN,
        not against the previous tick.

    `q_step` has NO default (P1): the quantum below which a weight change is noise is a property of
    the caller's learning rule, derived there, never a constant picked here. Reconstruction replays
    records onto a ZERO base (`apply`), so the first capture of a synapse carries its full
    quantized value.
    """

    TAG = 0x09
    _VER = 1

    def __init__(self, q_step):
        if not q_step or q_step <= 0:
            raise ValueError(
                "q_step must be a positive weight quantum DERIVED BY THE CALLER from its own "
                "learning rule (P1: this class refuses to pick one)"
            )
        self.q_step = float(q_step)
        self._last: dict[tuple[int, int], int] = {}

    def collect(self, w, pre_fired, post_fired):
        """One tick's record: dirty set -> quantize -> compare-drop -> packed bytes, or None when
        nothing the lake would learn from changed (writing nothing IS the mechanism)."""
        import numpy as np

        w = np.asarray(w)
        if w.ndim != 2:
            raise ValueError("w must be a 2-D [pre, post] weight array")
        n_pre, n_post = w.shape
        pre = sorted({int(i) for i in pre_fired})
        post = sorted({int(j) for j in post_fired})
        if (pre and (pre[0] < 0 or pre[-1] >= n_pre)) or (
            post and (post[0] < 0 or post[-1] >= n_post)
        ):
            raise ValueError("fired index out of range for w")
        entries, seen = [], set()
        for i in pre:
            row = np.rint(w[i] / self.q_step).astype(np.int64)
            for j in range(n_post):
                q = int(row[j])
                if self._last.get((i, j), 0) != q:
                    entries.append((i, j, q))
                    seen.add((i, j))
        for j in post:
            col = np.rint(w[:, j] / self.q_step).astype(np.int64)
            for i in range(n_pre):
                if (i, j) in seen:
                    continue
                q = int(col[i])
                if self._last.get((i, j), 0) != q:
                    entries.append((i, j, q))
        if not entries:
            return None
        for i, j, q in entries:
            self._last[(i, j)] = q
        out = [struct.pack("<BI", self._VER, len(entries))]
        out += [struct.pack("<IIq", i, j, q) for i, j, q in entries]
        return b"".join(out)

    @classmethod
    def entries(cls, payload):
        """-> [(pre, post, q)]. Refuses a malformed payload WHOLE (wrong version, short, trailing
        bytes) -- the same never-half-read rule every lake parser here follows."""
        try:
            ver, n = struct.unpack_from("<BI", payload, 0)
        except struct.error as exc:
            raise ValueError(f"SynapseDelta payload too short for its header: {exc}") from exc
        if ver != cls._VER:
            raise ValueError(f"SynapseDelta version {ver} is not {cls._VER}")
        if len(payload) != 5 + n * 16:
            raise ValueError(
                f"SynapseDelta length mismatch: {len(payload)} B for {n} entries -- refused whole"
            )
        return [struct.unpack_from("<IIq", payload, 5 + k * 16) for k in range(n)]

    def apply(self, payload, w):
        """Apply one record to `w` in place (zero-base reconstruction). Returns the entry count."""
        got = self.entries(payload)
        for i, j, q in got:
            w[i, j] = q * self.q_step
        return len(got)


# distinct-from: SynapseDelta above -- that is a spike-derived DELTA of individual weights inside
#   one matrix, compare-dropped tick by tick (tag 0x09); this is the WHOLE out-edge set of a block
#   of units, written once as the block's synaptic identity (tag 0x0C) and read back to compile
#   delivery bundles. Different tag, different cadence, different consumer.
# distinct-from: substrate/unit_blocks.py::UnitBlocks.group -- that PARTITIONS entries by the block
#   their post unit lives in and frames nothing; this is the on-disk record a block becomes. The
#   partition arithmetic is adopted from it (P2), not re-derived here.
# distinct-from: substrate/kernel_state.py::KernelState.to_blob -- the per-unit STATE frame
#   (_T_STATE, kernel_state.v1). This record deliberately carries NO state: a second copy of v/thr
#   inside a block record would be the P2/P4b duplicate. Sites read 2026-09-08.
class UnitBlockRecord:
    """One lake record per UNIT BLOCK -- `_T_UNITBLOCK` = 0x0C (specs/022 T066).

    LAYOUT (pre-registration 2.1): the class tag byte, then `struct '<BIIH'` = (version=1,
    block_id, n_units_in_block, block_size), then the block's out-edges as four parallel
    little-endian arrays of equal length -- `pre_local <u2` (the pre unit's index INSIDE the
    block), `post <u4` (a global unit index), `weight <f4`, `delay <u2`.

    THE TAG BYTE LEADS, and the pre-registration's 2.1 named the `<BIIH` header without saying
    so. It is added deliberately, not silently: the lake's dispatch convention is that a record's
    class is `payload[0]` (`lake_migrate._tag_of`, `lake_rebuild_gate.build_index` both read it),
    so without the byte this class's own `TAG_DISPOSITIONS[0x0C]` row would be unreachable and a
    migration would read these records as `_T_FACT` (version=1 = 0x01). Disclosed in the row's
    evidence file rather than left as a silent refinement.

    The edge COUNT is not in the header on purpose -- it is `(len(payload) - 12) / 12`, so a
    truncated or padded record fails the division and is refused WHOLE. Malformed means refused
    whole in every case (foreign tag, unknown version, short header, length mismatch, a
    `pre_local` outside the block), the same never-half-read rule `SynapseDelta.entries` follows:
    a block whose edge list half-parsed would compile a synaptic identity nobody ever wrote.

    WEIGHTS ARE STORED AT `<f4`. A caller whose weights are not exactly representable in float32
    will see them rounded at the record boundary; that is the declared dtype, and the paging
    row's byte-identity fixtures use multiples of 0.25 so the rounding cannot be mistaken for a
    paging defect.
    """

    TAG = 0x0C
    _VER = 1
    _HEAD = struct.Struct("<BIIH")
    _ITEM = 12  # pre_local u2 + post u4 + weight f4 + delay u2

    def __init__(self, block_id, n_units_in_block, block_size, pre_local, post, weight, delay):
        import numpy as np

        self.block_id = int(block_id)
        self.n_units_in_block = int(n_units_in_block)
        self.block_size = int(block_size)
        if self.block_id < 0 or self.block_size <= 0:
            raise ValueError("UnitBlockRecord needs a non-negative block id and a positive size")
        if not (0 < self.n_units_in_block <= self.block_size):
            raise ValueError(
                f"a block holds 1..{self.block_size} units (the tail block is SHORT, never "
                f"padded), got {self.n_units_in_block}"
            )
        self.pre_local = np.asarray(pre_local, dtype="<u2")
        self.post = np.asarray(post, dtype="<u4")
        self.weight = np.asarray(weight, dtype="<f4")
        self.delay = np.asarray(delay, dtype="<u2")
        lengths = {len(self.pre_local), len(self.post), len(self.weight), len(self.delay)}
        if len(lengths) != 1:
            raise ValueError(
                f"the four edge arrays must be parallel, got lengths {sorted(lengths)}"
            )
        if len(self.pre_local) and int(self.pre_local.max()) >= self.n_units_in_block:
            raise ValueError("a pre_local index falls outside its own block -- refused whole")

    def __len__(self):
        return int(len(self.pre_local))

    def pre_global(self):
        """The block's out-edge PRE indices in the kernel's global index space."""
        import numpy as np

        return self.block_id * self.block_size + self.pre_local.astype(np.int64)

    def encode(self) -> bytes:
        return b"".join(
            (
                bytes([self.TAG]),
                self._HEAD.pack(self._VER, self.block_id, self.n_units_in_block, self.block_size),
                self.pre_local.tobytes(),
                self.post.tobytes(),
                self.weight.tobytes(),
                self.delay.tobytes(),
            )
        )

    @classmethod
    def decode(cls, payload):
        import numpy as np

        if not isinstance(payload, (bytes, bytearray)) or len(payload) < 1 + cls._HEAD.size:
            raise ValueError("UnitBlockRecord payload is shorter than its header -- refused whole")
        payload = bytes(payload)
        if payload[0] != cls.TAG:
            raise ValueError(
                f"UnitBlockRecord tag {payload[0]:#04x} is not {cls.TAG:#04x} -- refused whole"
            )
        ver, block_id, n_units_in_block, block_size = cls._HEAD.unpack_from(payload, 1)
        if ver != cls._VER:
            raise ValueError(f"UnitBlockRecord version {ver} is not {cls._VER} -- refused whole")
        body = payload[1 + cls._HEAD.size :]
        if len(body) % cls._ITEM:
            raise ValueError(
                f"UnitBlockRecord length mismatch: {len(body)} array bytes is not a whole number "
                f"of {cls._ITEM}-byte edges -- refused whole"
            )
        n = len(body) // cls._ITEM
        cut = [0, 2 * n, 6 * n, 10 * n, 12 * n]
        return cls(
            block_id,
            n_units_in_block,
            block_size,
            np.frombuffer(body[cut[0] : cut[1]], dtype="<u2"),
            np.frombuffer(body[cut[1] : cut[2]], dtype="<u4"),
            np.frombuffer(body[cut[2] : cut[3]], dtype="<f4"),
            np.frombuffer(body[cut[3] : cut[4]], dtype="<u2"),
        )


OP_UPSERT = 0
OP_REMOVE = 1


# distinct-from: UnitBlockRecord directly above -- that is the WHOLE out-edge set of a block, its
#   synaptic identity, written once (0x0C); this is a CHANGE to that set, written per growth event
#   (0x0D), PRE-keyed the same way so the pager's pre_global() -> compile_bundles path composes
#   without a second compiler. Different tag, different cadence, and the merge below is what turns
#   a run of these back into one of those.
# distinct-from: SynapseDelta above -- that is the LEARNER's per-synapse delta over a dense
#   [pre, post] weight matrix, POST-grouped and quantized at the learner's own lr (0x09). This is
#   a STRUCTURAL delta over one block's out-edge list, PRE-keyed and stored at the record's own f4
#   grain. 0x09 is NOT retired and NOT reused: when the kernel becomes THE substrate at S6, 0x09's
#   "revisits with the kernel block's own proof" clause is discharged by this class and 0x09
#   retires (the P2 disposition, recorded here rather than left implicit).
# distinct-from: substrate/home.py's _T_GROWTH_DELTA chain -- that diffs whole npz MEMBERS of the
#   grown-substrate blob inside the BinLog container, with a sha256 manifest per link and a FULL
#   every 64. Its SHAPE is adopted here (additive tag, generation base, proof-then-commit,
#   latest-wins) and its CODE is not shared: a different container and a different grain, so a
#   common codec would be a merge across tag families this row does not attempt.
#   All sites read 2026-09-08.
class BlockDeltaRecord:
    """One change to ONE unit block's out-edges -- `_T_BLOCKDELTA` = 0x0D (specs/022 T067).

    LAYOUT (pre-registration 2.1): the class tag byte, then `struct '<BIIQ'` = (version=1,
    block_id, n_units_in_block, base_generation), then five parallel little-endian arrays of equal
    length -- `pre_local <u2` (the pre unit's index INSIDE the block), `post <u4` (a global unit
    index), `weight <f4`, `delay <u2`, `op <u1` (0 = upsert the edge (pre_local, post, delay) with
    this weight, 1 = remove it).

    AN ENTRY ADDRESSES AN EDGE BY THE TRIPLE `(pre_local, post, delay)`, which is why `edge_table`
    below refuses a base record that carries the same triple twice: two such edges are not
    separately addressable, and collapsing them silently would lose one.

    `base_generation` names the merged 0x0C generation this delta applies to -- 0 is the built
    record, and each merge of the block advances it by one. A delta whose base is not the block's
    CURRENT generation is REFUSED at apply time and never half-applied, which is the 0x08 chain's
    "a DELTA whose base is not what the chain holds" refusal at the block grain.

    The edge COUNT is not in the header, for the same reason it is not in `UnitBlockRecord`'s: it
    is `(len(payload) - 18) / 13`, so a truncated or padded record fails the division and is
    refused WHOLE. Malformed means refused whole in every case (foreign tag, unknown version,
    short header, length mismatch, a `pre_local` outside the block, an `op` that is neither).
    """

    TAG = 0x0D
    _VER = 1
    _HEAD = struct.Struct("<BIIQ")
    _ITEM = 13  # pre_local u2 + post u4 + weight f4 + delay u2 + op u1

    def __init__(
        self, block_id, n_units_in_block, base_generation, pre_local, post, weight, delay, op
    ):
        import numpy as np

        self.block_id = int(block_id)
        self.n_units_in_block = int(n_units_in_block)
        self.base_generation = int(base_generation)
        if self.block_id < 0 or self.n_units_in_block <= 0 or self.base_generation < 0:
            raise ValueError(
                "BlockDeltaRecord needs a non-negative block id, a positive unit count and a "
                "non-negative base generation"
            )
        self.pre_local = np.asarray(pre_local, dtype="<u2")
        self.post = np.asarray(post, dtype="<u4")
        self.weight = np.asarray(weight, dtype="<f4")
        self.delay = np.asarray(delay, dtype="<u2")
        self.op = np.asarray(op, dtype="<u1")
        lengths = {
            len(self.pre_local),
            len(self.post),
            len(self.weight),
            len(self.delay),
            len(self.op),
        }
        if len(lengths) != 1:
            raise ValueError(
                f"the five delta arrays must be parallel, got lengths {sorted(lengths)}"
            )
        if len(self.pre_local) and int(self.pre_local.max()) >= self.n_units_in_block:
            raise ValueError("a pre_local index falls outside its own block -- refused whole")
        if len(self.op) and not set(int(o) for o in self.op) <= {OP_UPSERT, OP_REMOVE}:
            raise ValueError(f"op is {OP_UPSERT} (upsert) or {OP_REMOVE} (remove) -- refused whole")

    def __len__(self):
        return int(len(self.pre_local))

    def encode(self) -> bytes:
        return b"".join(
            (
                bytes([self.TAG]),
                self._HEAD.pack(
                    self._VER, self.block_id, self.n_units_in_block, self.base_generation
                ),
                self.pre_local.tobytes(),
                self.post.tobytes(),
                self.weight.tobytes(),
                self.delay.tobytes(),
                self.op.tobytes(),
            )
        )

    @classmethod
    def decode(cls, payload):
        import numpy as np

        if not isinstance(payload, (bytes, bytearray)) or len(payload) < 1 + cls._HEAD.size:
            raise ValueError("BlockDeltaRecord payload is shorter than its header -- refused whole")
        payload = bytes(payload)
        if payload[0] != cls.TAG:
            raise ValueError(
                f"BlockDeltaRecord tag {payload[0]:#04x} is not {cls.TAG:#04x} -- refused whole"
            )
        ver, block_id, n_units_in_block, base_generation = cls._HEAD.unpack_from(payload, 1)
        if ver != cls._VER:
            raise ValueError(f"BlockDeltaRecord version {ver} is not {cls._VER} -- refused whole")
        body = payload[1 + cls._HEAD.size :]
        if len(body) % cls._ITEM:
            raise ValueError(
                f"BlockDeltaRecord length mismatch: {len(body)} array bytes is not a whole number "
                f"of {cls._ITEM}-byte entries -- refused whole"
            )
        n = len(body) // cls._ITEM
        cut = [0, 2 * n, 6 * n, 10 * n, 12 * n, 13 * n]
        return cls(
            block_id,
            n_units_in_block,
            base_generation,
            np.frombuffer(body[cut[0] : cut[1]], dtype="<u2"),
            np.frombuffer(body[cut[1] : cut[2]], dtype="<u4"),
            np.frombuffer(body[cut[2] : cut[3]], dtype="<f4"),
            np.frombuffer(body[cut[3] : cut[4]], dtype="<u2"),
            np.frombuffer(body[cut[4] : cut[5]], dtype="<u1"),
        )


# distinct-from: SynapseDelta.apply above -- that writes quantized weights into a dense [pre, post]
#   matrix in place on a ZERO base. These three functions carry a BLOCK's out-edge list, keyed by
#   (pre_local, post, delay), from one immutable record to the next. Site read 2026-09-08.
def edge_table(record) -> dict:
    """A block record's out-edges as `(pre_local, post, delay) -> f4 weight`, in the record's own
    array order. REFUSES a record that repeats a triple: the delta tier addresses an edge by that
    triple, so two edges sharing one are not addressable and collapsing them would lose an edge
    silently. `dict` keeps insertion order, which is what makes the re-encoded record
    deterministic."""
    import numpy as np

    table: dict[tuple[int, int, int], float] = {}
    for i in range(len(record)):
        key = (int(record.pre_local[i]), int(record.post[i]), int(record.delay[i]))
        if key in table:
            raise ValueError(
                f"block {record.block_id} carries the edge {key} twice -- the delta tier "
                "addresses an edge by (pre_local, post, delay) and cannot name either copy"
            )
        table[key] = float(np.float32(record.weight[i]))
    return table


def apply_deltas(base_edges, deltas, *, generation: int) -> dict:
    """THE ONE APPLY (P2): base edge table + 0x0D records -> the post-delta edge table.

    Used by `UnitBlockStore.fetch` (the fresh tier reads base + unmerged deltas) and by
    `lake_compactor.LakeCompactor.merge` (the fold writes exactly what the fetch would have read),
    so a merged record can never disagree with the record the pager was already serving.

    `generation` is the block's CURRENT merged generation and is a REQUIRED keyword: the
    pre-registration's signature named `(base_edges, deltas)`, and the refusal it also requires --
    a delta whose `base_generation` is not the current one -- is not computable without it. Stated
    as a deviation rather than dropped, because dropping it would delete the refusal.

    An upsert of an existing triple REPLACES its weight IN PLACE (the edge keeps its position); a
    new triple is appended; a remove deletes it. Refusal is WHOLE and before any entry lands."""
    import numpy as np

    for record in deltas:
        if int(record.base_generation) != int(generation):
            raise ValueError(
                f"delta for block {record.block_id} names base generation "
                f"{int(record.base_generation)} while the block is at {int(generation)} -- "
                "refused whole, never half-applied"
            )
    edges = dict(base_edges)
    for record in deltas:
        for i in range(len(record)):
            key = (int(record.pre_local[i]), int(record.post[i]), int(record.delay[i]))
            if int(record.op[i]) == OP_REMOVE:
                edges.pop(key, None)
            else:
                edges[key] = float(np.float32(record.weight[i]))
    return edges


def compare_drop(record, base_edges):
    """The delta the lake would LEARN from: entries that change nothing at the f4 record grain are
    dropped, and a record left with no entry is not written at all (D4's
    quantize-then-compare-then-append, at the grain this record actually stores). Returns a new
    `BlockDeltaRecord` or None."""
    import numpy as np

    keep = []
    for i in range(len(record)):
        key = (int(record.pre_local[i]), int(record.post[i]), int(record.delay[i]))
        op = int(record.op[i])
        weight = float(np.float32(record.weight[i]))
        if op == OP_REMOVE:
            if key not in base_edges:
                continue
        elif base_edges.get(key) == weight:
            continue
        keep.append((key[0], key[1], weight, key[2], op))
    if not keep:
        return None
    return BlockDeltaRecord(
        block_id=record.block_id,
        n_units_in_block=record.n_units_in_block,
        base_generation=record.base_generation,
        pre_local=[e[0] for e in keep],
        post=[e[1] for e in keep],
        weight=[e[2] for e in keep],
        delay=[e[3] for e in keep],
        op=[e[4] for e in keep],
    )


def record_from_edges(block_id, n_units_in_block, block_size, edges) -> UnitBlockRecord:
    """An edge table back into a 0x0C record, in the table's own order -- deterministic, so
    encoding the same table twice yields the same bytes and the relocation proof can compare
    them."""
    keys = list(edges)
    return UnitBlockRecord(
        block_id=block_id,
        n_units_in_block=n_units_in_block,
        block_size=block_size,
        pre_local=[k[0] for k in keys],
        post=[k[1] for k in keys],
        weight=[edges[k] for k in keys],
        delay=[k[2] for k in keys],
    )


# : How many of a tag's own WRITING ticks the running bar is computed over. Sized from what the bar
# needs --
# : a p95 is meaningless below ~20 samples -- and kept small enough that the window tracks a brain
# whose
# : frames grow: at the measured cadence (67 writing ticks in 1200) this is roughly the last 500
# ticks.
# : DISCLOSED, and the reason it is not larger: a longer window is a slower-staling fixed ceiling,
# which is
#: the defect this replaces.
RUNNING_WINDOW = 32


class AppendBudget:
    """Per-tag bytes/tick accounting with a flood assert against the tag's OWN RECENT rate.

    WHY NOT THE RECORDED BASELINE (specs/022 L4b, 2026-09-06). It was a fixed p95 measured once, and the
    quantity it bounds is not stationary: measured over 1200 ticks, per-tick 0x07 bytes climb monotonically
    52,763 -> 77,998 B with no settling, because _T_STATE carries WHOLE FRAMES that grow with the brain
    (growing_learners.json x2.3 over the run, coverage_cells x18.7). The live daemon duly reported 7 floods
    in 8 nights at 160,470 B against a 12,381 B ceiling recorded in August -- a stuck alarm, not a
    detection, and a stuck alarm is worse than none because it teaches its reader to ignore the real one.
    P1 names the fix: running-baseline-relative, never a magic constant held fixed.

    WHAT THIS DOES NOT DO. It does not detect a RAMP -- a class growing without bound over a brain's life,
    which is the pathology the module docstring names. A flood assert never detected that (a fixed p95 fires
    once, forever, and says nothing about the slope), and this one does not either. Two pathologies, and
    conflating them is what produced the stuck alarm.
    """

    def __init__(self, baseline: dict | None = None):
        if baseline is None:
            baseline = json.loads(APPEND_BASELINE.read_text()) if APPEND_BASELINE.exists() else {}
        # The recorded numbers PRE-FILL each tag's window rather than standing as a permanent
        # ceiling.
        # Day-one protection is preserved -- a seeded tag is bounded from its first write, and stays
        # bounded by the SEED for its first three (a 32-window's p95 is its 30th value) -- and it
        # then
        # MIGRATES to what the brain actually writes as real ticks push the seed out, so a stale
        # recording
        # self-corrects within a few writes instead of alarming forever. Measured migration cost is
        # in the
        # self-check below and in evidence/l4b-append-ceiling-nonstationary.md.
        self._p95 = {int(k): float(v) for k, v in baseline.get("per_tag_p95_bytes", {}).items()}
        self._recent: dict[int, list[int]] = {
            t: [int(v)] * RUNNING_WINDOW for t, v in self._p95.items()
        }
        self._seed_vals = {t: (float(v),) for t, v in self._p95.items()}
        self._tick_bytes: dict[int, int] = {}
        # THIS PROJECT RUNS A FREE-THREADING INTERPRETER, so read-modify-write on a dict is not
        # atomic.
        # Measured by an adversarial panel (2026-09-06): 8 threads x 20,000 appends counted 36,431
        # of
        # 160,000 -- 78% lost -- and end_tick could raise dict-changed-size out of the caller's
        # tick.
        self._lock = threading.Lock()

    def on_append(self, tag: int, n_bytes: int) -> None:
        with self._lock:
            self._tick_bytes[tag] = self._tick_bytes.get(tag, 0) + int(n_bytes)

    @staticmethod
    def _p95_of(xs) -> float:
        s = sorted(xs)
        return float(s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))])

    def bar(self, tag: int):
        """(bar_bytes, source) for `tag`, or (None, "unbaselined"). The tag's own recent writing ticks once
        it has enough of them; the recorded seed until then; nothing if it has neither."""
        recent = self._recent.get(tag)
        if not recent:
            return None, "unbaselined"
        # A TAG WITH NO RECORDED ROW STAYS UNBASELINED UNTIL IT HAS A FULL WINDOW OF ITS OWN.
        # Without this
        # gate the first form of this method handed such a tag a live bar computed from ONE sample,
        # because
        # end_tick's setdefault creates a window for anything that writes -- inverting "a budget
        # nobody
        # measured is not a budget", which is the rule that stops this instrument inventing a bound
        # for a
        # class nobody baselined. 0x01, the class L6 left unbounded, is precisely that class. Found
        # by an
        # adversarial panel, 2026-09-06.
        if tag not in self._p95 and len(recent) < RUNNING_WINDOW:
            return None, "unbaselined"
        seeded = sum(1 for x in recent if float(x) in self._seed_vals.get(tag, ()))
        src = "its own recent rate" if seeded == 0 else f"a window {seeded}/{len(recent)} seeded"
        return self._p95_of(recent) * FLOOD_FACTOR, src

    def end_tick(self) -> list[str]:
        """Close the tick; return violations (empty = green).

        A tag with neither a window nor a recorded seed never trips -- a budget nobody measured is not a
        budget, and the caller REPORTS those instead. The tick's bytes join the tag's window AFTER the
        comparison, so a tick is always judged against history rather than against itself."""
        violations = []
        with self._lock:  # swap out, then scan outside the lock
            taken, self._tick_bytes = self._tick_bytes, {}
        for tag, got in taken.items():
            bar, source = self.bar(tag)
            if bar is not None and got > bar:
                retention, _, _ = retention_of(tag)
                violations.append(
                    f"FLOOD: tag {tag:#04x} ({retention}) appended {got} B this tick "
                    f"> {FLOOD_FACTOR}x {source} ({bar / FLOOD_FACTOR:.0f} B)"
                )
            if (
                got > 0
            ):  # only WRITING ticks enter the window; silent ticks would drag the bar to zero
                w = self._recent.setdefault(tag, [])
                w.append(int(got))
                del w[:-RUNNING_WINDOW]
        return violations


def main() -> int:
    ok, msg = registry_consistent()
    if not ok:
        print(f"lake_engine self-check FAILED: {msg}")
        return 1
    # Constructed budget: calm control and the FLOOD red share one run.
    budget = AppendBudget(baseline={"per_tag_p95_bytes": {"1": 1000.0}})
    budget.on_append(0x01, 500)
    calm = budget.end_tick()
    budget.on_append(0x01, 9000)
    flood = budget.end_tick()
    if calm:
        print(f"lake_engine self-check FAILED: calm tick tripped the budget ({calm})")
        return 1
    if not flood:
        print("lake_engine self-check FAILED: the constructed flood did not trip")
        return 1
    # ---- L4b (2026-09-06): the bar is the tag's OWN RECENT RATE, so it owes three more arms
    # ----------
    # CONTROL 1 -- ORDINARY GROWTH MUST NOT FLAG. This INTERPOLATES the measured window-p95
    # trajectory of a 1200-tick brain; it is NOT the per-tick series, whose real shape alternates
    # small and large. Its largest step is 1.006x against an 8x bar and its maximum sits UNDER the
    # 99,048 B seed floor, so it cannot fail on two independent counts -- which the panel of
    # 2026-09-06 pointed out, and which is why the UNSEEDED arm below exists beside it.
    grow = [round(52763 * (77998 / 52763) ** (i / 66)) for i in range(67)]
    gb = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    grow_flags = 0
    for n in grow:
        gb.on_append(0x07, n)
        grow_flags += len(gb.end_tick())
    if grow_flags:
        print(f"lake_engine self-check FAILED: ordinary growth flagged {grow_flags} times")
        return 1
    # ...and the SAME series with NO seed, so the seed floor cannot carry it. Here the bar really is
    # the
    # series' own recent rate from the first full window on, and a flag would mean the adaptive bar
    # itself
    # false-alarms on ordinary growth. THIS is the arm that can fail.
    ub = AppendBudget(baseline={})
    unseeded_flags = 0
    for n in grow:
        ub.on_append(0x07, n)
        unseeded_flags += len(ub.end_tick())
    if unseeded_flags:
        print(
            f"lake_engine self-check FAILED: unseeded ordinary growth flagged {unseeded_flags} times"
        )
        return 1
    # RED -- a real spike, AFTER the window is all-real, must still flag.
    sb = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    for n in grow[:RUNNING_WINDOW]:
        sb.on_append(0x07, n)
        sb.end_tick()
    sb.on_append(0x07, int(max(grow[:RUNNING_WINDOW]) * FLOOD_FACTOR * 2))
    if not sb.end_tick():
        print("lake_engine self-check FAILED: a real spike did not trip the running bar")
        return 1
    # CONTROL 3 -- THE COST, MEASURED AND PRINTED RATHER THAN HIDDEN, and the first version of
    # this block printed the WRONG cost. _p95_of over a RUNNING_WINDOW of 32 returns sorted[29] --
    # the value from THREE writing ticks back -- so the no-flag condition is r**3 <= FLOOD_FACTOR:
    # ANY per-write growth ratio at or under 2.0 never flags, AT ANY MAGNITUDE. The 'goes quiet
    # after 3' below is therefore true only of a STEP to a new constant rate; a flood arriving over
    # three or more writing ticks costs ZERO alarms. The ramped arm underneath pins that so the
    # corrected statement cannot drift (adversarial panel, 2026-09-06).
    mb = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    sustained = 0
    for _ in range(40):
        mb.on_append(0x07, 160470)  # the live daemon's measured per-tick rate
        if mb.end_tick():
            sustained += 1
    if not 1 <= sustained <= RUNNING_WINDOW // 4:
        print(f"lake_engine self-check FAILED: migration took {sustained} ticks, expected 1..8")
        return 1
    # THE WORST CASE, pinned: the same increase delivered as a RAMP instead of a step is invisible.
    rb = AppendBudget(baseline={"per_tag_p95_bytes": {"7": 12381.0}})
    ramped = 0
    n = 12381.0
    for _ in range(67):  # the measured cadence of one 1200-tick run
        n *= 2.0  # r = 2.0 exactly: the boundary of r**3 <= FLOOD_FACTOR
        rb.on_append(0x07, int(n))
        ramped += len(rb.end_tick())
    if ramped:
        print(f"lake_engine self-check FAILED: the ramped arm flagged {ramped} times, expected 0")
        return 1
    print(
        f"  append budget: ordinary growth (52,763 -> 77,998 B over {len(grow)} writes) flags 0 both seeded and UNSEEDED; a real spike still flags; a stale seed costs {sustained} flood(s) to migrate. THE COST AT ITS WORST, not its best: the bar is 8x the p95 of a 32-window, i.e. the value from THREE writing ticks back, so a flood arriving in ONE tick costs {sustained} alarms while one arriving over >=3 writing ticks -- any per-write ratio <= 2.0 -- costs ZERO at ANY magnitude (doubling every write for 67 writes reaches ~9e23 B/tick and flags 0). A self-relative bar sees SLOPE over three writes, never magnitude; a RAMP needs a slope instrument."
    )
    # SynapseDelta (T010a): dirty-set wiring, compare-drop, the constructed pathology red, and a
    # zero-base reconstruction -- the cheap halves; full proofs live in tests/test_synapse_delta.py.
    import numpy as np

    sd = SynapseDelta(q_step=0.01)
    w = np.zeros((4, 4))
    w[0, 1] = 0.5
    w[3, 3] = 0.25
    recs = [sd.collect(w, [0], [])]
    touched = [(i, j) for i, j, _q in SynapseDelta.entries(recs[0])]
    if touched != [(0, 1)]:
        print(f"lake_engine self-check FAILED: dirty set not spike-derived ({touched})")
        return 1
    # The constructed 97.64% red: a frozen substrate whose endpoints keep firing must emit NOTHING
    # (a naive per-tick snapshot of the same dirty set would emit 50 ticks x 4 entries x 16 B).
    frozen = sum(len(r) for r in (sd.collect(w, [0], []) for _ in range(50)) if r)
    if frozen:
        print(f"lake_engine self-check FAILED: frozen substrate emitted {frozen} B (the pathology)")
        return 1
    rec2 = sd.collect(w, [3], [])  # (3,3) changed OUTSIDE the dirty set above -> captured NOW
    if rec2 is None or SynapseDelta.entries(rec2) != [(3, 3, 25)]:
        print("lake_engine self-check FAILED: deferred capture lost an out-of-dirty-set change")
        return 1
    recs.append(rec2)
    twin, sd_replay = np.zeros_like(w), SynapseDelta(q_step=0.01)
    for r in recs:
        sd_replay.apply(r, twin)
    if not np.array_equal(twin, np.rint(w / 0.01) * 0.01):
        print("lake_engine self-check FAILED: zero-base reconstruction diverged")
        return 1
    baseline_state = (
        "recorded"
        if APPEND_BASELINE.exists()
        else "NOT RECORDED (record on the pinned host before the assert binds)"
    )
    print(
        f"lake_engine self-check OK: {msg}; flood red fired; append baseline {baseline_state}; "
        "SynapseDelta dirty-set/compare-drop/deferred-capture/reconstruction OK, frozen-substrate "
        "red emitted 0 B"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Activation-driven paging of unit blocks (specs/022 T066 / Phase 3 S3, FR-010/FR-011).

WHAT IT IS. A block of units is RESIDENT when its compiled out-edge bundles sit in the kernel's
synapse table and its units integrate; PAGED-OUT when its bundles are gone from the table and its
units are masked out of propagate. `NeuronPager` decides which blocks are which, once per tick,
from what the network is actually doing: a paged-out unit with a non-zero pending arrival is DUE,
and the due set is fetched from the lake in ROUNDS under an integer per-tick budget.

THE PAGED QUANTITY IS EDGES, NOT STATE, and that is measured rather than assumed. Per unit the
kernel holds 21 B of `KernelState` and 8 B per partition of accumulator; each out-edge costs 16 B
in the bundle arrays, so at 32 out-edges per unit the edge arrays alone are 512 B/unit -- over
94% of resident bytes. Unit STATE stays resident for every unit (its durable owner is the
`kernel_state.v1` `_T_STATE` frame; a second copy inside a block record would be a P2/P4b
duplicate), and the self-check prints the resident FLOOR per unit that this leaves.

TWO RESOLUTIONS. A resident unit carries full state and edges. A paged-out unit carries exactly
the lossy summary: `KernelTick.pending[p][unit]` (int64, per partition, under the same
single-writer contract as the accumulator) plus its block's `last_active`. That summary is what
DECIDES fetches, and it is also what makes FR-011 true rather than aspirational -- a delivery to a
paged-out unit lands in `pending` and is drained into the accumulator before the next propagate,
so nothing is dropped and, within budget, the run is byte-identical to a fully resident twin.

A ROUND is one `UnitBlockStore.fetch` submission of at most `blocks_per_round` block reads. A tick
whose fetch set holds F blocks costs `ceil(F / blocks_per_round)` submissions. `max_rounds` is the
per-tick budget. Rounds are counted TWICE -- the planner's `rounds_used` (arithmetic over the due
set) and the store's `submissions` (incremented inside `fetch`) -- and no rounds number is
reported unless they agree on every tick. THE SEPARATION IS NARROW and is stated as narrow: the
two counts are one call apart, so they diverge on an empty round issued, a `fetch` that raises
part way through a tick, or an `_issue_rounds` override whose own count does not match what it
submitted (which is exactly what the budget-blind broken arm is). They are not independent
measurements of the same physical quantity, and reading them as one would be the mistake this
sentence exists to prevent.

BOUNDED STALENESS. A due block whose oldest pending arrival has waited `max_stale_ticks` ticks is
FORCED into rounds beyond the budget rather than deferred again, so observed staleness can never
exceed the bound; the cost of that guarantee is visible as a budget TRIP, never as a lost spike.
`max_stale_ticks = 0` therefore forces every drain onto its own tick, which is the regime in which
an overloaded run stays byte-identical to the twin while the budget trips.

NOTHING HERE HAS A DEFAULT (P1). `block_size`, `capacity_blocks`, `blocks_per_round`,
`max_rounds` and `max_stale_ticks` are the caller's, derived from its own bytes-per-block target
and its own per-tick I/O allowance; `main()` derives the two rounds constants from a MEASURED
per-block read latency against the recorded wake-tick p50 and prints the derivation.

NEVER MMAP, and that is the point of the row rather than a style preference: OS eviction under
mmap is blind to activation (research.md D1), so an mmap store would be a residency policy nobody
chose, measured as if it were one. `lake_segments.read_record` opens a plain file handle.

WHAT THIS ROW DOES NOT CLAIM: no paging WIN, no tick-time claim, no bytes-fetched/bytes-used ratio
(T067's metric), no live-tick charge (the pager joins the live tick at S6), no device tier, no
prefetch. It claims that both reds are reachable and counted, that a calm paged run is
byte-identical to the resident twin, that delivered charge is conserved under overload with
staleness bounded by an input, that rounds are integer, budgeted, double-counted and trip on
constructed overload, and that resident edge bytes fall, measured two ways.

  python -m snn_data_lake.substrate.neuron_pager
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from snn_data_lake.substrate.home import _T_UNITBLOCK
from snn_data_lake.substrate.kernel_state import KernelState
from snn_data_lake.substrate.kernel_tick import KernelTick, Synapses, compile_bundles
from snn_data_lake.substrate.lake_engine import UnitBlockRecord, retention_of
from snn_data_lake.substrate.lake_index import LakeIndex
from snn_data_lake.substrate.lake_rebuild_gate import INDEX_NAME
from snn_data_lake.substrate.lake_segments import SEGMENT_DIR, SegmentStore, read_record
from snn_data_lake.substrate.unit_blocks import UnitBlocks

REPO = Path(__file__).resolve().parents[3]
PERF_BASELINE = REPO / "tools" / "perf_baseline.json"

TAG = _T_UNITBLOCK[0]
#: `KernelState` lays every field block on a 128-byte boundary, and the `|i1` sign array is the
#: binding field: only multiples of 128 keep a BLOCK's start aligned in all five fields.
ALIGN_UNITS = 128
#: The share of a recorded wake tick this module's own self-check is willing to spend on paged
#: reads when it derives its rounds constants. CHOSEN and disclosed -- it is the self-check's
#: allowance, not a project constant, and a real caller states its own.
SELFCHECK_IO_ALLOWANCE = 0.05

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it. The gate's walk is
#: recall-biased and captures this module's `main`, so it is declared rather than left
#: UNDETERMINABLE -- the disposition `substrate/unit_blocks.py` gives its own self-check, for the
#: same reason: none of the four learning dispositions describes a self-check, and any of them
#: would be false here. The pager moves no weight and computes no update; it moves BYTES between
#: the lake and core, and the weights it compiles are the ones the record already held.
GRADIENT_CLAIM = {
    "main": (
        "NOT_A_RULE",
        "this module's self-check: it derives a rounds budget from a measured read latency, runs "
        "both pre-registered reds with their broken arms and their calm controls, and measures "
        "resident bytes two ways. It moves no weight and computes no update, so it is not a "
        "learning rule at all -- declared so the gate's UNDETERMINABLE ratchet stays honest debt",
    ),
}


class RoundsBudgetExceeded(RuntimeError):
    """The integer per-tick rounds budget was exceeded on at least one tick.

    RAISED ON DEMAND, never during the tick: exceeding the budget is how the mechanism keeps its
    staleness bound without dropping a delivery, so the tick must complete. `budget_trips` is the
    record and `assert_within_budget()` is the criterion that turns it red."""


# distinct-from: substrate/neuro_chip_stream.py::StreamingChip -- streams a RECORDED chip's spike
#   shards (input data) with one core resident. It holds no substrate unit, has no
#   activation-driven policy and no delivery path; its `max_resident_cores` counts context-manager
#   nesting, which is exactly the bookkeeping-integer shape the residency proof here must NOT
#   repeat (see `resident_edge_bytes` and the tracemalloc arm in main).
# distinct-from: substrate/proximity_index.py -- the tree's only mmap store (LRU over medoid
#   PATTERNS for a similarity index). It is the incumbent this row's "never mmap" clause refuses:
#   OS eviction is blind to activation. Not adopted, and not a rival -- it pages patterns, not
#   units.
# distinct-from: substrate/unit_blocks.py::UnitBlocks -- ADOPTED, not rivalled (P2): its
#   `block_of`/`units_in` IS the block arithmetic here, and this module is the first real consumer
#   of that partition. Its `resident()` context manager and `max_resident` counter are NOT used as
#   a residency instrument (they count nesting, its own panel's correction).
# distinct-from: substrate/lake_memory.py, cortex/theta_gamma_code.py,
#   hippocampus/affective_memory.py, hippocampus/episode_consolidation.py -- cognitive-level
#   eviction of traces / slots / episodes; none touches a substrate unit or a delivery.
#   All sites read 2026-09-08.
class UnitBlockStore:
    """The lake-side half: one `UnitBlockRecord` per block, INSIDE the lake (P4b).

    Records go through `lake_segments.SegmentStore` (WAL then a sealed Arrow segment, one record
    batch per record) and their pointers are folded into `lake_index.LakeIndex` as
    `(0x0C, block_id) -> (container, seq)` with the watermark advanced in the SAME transaction --
    no sidecar file, no second durable store.

    `root` is a lake home directory and there is NO DEFAULT: defaulting to `home()` would put a
    bench store inside the live brain's lake, so a missing root is refused."""

    def __init__(self, root):
        root = Path(root)
        if not root.is_dir():
            raise ValueError(
                f"UnitBlockStore needs an EXISTING lake home directory; {root} is not one. "
                "There is deliberately no default -- falling back to home() would write a bench "
                "store into the live brain's one lake"
            )
        self.root = root
        self.store = SegmentStore(root)
        self.index = LakeIndex(root / INDEX_NAME)
        #: the STORE-side rounds instrument: one increment per `fetch` submission
        self.submissions = 0
        self.blocks_read = 0
        self.blocks = None

    def build(self, pre, post, weight, delay, n_units, block_size):
        """Write ONE record per block and fold its pointer into the index. Returns the partition.

        Every block gets a record, INCLUDING blocks that own no out-edge. `UnitBlocks.group`
        omits empty blocks on purpose (a writer emitting a record for a block nothing happened in
        is the repeated-snapshot pathology in miniature), but the pager ADDRESSES blocks by id and
        an unwritten block would be an unfetchable one -- a different job, so a different rule,
        stated rather than inherited."""
        if not (
            isinstance(block_size, int)
            and not isinstance(block_size, bool)
            and block_size > 0
            and block_size % ALIGN_UNITS == 0
        ):
            raise ValueError(
                f"block_size must be a positive multiple of {ALIGN_UNITS} so every KernelState "
                "field block starts on a 128-byte boundary (the |i1 sign array is the binding "
                f"field; 32 would suffice for the 4- and 8-byte ones), got {block_size!r}. The "
                "caller derives it from its bytes-per-block target and the measured bytes/unit"
            )
        label, reclaimable, _reason = retention_of(_T_UNITBLOCK[0])
        if label != "IMMORTAL" or reclaimable:
            raise ValueError(
                f"the unit-block class is labeled {label}/reclaimable={reclaimable}; this store "
                "writes a block's synaptic identity, which nothing can reconstruct it from"
            )
        blocks = UnitBlocks(int(n_units), block_size)
        pre = np.asarray(pre, dtype=np.int64)
        post = np.asarray(post, dtype=np.int64)
        weight = np.asarray(weight, dtype=np.float64)
        delay = np.asarray(delay, dtype=np.int64)
        payloads = []
        for bid in range(blocks.n_blocks):
            units = blocks.units_in(bid)
            sel = (pre >= units.start) & (pre < units.stop)
            payloads.append(
                UnitBlockRecord(
                    block_id=bid,
                    n_units_in_block=len(units),
                    block_size=block_size,
                    pre_local=pre[sel] - units.start,
                    post=post[sel],
                    weight=weight[sel],
                    delay=delay[sel],
                ).encode()
            )
        self.store.append_many(payloads)
        sealed = self.store.seal()
        seq0 = int(sealed.stem.split("-")[1])
        self.index.put_batch(
            [(TAG, str(bid), sealed.name, seq0 + bid) for bid in range(blocks.n_blocks)],
            watermark=seq0 + blocks.n_blocks - 1,
        )
        self.blocks = blocks
        return blocks

    def fetch(self, block_ids):
        """ONE ROUND: the reads of `block_ids` submitted as one batch to a thread pool.

        "Async" means the reads of a round overlap EACH OTHER, not the tick's compute -- there is
        no prefetch in this row, and saying so is cheaper than letting the word imply one."""
        ids = [int(b) for b in block_ids]
        if not ids:
            return {}
        located = []
        for bid in ids:
            row = self.index.get(TAG, str(bid))
            if row is None:
                raise KeyError(
                    f"block {bid} has no pointer under tag {TAG:#04x} -- refused rather than "
                    "read as an empty edge set"
                )
            located.append((bid, self.root / SEGMENT_DIR / row[0], row[1]))
        self.submissions += 1
        self.blocks_read += len(ids)
        with ThreadPoolExecutor(max_workers=min(len(ids), 8)) as pool:
            payloads = list(pool.map(lambda loc: read_record(loc[1], loc[2]), located))
        return {
            loc[0]: UnitBlockRecord.decode(payload)
            for loc, payload in zip(located, payloads, strict=True)
        }

    def iter_payloads(self):
        """The store's own forward read, delegated -- one reader path (P2)."""
        return self.store.iter_payloads()

    def segment_digest(self) -> str:
        """sha256 over every sealed segment's bytes. The lake-identity instrument: a page-out
        DESTROYS NO RECORD, so this must not move across a full page-out/page-in cycle."""
        digest = hashlib.sha256()
        for seg in sorted((self.root / SEGMENT_DIR).glob("seg-*.lake")):
            digest.update(seg.read_bytes())
        return digest.hexdigest()

    def close(self):
        self.index.close()


class NeuronPager:
    """Residency policy over unit blocks, driven by activation (specs/022 T066).

    Owns `resident` (a bool mask SHARED with the KernelTick, so a page-in is visible to the very
    next phase) and per-block `last_active`. `step()` runs at the START of a tick, before
    propagate: plan the due set, fetch it in rounds, page blocks in (compiling their edges with
    the SAME `compile_bundles` the whole-table constructor uses -- one compiler, P2), drain their
    pending into the accumulators exactly, and demote at-rest blocks to make room.

    WHAT MAKES A BLOCK EVICTABLE, and one clause of it is a REFINEMENT of the pre-registration
    that is disclosed rather than slipped in. 2.1 requires a candidate's units to have
    `last_spike < tick - max_delay` and zero pending, which is the RING-SAFETY condition: no
    bundle key of that block can still be sitting in a delay slot, so removing it cannot orphan a
    scheduled delivery. That condition is necessary and NOT sufficient for the byte-identity 2.3
    also pre-registers: disclosure 2.6(b) records that a paged-out unit's membrane does not decay,
    so a block evicted while any unit holds charge would come back with a membrane the twin has
    already decayed. The candidate must therefore be AT REST -- v exactly zero, refrac zero, no
    charge in its accumulator -- and it is, or the calm arm's identity would be unreachable by
    construction rather than by defect."""

    def __init__(
        self,
        kt,
        store,
        *,
        blocks,
        capacity_blocks,
        blocks_per_round,
        max_rounds,
        max_stale_ticks,
        pinned=frozenset(),
    ):
        for name, value, floor in (
            ("capacity_blocks", capacity_blocks, 1),
            ("blocks_per_round", blocks_per_round, 1),
            ("max_rounds", max_rounds, 1),
            ("max_stale_ticks", max_stale_ticks, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < floor:
                raise ValueError(
                    f"{name} is an INPUT with NO DEFAULT (P1) and must be an int >= {floor}; the "
                    f"caller derives it from its own memory and I/O budget, got {value!r}"
                )
        if kt.resident is None:
            raise ValueError(
                "the pager needs a KernelTick built with a residency mask -- resident=None is "
                "the incumbent, unpaged, path"
            )
        pinned = frozenset(int(b) for b in pinned)
        bad = sorted(b for b in pinned if not 0 <= b < blocks.n_blocks)
        if bad:
            raise ValueError(f"pinned block(s) {bad} outside [0, {blocks.n_blocks})")
        if len(pinned) > capacity_blocks:
            raise ValueError(
                f"{len(pinned)} pinned blocks do not fit capacity_blocks={capacity_blocks} -- "
                "the hot set is COUNTED in capacity, never carried beside it"
            )
        self.kt = kt
        self.store = store
        self.blocks = blocks
        self.capacity_blocks = capacity_blocks
        self.blocks_per_round = blocks_per_round
        self.max_rounds = max_rounds
        self.max_stale_ticks = max_stale_ticks
        self.pinned = pinned
        self.resident = kt.resident
        self._units = [
            np.arange(blocks.units_in(b).start, blocks.units_in(b).stop, dtype=np.int64)
            for b in range(blocks.n_blocks)
        ]
        #: the PLANNER-side rounds instrument, per tick; `store_rounds_used` is the store's own
        self.rounds_used: list[int] = []
        self.store_rounds_used: list[int] = []
        #: (tick, rounds_needed, max_rounds) per tick on which the budget was exceeded
        self.budget_trips: list[tuple[int, int, int]] = []
        self.capacity_trips = 0
        self.page_ins = 0
        self.evictions = 0
        self.max_staleness = 0
        self.last_active = [0] * blocks.n_blocks
        self._last_pending_tick = [-1] * blocks.n_blocks
        self._due_tick: list[int | None] = [None] * blocks.n_blocks
        self._resident_blocks: set[int] = set()
        if pinned:
            for bid, record in sorted(self.store.fetch(sorted(pinned)).items()):
                self._page_in(bid, record)
        #: the pinned hot set is resident FROM CONSTRUCTION, so its load is not an
        #: activation-driven page-in; zeroing here keeps `page_ins > 0` a claim about paging
        self.page_ins = 0
        self.submissions_at_start = store.submissions
        self._prev_submissions = store.submissions

    # ---- planning ---------------------------------------------------------------------------
    def _forced(self, block, tick):
        due = self._due_tick[block]
        return due is not None and (tick - due) >= self.max_stale_ticks

    def plan(self, ext_q=None):
        """The due set, ordered: overdue first, then oldest pending arrival, then largest
        |pending|, then lowest block id -- total and deterministic, so two runs of the same
        fixture fetch in the same order.

        EXTERNAL DRIVE COUNTS AS DUE, which is an ADDITION to the pre-registered rule (2.1 names
        pending alone) and is disclosed as one. `ext_q` reaches a unit through propagate, not
        through a bundle, so a paged-out unit's external drive is not carried by the pending
        summary -- leaving it out would be a silent drop of exactly the kind S-R2 exists to
        forbid, differing from the S-R2 red only in which port the charge arrived on."""
        tick = self.kt.tick_index
        ordered = []
        for bid in range(self.blocks.n_blocks):
            if bid in self._resident_blocks:
                continue
            units = self._units[bid]
            waiting = 0
            for p in range(self.kt.p_parts):
                waiting += int(np.abs(self.kt.pending[p][units]).sum())
            driven = int(np.abs(ext_q[units]).sum()) if ext_q is not None else 0
            if not (waiting or driven):
                self._due_tick[bid] = None
                continue
            if waiting:
                self._last_pending_tick[bid] = tick
            if self._due_tick[bid] is None:
                self._due_tick[bid] = tick
            ordered.append(
                (0 if self._forced(bid, tick) else 1, self._due_tick[bid], -waiting, bid)
            )
        ordered.sort()
        return [row[3] for row in ordered]

    def _issue_rounds(self, due, tick):
        """Fetch the due set in rounds of `blocks_per_round`, at most `max_rounds` of them, and
        then in EXTRA rounds for any block already at its staleness bound. Returns
        `(records, rounds, trip)`; a trip is `(tick, rounds_needed, max_rounds)` or None.

        This is the one override point the budget-blind broken arm replaces."""
        width, budget = self.blocks_per_round, self.max_rounds
        budgeted = due[: width * budget]
        overdue = [b for b in due[width * budget :] if self._forced(b, tick)]
        batches = [budgeted[i : i + width] for i in range(0, len(budgeted), width)]
        extra = [overdue[i : i + width] for i in range(0, len(overdue), width)]
        records = {}
        for batch in batches + extra:
            for bid, record in self.store.fetch(batch).items():
                records[bid] = record
        rounds = len(batches) + len(extra)
        return records, rounds, ((tick, rounds, self.max_rounds) if extra else None)

    # ---- residency --------------------------------------------------------------------------
    def _at_rest(self, block, tick):
        """True when nothing about this block can change while it is out of core: no spike within
        max_delay (ring safety), refrac released, membrane exactly at zero and no charge waiting.
        The last three are the identity clause the class docstring discloses."""
        units = self._units[block]
        arrays = self.kt.state.arrays
        if int(arrays["last_spike"][units].max()) >= tick - self.kt.syn.max_delay:
            return False
        if int(arrays["refrac"][units].max()) != 0:
            return False
        if float(np.abs(arrays["v"][units]).max()) != 0.0:
            return False
        for p in range(self.kt.p_parts):
            if int(np.abs(self.kt._acc[p][units]).sum()):
                return False
            if int(np.abs(self.kt.pending[p][units]).sum()):
                return False
        return True

    def _page_in(self, block, record):
        """Compile the block's edges with the ONE compiler, insert them, mark its units resident
        and drain their pending into the accumulators EXACTLY (integer, so order-free)."""
        syn = self.kt.syn
        bundles, by_pre = compile_bundles(
            record.pre_global(),
            record.post,
            record.weight,
            record.delay,
            syn.n_units,
            syn.p_parts,
            syn.frac_bits,
            syn.max_delay,
        )
        for key, value in bundles.items():
            syn.bundles[key] = value
        for pre_unit, routes in by_pre.items():
            syn.by_pre[pre_unit] = routes
        units = self._units[block]
        self.resident[units] = True
        for p in range(self.kt.p_parts):
            self.kt._acc[p][units] += self.kt.pending[p][units]
            self.kt.pending[p][units] = 0
        self._resident_blocks.add(block)
        self._due_tick[block] = None
        self.page_ins += 1

    def _page_out(self, block):
        """DEMOTE a block out of core. Its bundles leave the table and its units are masked out of
        propagate -- and NOTHING IS DESTROYED: the record stays in the lake, byte for byte, which
        is why the class is IMMORTAL and why a lake digest taken before and after a full cycle
        must be identical."""
        syn = self.kt.syn
        for unit in self._units[block].tolist():
            for _part, _delay, key in syn.by_pre.pop(unit, ()):
                syn.bundles.pop(key, None)
        self.resident[self._units[block]] = False
        self._resident_blocks.discard(block)
        self.evictions += 1

    def _make_room(self, incoming, due, tick):
        arriving = [b for b in incoming if b not in self._resident_blocks]
        wanted = len(arriving) - (self.capacity_blocks - len(self._resident_blocks))
        if wanted <= 0:
            return
        due_set = set(due)
        candidates = [
            b
            for b in self._resident_blocks
            if b not in self.pinned and b not in due_set and self._at_rest(b, tick)
        ]
        candidates.sort(key=lambda b: (self.last_active[b], b))
        for block in candidates[:wanted]:
            self._page_out(block)
        if len(candidates) < wanted:
            #: over capacity rather than short a block: a fetch is never abandoned, so nothing is
            #: ever dropped for want of room -- the cost is counted here instead
            self.capacity_trips += 1

    def _refresh_last_active(self, tick):
        last = self.kt.state.arrays["last_spike"]
        for bid in range(self.blocks.n_blocks):
            units = self._units[bid]
            self.last_active[bid] = max(int(last[units].max()), self._last_pending_tick[bid])

    # ---- the tick ---------------------------------------------------------------------------
    def step(self, ext_q=None):
        """One planning step, at the START of a tick and before propagate."""
        tick = self.kt.tick_index
        due = self.plan(ext_q)
        records, rounds, trip = self._issue_rounds(due, tick)
        self.rounds_used.append(rounds)
        self.store_rounds_used.append(self.store.submissions - self._prev_submissions)
        self._prev_submissions = self.store.submissions
        if trip is not None:
            self.budget_trips.append(trip)
        if records:
            self._make_room(sorted(records), due, tick)
            for bid in sorted(records):
                waited = self._due_tick[bid]
                if waited is not None:
                    self.max_staleness = max(self.max_staleness, tick - waited)
                self._page_in(bid, records[bid])
        self._refresh_last_active(tick)
        return due

    def tick(self, ext_q=None):
        """`step()` then the kernel's own tick -- one call, so a caller cannot forget the half
        that makes residency safe. Returns what `KernelTick.tick` returns."""
        self.step(ext_q)
        return self.kt.tick(ext_q)

    # ---- readings ---------------------------------------------------------------------------
    def resident_edge_bytes(self) -> int:
        """The EXACT sum of `nbytes` over the resident bundle arrays. A memory reading, not a
        nesting counter -- the T064 debt this row was not allowed to repeat."""
        return int(sum(posts.nbytes + q.nbytes for posts, q in self.kt.syn.bundles.values()))

    def resident_fraction(self) -> float:
        return float(self.resident.mean())

    def resident_floor_bytes_per_unit(self) -> float:
        """State + accumulator + pending summary + the residency flag, per unit -- what stays in
        core for EVERY unit whatever the pager does. Paging reduces edge bytes only."""
        units = self.kt.state.n_units
        state = sum(a.nbytes for a in self.kt.state.arrays.values())
        acc = sum(a.nbytes for a in self.kt._acc)
        pending = sum(a.nbytes for a in self.kt.pending)
        return (state + acc + pending + self.resident.nbytes) / units

    def assert_within_budget(self) -> bool:
        if self.budget_trips:
            tick, needed, budget = self.budget_trips[0]
            raise RoundsBudgetExceeded(
                f"{len(self.budget_trips)} tick(s) exceeded the {budget}-round budget; first at "
                f"tick {tick} needing {needed} rounds (overflow {needed - budget})"
            )
        return True


# ---- the self-check's instruments -------------------------------------------------------------
class _MeteredTick(KernelTick):
    """Per-unit charge counters, differenced around the REAL phases so an instrument bug cannot
    look like a mechanism bug. `delivered` is charge that left a bundle for a post unit
    (accumulator or pending); `integrated` is charge that entered a membrane."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delivered = np.zeros(self.state.n_units, dtype=np.int64)
        self.integrated = np.zeros(self.state.n_units, dtype=np.int64)

    def _propagate(self, p, ext_q):
        members = self._members[p]
        self.integrated[members] += self._acc[p][members]
        return super()._propagate(p, ext_q)

    def _deliver_and_rotate(self, p):
        before_acc = self._acc[p].copy()
        before_pending = self.pending[p].copy()
        super()._deliver_and_rotate(p)
        self.delivered += (self._acc[p] - before_acc) + (self.pending[p] - before_pending)


class _DroppingTick(_MeteredTick):
    """THE S-R2 BROKEN ARM: the naive pager. Delivery DISCARDS every post that is not resident
    instead of conserving it, and counts exactly what it threw away."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discarded = np.zeros(self.state.n_units, dtype=np.int64)

    def _deliver_and_rotate(self, p):
        slot = self._rings[p][(self.tick_index + 1) % (self.syn.max_delay + 1)]
        acc, here = self._acc[p], self.resident
        for key in slot:
            posts, q = self.syn.bundles[key]
            keep = here[posts]
            np.add.at(acc, posts[keep], q[keep])
            np.add.at(self.delivered, posts[keep], q[keep])
            np.add.at(self.discarded, posts[~keep], q[~keep])
        slot.clear()


class _BudgetBlindPager(NeuronPager):
    """THE S-R1 BROKEN ARM: submits the whole due set as ONE round whatever `blocks_per_round`
    says, so the budget cannot be exceeded and no trip is ever recorded."""

    def _issue_rounds(self, due, tick):
        if not due:
            return {}, 0, None
        return self.store.fetch(due), 1, None


_FRAC = 16
_BLOCK = 128
_N_BLOCKS = 8
_N = _BLOCK * _N_BLOCKS
_P = 4
_MAX_DELAY = 3
_W = 1.5  # exactly representable in <f4, and one edge alone clears thr = 1.0


def _chain_edges(seed=11, fan=2):
    """Block b drives block b+1; block 7 terminal. Every block but the last OWNS edges, so paging
    one really moves compiled bundles."""
    rng = np.random.default_rng(seed)
    pre, post, weight, delay = [], [], [], []
    for b in range(_N_BLOCKS - 1):
        for unit in range(b * _BLOCK, (b + 1) * _BLOCK):
            for _ in range(fan):
                pre.append(unit)
                post.append(int(rng.integers((b + 1) * _BLOCK, (b + 2) * _BLOCK)))
                weight.append(_W)
                delay.append(int(rng.integers(1, _MAX_DELAY + 1)))
    return pre, post, weight, delay


def _fanout_edges(seed=13):
    """Block 0 into EVERY other block at delay 1, targets terminal: one pulse makes every block
    due on one tick, and a dropped delivery cannot change any later spike."""
    rng = np.random.default_rng(seed)
    pre, post, weight, delay = [], [], [], []
    for unit in range(_BLOCK):
        for b in range(1, _N_BLOCKS):
            pre.append(unit)
            post.append(int(rng.integers(b * _BLOCK, (b + 1) * _BLOCK)))
            weight.append(_W)
            delay.append(1)
    return pre, post, weight, delay


def _drive(units, level=2.0):
    ext = np.zeros(_N, dtype=np.int64)
    ext[list(units)] = int(level * (1 << _FRAC))
    return ext


def _fresh_state():
    state = KernelState([("net", _N)])
    state.arrays["thr"][:] = 1.0
    return state


def _twin(edges, max_delay=_MAX_DELAY):
    state = _fresh_state()
    syn = Synapses(*edges, _N, _P, _FRAC, max_delay)
    return _MeteredTick(state, syn, decay=0.5, v_reset=0.0, refrac_ticks=0)


def _new_store(root, name, edges):
    lake = Path(root) / name
    lake.mkdir(parents=True, exist_ok=True)
    store = UnitBlockStore(lake)
    return store, store.build(*edges, _N, _BLOCK)


def _new_pager(
    store, blocks, tick_cls=_MeteredTick, pager_cls=NeuronPager, max_delay=_MAX_DELAY, **budget
):
    state = _fresh_state()
    syn = Synapses.empty(_N, _P, _FRAC, max_delay)
    kt = tick_cls(
        state,
        syn,
        decay=0.5,
        v_reset=0.0,
        refrac_ticks=0,
        resident=np.zeros(_N, dtype=bool),
    )
    return kt, pager_cls(kt, store, blocks=blocks, **budget)


def _run(runner, drives, ticks):
    trains = []
    for t in range(ticks):
        trains.append(runner.tick(drives.get(t)).tobytes())
    return trains


def _derive_rounds(store, blocks):
    """blocks_per_round and max_rounds from a MEASURED per-block read latency against a stated
    share of the recorded wake-tick p50. Printed, never picked."""
    ids = list(range(blocks.n_blocks))
    singles = []
    for _ in range(3):
        for bid in ids:
            t0 = time.perf_counter()
            store.fetch([bid])
            singles.append(time.perf_counter() - t0)
    singles.sort()
    per_block = singles[len(singles) // 2]
    widths = {}
    for k in (1, 2, 4, 8):
        batch = (ids * (k // len(ids) + 1))[:k]
        t0 = time.perf_counter()
        store.fetch(batch)
        widths[k] = time.perf_counter() - t0
    width = max([k for k in widths if widths[k] <= 2.0 * per_block] or [1])
    baseline = json.loads(PERF_BASELINE.read_text())
    wake_p50 = baseline["ticks"]["wake"]["p50_ms"] / 1000.0
    allowance = SELFCHECK_IO_ALLOWANCE * wake_p50
    rounds = max(1, int(allowance // max(widths[width], 1e-9)))
    return {
        "per_block_s": per_block,
        "widths_s": widths,
        "blocks_per_round": width,
        "max_rounds": rounds,
        "wake_p50_s": wake_p50,
        "allowance_s": allowance,
    }


# distinct-from: cortex/brain.py::_regions_for -- the set of brain REGIONS that produced live
#   output this tick; a wiring/liveness census over a running brain, allocating nothing and
#   measuring no bytes.
# distinct-from: substrate/kernel_tick.py::Synapses -- the thing being MEASURED here, not a second
#   measurement of it: that class holds the bundles, this reads how many bytes of them the
#   allocator is still holding. Sites read 2026-09-08.
def _bundle_alloc_bytes():
    """LIVE traced bytes whose allocation site is `kernel_tick.py` -- the bundle arrays that
    `compile_bundles` created and that are still referenced.

    THE SECOND RESIDENCY INSTRUMENT, and it is filtered on purpose. An UNFILTERED tracemalloc peak
    over the same run counts everything an arm allocates, and the paged arm allocates a sqlite
    index, Arrow record batches and a thread pool that exist ONLY because it pages -- so the peak
    charges paging for its own I/O machinery and stops being a reading of resident edge bytes.
    Measured, not argued: it read 0.888 standalone and 1.262 for the same code under
    run_selfchecks. Filtered to the allocation site it measures the same quantity as the exact
    `nbytes` sum by a different mechanism (allocator accounting vs live-object arithmetic), which
    is what a second instrument is for."""
    from snn_data_lake.substrate import kernel_tick as compiler_module

    snapshot = tracemalloc.take_snapshot().filter_traces(
        (tracemalloc.Filter(True, compiler_module.__file__),)
    )
    return int(sum(stat.size for stat in snapshot.statistics("filename")))


def _fail(message):
    print(f"neuron_pager self-check FAILED: {message}")
    return 1


def main() -> int:
    from snn_data_lake.substrate.home import throwaway_home

    started = time.perf_counter()
    print("== T066 activation-driven paging ==")
    with throwaway_home() as scratch:
        root = Path(scratch)
        chain, fanout = _chain_edges(), _fanout_edges()

        # ---- the rounds constants, DERIVED in this same command ---------------------------
        cal_store, cal_blocks = _new_store(root, "cal", chain)
        derived = _derive_rounds(cal_store, cal_blocks)
        cal_store.close()
        print(
            f"derivation: per-block read {derived['per_block_s'] * 1e3:.3f} ms (median of 24 "
            f"single-block rounds); round widths ms "
            + ", ".join(f"{k}:{v * 1e3:.3f}" for k, v in sorted(derived["widths_s"].items()))
            + f" -> blocks_per_round={derived['blocks_per_round']} (widest still within 2x the "
            f"single-block latency); allowance {SELFCHECK_IO_ALLOWANCE:.0%} of the recorded "
            f"wake-tick p50 {derived['wake_p50_s'] * 1e3:.1f} ms = "
            f"{derived['allowance_s'] * 1e3:.2f} ms -> max_rounds={derived['max_rounds']}"
        )
        print(
            "  the CALM arm below runs on exactly these derived constants. The OVERLOAD arms do "
            "NOT, and cannot: a trip needs blocks_per_round * max_rounds < n_blocks, which a "
            "budget derived to be sufficient does not satisfy by construction. The derivation is "
            "what a caller would deploy; the overload budgets are what make the criterion "
            "falsifiable, and they are stated as constructed rather than measured"
        )

        # ---- S-R2: calm identity, and the DROPPING broken arm ------------------------------
        drives = {0: _drive([3, 17]), 22: _drive([41, 90])}
        twin = _twin(chain)
        twin_trains = _run(twin, drives, 45)
        store, blocks = _new_store(root, "calm", chain)
        _kt, pager = _new_pager(
            store,
            blocks,
            capacity_blocks=4,
            blocks_per_round=derived["blocks_per_round"],
            max_rounds=derived["max_rounds"],
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        paged_trains = _run(pager, drives, 45)
        if paged_trains != twin_trains:
            first = next(i for i, (a, b) in enumerate(zip(paged_trains, twin_trains)) if a != b)
            return _fail(f"calm paged run diverged from the resident twin at tick {first}")
        if not (pager.page_ins > 0 and pager.evictions > 0):
            return _fail(
                f"the calm arm never paged (page_ins={pager.page_ins}, "
                f"evictions={pager.evictions}) -- an identity that never paged proves nothing"
            )
        if not 0.0 < pager.resident_fraction() < 1.0:
            return _fail(f"resident fraction {pager.resident_fraction()} is not inside (0, 1)")
        print(
            f"S-R2 calm: 45 ticks byte-identical to the resident twin; page_ins="
            f"{pager.page_ins} evictions={pager.evictions} resident fraction "
            f"{pager.resident_fraction():.3f} of {_N_BLOCKS} blocks; rounds trips="
            f"{len(pager.budget_trips)}, CAPACITY trips={pager.capacity_trips}"
        )
        if pager.capacity_trips:
            print(
                f"  the capacity trips are the honest half of that fraction: capacity_blocks=4 "
                f"was exceeded on {pager.capacity_trips} tick(s) because no resident block was "
                "AT REST to demote, so the fetch went over capacity rather than deferring a "
                "block -- never a drop, and the reason the final fraction sits above 4/8. This "
                "chain fixture's wavefront keeps the last few blocks hot, which is disclosure "
                "2.6(d)'s regime (locality is T067's row) reported rather than smoothed"
            )

        drop_store, drop_blocks = _new_store(root, "drop", fanout)
        drop_twin = _twin(fanout, max_delay=1)
        drop_drives = {t: _drive(range(8)) for t in range(6)}
        drop_twin_trains = _run(drop_twin, drop_drives, 12)
        drop_kt, drop_pager = _new_pager(
            drop_store,
            drop_blocks,
            tick_cls=_DroppingTick,
            max_delay=1,
            capacity_blocks=_N_BLOCKS,
            blocks_per_round=4,
            max_rounds=2,
            max_stale_ticks=0,
            pinned=frozenset({0, 1}),
        )
        drop_trains = _run(drop_pager, drop_drives, 12)
        shortfall = drop_twin.delivered - drop_kt.delivered
        if not int(drop_kt.discarded.sum()):
            return _fail("the DROPPING broken arm discarded nothing -- it is not an arm")
        if not np.array_equal(shortfall, drop_kt.discarded):
            return _fail(
                "the dropping twin's shortfall does not equal its discard count -- the "
                "conservation instrument is blind, and no verdict follows"
            )
        if drop_trains == drop_twin_trains:
            return _fail("a dropped spike did not change the trains")
        print(
            f"S-R2 red: the DROPPING twin lost exactly {int(drop_kt.discarded.sum())} fixed-point "
            f"units of charge across {int((drop_kt.discarded != 0).sum())} units -- shortfall "
            "equals its own discard count, per unit, and its trains differ"
        )

        # ---- S-R2 under overload: conservation with a bounded staleness --------------------
        ov_store, ov_blocks = _new_store(root, "conserve", fanout)
        ov_drives = {t: _drive(range(8)) for t in range(10)}
        ov_kt, ov_pager = _new_pager(
            ov_store,
            ov_blocks,
            max_delay=1,
            capacity_blocks=_N_BLOCKS,
            blocks_per_round=1,
            max_rounds=1,
            max_stale_ticks=2,
            pinned=frozenset({0}),
        )
        _run(ov_pager, ov_drives, 20)
        leftover = sum(int(np.abs(p).sum()) for p in ov_kt.pending)
        if not ov_pager.budget_trips:
            return _fail("the constructed overload did not trip the rounds budget")
        if not 0 < ov_pager.max_staleness <= 2:
            return _fail(f"staleness {ov_pager.max_staleness} is outside (0, 2]")
        if not np.array_equal(ov_kt.delivered, ov_kt.integrated) or leftover:
            return _fail(
                f"charge was not conserved under overload (pending left {leftover} at the tail)"
            )
        print(
            f"S-R2 overload: {len(ov_pager.budget_trips)} trip(s), max staleness "
            f"{ov_pager.max_staleness} <= 2, every delivered unit of charge reached a membrane "
            "and pending drained to zero by the tail"
        )

        # ---- S-R1: the rounds budget, its calm control, and the BUDGET-BLIND arm ------------
        pulse = {0: _drive(range(8))}
        r1_store, r1_blocks = _new_store(root, "r1", fanout)
        r1_kt, r1_pager = _new_pager(
            r1_store,
            r1_blocks,
            max_delay=1,
            capacity_blocks=_N_BLOCKS,
            blocks_per_round=2,
            max_rounds=2,
            max_stale_ticks=0,
            pinned=frozenset({0}),
        )
        _run(r1_pager, pulse, 6)
        if not r1_pager.budget_trips:
            return _fail("blocks_per_round * max_rounds < n_blocks did not trip the budget")
        trip_tick, needed, budget = r1_pager.budget_trips[0]
        if not (needed > budget == r1_pager.max_rounds):
            return _fail(f"the trip record {r1_pager.budget_trips[0]} does not name an overflow")
        if r1_pager.rounds_used[trip_tick] != needed:
            return _fail("the trip's rounds do not match the tick's rounds_used")
        raised = False
        try:
            r1_pager.assert_within_budget()
        except RoundsBudgetExceeded:
            raised = True
        if not raised:
            return _fail("assert_within_budget() did not raise on a recorded trip")
        if not r1_kt.resident.all():
            return _fail("a forced round left a block paged out -- something was deferred away")

        calm_store, calm_blocks = _new_store(root, "r1calm", fanout)
        _ck, calm_pager = _new_pager(
            calm_store,
            calm_blocks,
            max_delay=1,
            capacity_blocks=_N_BLOCKS,
            blocks_per_round=4,
            max_rounds=2,
            max_stale_ticks=0,
            pinned=frozenset({0}),
        )
        _run(calm_pager, pulse, 6)
        if calm_pager.budget_trips or max(calm_pager.rounds_used) < 1:
            return _fail(
                f"the calm control is not a control: trips={len(calm_pager.budget_trips)}, "
                f"max rounds used={max(calm_pager.rounds_used)}"
            )

        blind_store, blind_blocks = _new_store(root, "blind", fanout)
        _bk, blind_pager = _new_pager(
            blind_store,
            blind_blocks,
            pager_cls=_BudgetBlindPager,
            max_delay=1,
            capacity_blocks=_N_BLOCKS,
            blocks_per_round=2,
            max_rounds=2,
            max_stale_ticks=0,
            pinned=frozenset({0}),
        )
        _run(blind_pager, pulse, 6)
        if blind_pager.budget_trips:
            return _fail("the BUDGET-BLIND arm recorded a trip; it cannot be the red")
        print(
            f"S-R1: overload tick {trip_tick} needed {needed} rounds against a {budget}-round "
            f"budget (overflow {needed - budget}) and assert_within_budget() RAISED; the calm "
            f"control tripped 0 times with max rounds {max(calm_pager.rounds_used)} >= 1; the "
            "BUDGET-BLIND arm read 0 trips on the identical overload"
        )

        # ---- both rounds instruments, on every tick of every arm ---------------------------
        for name, pgr, its_store in (
            ("calm", pager, store),
            ("overload", ov_pager, ov_store),
            ("r1", r1_pager, r1_store),
            ("r1calm", calm_pager, calm_store),
            ("blind", blind_pager, blind_store),
        ):
            if pgr.rounds_used != pgr.store_rounds_used:
                return _fail(f"{name}: planner and store rounds disagree -- report no number")
            if sum(pgr.rounds_used) != its_store.submissions - pgr.submissions_at_start:
                return _fail(f"{name}: total rounds do not match the store's submissions")
        print(
            "rounds: the planner's rounds_used and the store's submissions agree on EVERY tick "
            "of all five arms. The separation is NARROW, stated as narrow: they are one call "
            "apart (arithmetic over the due set vs a counter inside fetch), so they catch an "
            "empty round, a fetch that raises mid-tick, and an _issue_rounds override that "
            "miscounts -- not an independent measurement of the same quantity"
        )

        # ---- residency measured as MEMORY, two ways ----------------------------------------
        wide = _chain_edges(seed=5, fan=4)
        bytes_store, bytes_blocks = _new_store(root, "bytes", wide)
        tracemalloc.start()
        twin_wide = _twin(wide)
        _run(twin_wide, {0: _drive([1])}, 20)
        twin_traced = _bundle_alloc_bytes()
        twin_peak = tracemalloc.get_traced_memory()[1]
        twin_edge_bytes = int(sum(p.nbytes + q.nbytes for p, q in twin_wide.syn.bundles.values()))
        tracemalloc.stop()
        tracemalloc.start()
        _b_kt, b_pager = _new_pager(
            bytes_store,
            bytes_blocks,
            capacity_blocks=4,
            blocks_per_round=4,
            max_rounds=2,
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        _run(b_pager, {0: _drive([1])}, 20)
        paged_traced = _bundle_alloc_bytes()
        paged_peak = tracemalloc.get_traced_memory()[1]
        paged_edge_bytes = b_pager.resident_edge_bytes()
        tracemalloc.stop()
        below = paged_edge_bytes < twin_edge_bytes and paged_traced < twin_traced
        print(
            f"residency (NOT a win claim): resident edge bytes {paged_edge_bytes} vs the twin's "
            f"{twin_edge_bytes} (ratio {paged_edge_bytes / max(twin_edge_bytes, 1):.3f}, exact "
            f"nbytes sum over the live bundle arrays); LIVE ALLOCATIONS traced to "
            f"kernel_tick.py {paged_traced} vs {twin_traced} (ratio "
            f"{paged_traced / max(twin_traced, 1):.3f}, tracemalloc filtered to the bundle "
            f"allocation site); resident floor {b_pager.resident_floor_bytes_per_unit():.1f} "
            f"B/unit for EVERY unit (state 21 B + {_P} accumulators + {_P} pending + a flag) -- "
            f"paging reduces edge bytes only. Capacity trips {b_pager.capacity_trips} at "
            f"capacity_blocks=4 of {_N_BLOCKS}: the paged run held MORE than its capacity for "
            "part of the run, so the ratios are what this fixture actually cost, not a best case"
        )
        print(
            f"  THE UNFILTERED tracemalloc PEAK IS PRINTED AND NOT USED: {paged_peak} paged vs "
            f"{twin_peak} twin (ratio {paged_peak / max(twin_peak, 1):.3f}). The row "
            "pre-registered 'tracemalloc peak over the same run' as the second instrument; it "
            "was built, run, and MEASURED TO BE MEASURING SOMETHING ELSE -- it counts every "
            "allocation an arm makes, including the sqlite index, the Arrow batches and the "
            "thread pool that exist ONLY because the run pages, so it charges paging for its own "
            "I/O machinery. It read 0.888 standalone and 1.262 for the same code inside "
            "run_selfchecks, i.e. it disagreed with the exact instrument AND with itself across "
            "process histories. Replaced by the filtered snapshot above, which reads the SAME "
            "quantity by a different mechanism (allocator accounting at the bundle allocation "
            "site vs nbytes over the live arrays). Two instruments must AGREE or nothing is "
            "reported; the peak is kept visible so the replacement is not a silent one"
        )
        if not below:
            return _fail(
                "the two residency instruments do not both place the paged run below the twin"
            )

        # ---- the lake digest across a full page-out/page-in cycle --------------------------
        before = store.segment_digest()
        _cyc_kt, cyc_pager = _new_pager(
            store,
            blocks,
            capacity_blocks=3,
            blocks_per_round=4,
            max_rounds=2,
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        _run(cyc_pager, drives, 45)
        if not (cyc_pager.page_ins and cyc_pager.evictions):
            return _fail("the digest arm never completed a page-out/page-in cycle")
        if store.segment_digest() != before:
            return _fail("the lake's bytes moved across a page-out/page-in cycle")
        print(
            f"lake: {cyc_pager.page_ins} page-ins and {cyc_pager.evictions} page-outs left the "
            f"store's segment bytes IDENTICAL (sha256 {before[:16]}...) -- page-out changes "
            "residency in RAM and destroys no record, which is why 0x0C is IMMORTAL. The "
            "'MORTAL class evicted in the same pass' control is NOT APPLICABLE to a pass that "
            "destroys nothing (stated, not faked)"
        )
        for closing in (
            store,
            drop_store,
            ov_store,
            r1_store,
            calm_store,
            blind_store,
            bytes_store,
        ):
            closing.close()
    print(
        f"neuron_pager self-check OK in {time.perf_counter() - started:.2f} s: both reds counted "
        "with their broken arms and their calm controls in this one command"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

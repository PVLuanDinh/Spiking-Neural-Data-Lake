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

TWO RESOLUTIONS. A resident unit carries full state and edges. A paged-out unit loses its EDGES
and carries the lossy input summary: `KernelTick.pending[p][unit]` (int64, per partition, under
the same single-writer contract as the accumulator) plus its block's `last_active`. Its state does
not go anywhere -- the membrane keeps decaying in `KernelState` by the same op the resident twin
runs, so a paged-out unit ages rather than freezing. That summary is what DECIDES fetches, and it
is also what makes FR-011 true rather than aspirational: EVERY charge bound for a paged-out unit
lands in `pending` -- bundle deliveries from `_deliver_and_rotate` and external `ext_q` from
`_propagate` alike -- and is drained into the accumulator before the next propagate, so nothing is
dropped and, within budget, the run is byte-identical to a fully resident twin.

A ROUND is one `UnitBlockStore.fetch` submission of at most `blocks_per_round` block reads. A tick
whose fetch set holds F blocks costs `ceil(F / blocks_per_round)` submissions. `max_rounds` is the
per-tick budget. Rounds are counted TWICE -- the planner's `rounds_used` (arithmetic over the due
set) and the store's `submissions` (incremented inside `fetch`) -- and no rounds number is
reported unless they agree on every tick. THE SEPARATION IS NARROW and is stated as narrow: the
two counts are one call apart, so they diverge on an empty round issued or on a `fetch` that
raises part way through a tick. THEY DO NOT CATCH THE BUDGET-BLIND BROKEN ARM, and this paragraph
claimed they did until 2026-09-08: that arm reports 1 round and issues exactly 1 `fetch`, so its
own count matches its own submissions and both counters agree while the single submission carries
7 blocks against `blocks_per_round = 2`. They are not independent measurements of the same
physical quantity, and reading them as one would be the mistake this paragraph exists to prevent.

WIDTH IS THE READING THE PAIR CANNOT MAKE, so it is a third instrument rather than a rewording of
the second. `UnitBlockStore.fetch` records every submission's width; `blocks_read` is asserted
equal to their sum (so neither is a write-only counter); and `NeuronPager.assert_round_widths()`
refuses any submission THIS pager issued that carried more than `blocks_per_round` reads. The
constructor's pinned hot-set load is excluded by construction rather than by an aggregate bound:
it is one fetch of `len(pinned)` blocks, so an aggregate form would go red on a CORRECT pager
whenever the hot set is wider than one round. The budget-blind arm reads 7 > 2 on this
instrument, in the same command that reads 0 trips on the pair.

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
staleness bounded by an input, that rounds are integer, budgeted, double-counted, width-checked
and trip on constructed overload, and that resident edge bytes fall, measured TWO EXACT WAYS that
must agree to the byte -- a sum of `nbytes` over the bundle table, and a sum over the same arrays
reached by walking the routing index `by_pre`. The tracemalloc readings beside them are printed
as an allocator FOOTPRINT and are used as no criterion (see `_bundle_alloc_bytes`).

  python -m snn_data_lake.substrate.neuron_pager
"""

from __future__ import annotations

import hashlib
import json
import shutil
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


class RoundWidthExceeded(RuntimeError):
    """A submission carried more block reads than `blocks_per_round` allows.

    DISTINCT FROM `RoundsBudgetExceeded`, and deliberately a separate class: exceeding the ROUNDS
    budget is a legitimate, counted, disclosed trade the mechanism makes to keep its staleness
    bound. Exceeding a round's WIDTH is not a trade at all -- it means the caller's per-round I/O
    allowance was never honoured, which is exactly what the budget-blind broken arm does and what
    the two rounds counters agree about and therefore cannot see."""


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
    `(0x0C, block_id) -> (container, seq)` in ONE transaction -- no sidecar file, no second
    durable store. The applied WATERMARK is deliberately NOT advanced here (the reason is at the
    `put_batch` call): this store is a tenant of the lake's index, not the owner of its fold.

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
        #: the STORE-side WIDTH instrument: one entry per submission, its block count. Read by
        #: `NeuronPager.assert_round_widths` (per-round refusal) and by the `blocks_read`
        #: cross-check below -- the reading the two rounds COUNTERS structurally cannot make.
        self.submission_widths: list[int] = []
        self.blocks_read = 0
        #: the partition `build()` wrote, or None on a REOPENED store (the durable index serves
        #: `fetch` without it). Read by `NeuronPager.__init__` to refuse a pager addressing a
        #: different grain than the records describe.
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
        # watermark=None ON PURPOSE (corrected 2026-09-08): the applied watermark is ONE global
        # meta row naming how far the lake's fold has been applied, and this store does not own
        # that fold -- it is a tenant. Writing it here made a second tenant's progress marker go
        # BACKWARDS (measured 300 -> 4 with two real in-tree writers on one root), because the
        # two numbers are INCOMMENSURABLE: records-in-the-lake against this store's segment seq.
        # A stale-LOW marker only costs a replay of the tail; a marker advanced by the wrong
        # tenant records progress nobody made, so passing None is the safe direction, not the
        # lazy one. THE SHARED KEY SPACE IS STILL OWED AT S6, and it is the same defect from its
        # other side: these (0x0C, block_id) rows and `lake_rebuild_gate.build_index`'s
        # provisional (tag, GLOBAL ORDINAL) rows live in ONE lake_index.db under ONE tag space,
        # so a derived rebuild overwrites every pointer here and `fetch` then raises. It is inert
        # only while the store lives in throwaway homes. Whoever wires S6 must say which keying
        # owns tag 0x0C -- build_index cannot re-derive these rows today, though the lake itself
        # can (the forward scan sees the segment, the seq and the record's own block_id).
        self.index.put_batch(
            [(TAG, str(bid), sealed.name, seq0 + bid) for bid in range(blocks.n_blocks)],
            watermark=None,
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
        self.submission_widths.append(len(ids))
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

    def close(self):
        self.index.close()


# distinct-from: substrate/lake_index.py::LakeIndex.logical_dump_digest -- hashes the sorted
#   POINTER ROWS of the sqlite index (deliberately not the .db bytes, since page layout is engine
#   noise). This hashes the immutable SEGMENT BYTES on disk, which is the opposite choice for the
#   opposite reason: the records are what a page-out must not touch.
# distinct-from: basal_ganglia/self_build_loop.py::_firewalled_bytes/_crcs -- a crc32 fingerprint
#   of the five safety-envelope FILES, kept so the envelope can be RESTORED byte-exact after a
#   self-modification. Different files, different purpose, and it holds the bytes to undo with.
#   Both sites read 2026-09-08.
def segment_digest(root) -> tuple[str, int]:
    """(sha256 of every sealed segment's bytes under `root`, how many files went into it).

    THE ONE BODY (P2), taking a root rather than a store, so the byte-flip positive control runs
    the identical code the identity claim is made with. A page-out DESTROYS NO RECORD, so this
    must not move across a full page-out/page-in cycle -- and an arm that cannot MOVE it is not a
    measurement. The file count is returned rather than logged because a ZERO-file glob is
    REFUSED here: sha256-of-nothing equals sha256-of-nothing, so a blinded glob (a renamed
    SEGMENT_DIR, a store rooted elsewhere) would otherwise print IDENTICAL forever. Measured, not
    argued: the shipped arm passed and printed IDENTICAL over zero bytes when its glob was
    blinded (adversarial panel, 2026-09-08)."""
    files = sorted((Path(root) / SEGMENT_DIR).glob("seg-*.lake"))
    if not files:
        raise ValueError(
            f"no sealed segment under {Path(root) / SEGMENT_DIR} -- refusing to report "
            "sha256-of-nothing as a lake identity (an empty glob would compare EQUAL to any "
            "other empty glob, which is a criterion that cannot fail)"
        )
    digest = hashlib.sha256()
    for seg in files:
        digest.update(seg.read_bytes())
    return digest.hexdigest(), len(files)


class NeuronPager:
    """Residency policy over unit blocks, driven by activation (specs/022 T066).

    Owns `resident` (a bool mask SHARED with the KernelTick, so a page-in is visible to the very
    next phase) and per-block `last_active`. `step()` runs at the START of a tick, before
    propagate: plan the due set, fetch it in rounds, page blocks in (compiling their edges with
    the SAME `compile_bundles` the whole-table constructor uses -- one compiler, P2), drain their
    pending into the accumulators exactly, and demote EVICTABLE blocks to make room.

    WHAT MAKES A BLOCK EVICTABLE is exactly the RING-SAFETY condition 2.1 pre-registers, plus the
    two buffers that condition does not name. A candidate's units must have `last_spike < tick -
    max_delay` (no bundle key of that block can still be sitting in a delay slot, so removing it
    cannot orphan a scheduled delivery), `refrac == 0`, and zero charge in BOTH the accumulator
    and `pending` -- charge in either would be integrated by a membrane that is no longer being
    driven, and pending is the summary a page-in drains.

    THE MEMBRANE IS NOT PART OF THAT CONDITION, and that is a CORRECTION to what this class first
    shipped. The first version also required `v` to be exactly zero, because disclosure 2.6(b)
    recorded that a paged-out unit's membrane did not decay: a block demoted while any unit held
    charge would have come back holding a membrane the twin had already decayed, so the identity
    2.3 pre-registers would have been unreachable. That reason has been REMOVED rather than
    worked around -- `KernelTick._propagate` now decays every member, resident or not, with the
    same elementwise op the twin runs, and a non-resident member's `inj` is exactly zero (its
    accumulator is empty by the condition above and its external drive is routed to `pending`),
    so the decay is bit-identical rather than approximately right. The `v == 0` clause had to go
    with it: a float32 membrane under sustained sub-threshold charge decays TOWARD zero without
    reaching it, so the clause made eviction rare in precisely the regime the pager exists for.

    WHERE THAT COST IS AND IS NOT VISIBLE, because guessing it wrong is what produced the line
    this paragraph replaces. It is NOT visible in this module's calm arm: every edge in that
    fixture clears threshold, so each unit resets to exactly 0.0 and `v == 0` held anyway --
    removing the clause moved its page_ins, evictions and capacity trips by ZERO, measured on
    both sides. That arm's trips are the RING WINDOW (counted per trip by `_TripCensus` and
    printed). The cost is visible in the sub-threshold arm in tests/test_neuron_pager.py, where a
    block that received input, did not fire, and held no buffered charge was refused demotion and
    the fetch went over capacity.

    THE ONE LIMIT OF THE RING-SAFETY CONDITION IS NOW CLOSED, and it was closed the way it should
    have been opened -- by constructing the input, not by reasoning about it. The condition
    admitted a unit above threshold whose refractory had just been released: `_propagate` reads
    `gated = refrac > 0` BEFORE the decrement, so a unit whose refractory reaches 0 at tick t is
    gated at t and fires at t+1 if it is still at `thr`, and at t+1 its own last spike is already
    older than `max_delay` -- reachable only when `refrac_ticks > max_delay`, which no fixture in
    this tree ran, which is exactly why it stayed a paragraph instead of a measurement. Under
    capacity pressure on that tick `step()` demoted it before propagate and the spike was not
    deferred but LOST, because firing is restricted to resident units. `_evictable` now also
    requires every unit of a candidate block to hold `v < thr`: a unit at or above threshold with
    its refractory released is about to fire and stays in core. The clause is NARROW on purpose --
    sub-threshold charge still evicts freely, which is the whole point of removing the blanket
    `v == 0` clause above. The cost of holding such a block is a capacity trip, the same
    over-capacity trade `_make_room` already makes rather than abandon a fetch. Measured, not
    argued: tests/test_neuron_pager.py's refractory-window arm builds `max_delay=2` with
    `refrac_ticks=5` and read the twin firing `[128, 256]` against the paged run's `[256]` on the
    tree without this clause."""

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
        # THE GRAIN, checked here and NOT MOVED from the store (tests/test_neuron_pager.py pins
        # the store's own refusal, and a moved check would turn that red). `blocks` is the one
        # input every other check in this constructor is ABOUT: capacity_blocks, pinned ids and
        # the residency mask are all counted in it, and `_page_in` compiles edges at the RECORD's
        # grain while marking residency at the PAGER's, so a mismatch inserts bundles whose pre
        # units the mask excludes -- silently, measured, with no exception anywhere.
        if not (
            isinstance(blocks.block_size, int)
            and not isinstance(blocks.block_size, bool)
            and blocks.block_size > 0
            and blocks.block_size % ALIGN_UNITS == 0
        ):
            raise ValueError(
                f"blocks.block_size must be a positive multiple of {ALIGN_UNITS} so every "
                "KernelState field block starts on a 128-byte boundary (the |i1 sign array is "
                f"the binding field), got {blocks.block_size!r}"
            )
        if store.blocks is not None and (
            blocks.block_size != store.blocks.block_size or blocks.n_blocks != store.blocks.n_blocks
        ):
            raise ValueError(
                f"the pager addresses {blocks.n_blocks} blocks of {blocks.block_size} units "
                f"while its store wrote {store.blocks.n_blocks} of {store.blocks.block_size} -- "
                "a fetch would return a record describing different units than the residency "
                "mask marks. A REOPENED store (store.blocks is None, the durable index serving "
                "fetch on its own) is accepted: that is the path the pager takes at S6, and "
                "refusing it would trade this silent defect for a loud one"
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
        #: WHERE THIS PAGER'S OWN SUBMISSIONS START in the store's width log. The hot-set load
        #: above is ONE fetch of len(pinned) blocks and is deliberately excluded: it is not a
        #: round, and an aggregate width bound that included it would go red on a CORRECT pager
        #: whenever the hot set is wider than blocks_per_round (measured, adversarial panel).
        self.widths_at_start = len(store.submission_widths)
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
        through a bundle, so a driven paged-out block would otherwise be invisible to a planner
        reading pending alone, and its charge would sit in the summary a whole tick longer than
        it needs to. It is a LATENCY clause, not the conservation clause it was first written as:
        `_propagate` routes a non-resident member's `ext_q` into `pending[p]`, so deferring the
        fetch costs staleness and never a unit of charge (the arm that made that distinction
        measurable is tests/test_neuron_pager.py's deferred-drive conservation test, which read
        18874368 fixed-point units lost before the routing existed)."""
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
    # distinct-from: brainstem/so101_arm.py::trajectory_breach -- a SAFETY predicate over joint
    #   positions against a derived droop envelope; it decides whether a physical arm has been
    #   disturbed, holds no unit, and no residency decision reads it.
    # distinct-from: cortex/nightly_skills.py::load -- a persistence RESTORE that rebuilds a
    #   router and a curiosity log from home(); it moves saved state back into a faculty and
    #   answers no question about whether anything may leave core.
    # distinct-from: cortex/time_cells.py::nominal_estimate -- a noise-free decode of what a
    #   time-cell ladder would report for an interval; a readout over thresholds, not a
    #   predicate over a substrate block's buffers. All three sites read 2026-09-08, and re-read
    #   2026-09-08 when the membrane clause below was added: none of them reads a membrane
    #   against a threshold to decide residency, so none is a second mechanism for this job.
    def _evictable(self, block, tick):
        """True when nothing this block does out of core can differ from what it would do in it:
        no spike within max_delay (ring safety), refrac released, no charge waiting in either
        buffer, and no unit already AT threshold.

        The membrane is tested for ONE thing and not for the other. Sub-threshold charge is not a
        reason to hold a block -- it keeps decaying while paged out, by the same op the twin runs,
        so it comes back holding what the twin holds. Charge that has already reached `thr` is a
        different quantity: with the refractory released, that unit fires on the NEXT propagate,
        and firing is restricted to RESIDENT units, so demoting it silences a spike rather than
        deferring one. The class docstring carries why the blanket `v == 0` clause existed, why
        removing its CAUSE removed it, and how this narrow clause closes the regime that removal
        left open."""
        units = self._units[block]
        arrays = self.kt.state.arrays
        if int(arrays["last_spike"][units].max()) >= tick - self.kt.syn.max_delay:
            return False
        if int(arrays["refrac"][units].max()) != 0:
            return False
        if bool(np.any(arrays["v"][units] >= arrays["thr"][units])):
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
            if b not in self.pinned and b not in due_set and self._evictable(b, tick)
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
        return edge_bytes_over_table(self.kt.syn)

    def resident_fraction(self) -> float:
        return float(self.resident.mean())

    # distinct-from: `resident_edge_bytes` two lines above -- that is the FIRST reading and this
    #   is deliberately a SECOND reading of the SAME quantity, which is the point: they must
    #   agree to the byte or nothing is reported. `edge_bytes_over_routes` carries how the two
    #   fail differently.
    # distinct-from: insula/interoceptive_integration.py::_measure_edge_charge -- also a paired
    #   reading, but its pair is an A/B over a VARIED producer (live vs pinned), not two paths to
    #   one static quantity; it measures whether a wire carries influence, not how many bytes are
    #   resident.
    # distinct-from: cortex/cortical_prior.py::e_fraction -- returns None for NOT MEASURED over a
    #   dataset lookup; no allocation, no second path, and nothing here may be None. Sites read
    #   2026-09-08.
    def resident_edge_bytes_via_routes(self) -> tuple[int, int]:
        """(bytes, dangling routes), reached through `by_pre` instead of the bundle table."""
        return edge_bytes_over_routes(self.kt.syn)

    def resident_floor_bytes_per_unit(self) -> float:
        """State + accumulator + pending summary + the residency flag, per unit -- what stays in
        core for EVERY unit whatever the pager does. Paging reduces edge bytes only."""
        units = self.kt.state.n_units
        state = sum(a.nbytes for a in self.kt.state.arrays.values())
        acc = sum(a.nbytes for a in self.kt._acc)
        pending = sum(a.nbytes for a in self.kt.pending)
        return (state + acc + pending + self.resident.nbytes) / units

    # distinct-from: `assert_within_budget` below -- that criterion reads how MANY rounds a tick
    #   needed against max_rounds, which is a trade the mechanism makes deliberately. These read
    #   how WIDE each round was against blocks_per_round, which is never a trade. The
    #   budget-blind arm is green on the first and RED on this one, which is exactly why both
    #   exist rather than one.
    # distinct-from: substrate/unit_blocks.py::UnitBlocks.units_in -- also a per-block width, but
    #   a STATIC property of the partition (which units a block owns, short at the tail). This
    #   counts what a fetch actually submitted at runtime; a correct partition says nothing about
    #   whether the I/O allowance was honoured.
    # distinct-from: substrate/lake_engine.py::AppendBudget -- a per-tag BYTES/tick flood assert
    #   against a tag's own recent rate; a rate over a non-stationary quantity, not an integer
    #   bound on one submission. Sites read 2026-09-08.
    def round_widths(self) -> list[int]:
        """The block count of every submission THIS pager issued (hot-set load excluded)."""
        return list(self.store.submission_widths[self.widths_at_start :])

    def assert_round_widths(self) -> bool:
        """A round is one fetch of AT MOST `blocks_per_round` block reads -- CHARGED here.

        The two rounds counters agree on the budget-blind arm because they sit one call apart;
        this is the reading they cannot make, and it goes red on that arm (7 > 2) in the same
        command. `blocks_read` gets its consumer in the same pass -- it must equal the sum of the
        store's recorded widths -- so neither is a write-only bookkeeping integer (P3 shape 4)."""
        widths = self.round_widths()
        if self.store.blocks_read != sum(self.store.submission_widths):
            raise RoundWidthExceeded(
                f"the store's blocks_read ({self.store.blocks_read}) does not equal the sum of "
                f"its recorded submission widths ({sum(self.store.submission_widths)}) -- the "
                "two width bookkeepers disagree, so no width verdict follows"
            )
        over = [(i, w) for i, w in enumerate(widths) if w > self.blocks_per_round]
        if over:
            i, w = over[0]
            raise RoundWidthExceeded(
                f"{len(over)} of {len(widths)} submissions carried more than "
                f"blocks_per_round={self.blocks_per_round} block reads; first at round {i} with "
                f"{w} blocks. A round is DEFINED as at most blocks_per_round reads, so a wider "
                "submission means the per-round I/O allowance was never honoured -- which the "
                "planner/store rounds pair cannot see, because on that arm it agrees"
            )
        return True

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


class _TripCensus(NeuronPager):
    """RECORDING ONLY: which clause refused to demote anything on a capacity trip.

    `capacity_trips` says a fetch went over capacity; it does not say WHY nothing was demotable,
    and this module's first printed explanation named the wrong clause (the membrane) because
    nobody had counted. On each trip this walks the resident blocks and buckets them with
    `_evictable` ITSELF rather than a second copy of its clauses -- pinned or in this tick's due
    set, refused by the predicate, and of those, how many had spiked inside the ring window
    `tick - max_delay`, which is the one clause a wavefront fixture keeps true. It overrides no
    decision: `_make_room` runs the incumbent and this class only reads the same state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trip_skipped = 0
        self.trip_refused = 0
        self.trip_in_ring_window = 0

    def _make_room(self, incoming, due, tick):
        before = self.capacity_trips
        due_set = set(due)
        last = self.kt.state.arrays["last_spike"]
        skipped, refused, ringed = 0, 0, 0
        for bid in sorted(self._resident_blocks):
            if bid in self.pinned or bid in due_set:
                skipped += 1
            elif not self._evictable(bid, tick):
                refused += 1
                if int(last[self._units[bid]].max()) >= tick - self.kt.syn.max_delay:
                    ringed += 1
        super()._make_room(incoming, due, tick)
        if self.capacity_trips > before:
            self.trip_skipped += skipped
            self.trip_refused += refused
            self.trip_in_ring_window += ringed


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


def _rings_of(kt):
    return [[list(slot) for slot in ring] for ring in kt._rings]


# distinct-from: basal_ganglia/evidence_accumulation.py::_patience_consumer_factory -- the "ring"
#   there is a SignalRing on a declared inter-region port, built fresh per replay arm; the rings
#   read here are KernelTick's DELAY-SLOT lists, an execution detail of one tick, and nothing is
#   constructed.
# distinct-from: substrate/selfcheck_cache.py::closure_key -- also derived per module run, but it
#   hashes a file closure to decide whether a self-check may be SKIPPED; these collect a run's own
#   per-tick series so an arm can print them. Sites read 2026-09-08.
def _run_twin(kt, drives, ticks):
    """The resident twin's trains AND its ring-slot key lists, tick by tick."""
    trains, rings = [], []
    for t in range(ticks):
        trains.append(kt.tick(drives.get(t)).tobytes())
        rings.append(_rings_of(kt))
    return trains, rings


def _run_paged(pager, drives, ticks):
    """The paged run's trains plus the three per-tick SERIES a counter cannot carry.

    `capacity_trips` is an EVENT count and `page_ins` a running total, so neither can say how far
    over capacity residency stood or when loading stopped. Rings, resident block count and
    per-tick page-ins are collected here so the arms can print series instead of end points."""
    trains, rings, held, page_ins = [], [], [], []
    last = pager.page_ins
    for t in range(ticks):
        trains.append(pager.tick(drives.get(t)).tobytes())
        rings.append(_rings_of(pager.kt))
        held.append(len(pager._resident_blocks))
        page_ins.append(pager.page_ins - last)
        last = pager.page_ins
    return trains, rings, held, page_ins


def _residency_line(name, held, capacity):
    """One printed line reconciling the residency SERIES with the trip COUNTER."""
    return (
        f"{name}: peak {max(held)} of {_N_BLOCKS} blocks resident, time-mean "
        f"{sum(held) / len(held):.2f}, and residency stood ABOVE capacity_blocks={capacity} on "
        f"{sum(1 for h in held if h > capacity)} of {len(held)} ticks. capacity_trips is NOT that "
        "number, and this line exists because the record once read it as if it were: it counts "
        "TICKS ON WHICH A FETCH FOUND TOO FEW EVICTABLE CANDIDATES, an event, so it can read "
        "small while residency sits over capacity for most of the run"
    )


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


# distinct-from: substrate/kernel_tick.py::Synapses -- the thing being MEASURED, not a second
#   measurement of it: that class HOLDS the bundle table and the routing index; these two read how
#   many bytes of it are resident. Site read 2026-09-08.
def edge_bytes_over_table(syn) -> int:
    """READING ONE of resident edge bytes: `nbytes` summed over `syn.bundles.values()`."""
    return int(sum(posts.nbytes + q.nbytes for posts, q in syn.bundles.values()))


# distinct-from: `edge_bytes_over_table` directly above -- the SAME quantity, deliberately, by a
#   path that fails differently. The table sum walks the dict a page-in writes into; this walks
#   the ROUTING INDEX a spike actually travels, so the two disagree exactly where the paging
#   bookkeeping is inconsistent: a bundle left in the table with no route into it is invisible to
#   the routes walk (the byte sums differ), and a route left in `by_pre` after its bundle was
#   popped is invisible to the table walk (`dangling` rises). `_page_out` pops both together, so
#   on a correct run they are EQUAL with dangling 0 -- and that equality is the criterion, in
#   place of the tracemalloc reading that claimed to be a second reading of this and was not.
# distinct-from: substrate/kernel_tick.py::Synapses -- as above, the measured object.
def edge_bytes_over_routes(syn) -> tuple[int, int]:
    """READING TWO: (bytes, dangling), reached by walking `syn.by_pre` -> keys -> arrays."""
    total, dangling = 0, 0
    for routes in syn.by_pre.values():
        for _part, _delay, key in routes:
            entry = syn.bundles.get(key)
            if entry is None:
                dangling += 1
                continue
            posts, q = entry
            total += posts.nbytes + q.nbytes
    return int(total), dangling


# distinct-from: cortex/brain.py::_regions_for -- the set of brain REGIONS that produced live
#   output this tick; a wiring/liveness census over a running brain, allocating nothing and
#   measuring no bytes.
# distinct-from: substrate/kernel_tick.py::Synapses -- the thing being MEASURED here, not a second
#   measurement of it: that class holds the bundles, this reads how many bytes of them the
#   allocator is still holding. Sites read 2026-09-08.
def _bundle_alloc_bytes():
    """LIVE traced bytes whose allocation site is `kernel_tick.py` -- the bundle arrays that
    `compile_bundles` created and that are still referenced.

    AN ALLOCATOR FOOTPRINT OF THE BUNDLE TABLE, PER-OBJECT OVERHEAD INCLUDED, AND USED AS NO
    CRITERION. Until 2026-09-08 this docstring and the printed paragraph both said the filtered
    snapshot "measures the same quantity as the exact `nbytes` sum by a different mechanism". It
    does not, and the correction is measured rather than softened: on this module's own fixture
    it reads ~34x the exact edge bytes (twin 1,979,364 traced against 57,344 B of array data --
    only 2.9% of the reading is the named quantity), because the filter is by FILENAME and so
    also counts ndarray object headers, the bundle dict's key tuples, the whole `by_pre` routing
    index, and KernelTick's own `_acc` / `pending` / `_members` / `_rings`, which are the
    resident FLOOR paging does not reduce. It also tracks bundle COUNT rather than bundle bytes
    (holding edge bytes fixed and moving the count 256 -> 3,072 moves this reading 5.0x while the
    exact instrument does not move), and it varies with PROCESS HISTORY (three identical
    repetitions in one process read 0.788 / 0.605 / 0.605 against an exact 0.857 every time) --
    which is the same defect that retired the unfiltered peak below.

    IT STAYS PRINTED, because deleting a reading the record quoted would hide the correction
    rather than make it. The criterion's two instruments are `edge_bytes_over_table` and
    `edge_bytes_over_routes`, both EXACT and required to agree to the byte."""
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
        twin_trains, twin_rings = _run_twin(twin, drives, 45)
        store, blocks = _new_store(root, "calm", chain)
        calm_capacity = 4
        _kt, pager = _new_pager(
            store,
            blocks,
            capacity_blocks=calm_capacity,
            blocks_per_round=derived["blocks_per_round"],
            max_rounds=derived["max_rounds"],
            max_stale_ticks=1,
            pinned=frozenset({0}),
            pager_cls=_TripCensus,
        )
        paged_trains, paged_rings, calm_held, _calm_page_ins = _run_paged(pager, drives, 45)
        if paged_trains != twin_trains:
            first = next(i for i, (a, b) in enumerate(zip(paged_trains, twin_trains)) if a != b)
            return _fail(f"calm paged run diverged from the resident twin at tick {first}")
        # THE RING CLAUSE, in THIS command. Section 3.2 credits the calm row with "trains AND
        # ring-slot key lists"; until 2026-09-08 that second half existed only in pytest, i.e. in
        # a different command than the heading names.
        if paged_rings != twin_rings:
            first = next(i for i, (a, b) in enumerate(zip(paged_rings, twin_rings)) if a != b)
            return _fail(f"calm ring-slot key lists differ from the twin's at tick {first}")
        if not (pager.page_ins > 0 and pager.evictions > 0):
            return _fail(
                f"the calm arm never paged (page_ins={pager.page_ins}, "
                f"evictions={pager.evictions}) -- an identity that never paged proves nothing"
            )
        if not 0.0 < pager.resident_fraction() < 1.0:
            return _fail(f"resident fraction {pager.resident_fraction()} is not inside (0, 1)")
        print(
            f"S-R2 calm: 45 ticks byte-identical to the resident twin in TRAINS AND in RING-SLOT "
            f"KEY LISTS; page_ins={pager.page_ins} evictions={pager.evictions} resident fraction "
            f"{pager.resident_fraction():.3f} of {_N_BLOCKS} blocks; rounds trips="
            f"{len(pager.budget_trips)}, CAPACITY trips={pager.capacity_trips}"
        )
        print(f"  {_residency_line('calm arm residency', calm_held, calm_capacity)}")
        if pager.capacity_trips:
            print(
                f"  the capacity trips are the honest half of that fraction: on "
                f"{pager.capacity_trips} tick(s) a fetch found too few evictable candidates, so "
                "it went over capacity rather than deferring a block -- never a drop, and the "
                f"reason the final fraction sits above {calm_capacity}/{_N_BLOCKS}. WHICH CLAUSE "
                f"REFUSED, counted rather than asserted: across those trips {pager.trip_skipped} "
                "resident block-visits were pinned or already in that tick's due set and "
                f"{pager.trip_refused} were refused by the eviction predicate, of which "
                f"{pager.trip_in_ring_window} had spiked inside the ring window tick - "
                "max_delay. THAT CENSUS IS THE ATTRIBUTION, and this line used to give a "
                "different one: it credited the trips to the no-locality regime of disclosure "
                "2.6(d) and routed them to T067, on a fixture whose block-to-block reach is "
                "exactly 1 -- a nearest-neighbour chain, the most LOCAL topology available, so "
                "connectivity locality is neither the variable here nor varied anywhere in this "
                "row. The measured cause is the ring-safety clause under a firing wavefront: "
                "every refusal counted above lies inside tick - max_delay. THE MEMBRANE IS NOT "
                "THE REASON either, and this line used to say that too: dropping the old "
                "`v == 0` clause moved this arm's page_ins, evictions and capacity trips by "
                "ZERO, because a fixture whose every edge clears threshold resets each unit to "
                "exactly 0 anyway. That clause's cost is real but visible elsewhere -- in the "
                "sub-threshold arm in tests/test_neuron_pager.py, where it pinned a charged "
                "block in core and the fetch went over capacity"
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

        # ---- S-R2 under overload at max_stale_ticks = 0: IDENTITY while the budget trips ----
        # Section 3.4 says the overload arm "runs in two regimes ... both in the same command".
        # Until 2026-09-08 only the max_stale_ticks=2 regime was in this command and the =0
        # regime lived in pytest, so the sentence was true only if "the same command" silently
        # meant a different one. This is that second regime, here.
        z_store, z_blocks = _new_store(root, "stale0", fanout)
        z_twin = _twin(fanout, max_delay=1)
        z_twin_trains, _z_twin_rings = _run_twin(z_twin, ov_drives, 20)
        z_kt, z_pager = _new_pager(
            z_store,
            z_blocks,
            max_delay=1,
            capacity_blocks=_N_BLOCKS,
            blocks_per_round=1,
            max_rounds=1,
            max_stale_ticks=0,
            pinned=frozenset({0}),
        )
        z_trains, _z_rings, _z_held, _z_page_ins = _run_paged(z_pager, ov_drives, 20)
        if not z_pager.budget_trips:
            return _fail("max_stale_ticks=0 forced no round past the budget -- no trip to report")
        if z_pager.max_staleness != 0:
            return _fail(f"max_stale_ticks=0 left staleness {z_pager.max_staleness}, not 0")
        if z_trains != z_twin_trains:
            first = next(i for i, (a, b) in enumerate(zip(z_trains, z_twin_trains)) if a != b)
            return _fail(f"the max_stale_ticks=0 overload arm diverged from the twin at {first}")
        if not np.array_equal(z_kt.delivered, z_twin.delivered):
            return _fail("per-unit delivered charge under overload did not equal the twin's")
        print(
            f"S-R2 overload at max_stale_ticks=0: {len(z_pager.budget_trips)} trip(s) and "
            "staleness 0, byte-identical to the resident twin with per-unit delivered charge "
            "equal to the twin's exactly -- identity WITH trips, the second regime, this command"
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
            "empty round and a fetch that raises mid-tick -- not an independent measurement of "
            "the same quantity, and NOT the budget-blind arm, which they agree about"
        )

        # ---- the WIDTH instrument: the reading the rounds PAIR structurally cannot make -----
        for name, pgr in (
            ("calm", pager),
            ("overload", ov_pager),
            ("stale0", z_pager),
            ("r1", r1_pager),
            ("r1calm", calm_pager),
        ):
            try:
                pgr.assert_round_widths()
            except RoundWidthExceeded as exc:
                return _fail(f"the CORRECT arm {name!r} exceeded a round's width: {exc}")
        blind_widths = blind_pager.round_widths()
        try:
            blind_pager.assert_round_widths()
            return _fail("the BUDGET-BLIND arm passed the width instrument; it cannot be the red")
        except RoundWidthExceeded:
            pass
        print(
            f"width: every submission of the four correct arms carried at most its own "
            f"blocks_per_round, and blocks_read equals the sum of the recorded widths on every "
            f"store (so neither counter is write-only). The BUDGET-BLIND arm reads widths "
            f"{blind_widths} against blocks_per_round={blind_pager.blocks_per_round} and RAISES "
            "-- a red the planner/store pair cannot see, because that arm reports 1 round and "
            "issues exactly 1 fetch, so its two counts agree while the one submission carries "
            f"{max(blind_widths)} blocks. The constructor's pinned hot-set load is excluded by "
            "construction: it is one fetch of len(pinned), and an aggregate bound including it "
            "would go red on a CORRECT pager whose hot set is wider than a round"
        )

        # ---- residency measured as MEMORY, by TWO EXACT readings that must agree ------------
        wide = _chain_edges(seed=5, fan=4)
        bytes_capacity = 4
        bytes_store, bytes_blocks = _new_store(root, "bytes", wide)
        tracemalloc.start()
        twin_wide = _twin(wide)
        _run(twin_wide, {0: _drive([1])}, 20)
        twin_traced = _bundle_alloc_bytes()
        twin_peak = tracemalloc.get_traced_memory()[1]
        twin_edge_bytes = edge_bytes_over_table(twin_wide.syn)
        twin_route_bytes, twin_dangling = edge_bytes_over_routes(twin_wide.syn)
        tracemalloc.stop()
        tracemalloc.start()
        _b_kt, b_pager = _new_pager(
            bytes_store,
            bytes_blocks,
            capacity_blocks=bytes_capacity,
            blocks_per_round=4,
            max_rounds=2,
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        _bt, _br, bytes_held, _b_page_ins = _run_paged(b_pager, {0: _drive([1])}, 20)
        paged_traced = _bundle_alloc_bytes()
        paged_peak = tracemalloc.get_traced_memory()[1]
        paged_edge_bytes = b_pager.resident_edge_bytes()
        paged_route_bytes, paged_dangling = b_pager.resident_edge_bytes_via_routes()
        tracemalloc.stop()
        edge_owning = len({key[0] // _BLOCK for key in twin_wide.syn.bundles})
        agree = (
            paged_edge_bytes == paged_route_bytes
            and twin_edge_bytes == twin_route_bytes
            and not paged_dangling
            and not twin_dangling
        )
        below = paged_edge_bytes < twin_edge_bytes and paged_route_bytes < twin_route_bytes
        print(
            f"residency (NOT a win claim): resident edge bytes {paged_edge_bytes} vs the twin's "
            f"{twin_edge_bytes} (ratio {paged_edge_bytes / max(twin_edge_bytes, 1):.3f}), READ "
            f"TWICE AND REQUIRED TO AGREE TO THE BYTE. Reading one sums nbytes over the bundle "
            f"TABLE; reading two sums the same arrays reached by walking the ROUTING INDEX "
            f"by_pre, and reads {paged_route_bytes} paged / {twin_route_bytes} twin with "
            f"{paged_dangling}/{twin_dangling} dangling routes. They fail DIFFERENTLY, which is "
            "the point: a bundle orphaned in the table moves only the first, a by_pre entry left "
            "behind after a page-out moves only the second. Agreement="
            f"{agree}, both-below-twin={below}. Resident floor "
            f"{b_pager.resident_floor_bytes_per_unit():.1f} B/unit for EVERY unit (state 21 B + "
            f"{_P} accumulators + {_P} pending + a flag) -- paging reduces edge bytes only"
        )
        print(
            f"  {_residency_line('residency arm', bytes_held, bytes_capacity)}; page_ins="
            f"{b_pager.page_ins} evictions={b_pager.evictions} capacity trips="
            f"{b_pager.capacity_trips}. THE RATIO'S SHAPE, stated rather than left to be "
            f"inferred: {b_pager.evictions} block(s) of the {edge_owning} that own edges were "
            f"demoted, so this reading is what the fixture actually cost at a nominal capacity "
            f"of {bytes_capacity}, not a best case"
        )
        print(
            f"  THE TRACEMALLOC READINGS ARE PRINTED AND USED AS NO CRITERION: filtered to "
            f"kernel_tick.py {paged_traced} vs {twin_traced} (ratio "
            f"{paged_traced / max(twin_traced, 1):.3f}); unfiltered peak {paged_peak} vs "
            f"{twin_peak} (ratio {paged_peak / max(twin_peak, 1):.3f}). Both are an ALLOCATOR "
            "FOOTPRINT OF THE BUNDLE TABLE INCLUDING PER-OBJECT OVERHEAD, not a second reading "
            f"of resident edge bytes: the filtered one runs about "
            f"{twin_traced / max(twin_edge_bytes, 1):.0f}x the exact quantity on this fixture "
            "because a filename filter also counts ndarray headers, the dict's key tuples, the "
            "whole by_pre index and KernelTick's own _acc/pending/_members/_rings; it tracks "
            "bundle COUNT rather than bundle bytes; and it varies with process history (0.788 / "
            "0.605 / 0.605 across three identical repetitions in one process against an exact "
            "0.857 every time). The row pre-registered the peak as the second instrument and "
            "then the filtered snapshot as its replacement, calling it 'the SAME quantity by a "
            "different mechanism' -- it was neither, and both are kept visible so the correction "
            "is not a silent one. The criterion above uses the two EXACT readings"
        )
        if not agree:
            return _fail(
                f"the two exact residency instruments disagree (table {paged_edge_bytes} vs "
                f"routes {paged_route_bytes} paged, {twin_edge_bytes} vs {twin_route_bytes} "
                f"twin, dangling {paged_dangling}/{twin_dangling}) -- report no number"
            )
        if not below:
            return _fail(
                "the two exact residency instruments do not both place the paged run below the twin"
            )

        # ---- the THIRD pre-registered attribution arm (P8 clause 3): eviction NEUTERED ------
        # 2.6 pre-registered three per-item neuters and this one was dropped at build time, so
        # 3.4's "every deviation" list was short by one. Built now: capacity = n_blocks means no
        # block can ever be demoted, so if the ratio above is caused by EVICTION the neutered arm
        # must read the twin's bytes exactly.
        neuter_store, neuter_blocks = _new_store(root, "neuter", wide)
        _n_kt, n_pager = _new_pager(
            neuter_store,
            neuter_blocks,
            capacity_blocks=_N_BLOCKS,
            blocks_per_round=4,
            max_rounds=2,
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        _nt, _nr, _n_held, n_page_ins = _run_paged(n_pager, {0: _drive([1])}, 20)
        neuter_edge_bytes = n_pager.resident_edge_bytes()
        neuter_route_bytes, neuter_dangling = n_pager.resident_edge_bytes_via_routes()
        warm = 10
        print(
            f"attribution (eviction NEUTERED, capacity_blocks={_N_BLOCKS} of {_N_BLOCKS}) vs the "
            f"INCUMBENT (capacity_blocks={bytes_capacity}), side by side on one fixture: "
            f"resident edge bytes {neuter_edge_bytes} vs {paged_edge_bytes}, against the twin's "
            f"{twin_edge_bytes} -- EQUAL to the twin with eviction off "
            f"({neuter_edge_bytes == twin_edge_bytes}), BELOW it with eviction on "
            f"({paged_edge_bytes < twin_edge_bytes}); evictions {n_pager.evictions} vs "
            f"{b_pager.evictions}; second reading {neuter_route_bytes} with {neuter_dangling} "
            f"dangling. So the ratio is caused by eviction and by nothing else. THE OTHER HALF "
            f"OF THE PRE-REGISTERED READING DOES NOT DISCRIMINATE HERE, printed rather than "
            f"quietly dropped: page-ins after warm-up (t >= {warm}) are "
            f"{sum(n_page_ins[warm:])} neutered and {sum(_b_page_ins[warm:])} incumbent -- both "
            "zero on this fixture, so that clause separates nothing and only the bytes clause is "
            "load-bearing"
        )
        if not (
            neuter_edge_bytes == twin_edge_bytes
            and neuter_route_bytes == twin_route_bytes
            and n_pager.evictions == 0
            and paged_edge_bytes < twin_edge_bytes
        ):
            return _fail(
                f"the neutered attribution arm did not read the twin's bytes ({neuter_edge_bytes}"
                f" vs {twin_edge_bytes}, evictions {n_pager.evictions}) -- the ratio is not "
                "attributable to eviction"
            )

        # ---- the lake digest across a full page-out/page-in cycle, WITH ITS CONTROL --------
        before, n_segments = segment_digest(store.root)
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
        after, after_n = segment_digest(store.root)
        if after != before or after_n != n_segments:
            return _fail("the lake's bytes moved across a page-out/page-in cycle")
        # THE POSITIVE CONTROL, in this same command. The identity above is TRUE by construction
        # -- the pager performs no write at all -- so on its own it is a criterion with no
        # reachable red, which P8 forbids: the arm passed unchanged with every write path of the
        # store replaced by raise-on-call, and passed printing IDENTICAL over ZERO BYTES when its
        # glob was blinded. Two reds now exist: a flipped byte on a COPY must move the digest,
        # and an empty glob must be REFUSED rather than compared.
        control_root = root / "digest-control"
        shutil.copytree(store.root / SEGMENT_DIR, control_root / SEGMENT_DIR)
        control_before, control_n = segment_digest(control_root)
        victim = sorted((control_root / SEGMENT_DIR).glob("seg-*.lake"))[0]
        raw = bytearray(victim.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        victim.write_bytes(bytes(raw))
        control_after, _ = segment_digest(control_root)
        blind_refused = False
        try:
            segment_digest(root / "digest-control-empty")
        except ValueError:
            blind_refused = True
        if control_before != before or control_n != n_segments:
            return _fail("the digest control's copy is not byte-identical to the store it copies")
        if control_after == control_before:
            return _fail("a flipped byte did not move the digest -- the instrument is blind")
        if not blind_refused:
            return _fail("an EMPTY segment glob was digested instead of refused")
        print(
            f"lake: {cyc_pager.page_ins} page-ins and {cyc_pager.evictions} page-outs left the "
            f"store's {n_segments} segment file(s) IDENTICAL (sha256 {before[:16]}...) -- "
            "page-out changes residency in RAM and destroys no record, which is why 0x0C is "
            "IMMORTAL. CONTROL, same command, same digest function over a root: one byte flipped "
            f"in a scratch COPY moves it to {control_after[:16]}..., and a zero-file glob is "
            "REFUSED rather than reported as sha256-of-nothing == sha256-of-nothing. The "
            "'MORTAL class evicted in the same pass' control stays NOT APPLICABLE to a pass that "
            "destroys nothing (stated, not faked)"
        )
        for closing in (
            store,
            drop_store,
            ov_store,
            z_store,
            r1_store,
            calm_store,
            blind_store,
            bytes_store,
            neuter_store,
        ):
            closing.close()
    print(
        f"neuron_pager self-check OK in {time.perf_counter() - started:.2f} s: both reds counted "
        "with their broken arms and their calm controls in this one command"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

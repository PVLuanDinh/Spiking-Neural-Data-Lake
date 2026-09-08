"""The sleep-night merge of the fresh tier and the locality reorder (specs/022 T067 / Phase 3 S4).

WHAT IT IS. `UnitBlockStore` writes one 0x0C record per unit block and, since T067, 0x0D deltas
beside it; `fetch` serves base + unmerged deltas through the ONE apply. This module is the other
half: a duty that FOLDS a block's unmerged deltas into a NEW 0x0C record so the fresh tier stays
short, and a pass that REORDERS the blocks' records so the ones a tick fetches together sit
together in the lake. Both are APPEND-ONLY. Nothing here destroys or rewrites a record: the old
0x0C record and every 0x0D record stay in their segments byte for byte, and what moves is a
POINTER in the derived index (D1 -- the lake is never rewritten in place).

THE RELOCATION PROOF COMES BEFORE THE POINTER MOVE, and it is the reason the merge can be
believed. After the merged record is sealed it is decoded BACK FROM DISK and required to be (a)
byte-identical to what this pass built and (b) equal to base + deltas RE-APPLIED FROM THE LAKE
with `lake_engine.apply_deltas` -- the INCUMBENT apply, deliberately not the seam a subclass can
replace, which is what makes a merge that drops a delta reachable as a red rather than a
possibility. A block whose proof fails is SKIPPED with its pointer untouched and counted; it is
never half-moved.

"BACKGROUND" IS A SLEEP-NIGHT DUTY UNDER AN INTEGER BUDGET, NOT A THREAD. `budget_blocks` is an
INPUT WITH NO DEFAULT (P1); a dirty set larger than the budget is DEFERRED to the next call and
counted. `merge` never raises past its own guards -- the nightly-duty shape `cortex/brain.py`'s
`_enter_sleep` blocks already use. THE SLEEP WIRE IS NOT MADE HERE and that is deliberate, not an
omission: the live lake has no segment tier at all (`~/.snn_data_lake` holds `brain_lake.bin` and
neither `segments/` nor `lake_index.db`), so a guarded import from `_enter_sleep` would flip this
module REACHABLE for `dormancy_gate` -- whose AST walk is unconditional -- and leave a hollow
green over a duty with nothing to do. The `wire` row in `dormancy_gate.EXEMPT` dates it to S6.

READS PER ROUND IS THE QUANTITY THE REORDER MOVES. `UnitBlockStore.fetch` coalesces a round's
needed `(segment, seq)` pairs into maximal CONTIGUOUS runs and issues one `read_records` per run,
so a tick that fetches k index-distant blocks costs k reads and the same tick after a reorder that
made them adjacent costs 1. `bytes_fetched / bytes_used` is 1.0 BY CONSTRUCTION here -- a run is
built only from seqs the round asked for, so there is no over-read -- and it is printed as such
rather than presented as a result.

ADOPTION IS A RECORD, NOT A POSITION -- D2's epoch fence, and it closes the hazard this row's
first pass opened. A block whose relocation proof FAILS leaves an unadopted 0x0C record in the
lake, and `build_index` used to take the LATEST 0x0C record per block by forward-scan position, so
a rebuild of such a store derived a pointer the store never adopted AND a superseded watermark to
match it -- a reclaimer reading that mark could name deltas as folded that the current record does
not hold. `UnitBlockStore.adopt_merged` now appends a `lake_engine.BlockAdoptRecord` (0x0D,
version byte 2, carrying the block, the generation, the adopted `(container, seq)` and how many
deltas that generation folded) BEFORE it upserts the pointer, and `build_index` adopts a 0x0C
record ONLY where such a record names it. `build` fences generation 0 the same way, so there is no
special case for the built record. The ordering is also the recovery property: a crash between the
ADOPT record and the upsert leaves the truth derivable from the lake alone, and the sanctioned
repair is a rebuild.

WHAT THE FENCE DOES NOT CLOSE, measured rather than read: `RETENTION[0x0D].reclaimable` STAYS
False. A reclaim of exactly what that exemption licenses -- a block's folded delta prefix -- was
constructed and run, and because a sealed segment cannot lose a record in place a reclaimer must
re-containerise the survivors; the ADOPT record's stored container then names a segment the new
root lacks, and the watermark it carries (folded_through) exceeds the deltas a rebuild can still
see, so the fresh tier drops the UNMERGED tail. The reason on `RETENTION[0x0D]` carries both
readings and the fix a reclaimer row owes.

WHAT THIS ROW DOES NOT CLAIM: no paging WIN (T068's), no tick-time claim, no live producer of
kernel-tier deltas (S6/T062), no live sleep duty and no cockpit row (S6), no reclaim of
superseded records (the label and the proof land; the segment family has no reclaimer and gains
none here), no over-read trade, and no claim that the co-fetch order is the right locality signal
beyond the stream it watched.

  python -m snn_data_lake.substrate.lake_compactor
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from snn_data_lake.substrate.home import _T_BLOCKDELTA, _T_UNITBLOCK
from snn_data_lake.substrate.kernel_state import KernelState
from snn_data_lake.substrate.kernel_tick import KernelTick, Synapses
from snn_data_lake.substrate.lake_engine import (
    BlockDeltaRecord,
    UnitBlockRecord,
    apply_deltas,
    edge_table,
    record_from_edges,
    retention_of,
)
from snn_data_lake.substrate.neuron_pager import PERF_BASELINE, NeuronPager, UnitBlockStore

#: The share of one subjective NIGHT this self-check is willing to spend folding the fresh tier
#: when it derives `budget_blocks`. CHOSEN and disclosed -- the self-check's own allowance, not a
#: project constant, exactly as `neuron_pager.SELFCHECK_IO_ALLOWANCE` is for the rounds budget. A
#: real caller states its own.
SELFCHECK_NIGHT_SHARE = 0.10
#: A subjective night in SLEEP TICKS, taken from the incumbent phase machine rather than picked
#: here: `cortex/brain.py::Brain.day_period` defaults to 200 and
#: `hypothalamus/circadian_clock.light_cycle` runs a 0.5 duty, so a night is about 100 sleep ticks
#: (the recorded baseline measured 59 sleep of 120 ticks, 0.492). NAMED rather than imported:
#: importing `cortex.brain` from the substrate tier to read a constructor default would drag a
#: live root's whole import graph in for one number.
SELFCHECK_NIGHT_TICKS = 100

#: Read statically by `substrate/gradient_claim_gate.py`; never imported by it. The gate's walk is
#: recall-biased and captures this module's `main`, so it is declared rather than left
#: UNDETERMINABLE -- the same disposition `substrate/neuron_pager.py` and `substrate/unit_blocks.py`
#: give their own self-checks, for the same reason: none of the four learning dispositions
#: describes a self-check, and any of them would be false here.
GRADIENT_CLAIM = {
    "main": (
        "NOT_A_RULE",
        "this module's self-check: it derives a merge budget from a measured per-block fold cost, "
        "runs the three pre-registered reds with their broken arms and their controls, and reads "
        "reads-per-round before and after a locality reorder. It moves no weight and computes no "
        "update -- the weights it carries are the ones the records already held, moved from one "
        "immutable record to the next -- so it is not a learning rule at all",
    ),
}

_TAG_BLOCK = _T_UNITBLOCK[0]
_TAG_DELTA = _T_BLOCKDELTA[0]


def _labels():
    """READ THE RETENTION LABEL FOR EVERY TAG THIS MODULE MOVES, per tag.

    L6's correction, inherited rather than re-derived: a label read is credited to the TAG it
    names, never to the function that contains it, so a pass that touches two classes reads two
    labels. 0x0C must be IMMORTAL (this pass may never destroy one) and 0x0D must be COMPACTABLE
    (its superseded prefix is what the merge's proof licenses); either label changing under this
    module is a refusal at the door, not a surprise at the write."""
    return retention_of(_TAG_BLOCK), retention_of(_TAG_DELTA)


def _refuse_on_label():
    block, delta = _labels()
    if block[0] != "IMMORTAL" or block[1]:
        raise ValueError(
            f"tag {_TAG_BLOCK:#04x} is labeled {block[0]}/reclaimable={block[1]}; this pass "
            "appends a NEW record and moves a pointer, and it refuses to run against a class "
            "whose label no longer says the old record must survive"
        )
    if delta[0] != "COMPACTABLE":
        raise ValueError(
            f"tag {_TAG_DELTA:#04x} is labeled {delta[0]}; a fold that supersedes a delta needs "
            "the class whose contract permits it"
        )


def _budget_of(budget_blocks) -> int:
    if isinstance(budget_blocks, bool) or not isinstance(budget_blocks, int) or budget_blocks < 1:
        raise ValueError(
            "budget_blocks is an INPUT with NO DEFAULT (P1) and must be an int >= 1; the caller "
            "derives it from a measured per-block fold cost and its own sleep-night allowance, "
            f"got {budget_blocks!r}"
        )
    return int(budget_blocks)


def block_ids(store) -> list[int]:
    """Every block id the store addresses, through its public pointer lookup -- so a REOPENED
    store (whose `blocks` partition is None because the durable index serves `fetch` on its own)
    is enumerated the same way a freshly built one is."""
    if store.blocks is not None:
        return list(range(store.blocks.n_blocks))
    ids: list[int] = []
    while True:
        try:
            store.block_pointer(len(ids))
        except KeyError:
            return ids
        ids.append(len(ids))


# distinct-from: substrate/neuron_pager.py::NeuronPager.plan -- that orders a tick's DUE SET for
#   fetching, by staleness and pending charge, and is recomputed every tick. This orders the
#   BLOCKS' PLACEMENT IN THE LAKE once, from a run's recorded fetch sets, and the pager's ordering
#   is unaffected by it.
# distinct-from: substrate/unit_blocks.py::UnitBlocks -- the PARTITION (which unit lives in which
#   block), which this never changes: a reorder moves a block's record, never a unit's membership,
#   so `block_of`, the pager's lowest-block-id tiebreak and every `KernelState` index are
#   untouched. Both sites read 2026-09-08.
def co_fetch_order(due_sets, n_blocks) -> list[int]:
    """A block placement order from a run's per-tick fetch sets: blocks fetched TOGETHER go
    ADJACENT.

    A deterministic greedy walk over the co-fetch count graph -- from the lowest-id fetched block,
    take the unplaced neighbour it was fetched with most often, ties by LOWEST ID; when the walk
    strands, restart from the lowest unplaced fetched block; blocks that were never fetched keep
    INDEX ORDER at the tail. The co-fetch graph is the only locality signal this row has ("block
    adjacency until the prior lands"): the H01 prior (T062) may replace it as the order's INPUT
    later without changing this mechanism.

    `n_blocks` is a second argument the pre-registration's `co_fetch_order(due_sets)` did not
    name, and it is required rather than inferred: without it the never-fetched tail cannot be
    enumerated and the result would not be a permutation, which the caller asserts."""
    n_blocks = int(n_blocks)
    counts: dict[tuple[int, int], int] = {}
    fetched: set[int] = set()
    for group in due_sets:
        ids = sorted({int(b) for b in group})
        fetched.update(ids)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                counts[(a, b)] = counts.get((a, b), 0) + 1
    outside = sorted(b for b in fetched if not 0 <= b < n_blocks)
    if outside:
        raise ValueError(f"fetch set names block(s) {outside} outside [0, {n_blocks})")

    def together(a: int, b: int) -> int:
        return counts.get((a, b) if a < b else (b, a), 0)

    order: list[int] = []
    unplaced = set(fetched)
    while unplaced:
        current = min(unplaced)
        order.append(current)
        unplaced.discard(current)
        while True:
            best, best_count = None, 0
            for candidate in sorted(unplaced):
                seen = together(current, candidate)
                if seen > best_count:  # strictly greater, so a tie keeps the LOWEST id
                    best, best_count = candidate, seen
            if best is None:
                break
            order.append(best)
            unplaced.discard(best)
            current = best
    order += [b for b in range(n_blocks) if b not in fetched]
    return order


# distinct-from: hippocampus/lake_consolidation.py::LakeConsolidation.consolidate -- the sleep
#   PRUNE POLICY over FACT dicts: which learned/re-derivable facts survive a cap. It is pure, it
#   decides what to DROP, and `home.KnowledgeLog.consolidate` performs the drop by rewriting
#   brain_lake.bin. This drops nothing at all: it appends a folded record and moves a pointer in
#   the segment family, which that pass never touches.
# distinct-from: substrate/home.py::KnowledgeLog.consolidate and substrate/binlog.py::BinLog.rewrite
#   -- the tree's ONE durable reclaimer: a whole-log rewrite of brain_lake.bin to an explicit
#   survivor list, which is how this repo reclaims. Nothing here rewrites a file, deletes a record
#   or shortens a log; the old bytes stay and only the index's pointer moves, which is why this
#   module never enters the `immortal_retention` reclaimer census.
# distinct-from: substrate/brain_telemetry.py::_compact/_start_compaction -- rotate-and-decimate
#   over a JSONL TELEMETRY file on a daemon thread, keeping 1 in N lines. It is the tree's only
#   background-compaction precedent and it is not a lake record anywhere in it; this is a
#   foreground sleep-NIGHT duty under an integer budget, not a thread.
# distinct-from: substrate/lake_migrate.py::migrate -- an identity COPY of every record, read-as-is
#   per tag, from one root into a fresh SegmentStore with per-tag digest equality both ways. It
#   re-containerises and never merges two records, changes a record's position, or writes a second
#   record for a key that had one -- which are exactly the three things this does.
# distinct-from: substrate/lake_engine.py::SynapseDelta -- the LEARNER's per-synapse delta over a
#   dense [pre, post] matrix (0x09), POST-grouped and quantized at the learner's lr, with no
#   writer anywhere. This folds STRUCTURAL out-edge deltas (0x0D) at the block grain.
#   All five sites read 2026-09-08.
class LakeCompactor:
    """The fold and the placement pass. Neither destroys anything; both prove before they commit.

    Two override SEAMS exist so the reds are constructible, and both are named rather than
    discovered: `_apply` (what a merge folds with) and `_prove` (the relocation proof). The proof
    deliberately re-applies with `lake_engine.apply_deltas` rather than through `_apply`, so a
    subclass that drops a delta cannot also silence the arm that would catch it."""

    # ---- the seams -----------------------------------------------------------------------------
    def _apply(self, base_edges, deltas, *, generation):
        """The fold's apply -- THE ONE APPLY (P2), the same one `UnitBlockStore.fetch` serves
        with, so a folded record can never disagree with the record the pager was reading."""
        return apply_deltas(base_edges, deltas, generation=generation)

    def _prove(self, store, bid, container, seq, payload, generation) -> bool:
        """THE RELOCATION PROOF, run BEFORE the pointer moves and never after.

        Three things, all from disk: the sealed record reads back BYTE-IDENTICAL to what this pass
        built; it decodes to a record describing THIS block at the same grain; and its edge set
        equals base + deltas re-applied from the lake with the INCUMBENT apply. The third is the
        one that makes a dropping fold reachable, which is why it does not go through `_apply`."""
        sealed = store.payload_at(container, seq)
        if sealed != payload:
            return False
        back = UnitBlockRecord.decode(sealed)
        lake_base = UnitBlockRecord.decode(store.payload_at(*store.block_pointer(bid)))
        if (
            back.block_id != bid
            or back.n_units_in_block != lake_base.n_units_in_block
            or back.block_size != lake_base.block_size
        ):
            return False
        lake_deltas = [
            BlockDeltaRecord.decode(store.payload_at(*loc)) for loc in store.unmerged_pointers(bid)
        ]
        want = apply_deltas(edge_table(lake_base), lake_deltas, generation=generation)
        return edge_table(back) == want

    def _budgeted(self, dirty, budget_blocks):
        """(this call's blocks, the ones deferred to the next). The seam the budget-blind arm
        replaces."""
        dirty = list(dirty)
        return dirty[:budget_blocks], dirty[budget_blocks:]

    def _folded(self, store, bid):
        """(the block's base record, its edge set with every unmerged delta folded in)."""
        base = UnitBlockRecord.decode(store.payload_at(*store.block_pointer(bid)))
        deltas = [
            BlockDeltaRecord.decode(store.payload_at(*loc)) for loc in store.unmerged_pointers(bid)
        ]
        return base, self._apply(edge_table(base), deltas, generation=store.generation(bid))

    # ---- the sleep-night fold -------------------------------------------------------------------
    def merge(self, store, *, budget_blocks, resident_blocks=()):
        """Fold up to `budget_blocks` DIRTY blocks' unmerged deltas into new 0x0C records.

        Per block, in block-id order: build the merged edge set through the one apply, encode it,
        append and seal it, PROVE RELOCATION, and only then upsert the pointer, advance the
        generation and mark the folded deltas superseded. A failed proof skips the block with its
        pointer untouched and counts a refusal. Returns the duty's counters and never raises past
        its own guards -- except the two refusals that are configuration and not runtime: a
        missing or non-integer budget, and a retention label that no longer permits the pass.

        `resident_blocks` is the caller's pager residency set, and it is a fact rather than a
        picked constant: with no pager attached it is empty. A merged block that is CURRENTLY
        resident keeps its old compiled bundles until it is evicted and paged back in, so it is
        counted as `stale_resident` and reported rather than assumed refreshed (2.6(g))."""
        budget_blocks = _budget_of(budget_blocks)
        _refuse_on_label()
        started = time.perf_counter()
        resident = {int(b) for b in resident_blocks}
        dirty = [bid for bid in block_ids(store) if store.n_unmerged(bid)]
        chosen, deferred = self._budgeted(dirty, budget_blocks)
        merged_ids: list[int] = []
        refusals, bytes_appended, stale_resident = 0, 0, 0
        refusal_last = ""
        for bid in chosen:
            try:
                base, folded = self._folded(store, bid)
                generation = store.generation(bid)
                payload = record_from_edges(
                    bid, base.n_units_in_block, base.block_size, folded
                ).encode()
                container, seqs = store.seal_and_locate([payload])
                bytes_appended += len(payload)
                if not self._prove(store, bid, container, seqs[0], payload, generation):
                    refusals += 1
                    refusal_last = f"block {bid}: the relocation proof failed; pointer untouched"
                    continue
                store.adopt_merged(bid, container, seqs[0])
                merged_ids.append(bid)
                if bid in resident:
                    stale_resident += 1
            except Exception as exc:  # noqa: BLE001 - a nightly duty counts, it never kills a tick
                refusals += 1
                refusal_last = f"block {bid}: {exc!r}"[:200]
        return {
            "merged": len(merged_ids),
            "deferred": len(deferred),
            "refusals": refusals,
            "bytes_appended": bytes_appended,
            "seconds": time.perf_counter() - started,
            "stale_resident": stale_resident,
            #: the ids, beside the counts: S-R7's criterion is that the UNION of two calls' merged
            #: sets equals the dirty set, which a count cannot express.
            "merged_ids": tuple(merged_ids),
            "deferred_ids": tuple(deferred),
            "refusal_last": refusal_last,
        }

    # ---- the locality reorder --------------------------------------------------------------------
    def _reorder_payloads(self, store, order):
        """[(block_id, its CURRENT record's bytes)] in `order`.

        A reorder is a merge of EVERYTHING, so it goes through the same apply and the same proof:
        a block with unmerged deltas is folded here exactly as `merge` folds it, and a block with
        none re-encodes to the bytes it already had. The seam a dropping arm replaces."""
        rows = []
        for bid in order:
            base, folded = self._folded(store, bid)
            rows.append(
                (
                    bid,
                    record_from_edges(bid, base.n_units_in_block, base.block_size, folded).encode(),
                )
            )
        return rows

    def reorder(self, store, order):
        """Write every block's CURRENT record into ONE new sealed segment in `order`, proving each
        relocated record before moving its pointer. Old segments stay byte for byte.

        `order` must be a PERMUTATION of the store's block ids and is refused otherwise -- an
        order that dropped or repeated a block would place a record nobody can address, and
        refusing at the call is cheaper than discovering it at the next page-in."""
        _refuse_on_label()
        ids = block_ids(store)
        order = [int(b) for b in order]
        if sorted(order) != sorted(ids):
            raise ValueError(
                f"reorder needs a PERMUTATION of the store's {len(ids)} block ids; got "
                f"{len(order)} entries covering {len(set(order))} distinct ids"
            )
        started = time.perf_counter()
        rows = self._reorder_payloads(store, order)
        payloads = [payload for _bid, payload in rows if payload is not None]
        if not payloads:
            raise ValueError("a reorder that writes nothing is not a reorder")
        container, seqs = store.seal_and_locate(payloads)
        # A BLOCK'S NEW ADDRESS IS ITS POSITION IN `order`, which is what makes this a reorder. For
        # a pass that writes one record per position this is exactly `seqs[position]`, because
        # `seal_and_locate` returns a contiguous run; a constructed arm that writes FEWER records
        # than it claims positions for shifts every pointer past the gap, and the per-record
        # relocation proof below is what has to catch that rather than the arithmetic.
        first = seqs[0]
        moved: list[int] = []
        refusals, bytes_appended = 0, sum(len(p) for p in payloads)
        refusal_last = ""
        for position, (bid, _payload) in enumerate(rows):
            seq = first + position
            try:
                built = rows[position][1]
                if built is None or not self._prove(
                    store, bid, container, seq, built, store.generation(bid)
                ):
                    refusals += 1
                    refusal_last = f"block {bid}: the relocation proof failed; pointer untouched"
                    continue
                store.adopt_merged(bid, container, seq)
                moved.append(bid)
            except Exception as exc:  # noqa: BLE001 - same nightly-duty shape as merge
                refusals += 1
                refusal_last = f"block {bid}: {exc!r}"[:200]
        return {
            "merged": len(moved),
            "deferred": 0,
            "refusals": refusals,
            "bytes_appended": bytes_appended,
            "seconds": time.perf_counter() - started,
            "stale_resident": 0,
            "merged_ids": tuple(moved),
            "deferred_ids": (),
            "refusal_last": refusal_last,
        }


# ---- the self-check's fixtures and arms ---------------------------------------------------------
_FRAC = 16
_BLOCK = 128
_N_BLOCKS = 8
_N = _BLOCK * _N_BLOCKS
_P = 4
_MAX_DELAY = 3
_W = 1.5  # exactly representable in <f4, and one edge alone clears thr = 1.0
_W_SUB = 0.25
_R8_BLOCKS = 16
_R8_N = _BLOCK * _R8_BLOCKS
_CO_FETCH_K = 4
#: 2.4's pre-registered control margin, as an INTEGER difference in reads per round: the
#: structured stream must reach exactly 1 from k, and the random stream's reduction must fall
#: short of that by at least this much. Stated for k = 4 before the run.
_CONTROL_MARGIN = _CO_FETCH_K - 2


def _edges(n_blocks=_N_BLOCKS):
    """Block b drives block b+1 at delays 1 and 2. Every (pre, post, delay) triple is UNIQUE,
    which is a property the delta tier needs rather than a fixture convenience: a 0x0D entry
    addresses an edge by that triple."""
    pre, post, weight, delay = [], [], [], []
    for b in range(n_blocks - 1):
        for i in range(_BLOCK):
            unit = b * _BLOCK + i
            for d in (1, 2):
                pre.append(unit)
                post.append((b + 1) * _BLOCK + ((i * 7 + d * 13) % _BLOCK))
                weight.append(_W)
                delay.append(d)
    return pre, post, weight, delay


_SPEC = (
    (0, 0, ((0, 1 * _BLOCK + 5, _W, 3, 0),)),  # a NEW edge: the train-visible one
    (1, 0, ((0, 2 * _BLOCK + 13, _W_SUB, 1, 0),)),  # a weight CHANGE on an existing triple
    (2, 0, ((0, 3 * _BLOCK + 13, 0.0, 1, 1),)),  # a REMOVE
    (3, 0, ((0, 4 * _BLOCK + 13, _W, 1, 0),)),  # already at this weight: compare-dropped
)


def _deltas(spec):
    return [
        BlockDeltaRecord(
            block_id=bid,
            n_units_in_block=_BLOCK,
            base_generation=gen,
            pre_local=[e[0] for e in entries],
            post=[e[1] for e in entries],
            weight=[e[2] for e in entries],
            delay=[e[3] for e in entries],
            op=[e[4] for e in entries],
        )
        for bid, gen, entries in spec
    ]


def _twin_edges(edges, spec):
    """The TWIN's post-delta edge set, walked with a plain dict and NO compactor code in it --
    the control that is not the method (2.5)."""
    table, order = {}, []
    for p, q, w, d in zip(*edges, strict=True):
        key = (int(p), int(q), int(d))
        table[key] = float(w)
        order.append(key)
    for bid, _gen, entries in spec:
        for pre_local, q, w, d, op in entries:
            key = (bid * _BLOCK + pre_local, q, d)
            if op == 1:
                if key in table:
                    del table[key]
                    order = [k for k in order if k != key]
            else:
                if key not in table:
                    order.append(key)
                table[key] = w
    return (
        [k[0] for k in order],
        [k[1] for k in order],
        [table[k] for k in order],
        [k[2] for k in order],
    )


def _drive(units, level=2.0, n_units=_N):
    ext = np.zeros(n_units, dtype=np.int64)
    ext[list(units)] = int(level * (1 << _FRAC))
    return ext


def _twin(edges, n_units=_N):
    state = KernelState([("net", n_units)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses(*edges, n_units, _P, _FRAC, _MAX_DELAY)
    return KernelTick(state, syn, decay=0.5, v_reset=0.0, refrac_ticks=0)


def _store(root, name, edges, n_units=_N, store_cls=UnitBlockStore):
    lake = Path(root) / name
    lake.mkdir(parents=True, exist_ok=True)
    store = store_cls(lake)
    return store, store.build(*edges, n_units, _BLOCK)


def _paged(store, blocks, n_units=_N, **budget):
    state = KernelState([("net", n_units)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses.empty(n_units, _P, _FRAC, _MAX_DELAY)
    kt = KernelTick(
        state,
        syn,
        decay=0.5,
        v_reset=0.0,
        refrac_ticks=0,
        resident=np.zeros(n_units, dtype=bool),
    )
    return kt, NeuronPager(kt, store, blocks=blocks, **budget)


def _run(runner, drives, ticks):
    return [runner.tick(drives.get(t)).tobytes() for t in range(ticks)]


def _per_file_sha(root):
    from snn_data_lake.substrate.lake_segments import SEGMENT_DIR

    return {
        seg.name: hashlib.sha256(seg.read_bytes()).hexdigest()
        for seg in sorted((Path(root) / SEGMENT_DIR).glob("seg-*.lake"))
    }


def _logical(store, n_blocks=_N_BLOCKS):
    got = []
    for _bid, record in sorted(store.fetch(range(n_blocks)).items()):
        pre = record.pre_global()
        for j in range(len(record)):
            got.append(
                (int(pre[j]), int(record.post[j]), float(record.weight[j]), int(record.delay[j]))
            )
    return sorted(got)


class _DroppingCompactor(LakeCompactor):
    """S-R6's RED: the fold skips the first delta of every block. Its relocation proof re-applies
    from the lake with the incumbent apply, which is what makes the drop visible."""

    def _apply(self, base_edges, deltas, *, generation):
        return apply_deltas(base_edges, list(deltas)[1:], generation=generation)


class _DroppingNoProofCompactor(_DroppingCompactor):
    """2.6's per-item NEUTER: the same dropping fold with the proof off. It must read refusals 0
    and adopt a record that is not base + deltas -- the proof is what makes the red reachable."""

    def _prove(self, *args, **kwargs):
        return True


class _BudgetBlindCompactor(LakeCompactor):
    """S-R7's RED: folds the whole dirty set whatever `budget_blocks` says, so nothing defers."""

    def _budgeted(self, dirty, budget_blocks):
        return list(dirty), []


class _DroppingReorderCompactor(LakeCompactor):
    """S-R8's RED: skips one block's record while still claiming every block's position."""

    skip = 0

    def _reorder_payloads(self, store, order):
        return [
            (bid, None if bid == self.skip else payload)
            for bid, payload in super()._reorder_payloads(store, order)
        ]


class _DroppingReorderNoProofCompactor(_DroppingReorderCompactor):
    """...and the same drop with the proof off, which is where the damage becomes observable."""

    def _prove(self, *args, **kwargs):
        return True


class _DroppingFetchStore(UnitBlockStore):
    """2.6's per-item NEUTER on the READ side: the fresh-tier apply ignores the first delta of
    every block, so a paged run must diverge from the twin."""

    def _apply_fetch(self, base_edges, deltas, *, generation):
        return apply_deltas(base_edges, list(deltas)[1:], generation=generation)


def _dirty_spec(n_dirty):
    return tuple(
        (bid, 0, ((1, (bid + 1) * _BLOCK + 17 + bid, _W_SUB, 3, 0),)) for bid in range(n_dirty)
    )


def _structured_sets(ticks, n_blocks=_R8_BLOCKS):
    step = n_blocks // _CO_FETCH_K
    return [tuple(sorted(i * step for i in range(_CO_FETCH_K))) for _ in range(ticks)]


def _random_sets(ticks, n_blocks=_R8_BLOCKS, seed=17):
    rng = np.random.default_rng(seed)
    return [
        tuple(sorted(int(b) for b in rng.choice(n_blocks, size=_CO_FETCH_K, replace=False)))
        for _ in range(ticks)
    ]


def _reads_per_round(store, due_sets):
    start = len(store.reads_per_round)
    for group in due_sets:
        store.fetch(group)
    got = store.reads_per_round[start:]
    return sum(got) / len(got)


def _derive_budget(store):
    """budget_blocks from a MEASURED per-block fold cost against a stated share of a subjective
    night at the RECORDED SLEEP-TICK p50 (tools/perf_baseline.json ticks.sleep.p50_ms -- the
    reference named the way `neuron_pager._derive_rounds` names the wake p50). Printed, never
    picked."""
    ids = block_ids(store)
    store.append_deltas(
        _deltas([(bid, store.generation(bid), ((2, 0, _W_SUB, 3, 0),)) for bid in ids[:-1]])
    )
    started = time.perf_counter()
    result = LakeCompactor().merge(store, budget_blocks=len(ids))
    elapsed = time.perf_counter() - started
    per_block = elapsed / max(result["merged"], 1)
    baseline = json.loads(PERF_BASELINE.read_text())
    sleep_p50 = baseline["ticks"]["sleep"]["p50_ms"] / 1000.0
    allowance = SELFCHECK_NIGHT_SHARE * SELFCHECK_NIGHT_TICKS * sleep_p50
    return {
        "per_block_s": per_block,
        "merged": result["merged"],
        "sleep_p50_s": sleep_p50,
        "allowance_s": allowance,
        "budget_blocks": max(1, int(allowance // max(per_block, 1e-9))),
    }


def _fail(message):
    print(f"lake_compactor self-check FAILED: {message}")
    return 1


def main() -> int:
    from snn_data_lake.substrate.home import throwaway_home

    started = time.perf_counter()
    print("== T067 the growth delta tier, the sleep-night merge, the locality reorder ==")
    with throwaway_home() as scratch:
        root = Path(scratch)
        edges = _edges()
        twin_edges = _twin_edges(edges, _SPEC)

        # ---- the budget, DERIVED in this same command -------------------------------------
        cal_store, _cal_blocks = _store(root, "cal", edges)
        derived = _derive_budget(cal_store)
        cal_store.close()
        print(
            f"derivation: {derived['merged']} blocks folded in "
            f"{derived['per_block_s'] * 1e3:.3f} ms each (measured, this host); a subjective "
            f"night is ~{SELFCHECK_NIGHT_TICKS} sleep ticks at the RECORDED sleep-tick p50 "
            f"{derived['sleep_p50_s'] * 1e3:.2f} ms, and this check's stated share of one night "
            f"is {SELFCHECK_NIGHT_SHARE:.0%} = {derived['allowance_s'] * 1e3:.1f} ms -> "
            f"budget_blocks={derived['budget_blocks']}. The OVERLOAD arm below does NOT run on "
            "that budget and cannot: a budget derived to be sufficient never defers, which is "
            "what makes the deferral criterion falsifiable only under a constructed one"
        )

        # ---- S-R6: the fresh tier, the merge, and the dropping arm -------------------------
        store, blocks = _store(root, "fresh", edges)
        wrote = store.append_deltas(_deltas(_SPEC))
        if wrote != 3:
            return _fail(f"append_deltas wrote {wrote} of 4 records; one must be compare-dropped")
        drives = {0: _drive([0, 1, 2])}
        twin_trains = _run(_twin(twin_edges), drives, 20)
        base_trains = _run(_twin(edges), drives, 20)
        if base_trains == twin_trains:
            return _fail("the delta set changes no train -- the identity below would be vacuous")
        _kt, pager = _paged(
            store,
            blocks,
            capacity_blocks=4,
            blocks_per_round=8,
            max_rounds=4,
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        fresh_trains = _run(pager, drives, 20)
        if fresh_trains != twin_trains:
            return _fail("the fresh-tier paged run diverged from the post-delta resident twin")
        if not pager.page_ins:
            return _fail("the fresh-tier arm never paged -- an identity that never paged is none")
        before_files = _per_file_sha(store.root)
        want_edges = _logical(store)
        result = LakeCompactor().merge(store, budget_blocks=_N_BLOCKS)
        after_files = _per_file_sha(store.root)
        unmerged = [bid for bid in range(_N_BLOCKS) if store.n_unmerged(bid)]
        _kt2, pager2 = _paged(
            store,
            blocks,
            capacity_blocks=4,
            blocks_per_round=8,
            max_rounds=4,
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        merged_trains = _run(pager2, drives, 20)
        moved = [n for n, d in before_files.items() if after_files.get(n) != d]
        if result["refusals"] or result["merged"] != 3 or result["deferred"]:
            return _fail(f"the merge did not fold the dirty set cleanly: {result}")
        if unmerged:
            return _fail(f"blocks {unmerged} kept an unmerged delta after the fold")
        if merged_trains != twin_trains:
            return _fail("the merged paged run diverged from the same resident twin")
        if moved:
            return _fail(f"pre-existing segment file(s) {moved} moved -- a merge destroys nothing")
        if _logical(store) != want_edges:
            return _fail("the logical record set through the index changed across the merge")
        print(
            f"S-R6: {wrote} of 4 deltas written (one compare-dropped at the f4 record grain); the "
            f"fresh-tier paged run is byte-identical to a twin compiled directly on the post-delta "
            f"edge set over 20 ticks (page_ins={pager.page_ins}) and so is the MERGED run "
            f"(merged={result['merged']} refusals={result['refusals']} "
            f"bytes_appended={result['bytes_appended']} seconds={result['seconds']:.3f} "
            f"stale_resident={result['stale_resident']}); every one of the "
            f"{len(before_files)} pre-existing segment files is byte-identical; the logical record "
            "set through the index is unchanged. The PRE-delta twin differs, so none of that is "
            "vacuous"
        )

        stale = _deltas([(0, 0, ((1, 1 * _BLOCK + 9, _W, 1, 0),))])
        try:
            store.append_deltas(stale)
            return _fail("a delta whose base_generation is stale was accepted")
        except ValueError:
            pass
        if (
            store.append_deltas(
                _deltas([(0, store.generation(0), ((1, 1 * _BLOCK + 9, _W, 1, 0),))])
            )
            != 1
        ):
            return _fail("the current-generation positive control did not land")
        print(
            f"  the generation refusal fires on a stale base and its positive control at the "
            f"current generation ({store.generation(0)}) lands"
        )

        drop_store, _drop_blocks = _store(root, "dropmerge", edges)
        drop_store.append_deltas(_deltas(_SPEC))
        pointers = {bid: drop_store.block_pointer(bid) for bid in range(_N_BLOCKS)}
        generations = {bid: drop_store.generation(bid) for bid in range(_N_BLOCKS)}
        dropped = _DroppingCompactor().merge(drop_store, budget_blocks=_N_BLOCKS)
        untouched = all(
            drop_store.block_pointer(bid) == pointers[bid]
            and drop_store.generation(bid) == generations[bid]
            for bid in range(_N_BLOCKS)
        )
        if not (dropped["refusals"] >= 1 and dropped["merged"] == 0 and untouched):
            return _fail(
                f"the DROPPING merge was not refused with its pointers untouched: {dropped}"
            )

        neuter_store, _neuter_blocks = _store(root, "noproof", edges)
        neuter_store.append_deltas(_deltas(_SPEC))
        neuter_want = _logical(neuter_store)
        neutered = _DroppingNoProofCompactor().merge(neuter_store, budget_blocks=_N_BLOCKS)
        neuter_got = _logical(neuter_store)
        if neutered["refusals"] or neutered["merged"] < 1:
            return _fail(f"the PROOF-NEUTERED arm refused something: {neutered}")
        if neuter_got == neuter_want:
            return _fail(
                "the proof-neutered dropping fold adopted a correct record -- no attribution"
            )
        print(
            f"S-R6 red, and the attribution beside it: the DROPPING fold reads "
            f"refusals={dropped['refusals']} merged={dropped['merged']} with every pointer and "
            f"generation UNTOUCHED, while the SAME fold with the PROOF NEUTERED reads "
            f"refusals={neutered['refusals']} merged={neutered['merged']} and adopts a record set "
            f"that differs from base + deltas by "
            f"{len(set(neuter_want) ^ set(neuter_got))} edge(s). THE PROOF IS WHAT MAKES THE RED "
            f"REACHABLE: {dropped['refusal_last']}"
        )

        read_store, read_blocks = _store(root, "dropfetch", edges, store_cls=_DroppingFetchStore)
        read_store.append_deltas(_deltas(_SPEC))
        _rkt, read_pager = _paged(
            read_store,
            read_blocks,
            capacity_blocks=8,
            blocks_per_round=8,
            max_rounds=4,
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        if _run(read_pager, drives, 20) == twin_trains:
            return _fail("the READ-side dropping neuter matched the twin -- it is not an arm")
        print(
            "  the per-item neuter on the READ side (the fresh-tier apply ignores one delta) makes "
            "the paged run DIFFER from the twin, so the identity above is attributable to the apply"
        )

        # ---- S-R7: the budget, its calm control, and the budget-blind arm ------------------
        ov_store, _ov_blocks = _store(root, "overload", edges)
        ov_store.append_deltas(_deltas(_dirty_spec(6)))
        dirty = {bid for bid in range(_N_BLOCKS) if ov_store.n_unmerged(bid)}
        first = LakeCompactor().merge(ov_store, budget_blocks=3)
        second = LakeCompactor().merge(ov_store, budget_blocks=3)
        union = set(first["merged_ids"]) | set(second["merged_ids"])
        calm_store, _calm_blocks = _store(root, "calm", edges)
        calm_store.append_deltas(_deltas(_dirty_spec(2)))
        calm_dirty = {bid for bid in range(_N_BLOCKS) if calm_store.n_unmerged(bid)}
        calm = LakeCompactor().merge(calm_store, budget_blocks=3)
        blind_store, _blind_blocks = _store(root, "blind", edges)
        blind_store.append_deltas(_deltas(_dirty_spec(6)))
        blind = _BudgetBlindCompactor().merge(blind_store, budget_blocks=3)
        if not (first["deferred"] > 0 and second["deferred"] == 0 and union == dirty):
            return _fail(
                f"the constructed overload did not defer and then finish: {first} {second}"
            )
        if not (calm["deferred"] == 0 and set(calm["merged_ids"]) == calm_dirty):
            return _fail(f"the calm control is not a control: {calm}")
        if blind["deferred"]:
            return _fail(f"the BUDGET-BLIND arm deferred; it cannot be the red: {blind}")
        stale_store, _stale_blocks = _store(root, "resident", edges)
        stale_store.append_deltas(_deltas(_dirty_spec(3)))
        stale_result = LakeCompactor().merge(
            stale_store, budget_blocks=8, resident_blocks=frozenset({0, 1})
        )
        if stale_result["stale_resident"] != 2:
            return _fail(f"stale_resident read {stale_result['stale_resident']}, expected 2")
        print(
            f"S-R7: {len(dirty)} dirty blocks against budget_blocks=3 defer "
            f"{first['deferred']} on the first call and {second['deferred']} on the second, and "
            f"the union of both calls' merged sets IS the dirty set ({sorted(union)}); the calm "
            f"control at {len(calm_dirty)} <= 3 defers {calm['deferred']} and folds all of them; "
            f"the BUDGET-BLIND arm reads deferred={blind['deferred']} merged={blind['merged']} on "
            f"the identical overload. stale_resident counts {stale_result['stale_resident']} of "
            "the 2 merged blocks a caller reported resident -- their compiled bundles are the OLD "
            "record's until eviction (2.6(g))"
        )
        refused_budget = 0
        for bad in (0, -1, 2.0, True, "3", None):
            try:
                LakeCompactor().merge(calm_store, budget_blocks=bad)
            except ValueError:
                refused_budget += 1
        try:
            LakeCompactor().merge(calm_store)
        except TypeError:
            refused_budget += 1
        if refused_budget != 7:
            return _fail(f"only {refused_budget} of 7 bad budgets were refused")

        # ---- S-R8: the locality reorder, its null, and its red -----------------------------
        r8_edges = _edges(_R8_BLOCKS)
        s_store, s_blocks = _store(root, "structured", r8_edges, n_units=_R8_N)
        s_sets = _structured_sets(20)
        s_before = _reads_per_round(s_store, s_sets)
        s_files = _per_file_sha(s_store.root)
        s_edges_before = _logical(s_store, _R8_BLOCKS)
        order = co_fetch_order(s_sets, _R8_BLOCKS)
        s_result = LakeCompactor().reorder(s_store, order)
        s_after = _reads_per_round(s_store, s_sets)
        s_moved = [n for n, d in s_files.items() if _per_file_sha(s_store.root).get(n) != d]
        if sorted(order) != list(range(_R8_BLOCKS)):
            return _fail("co_fetch_order did not return a permutation")
        if s_result["refusals"]:
            return _fail(f"the reorder refused a block: {s_result}")
        if _logical(s_store, _R8_BLOCKS) != s_edges_before:
            return _fail("the reorder changed a record")
        if s_moved:
            return _fail(f"pre-existing segment file(s) {s_moved} moved across the reorder")
        if s_store.bytes_fetched != s_store.bytes_used:
            return _fail("bytes fetched != bytes used -- the round over-read")
        s_store.assert_read_accounting()

        r_store, _r_blocks = _store(root, "random", r8_edges, n_units=_R8_N)
        r_sets = _random_sets(20)
        r_before = _reads_per_round(r_store, r_sets)
        LakeCompactor().reorder(r_store, co_fetch_order(r_sets, _R8_BLOCKS))
        r_after = _reads_per_round(r_store, r_sets)

        # 2.6's per-item NEUTER for the reorder: the ORDER FUNCTION replaced by IDENTITY, through
        # the incumbent compactor -- the rewrite happens, only the placement does not, so a
        # reduction that survived here would belong to the rewrite and not to the order.
        i_store, _i_blocks = _store(root, "identity-order", r8_edges, n_units=_R8_N)
        i_before = _reads_per_round(i_store, s_sets)
        LakeCompactor().reorder(i_store, list(range(_R8_BLOCKS)))
        i_after = _reads_per_round(i_store, s_sets)

        margin = (s_before - s_after) - (r_before - r_after)
        if s_after != 1.0 or s_before != float(_CO_FETCH_K):
            return _fail(
                f"the structured stream read {s_before} -> {s_after}, not {_CO_FETCH_K} -> 1"
            )
        # 2.5 pre-registers the null's expected reading as "random reorder >= 3 -> >= 3 (WITHIN
        # ONE)", and the within-one clause is the bar rather than a hard 3.0: a greedy order
        # fitted to noise still puts SOME co-occurring pair adjacent by chance, so the random
        # control moves a little and pre-registering that it does not would have been wrong. The
        # load-bearing criterion is 2.4's INTEGER MARGIN between the two reductions, below.
        if r_after < r_before - 1.0:
            return _fail(
                f"the random control's reads per round fell {r_before:.2f} -> {r_after:.2f}, more "
                "than the one read 2.5 allows a noise-fitted order"
            )
        if margin < _CONTROL_MARGIN:
            return _fail(f"the structured-minus-random margin is {margin}, under {_CONTROL_MARGIN}")
        if i_after != i_before:
            return _fail(f"the IDENTITY order moved reads per round {i_before} -> {i_after}")
        print(
            f"S-R8: a co-fetch stream of {_CO_FETCH_K} index-distant blocks costs {s_before:.2f} "
            f"reads per round before the reorder and {s_after:.2f} after "
            f"(bytes-fetched/bytes-used 1.0 both times, by construction -- no over-read this row); "
            f"every record decodes identically, every pre-existing segment file is byte-identical, "
            f"and {s_result['merged']} of {_R8_BLOCKS} blocks were relocated with "
            f"{s_result['refusals']} refusals. THE NULL, in the same command: a RANDOM-fetch "
            f"stream reorders {r_before:.2f} -> {r_after:.2f}, so the structured reduction "
            f"{s_before - s_after:.2f} exceeds the random reduction {r_before - r_after:.2f} by "
            f"{margin:.2f} against a pre-registered margin of {_CONTROL_MARGIN}. THE PER-ITEM "
            f"NEUTER: the order replaced by IDENTITY reads {i_before:.2f} -> {i_after:.2f}, i.e. "
            "the reduction is the ORDER's and not the rewrite's"
        )

        twin_r8 = _run(_twin(r8_edges, n_units=_R8_N), {0: _drive([0, 1, 2], n_units=_R8_N)}, 24)
        id_store, id_blocks = _store(root, "reorder-identity", r8_edges, n_units=_R8_N)
        _ikt, id_pager = _paged(
            id_store,
            id_blocks,
            n_units=_R8_N,
            capacity_blocks=6,
            blocks_per_round=8,
            max_rounds=4,
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        id_before = _run(id_pager, {0: _drive([0, 1, 2], n_units=_R8_N)}, 24)
        LakeCompactor().reorder(id_store, co_fetch_order(_structured_sets(8), _R8_BLOCKS))
        _ikt2, id_pager2 = _paged(
            id_store,
            id_blocks,
            n_units=_R8_N,
            capacity_blocks=6,
            blocks_per_round=8,
            max_rounds=4,
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        id_after = _run(id_pager2, {0: _drive([0, 1, 2], n_units=_R8_N)}, 24)
        if not (id_before == twin_r8 == id_after):
            return _fail("a paged run stopped matching its resident twin across the reorder")

        d_store, _d_blocks = _store(root, "dropreorder", r8_edges, n_units=_R8_N)
        d_before = _logical(d_store, _R8_BLOCKS)
        dropper = _DroppingReorderCompactor()
        dropper.skip = order[1]
        d_result = dropper.reorder(d_store, order)
        caught = d_result["refusals"] >= 1 and _logical(d_store, _R8_BLOCKS) == d_before
        b_store, _b_blocks = _store(root, "dropreorder-noproof", r8_edges, n_units=_R8_N)
        blind_dropper = _DroppingReorderNoProofCompactor()
        blind_dropper.skip = order[1]
        blind_dropper.reorder(b_store, order)
        unreadable = False
        try:
            _logical(b_store, _R8_BLOCKS)
        except (KeyError, ValueError) as exc:
            unreadable = True
            unreadable_why = f"{type(exc).__name__}: {str(exc)[:90]}"
        refused_order = 0
        for bad in (
            list(range(_R8_BLOCKS - 1)),
            [0] + list(range(_R8_BLOCKS - 1)),
            list(range(1, _R8_BLOCKS + 1)),
        ):
            try:
                LakeCompactor().reorder(s_store, bad)
            except ValueError:
                refused_order += 1
        if not caught:
            return _fail(f"the DROPPING reorder was not refused by the proof: {d_result}")
        if not unreadable:
            return _fail("the proof-neutered dropping reorder left every block readable")
        if refused_order != 3:
            return _fail(f"only {refused_order} of 3 non-permutation orders were refused")
        print(
            f"S-R8 red: a DROPPING reorder (one block's record skipped while every position is "
            f"still claimed) is caught by the per-record relocation proof -- refusals="
            f"{d_result['refusals']} of {_R8_BLOCKS}, merged={d_result['merged']}, every record "
            f"still readable and identical. WITH THE PROOF NEUTERED the same drop leaves a block "
            f"UNREADABLE, refused rather than read as an empty edge set ({unreadable_why}). A "
            "LOSSY order -- short, repeating or out of range -- is refused at the call, 3 of 3"
        )

        print(
            "counters: "
            + ", ".join(
                f"{name} rounds={s.submissions} reads={sum(s.reads_per_round)} "
                f"records={s.records_read} blocks={s.blocks_read} bytes={s.bytes_fetched}"
                for name, s in (
                    ("fresh", store),
                    ("structured", s_store),
                    ("random", r_store),
                )
            )
        )
        print(
            "DISCLOSED, every run: (a) no live producer of kernel-tier deltas exists until "
            "S6/T062 -- this check, its arms and the tests are the writer's callers today; (b) the "
            "duty is NOT wired into Brain._enter_sleep, because the live lake has no segment tier "
            "and a guarded import would be a hollow reachability; (c) bytes-fetched/bytes-used is "
            "1.0 BY CONSTRUCTION without over-read and reads per round is the quantity this row "
            "moves; (d) NOTHING reclaims a superseded record -- the label and the proof land, the "
            "reclaim is a separate row; (e) the append budget is a FLOOD assert, so 0x0D's bound "
            "is a flood bound and not a slope bound; (f) the order is only as good as the run it "
            "watched, which the random control measures; (g) a merged block that is resident keeps "
            "its old bundles until eviction, counted as stale_resident; (h) a block whose proof "
            "FAILS leaves an unadopted 0x0C record, and since the ADOPTION FENCE landed no "
            "rebuild adopts one -- a 0x0C record is adopted only where a 0x0D ADOPT record names "
            "it, generation 0 included -- but 0x0D's reclaimable STAYS False on a MEASURED "
            "divergence of its own: a reclaim of a folded delta prefix must re-containerise the "
            "survivors, which invalidates the ADOPT record's stored container and leaves its "
            "folded_through watermark above the deltas a rebuild can see"
        )
        for closing in (
            store,
            drop_store,
            neuter_store,
            read_store,
            ov_store,
            calm_store,
            blind_store,
            stale_store,
            s_store,
            r_store,
            i_store,
            id_store,
            d_store,
            b_store,
        ):
            closing.close()
    print(
        f"lake_compactor self-check OK in {time.perf_counter() - started:.2f} s: S-R6, S-R7 and "
        "S-R8 with their broken arms, their controls and their per-item neuters in this one command"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

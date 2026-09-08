"""T067 / Phase 3 S4 -- the growth delta tier, the sleep-phase merge, the locality reorder.

Section 2.3 of specs/022-unified-brain-arc/evidence/s4-lake-compactor.md, written against the API
section 2.1 names and RUN RED on the tree that has no `substrate/lake_compactor.py`, no
`_T_BLOCKDELTA`, no `BlockDeltaRecord` and no `read_records`. Each criterion carries the
constructed BROKEN ARM that makes it a measurement:

  S-R6  a merge (or the fresh-tier read) loses or mis-applies a delta -- the DROPPING merge (a
        LakeCompactor subclass whose apply skips one delta) must fail its relocation proof with
        the pointer untouched, and a store whose FETCH-side apply drops the same delta must
        produce a paged run that DIFFERS from the resident twin.
  S-R7  the merge budget never trips on constructed overload -- the BUDGET-BLIND merge (a
        subclass that ignores `budget_blocks`) must read `deferred == 0` on the identical
        overload the incumbent defers on.
  S-R8  a reorder changes or loses a record, or does not reduce reads on a co-fetched stream, or
        reduces them on a stream with no co-fetch structure -- the RANDOM-fetch control is the
        null, the DROPPING reorder is the red, and a non-permutation order is refused at the call.

Every fixture weight is a multiple of 0.25 and so exactly representable in `<f4`, the dtype both
record classes store weights in. EVERY (pre, post, delay) TRIPLE IN A BLOCK IS UNIQUE, and that is
a property of the delta tier and not of the fixture: a 0x0D entry ADDRESSES an edge by that
triple, so a base record carrying the same triple twice is not addressable and must be refused
rather than silently collapsed.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from snn_data_lake.substrate import home as home_mod

# IMPORTED AS A MODULE, not `from ... import LakeCompactor`, so the import block's ORDER does not
# move between the commit that runs these RED (no substrate/lake_compactor.py on the tree, which
# ruff's isort would classify as third-party) and the commit that makes them green.
from snn_data_lake.substrate import lake_compactor as compactor_module
from snn_data_lake.substrate.home import _T_BLOCKDELTA, _T_UNITBLOCK, observe_append
from snn_data_lake.substrate.kernel_state import KernelState
from snn_data_lake.substrate.kernel_tick import KernelTick, Synapses
from snn_data_lake.substrate.lake_engine import (
    AppendBudget,
    BlockDeltaRecord,
    UnitBlockRecord,
    apply_deltas,
    edge_table,
    record_from_edges,
    registry_consistent,
    retention_of,
)
from snn_data_lake.substrate.lake_migrate import TAG_DISPOSITIONS
from snn_data_lake.substrate.lake_rebuild_gate import build_index
from snn_data_lake.substrate.lake_segments import SEGMENT_DIR, read_records, read_segment
from snn_data_lake.substrate.neuron_pager import NeuronPager, UnitBlockStore

FRAC = 16
BLOCK = 128
N_BLOCKS = 8
N = BLOCK * N_BLOCKS
P_PARTS = 4
MAX_DELAY = 3
W_FIRE = 1.5  # one edge alone clears thr = 1.0
W_SUB = 0.25  # one edge alone stays under it
TAG_BLOCK = _T_UNITBLOCK[0]
TAG_DELTA = _T_BLOCKDELTA[0]

#: S-R8's pre-registered control margin (2.4): the structured stream must reach exactly 1 read per
#: round from k, and the random stream must not fall by more than (k - 1) - MARGIN. Stated for
#: k = 4 as an integer difference in reads per round, never a ratio.
CO_FETCH_K = 4
CONTROL_MARGIN = CO_FETCH_K - 2


# ---- fixtures --------------------------------------------------------------------------------
def _base_edges(n_blocks=N_BLOCKS):
    """Block b drives block b+1 with two edges per unit at delays 1 and 2. Every (pre, post,
    delay) triple is unique because a pre unit's two edges differ in delay."""
    pre, post, weight, delay = [], [], [], []
    for b in range(n_blocks - 1):
        for i in range(BLOCK):
            unit = b * BLOCK + i
            for d in (1, 2):
                pre.append(unit)
                post.append((b + 1) * BLOCK + ((i * 7 + d * 13) % BLOCK))
                weight.append(W_FIRE)
                delay.append(d)
    return pre, post, weight, delay


def _edge_tuples(edges):
    return [(int(p), int(q), float(w), int(d)) for p, q, w, d in zip(*edges, strict=True)]


#: (block_id, base_generation, [(pre_local, post, weight, delay, op)]) -- the S-R6 delta set:
#: a NEW edge that changes the trains, a WEIGHT CHANGE that does not, a REMOVE, and a
#: compare-drop candidate (an edge already at exactly that weight).
def _delta_spec():
    return [
        (0, 0, [(0, 1 * BLOCK + 5, W_FIRE, 3, 0)]),  # NEW: unit 0 -> 133, the train-visible one
        (1, 0, [(0, 2 * BLOCK + 13, W_SUB, 1, 0)]),  # WEIGHT CHANGE on an existing triple
        (2, 0, [(0, 3 * BLOCK + 13, 0.0, 1, 1)]),  # REMOVE an existing triple
        (3, 0, [(0, 4 * BLOCK + 13, W_FIRE, 1, 0)]),  # COMPARE-DROP: already at this weight
    ]


def _delta_records(spec):
    return [
        BlockDeltaRecord(
            block_id=bid,
            n_units_in_block=BLOCK,
            base_generation=gen,
            pre_local=[e[0] for e in entries],
            post=[e[1] for e in entries],
            weight=[e[2] for e in entries],
            delay=[e[3] for e in entries],
            op=[e[4] for e in entries],
        )
        for bid, gen, entries in spec
    ]


def _apply_in_python(edges, spec):
    """The TWIN's post-delta edge set, computed by a plain dict walk with NO compactor code in
    it -- the control that is not the method (2.5)."""
    table, order = {}, []
    for p, q, w, d in _edge_tuples(edges):
        key = (p, q, d)
        assert key not in table, "the fixture must not repeat an edge triple"
        table[key] = w
        order.append(key)
    for bid, _gen, entries in spec:
        for pre_local, q, w, d, op in entries:
            key = (bid * BLOCK + pre_local, q, d)
            if op == 1:
                if key in table:
                    del table[key]
                    order.remove(key)
            else:
                if key not in table:
                    order.append(key)
                table[key] = w
    pre = [k[0] for k in order]
    post = [k[1] for k in order]
    weight = [table[k] for k in order]
    delay = [k[2] for k in order]
    return pre, post, weight, delay


def _drive(units, level=2.0, n_units=N):
    ext = np.zeros(n_units, dtype=np.int64)
    ext[list(units)] = int(level * (1 << FRAC))
    return ext


def _twin(edges, n_units=N, max_delay=MAX_DELAY):
    state = KernelState([("net", n_units)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses(*edges, n_units, P_PARTS, FRAC, max_delay)
    return KernelTick(state, syn, decay=0.5, v_reset=0.0, refrac_ticks=0)


def _store_for(tmp_path, edges, name="lake", n_units=N):
    root = Path(tmp_path) / name
    root.mkdir(parents=True, exist_ok=True)
    store = UnitBlockStore(root)
    blocks = store.build(*edges, n_units, BLOCK)
    return store, blocks


def _paged(store, blocks, *, n_units=N, max_delay=MAX_DELAY, **budget):
    state = KernelState([("net", n_units)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses.empty(n_units, P_PARTS, FRAC, max_delay)
    kt = KernelTick(
        state,
        syn,
        decay=0.5,
        v_reset=0.0,
        refrac_ticks=0,
        resident=np.zeros(n_units, dtype=bool),
    )
    return kt, NeuronPager(kt, store, blocks=blocks, **budget)


def _rings_of(kt):
    return [[list(slot) for slot in ring] for ring in kt._rings]


def _run(runner, drives, ticks):
    """Trains AND ring-slot key lists -- S-R6(i) names both."""
    trains, rings = [], []
    for t in range(ticks):
        trains.append(runner.tick(drives.get(t)).tobytes())
        rings.append(_rings_of(runner if isinstance(runner, KernelTick) else runner.kt))
    return trains, rings


def _per_file_sha(root):
    return {
        seg.name: hashlib.sha256(seg.read_bytes()).hexdigest()
        for seg in sorted((Path(root) / SEGMENT_DIR).glob("seg-*.lake"))
    }


def _logical_edges(store, n_blocks=N_BLOCKS):
    """The logical record set reachable THROUGH THE INDEX, as global edge tuples."""
    got = []
    for _bid, record in sorted(store.fetch(range(n_blocks)).items()):
        pre = record.pre_global()
        for j in range(len(record)):
            got.append(
                (int(pre[j]), int(record.post[j]), float(record.weight[j]), int(record.delay[j]))
            )
    return sorted(got)


def _block_rows_digest(index, n_blocks):
    """The (0x0C, *) and (0x0D, *) rows the store owns, enumerated deterministically and hashed.

    `LakeIndex.logical_dump_digest` hashes EVERY row plus the watermark, and a rebuild writes
    ordinal-keyed rows for foreign tags and a watermark the store deliberately never advances --
    so the rebuild-identity claim is about the BLOCK TIER's rows and this is the comparator for
    it, built from nothing but the public `get`."""
    h = hashlib.sha256()
    for bid in range(n_blocks):
        for tag, key in (
            (TAG_BLOCK, str(bid)),
            (TAG_BLOCK, f"{bid}/gen"),
            (TAG_DELTA, f"{bid}/count"),
        ):
            h.update(repr((tag, key, index.get(tag, key))).encode())
        count = index.get(TAG_DELTA, f"{bid}/count")
        for n in range(0 if count is None else int(count[1])):
            key = f"{bid}/{n}"
            h.update(repr((TAG_DELTA, key, index.get(TAG_DELTA, key))).encode())
    return h.hexdigest()


# ---- broken arms -----------------------------------------------------------------------------
class _DroppingCompactor(compactor_module.LakeCompactor):
    """THE S-R6 RED: its apply SKIPS the first delta of every block, so the record it seals is
    not base + deltas. The relocation proof re-applies from the lake with the INCUMBENT apply,
    which is what makes this reachable."""

    def _apply(self, base_edges, deltas, *, generation):
        return apply_deltas(base_edges, list(deltas)[1:], generation=generation)


class _NoProofCompactor(compactor_module.LakeCompactor):
    """The 2.6 per-item NEUTER: the pointer moves without the relocation proof. Paired with the
    dropping apply it must read refusals == 0 -- the proof is what makes the red reachable."""

    def _prove(self, *args, **kwargs):
        return True


class _DroppingNoProofCompactor(_NoProofCompactor):
    def _apply(self, base_edges, deltas, *, generation):
        return apply_deltas(base_edges, list(deltas)[1:], generation=generation)


class _BudgetBlindCompactor(compactor_module.LakeCompactor):
    """THE S-R7 RED: merges the whole dirty set whatever `budget_blocks` says, so nothing is
    ever deferred."""

    def _budgeted(self, dirty, budget_blocks):
        return list(dirty), []


class _DroppingReorderCompactor(compactor_module.LakeCompactor):
    """THE S-R8 RED: skips one block's record in the new segment while still claiming every
    block's position in `order`, so the pointers past the gap address the wrong record and the
    tail addresses none at all."""

    skip = 0

    def _reorder_payloads(self, store, order):
        rows = super()._reorder_payloads(store, order)
        return [(bid, None if bid == self.skip else payload) for bid, payload in rows]


class _DroppingFetchStore(UnitBlockStore):
    """The 2.6 per-item NEUTER on the READ side: the fresh-tier apply ignores the first delta of
    every block, so a paged run must diverge from the twin."""

    def _apply_fetch(self, base_edges, deltas, *, generation):
        return apply_deltas(base_edges, list(deltas)[1:], generation=generation)


# ---- the registry ------------------------------------------------------------------------------
def test_the_block_delta_class_is_registered_in_both_registries_as_compactable():
    label, reclaimable, reason = retention_of(TAG_DELTA)
    assert label == "COMPACTABLE", "a delta superseded by a merged record is reconstructible"
    assert isinstance(reclaimable, bool)
    assert len(reason) >= 20, "a reason shorter than 20 chars is a label, not a reason"
    assert "proof" in reason.lower(), "the reason must name the reconstruction proof"
    assert TAG_DELTA in TAG_DISPOSITIONS
    assert TAG_DISPOSITIONS[TAG_DELTA][0] == "read-as-is"
    ok, message = registry_consistent()
    assert ok, message


def test_the_delta_record_refuses_a_malformed_payload_whole():
    record = _delta_records(_delta_spec())[0]
    payload = record.encode()
    assert payload[0] == TAG_DELTA, "the tag byte leads, as the lake's dispatch convention needs"
    assert BlockDeltaRecord.decode(payload).encode() == payload
    with pytest.raises(ValueError):
        BlockDeltaRecord.decode(payload[:6])  # short header
    with pytest.raises(ValueError):
        BlockDeltaRecord.decode(bytes([TAG_BLOCK]) + payload[1:])  # foreign tag
    with pytest.raises(ValueError):
        BlockDeltaRecord.decode(payload + b"\x00")  # trailing byte, not a whole edge
    with pytest.raises(ValueError):
        BlockDeltaRecord(
            block_id=0,
            n_units_in_block=BLOCK,
            base_generation=0,
            pre_local=[BLOCK],  # outside its own block
            post=[7],
            weight=[W_FIRE],
            delay=[1],
            op=[0],
        )
    with pytest.raises(ValueError):
        BlockDeltaRecord(
            block_id=0,
            n_units_in_block=BLOCK,
            base_generation=0,
            pre_local=[0],
            post=[7],
            weight=[W_FIRE],
            delay=[1],
            op=[2],  # neither upsert nor remove
        )


def test_an_edge_table_refuses_a_base_record_that_repeats_a_triple():
    record = UnitBlockRecord(
        block_id=0,
        n_units_in_block=BLOCK,
        block_size=BLOCK,
        pre_local=[0, 0],
        post=[7, 7],
        weight=[W_FIRE, W_SUB],
        delay=[1, 1],
    )
    with pytest.raises(ValueError):
        edge_table(record)


# ---- S-R6: the delta tier's identity, its merge, and the dropping arm ---------------------------
def test_the_fresh_tier_paged_run_is_byte_identical_to_the_post_delta_resident_twin(tmp_path):
    edges = _base_edges()
    spec = _delta_spec()
    store, blocks = _store_for(tmp_path, edges, name="fresh")
    store.append_deltas(_delta_records(spec))
    twin_trains, twin_rings = _run(_twin(_apply_in_python(edges, spec)), {0: _drive([0, 1, 2])}, 20)
    _kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=4,
        blocks_per_round=8,
        max_rounds=4,
        max_stale_ticks=1,
        pinned=frozenset({0}),
    )
    paged_trains, paged_rings = _run(pager, {0: _drive([0, 1, 2])}, 20)
    assert pager.page_ins > 0, "an identity that never paged proves nothing"
    assert paged_trains == twin_trains, "the fresh tier lost or mis-applied a delta"
    assert paged_rings == twin_rings, "the ring-slot key lists differ from the twin's"


def test_a_dropping_fresh_tier_apply_diverges_from_the_twin(tmp_path):
    """The read-side neuter (2.6): deltas ignored -> identity to the twin FAILS."""
    edges = _base_edges()
    spec = _delta_spec()
    root = Path(tmp_path) / "dropfetch"
    root.mkdir(parents=True, exist_ok=True)
    store = _DroppingFetchStore(root)
    blocks = store.build(*edges, N, BLOCK)
    store.append_deltas(_delta_records(spec))
    twin_trains, _twin_rings = _run(
        _twin(_apply_in_python(edges, spec)), {0: _drive([0, 1, 2])}, 20
    )
    _kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=8,
        blocks_per_round=8,
        max_rounds=4,
        max_stale_ticks=1,
        pinned=frozenset({0}),
    )
    dropped_trains, _rings = _run(pager, {0: _drive([0, 1, 2])}, 20)
    assert dropped_trains != twin_trains, "a dropped delta did not change the trains"


def test_a_merge_reaches_the_twin_leaves_every_prior_segment_byte_identical_and_refuses_nothing(
    tmp_path,
):
    edges = _base_edges()
    spec = _delta_spec()
    store, blocks = _store_for(tmp_path, edges, name="merged")
    store.append_deltas(_delta_records(spec))
    before = _per_file_sha(store.root)
    result = compactor_module.LakeCompactor().merge(store, budget_blocks=N_BLOCKS)
    assert result["refusals"] == 0, result
    assert result["merged"] >= 3, "the compare-dropped block is not dirty; the other three are"
    assert result["deferred"] == 0
    assert result["bytes_appended"] > 0 and result["seconds"] >= 0.0
    for bid in range(N_BLOCKS):
        assert store.n_unmerged(bid) == 0, f"block {bid} kept an unmerged delta"
    after = _per_file_sha(store.root)
    for name, digest in before.items():
        assert after[name] == digest, f"{name} moved -- a merge destroys nothing (D1)"
    assert set(after) > set(before), "a merge appends a new sealed segment"

    twin_trains, twin_rings = _run(_twin(_apply_in_python(edges, spec)), {0: _drive([0, 1, 2])}, 20)
    _kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=4,
        blocks_per_round=8,
        max_rounds=4,
        max_stale_ticks=1,
        pinned=frozenset({0}),
    )
    merged_trains, merged_rings = _run(pager, {0: _drive([0, 1, 2])}, 20)
    assert merged_trains == twin_trains and merged_rings == twin_rings
    assert _logical_edges(store) == sorted(_edge_tuples(_apply_in_python(edges, spec)))


def test_a_delta_whose_base_generation_is_stale_is_refused_and_never_applied(tmp_path):
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="stalegen")
    store.append_deltas(_delta_records(_delta_spec()))
    compactor_module.LakeCompactor().merge(store, budget_blocks=N_BLOCKS)
    assert store.generation(0) == 1, "a merged block's generation advances"
    stale = _delta_records([(0, 0, [(1, 1 * BLOCK + 9, W_FIRE, 1, 0)])])
    with pytest.raises(ValueError):
        store.append_deltas(stale)
    assert store.n_unmerged(0) == 0, "a refused delta must not land"
    fresh = _delta_records([(0, 1, [(1, 1 * BLOCK + 9, W_FIRE, 1, 0)])])
    assert store.append_deltas(fresh) == 1, "the positive control: the current generation lands"
    assert store.n_unmerged(0) == 1


def test_the_dropping_merge_fails_its_relocation_proof_with_the_pointer_untouched(tmp_path):
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="dropmerge")
    store.append_deltas(_delta_records(_delta_spec()))
    pointers = {bid: store.block_pointer(bid) for bid in range(N_BLOCKS)}
    generations = {bid: store.generation(bid) for bid in range(N_BLOCKS)}
    unmerged = {bid: store.n_unmerged(bid) for bid in range(N_BLOCKS)}
    result = _DroppingCompactor().merge(store, budget_blocks=N_BLOCKS)
    assert result["refusals"] >= 1, "the dropping merge was not caught by the relocation proof"
    assert result["merged"] == 0, "no block may be adopted on a failed proof"
    for bid in range(N_BLOCKS):
        assert store.block_pointer(bid) == pointers[bid], f"block {bid}'s pointer moved"
        assert store.generation(bid) == generations[bid]
        assert store.n_unmerged(bid) == unmerged[bid]
    assert _logical_edges(store) == sorted(_edge_tuples(_apply_in_python(edges, _delta_spec()))), (
        "the fresh tier still reads base + deltas, because nothing was adopted"
    )


def test_the_proof_is_what_makes_the_dropping_merge_reachable(tmp_path):
    """2.6 per-item neuter, printed side by side with the incumbent: with the proof neutered the
    dropping merge is NOT caught (refusals 0) and the wrong record is adopted."""
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="noproof")
    store.append_deltas(_delta_records(_delta_spec()))
    result = _DroppingNoProofCompactor().merge(store, budget_blocks=N_BLOCKS)
    assert result["refusals"] == 0, "with the proof off nothing can refuse"
    assert result["merged"] >= 1
    assert _logical_edges(store) != sorted(_edge_tuples(_apply_in_python(edges, _delta_spec()))), (
        "the neutered arm adopted a record that is not base + deltas, and nothing said so"
    )


# ---- S-R7: the merge budget, its calm control, and the budget-blind arm -------------------------
def _dirty_spec(n_dirty):
    return [(bid, 0, [(1, (bid + 1) * BLOCK + 17 + bid, W_SUB, 3, 0)]) for bid in range(n_dirty)]


def test_a_constructed_overload_defers_and_the_next_call_merges_the_rest(tmp_path):
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="overload")
    store.append_deltas(_delta_records(_dirty_spec(6)))
    dirty = {bid for bid in range(N_BLOCKS) if store.n_unmerged(bid)}
    assert len(dirty) == 6, dirty
    first = compactor_module.LakeCompactor().merge(store, budget_blocks=3)
    assert first["deferred"] > 0, "the constructed overload did not defer"
    assert first["merged"] == 3
    second = compactor_module.LakeCompactor().merge(store, budget_blocks=3)
    assert second["deferred"] == 0
    assert set(first["merged_ids"]) | set(second["merged_ids"]) == dirty
    assert all(store.n_unmerged(bid) == 0 for bid in dirty)


def test_the_calm_control_defers_nothing_and_merges_the_whole_dirty_set(tmp_path):
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="calm")
    store.append_deltas(_delta_records(_dirty_spec(2)))
    dirty = {bid for bid in range(N_BLOCKS) if store.n_unmerged(bid)}
    result = compactor_module.LakeCompactor().merge(store, budget_blocks=3)
    assert result["deferred"] == 0
    assert set(result["merged_ids"]) == dirty and result["merged"] == len(dirty)


def test_a_budget_blind_merge_reads_zero_deferred_on_the_identical_overload(tmp_path):
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="blind")
    store.append_deltas(_delta_records(_dirty_spec(6)))
    result = _BudgetBlindCompactor().merge(store, budget_blocks=3)
    assert result["deferred"] == 0, "the budget-blind arm cannot be the red if it defers"
    assert result["merged"] == 6


def test_a_merged_block_that_is_still_resident_is_counted_as_stale(tmp_path):
    """2.6(g): the pager keeps old bundles until eviction, so a merged resident block is
    reported rather than silently assumed refreshed."""
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="stale")
    store.append_deltas(_delta_records(_dirty_spec(3)))
    result = compactor_module.LakeCompactor().merge(
        store, budget_blocks=8, resident_blocks=frozenset({0, 1})
    )
    assert result["stale_resident"] == 2, result
    quiet = compactor_module.LakeCompactor()
    store.append_deltas(_delta_records([(4, 0, [(1, 5 * BLOCK + 3, W_SUB, 3, 0)])]))
    assert quiet.merge(store, budget_blocks=8)["stale_resident"] == 0


def test_the_merge_refuses_a_budget_that_is_missing_or_not_an_integer(tmp_path):
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="badbudget")
    with pytest.raises(TypeError):
        compactor_module.LakeCompactor().merge(store)
    for bad in (0, -1, 2.0, True, "3", None):
        with pytest.raises(ValueError):
            compactor_module.LakeCompactor().merge(store, budget_blocks=bad)


# ---- S-R8: the locality reorder, its control, and its reds --------------------------------------
R8_BLOCKS = 16
R8_N = BLOCK * R8_BLOCKS


def _r8_store(tmp_path, name):
    edges = _base_edges(R8_BLOCKS)
    root = Path(tmp_path) / name
    root.mkdir(parents=True, exist_ok=True)
    store = UnitBlockStore(root)
    blocks = store.build(*edges, R8_N, BLOCK)
    return store, blocks, edges


def _structured_sets(ticks):
    step = R8_BLOCKS // CO_FETCH_K
    return [tuple(sorted(i * step for i in range(CO_FETCH_K))) for _ in range(ticks)]


def _random_sets(ticks, seed=17):
    rng = np.random.default_rng(seed)
    return [
        tuple(sorted(int(b) for b in rng.choice(R8_BLOCKS, size=CO_FETCH_K, replace=False)))
        for _ in range(ticks)
    ]


def _reads_per_round(store, due_sets):
    start = len(store.reads_per_round)
    for group in due_sets:
        store.fetch(group)
    got = store.reads_per_round[start:]
    return sum(got) / len(got)


def test_the_locality_reorder_takes_a_co_fetched_stream_from_k_reads_a_round_to_one(tmp_path):
    store, blocks, edges = _r8_store(tmp_path, "structured")
    due_sets = _structured_sets(20)
    before_reads = _reads_per_round(store, due_sets)
    assert before_reads == float(CO_FETCH_K), "index-distant groups must cost k reads a round"
    assert store.bytes_fetched == store.bytes_used, "no over-read: the ratio is 1.0 by construction"

    before_files = _per_file_sha(store.root)
    before_edges = _logical_edges(store, R8_BLOCKS)
    order = compactor_module.co_fetch_order(due_sets, R8_BLOCKS)
    assert sorted(order) == list(range(R8_BLOCKS)), "co_fetch_order must return a permutation"
    result = compactor_module.LakeCompactor().reorder(store, order)
    assert result["refusals"] == 0, result

    after_reads = _reads_per_round(store, due_sets)
    assert after_reads == 1.0, f"the group is not contiguous after the reorder ({after_reads})"
    assert store.bytes_fetched == store.bytes_used
    assert _logical_edges(store, R8_BLOCKS) == before_edges, "a reorder changes no record"
    after_files = _per_file_sha(store.root)
    for name, digest in before_files.items():
        assert after_files[name] == digest, f"{name} moved -- a reorder destroys nothing"


def test_the_random_fetch_control_is_not_helped_by_a_locality_order_fitted_to_it(tmp_path):
    """2.5's null: an order fitted to noise must not buy what the structured order buys."""
    structured_store, _b, _e = _r8_store(tmp_path, "margin-structured")
    structured_sets = _structured_sets(20)
    s_before = _reads_per_round(structured_store, structured_sets)
    compactor_module.LakeCompactor().reorder(
        structured_store, compactor_module.co_fetch_order(structured_sets, R8_BLOCKS)
    )
    s_after = _reads_per_round(structured_store, structured_sets)

    random_store, _rb, _re = _r8_store(tmp_path, "margin-random")
    random_sets = _random_sets(20)
    r_before = _reads_per_round(random_store, random_sets)
    compactor_module.LakeCompactor().reorder(
        random_store, compactor_module.co_fetch_order(random_sets, R8_BLOCKS)
    )
    r_after = _reads_per_round(random_store, random_sets)

    assert s_after == 1.0 and s_before == float(CO_FETCH_K)
    assert r_after >= 3.0, f"the random control's reads per round fell to {r_after}"
    assert (s_before - s_after) - (r_before - r_after) >= CONTROL_MARGIN, (
        f"structured reduction {s_before - s_after} vs random {r_before - r_after} -- the "
        f"pre-registered margin is {CONTROL_MARGIN} reads per round"
    )


def test_a_paged_run_is_byte_identical_to_its_twin_before_and_after_a_reorder(tmp_path):
    store, blocks, edges = _r8_store(tmp_path, "reorder-identity")
    drives = {0: _drive([0, 1, 2], n_units=R8_N)}
    twin_trains, _twin_rings = _run(_twin(edges, n_units=R8_N), drives, 24)
    _kt, pager = _paged(
        store,
        blocks,
        n_units=R8_N,
        capacity_blocks=6,
        blocks_per_round=8,
        max_rounds=4,
        max_stale_ticks=1,
        pinned=frozenset({0}),
    )
    before_trains, _r = _run(pager, drives, 24)
    assert before_trains == twin_trains
    compactor_module.LakeCompactor().reorder(
        store, compactor_module.co_fetch_order(_structured_sets(8), R8_BLOCKS)
    )
    _kt2, pager2 = _paged(
        store,
        blocks,
        n_units=R8_N,
        capacity_blocks=6,
        blocks_per_round=8,
        max_rounds=4,
        max_stale_ticks=1,
        pinned=frozenset({0}),
    )
    after_trains, _r2 = _run(pager2, drives, 24)
    assert after_trains == twin_trains, "the reorder changed what the substrate computes"


def test_a_dropping_reorder_makes_a_block_unreadable_and_fails_the_record_identity(tmp_path):
    store, _blocks, _edges = _r8_store(tmp_path, "dropreorder")
    before_edges = _logical_edges(store, R8_BLOCKS)
    order = compactor_module.co_fetch_order(_structured_sets(8), R8_BLOCKS)
    dropper = _DroppingReorderCompactor()
    dropper.skip = order[1]
    dropper.reorder(store, order)
    # REFUSED, never read as an empty edge set. The exact type is the segment's own range
    # refusal (ValueError) or the index's missing-pointer refusal (KeyError) depending on where
    # the shifted pointer lands; the criterion is that a read cannot silently succeed.
    with pytest.raises((KeyError, ValueError)):
        _logical_edges(store, R8_BLOCKS)
    with pytest.raises((KeyError, ValueError, AssertionError)):
        assert _logical_edges(store, R8_BLOCKS) == before_edges


def test_a_reorder_refuses_an_order_that_is_not_a_permutation(tmp_path):
    store, _blocks, _edges = _r8_store(tmp_path, "lossy")
    for bad in (
        list(range(R8_BLOCKS - 1)),  # short
        [0] + list(range(R8_BLOCKS - 1)),  # repeats 0
        list(range(1, R8_BLOCKS + 1)),  # out of range
    ):
        with pytest.raises(ValueError):
            compactor_module.LakeCompactor().reorder(store, bad)


def test_the_co_fetch_order_is_deterministic_and_places_co_fetched_blocks_together():
    due_sets = _structured_sets(5)
    order = compactor_module.co_fetch_order(due_sets, R8_BLOCKS)
    assert order == compactor_module.co_fetch_order(due_sets, R8_BLOCKS), (
        "the order function must be deterministic"
    )
    assert sorted(order) == list(range(R8_BLOCKS))
    group = set(due_sets[0])
    positions = sorted(order.index(b) for b in group)
    assert positions == list(range(positions[0], positions[0] + len(group))), (
        "the co-fetched group must be contiguous in the order"
    )
    # blocks never fetched keep index order at the tail
    tail = [b for b in order if b not in group]
    assert tail == sorted(tail)


# ---- the coalesced range read --------------------------------------------------------------------
def test_read_records_returns_a_span_and_refuses_one_outside_the_segment(tmp_path):
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="ranges")
    segment = sorted((store.root / SEGMENT_DIR).glob("seg-*.lake"))[0]
    whole = list(read_segment(segment))
    base = int(segment.stem.split("-")[1])
    assert read_records(segment, base, base + 2) == whole[:3]
    assert read_records(segment, base + 1, base + 1) == whole[1:2]
    for lo, hi in ((base - 1, base + 1), (base, base + len(whole)), (base + 2, base + 1)):
        with pytest.raises(ValueError):
            read_records(segment, lo, hi)


def test_a_fetch_reads_each_blocks_deltas_in_the_same_round_and_never_over_reads(tmp_path):
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="rounds")
    store.append_deltas(_delta_records(_delta_spec()))
    start_records = store.records_read
    store.fetch([0, 1])
    # block 0 and block 1 each carry one unmerged delta: 1 + k records in the same round
    assert store.records_read - start_records == 4
    assert store.submissions >= 1
    assert store.submission_widths[-1] == 2, "the WIDTH bound stays a BLOCK bound (T066)"
    assert store.submission_records[-1] == 4
    assert store.bytes_fetched == store.bytes_used, "no over-read this row: the ratio is 1.0"
    store.assert_read_accounting()


# ---- the rebuild identity (D1) ---------------------------------------------------------------
def test_build_index_reproduces_the_stores_block_rows_after_build_deltas_merge_and_reorder(
    tmp_path,
):
    edges = _base_edges()
    store, blocks = _store_for(tmp_path, edges, name="rebuild")
    store.append_deltas(_delta_records(_delta_spec()))
    merged = compactor_module.LakeCompactor().merge(store, budget_blocks=N_BLOCKS)
    assert merged["refusals"] == 0, "the rebuild identity is stated for a store with no refusal"
    store.append_deltas(_delta_records([(5, 0, [(2, 6 * BLOCK + 11, W_SUB, 2, 0)])]))
    compactor_module.LakeCompactor().reorder(
        store, compactor_module.co_fetch_order([(0, 2, 4)], N_BLOCKS)
    )

    from snn_data_lake.substrate.lake_index import LakeIndex

    want = _block_rows_digest(store.index, N_BLOCKS)
    build_index(store.root, Path(tmp_path) / "rebuilt.db")
    rebuilt = LakeIndex(Path(tmp_path) / "rebuilt.db")
    try:
        assert _block_rows_digest(rebuilt, N_BLOCKS) == want, (
            "build_index did not re-derive the store's (0x0C, *) and (0x0D, *) rows"
        )
        # THE PLANTED-ROW RED IS KEPT: a row inserted behind the gate's back must move the digest.
        rebuilt.put_batch([(TAG_DELTA, "0/planted", "nowhere", 0)])
        rebuilt.put_batch([(TAG_BLOCK, "0/gen", "meta", 99)])
        assert _block_rows_digest(rebuilt, N_BLOCKS) != want
    finally:
        rebuilt.close()


def test_a_store_rebuilt_from_the_lake_alone_pages_in_identically(tmp_path):
    edges = _base_edges()
    spec = _delta_spec()
    store, blocks = _store_for(tmp_path, edges, name="rebuild-page")
    store.append_deltas(_delta_records(spec))
    compactor_module.LakeCompactor().merge(store, budget_blocks=N_BLOCKS)
    want = _logical_edges(store)
    store.index.close()
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(store.root / "lake_index.db") + suffix)
        if path.exists():
            path.unlink()
    build_index(store.root, store.root / "lake_index.db")
    reopened = UnitBlockStore(store.root)
    try:
        assert _logical_edges(reopened) == want
        twin_trains, _tr = _run(_twin(_apply_in_python(edges, spec)), {0: _drive([0, 1, 2])}, 20)
        _kt, pager = _paged(
            reopened,
            blocks,
            capacity_blocks=4,
            blocks_per_round=8,
            max_rounds=4,
            max_stale_ticks=1,
            pinned=frozenset({0}),
        )
        paged_trains, _pr = _run(pager, {0: _drive([0, 1, 2])}, 20)
        assert paged_trains == twin_trains
    finally:
        reopened.close()


# ---- refusals and the observer ---------------------------------------------------------------
def test_a_delta_for_an_unknown_block_is_refused(tmp_path):
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="unknown")
    with pytest.raises((KeyError, ValueError)):
        store.append_deltas(_delta_records([(N_BLOCKS + 3, 0, [(0, 5, W_SUB, 1, 0)])]))


def test_the_delta_writer_reaches_the_one_append_observer(tmp_path):
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="observed")
    seen = []
    prev = home_mod.set_append_observer(lambda tag, n: seen.append((tag, n)))
    try:
        store.append_deltas(_delta_records(_delta_spec()))
    finally:
        home_mod.set_append_observer(prev)
    assert seen, "the block-grain delta writer is off the observer path"
    assert {tag for tag, _n in seen} == {TAG_DELTA}
    assert all(n > 0 for _tag, n in seen)
    budget = AppendBudget(baseline={"per_tag_p95_bytes": {str(TAG_DELTA): 10.0}})
    for tag, n in seen:
        budget.on_append(tag, n)
    assert budget.end_tick(), "a constructed flood on the new class must reach the budget"


def test_observe_append_is_the_one_documented_observer_path():
    seen = []
    prev = home_mod.set_append_observer(lambda tag, n: seen.append((tag, n)))
    try:
        observe_append(_T_BLOCKDELTA, 4096)
    finally:
        home_mod.set_append_observer(prev)
    assert seen == [(TAG_DELTA, 4096)]


def test_a_second_write_to_a_store_with_a_non_empty_wal_indexes_every_block_to_its_true_seq(
    tmp_path,
):
    """The seq-arithmetic fix: `seq0 + bid` assumes an empty WAL, and this row is the first
    second-writer."""
    root = Path(tmp_path) / "seqfix"
    root.mkdir(parents=True, exist_ok=True)
    store = UnitBlockStore(root)
    store.store.append(b"\x01a-foreign-record-already-in-the-wal")
    blocks = store.build(*_base_edges(), N, BLOCK)
    assert blocks.n_blocks == N_BLOCKS
    for bid, record in sorted(store.fetch(range(N_BLOCKS)).items()):
        assert record.block_id == bid, (
            f"block {bid}'s pointer resolves to record {record.block_id} -- the seq arithmetic "
            "is off by the WAL's prior count"
        )


def test_record_from_edges_round_trips_through_apply_deltas(tmp_path):
    """The ONE apply, exercised without a store: base -> table -> apply -> record -> table."""
    edges = _base_edges()
    store, _blocks = _store_for(tmp_path, edges, name="apply")
    base = store.fetch([1])[1]
    table = edge_table(base)
    deltas = _delta_records([(1, 0, [(0, 2 * BLOCK + 13, W_SUB, 1, 0)])])
    applied = apply_deltas(table, deltas, generation=0)
    record = record_from_edges(1, base.n_units_in_block, base.block_size, applied)
    assert edge_table(record) == applied
    assert UnitBlockRecord.decode(record.encode()).encode() == record.encode()
    with pytest.raises(ValueError):
        apply_deltas(table, deltas, generation=1)

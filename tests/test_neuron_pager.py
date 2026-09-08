"""T066 / Phase 3 S3 -- activation-driven paging: BOTH reds, in tree before the mechanism.

Section 2.3 of specs/022-unified-brain-arc/evidence/s3-neuron-pager.md, written against the API
that pre-registration names and RUN RED on the tree that has no `substrate/neuron_pager.py`.
Each criterion carries the constructed BROKEN ARM that makes it a measurement:

  S-R2  a spike to a paged-out unit is dropped silently -- the DROPPING twin (a KernelTick
        subclass whose deliver phase discards non-resident posts, the naive pager) must show a
        per-unit charge shortfall EQUAL to its own discard count, and trains that differ from
        the resident twin.
  S-R1  the rounds budget never trips on constructed overload -- the BUDGET-BLIND pager (a
        subclass that submits the whole due set in one round whatever `blocks_per_round` says)
        must read 0 trips on the identical overload, so the criterion is seen to fail.

Every fixture weight is a multiple of 0.25 and so exactly representable in `<f4`, which is the
dtype the block record stores weights in: a fixture whose weights rounded at the record boundary
would fail the byte-identity criterion for a reason that has nothing to do with paging.

The fixtures set `refrac_ticks=0` and give every edge a weight above threshold on purpose. A unit
that fires resets to `v_reset` (0.0 here), so the network returns to EXACT rest between waves --
which is what makes a block evictable at all, since a paged-out unit's membrane does not decay
(disclosure 2.6(b)) and an evicted block that is not at rest could not be byte-identical to the
twin.
"""

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
from snn_data_lake.substrate.neuron_pager import (
    NeuronPager,
    RoundsBudgetExceeded,
    UnitBlockStore,
)

from snn_data_lake.substrate import kernel_tick as kernel_tick_module
from snn_data_lake.substrate import neuron_pager as neuron_pager_module
from snn_data_lake.substrate.home import _T_UNITBLOCK
from snn_data_lake.substrate.kernel_state import KernelState
from snn_data_lake.substrate.kernel_tick import KernelTick, Synapses, compile_bundles
from snn_data_lake.substrate.lake_bridge import iter_records
from snn_data_lake.substrate.lake_engine import (
    UnitBlockRecord,
    registry_consistent,
    retention_of,
)
from snn_data_lake.substrate.lake_rebuild_gate import assert_rebuild
from snn_data_lake.substrate.lake_segments import read_record, read_segment

FRAC = 16
BLOCK = 128
N_BLOCKS = 8
N = BLOCK * N_BLOCKS
P_PARTS = 4
MAX_DELAY = 3
WEIGHT = 1.5  # exactly representable in float32; one edge alone clears thr = 1.0
TAG = _T_UNITBLOCK[0]


# ---- instruments -----------------------------------------------------------------------------
class _MeteredTick(KernelTick):
    """The incumbent tick plus two per-unit counters, both differenced around the real phases.

    `delivered` is charge that LEFT a bundle for a post unit (accumulator or pending); `integrated`
    is charge that entered a unit's membrane. Conservation is `delivered == integrated` once every
    pending has drained. The counters wrap the incumbent phases rather than re-implementing them,
    so an instrument bug cannot look like a mechanism bug."""

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
    """THE S-R2 RED: the naive pager. Its deliver phase DISCARDS every post that is not resident
    instead of conserving it, and counts exactly what it threw away."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.discarded = np.zeros(self.state.n_units, dtype=np.int64)

    def _deliver_and_rotate(self, p):
        np_mod = self._np
        slot = self._rings[p][(self.tick_index + 1) % (self.syn.max_delay + 1)]
        acc = self._acc[p]
        keep_mask = self.resident
        for key in slot:
            posts, q = self.syn.bundles[key]
            keep = keep_mask[posts]
            np_mod.add.at(acc, posts[keep], q[keep])
            np_mod.add.at(self.delivered, posts[keep], q[keep])
            np_mod.add.at(self.discarded, posts[~keep], q[~keep])
        slot.clear()


class _BudgetBlindPager(NeuronPager):
    """THE S-R1 RED: submits the whole due set as ONE round whatever `blocks_per_round` says, so
    the budget can never be exceeded and no trip is ever recorded."""

    def _issue_rounds(self, due, tick):
        if not due:
            return {}, 0, None
        return self.store.fetch(due), 1, None


# ---- fixtures --------------------------------------------------------------------------------
def _chain_edges(seed=11, fan=2):
    """A layered chain: block b's units drive block b+1's, block 7 terminal. Every block except
    the last OWNS edges, so paging a block really moves compiled bundles."""
    rng = np.random.default_rng(seed)
    pre, post, weight, delay = [], [], [], []
    for b in range(N_BLOCKS - 1):
        for unit in range(b * BLOCK, (b + 1) * BLOCK):
            for _ in range(fan):
                pre.append(unit)
                post.append(int(rng.integers((b + 1) * BLOCK, (b + 2) * BLOCK)))
                weight.append(WEIGHT)
                delay.append(int(rng.integers(1, MAX_DELAY + 1)))
    return pre, post, weight, delay


def _fanout_edges(seed=13):
    """Block 0 fans out into EVERY other block at delay 1, and its targets are terminal. One drive
    pulse therefore makes every block due on one tick (the constructed overload), and a dropped
    delivery cannot change any later spike -- which is what makes the dropping twin's shortfall
    comparable to the resident twin's charge at all."""
    rng = np.random.default_rng(seed)
    pre, post, weight, delay = [], [], [], []
    for unit in range(BLOCK):
        for b in range(1, N_BLOCKS):
            pre.append(unit)
            post.append(int(rng.integers(b * BLOCK, (b + 1) * BLOCK)))
            weight.append(WEIGHT)
            delay.append(1)
    return pre, post, weight, delay


def _drive(units, level=2.0):
    ext = np.zeros(N, dtype=np.int64)
    ext[list(units)] = int(level * (1 << FRAC))
    return ext


def _twin(edges, *, cls=None, max_delay=MAX_DELAY):
    state = KernelState([("net", N)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses(*edges, N, P_PARTS, FRAC, max_delay)
    tick_cls = cls or _MeteredTick
    return tick_cls(state, syn, decay=0.5, v_reset=0.0, refrac_ticks=0)


def _store_for(tmp_path, edges, name="lake"):
    root = Path(tmp_path) / name
    root.mkdir(parents=True, exist_ok=True)
    store = UnitBlockStore(root)
    blocks = store.build(*edges, N, BLOCK)
    return store, blocks


def _paged(
    store,
    blocks,
    *,
    capacity_blocks,
    blocks_per_round,
    max_rounds,
    max_stale_ticks,
    pinned,
    tick_cls=None,
    pager_cls=None,
    max_delay=MAX_DELAY,
):
    state = KernelState([("net", N)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses.empty(N, P_PARTS, FRAC, max_delay)
    resident = np.zeros(N, dtype=bool)
    kt = (tick_cls or _MeteredTick)(
        state, syn, decay=0.5, v_reset=0.0, refrac_ticks=0, resident=resident
    )
    pager = (pager_cls or NeuronPager)(
        kt,
        store,
        blocks=blocks,
        capacity_blocks=capacity_blocks,
        blocks_per_round=blocks_per_round,
        max_rounds=max_rounds,
        max_stale_ticks=max_stale_ticks,
        pinned=pinned,
    )
    return kt, pager


def _rings_of(kt):
    return [[list(slot) for slot in ring] for ring in kt._rings]


def _run_plain(kt, drives, ticks):
    trains, rings = [], []
    for t in range(ticks):
        trains.append(kt.tick(drives.get(t)).tobytes())
        rings.append(_rings_of(kt))
    return trains, rings


def _run_paged(pager, drives, ticks):
    trains, rings, pending_zero = [], [], []
    for t in range(ticks):
        trains.append(pager.tick(drives.get(t)).tobytes())
        rings.append(_rings_of(pager.kt))
        pending_zero.append(all(not int(np.abs(p).sum()) for p in pager.kt.pending))
    return trains, rings, pending_zero


def _calm_drives():
    return {0: _drive([3, 17]), 22: _drive([41, 90])}


def _pulse_drives(n_ticks):
    return {t: _drive(range(0, 8)) for t in range(n_ticks)}


# ---- S-R2: no silent drop --------------------------------------------------------------------
def test_a_calm_paged_run_is_byte_identical_to_the_resident_twin(tmp_path):
    edges = _chain_edges()
    drives = _calm_drives()
    twin = _twin(edges)
    twin_trains, twin_rings = _run_plain(twin, drives, 45)
    store, blocks = _store_for(tmp_path, edges)
    kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=4,
        blocks_per_round=4,
        max_rounds=2,
        max_stale_ticks=1,
        pinned=frozenset({0}),
    )
    paged_trains, paged_rings, _ = _run_paged(pager, drives, 45)
    assert paged_trains == twin_trains, "a calm paged run must be byte-identical to the twin"
    assert paged_rings == twin_rings, "ring-slot key lists must match the twin tick for tick"
    assert not pager.budget_trips, "the calm arm must not trip the rounds budget"
    assert np.array_equal(kt.delivered, twin.delivered)


def test_the_calm_arm_actually_paged_rather_than_holding_everything_resident(tmp_path):
    edges = _chain_edges()
    store, blocks = _store_for(tmp_path, edges)
    kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=4,
        blocks_per_round=4,
        max_rounds=2,
        max_stale_ticks=1,
        pinned=frozenset({0}),
    )
    _run_paged(pager, _calm_drives(), 45)
    fraction = float(kt.resident.mean())
    assert pager.page_ins > 0, "an identity that never paged proves nothing"
    assert pager.evictions > 0, "an identity that never evicted proves nothing"
    assert 0.0 < fraction < 1.0, f"resident fraction must be strictly inside (0, 1): {fraction}"


def test_overload_conserves_every_units_charge_and_bounds_staleness(tmp_path):
    edges = _fanout_edges()
    drives = _pulse_drives(10)
    store, blocks = _store_for(tmp_path, edges, name="ov")
    kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=N_BLOCKS,
        blocks_per_round=1,
        max_rounds=1,
        max_stale_ticks=2,
        pinned=frozenset({0}),
        max_delay=1,
    )
    _trains, _rings, pending_zero = _run_paged(pager, drives, 20)
    assert pager.budget_trips, "the constructed overload must trip the rounds budget"
    assert pager.max_staleness <= 2, f"staleness exceeded its bound: {pager.max_staleness}"
    assert pager.max_staleness > 0, "an overload with no deferral has not tested the bound"
    assert np.array_equal(kt.delivered, kt.integrated), (
        "every delivered charge must reach a membrane -- nothing may be dropped"
    )
    # the stated tail: the last delivery lands at the end of tick 10, and a block deferred to its
    # bound is forced max_stale_ticks later, so every pending must be zero from tick 13 on
    assert all(pending_zero[13:]), "pending did not drain to zero by the stated tail"


def test_overload_at_zero_staleness_stays_byte_identical_to_the_twin(tmp_path):
    edges = _fanout_edges()
    drives = _pulse_drives(10)
    twin = _twin(edges, max_delay=1)
    twin_trains, _rings = _run_plain(twin, drives, 20)
    store, blocks = _store_for(tmp_path, edges, name="ov0")
    kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=N_BLOCKS,
        blocks_per_round=1,
        max_rounds=1,
        max_stale_ticks=0,
        pinned=frozenset({0}),
        max_delay=1,
    )
    paged_trains, _pr, _pz = _run_paged(pager, drives, 20)
    assert pager.budget_trips, "forcing every due block must exceed a one-round budget"
    assert pager.max_staleness == 0, "max_stale_ticks=0 forces every drain on its own tick"
    assert paged_trains == twin_trains
    assert np.array_equal(kt.delivered, twin.delivered), (
        "per-unit delivered charge under overload must equal the twin's exactly"
    )


def test_the_dropping_twin_loses_exactly_the_charge_it_discards(tmp_path):
    edges = _fanout_edges()
    drives = _pulse_drives(6)
    twin = _twin(edges, max_delay=1)
    twin_trains, _rings = _run_plain(twin, drives, 12)
    store, blocks = _store_for(tmp_path, edges, name="drop")
    kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=N_BLOCKS,
        blocks_per_round=4,
        max_rounds=2,
        max_stale_ticks=0,
        pinned=frozenset({0, 1}),
        tick_cls=_DroppingTick,
        max_delay=1,
    )
    dropping_trains, _dr, _dz = _run_paged(pager, drives, 12)
    shortfall = twin.delivered - kt.delivered
    assert int(kt.discarded.sum()) > 0, "the broken arm must actually discard something"
    assert np.array_equal(shortfall, kt.discarded), (
        "the naive pager's charge shortfall must equal its own discard count, per unit"
    )
    assert dropping_trains != twin_trains, "a dropped spike must change the trains"


# ---- S-R1: the rounds budget -----------------------------------------------------------------
def test_the_rounds_budget_trips_on_constructed_overload_and_the_assert_raises(tmp_path):
    edges = _fanout_edges()
    store, blocks = _store_for(tmp_path, edges, name="r1")
    kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=N_BLOCKS,
        blocks_per_round=2,
        max_rounds=2,
        max_stale_ticks=0,
        pinned=frozenset({0}),
        max_delay=1,
    )
    _run_paged(pager, {0: _drive(range(0, 8))}, 6)
    assert len(pager.budget_trips) >= 1, "blocks_per_round * max_rounds < n_blocks must trip"
    tick, rounds_needed, budget = pager.budget_trips[0]
    assert rounds_needed > budget, "a trip whose rounds fit the budget is not a trip"
    assert pager.rounds_used[tick] == rounds_needed > pager.max_rounds
    assert budget == pager.max_rounds
    with pytest.raises(RoundsBudgetExceeded):
        pager.assert_within_budget()
    assert kt.resident.all(), "nothing may be left paged out by a forced round"


def test_the_calm_control_never_trips_the_budget_and_still_exercises_it(tmp_path):
    edges = _fanout_edges()
    store, blocks = _store_for(tmp_path, edges, name="r1calm")
    _kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=N_BLOCKS,
        blocks_per_round=4,
        max_rounds=2,
        max_stale_ticks=0,
        pinned=frozenset({0}),
        max_delay=1,
    )
    _run_paged(pager, {0: _drive(range(0, 8))}, 6)
    assert not pager.budget_trips, "the calm control must record zero trips"
    assert all(r <= pager.max_rounds for r in pager.rounds_used)
    assert max(pager.rounds_used) >= 1, "a budget that is never exercised is not a control"
    assert pager.assert_within_budget() is True


def test_a_budget_blind_pager_reads_zero_trips_on_the_identical_overload(tmp_path):
    edges = _fanout_edges()
    store, blocks = _store_for(tmp_path, edges, name="blind")
    _kt, blind = _paged(
        store,
        blocks,
        capacity_blocks=N_BLOCKS,
        blocks_per_round=2,
        max_rounds=2,
        max_stale_ticks=0,
        pinned=frozenset({0}),
        pager_cls=_BudgetBlindPager,
        max_delay=1,
    )
    _run_paged(blind, {0: _drive(range(0, 8))}, 6)
    assert not blind.budget_trips, "the broken arm must read 0 trips -- that is the red"
    assert max(blind.rounds_used) == 1, "the blind arm submits the whole due set in one round"


def test_both_rounds_instruments_agree_on_every_tick_in_every_arm(tmp_path):
    edges = _fanout_edges()
    arms = {
        "overload": dict(blocks_per_round=2, max_rounds=2, max_stale_ticks=0, pager_cls=None),
        "calm": dict(blocks_per_round=4, max_rounds=2, max_stale_ticks=0, pager_cls=None),
        "blind": dict(
            blocks_per_round=2, max_rounds=2, max_stale_ticks=0, pager_cls=_BudgetBlindPager
        ),
    }
    for name, cfg in arms.items():
        store, blocks = _store_for(tmp_path, edges, name=f"agree-{name}")
        _kt, pager = _paged(
            store, blocks, capacity_blocks=N_BLOCKS, pinned=frozenset({0}), max_delay=1, **cfg
        )
        _run_paged(pager, {0: _drive(range(0, 8))}, 6)
        assert pager.rounds_used == pager.store_rounds_used, (
            f"{name}: the planner and the store disagree on rounds -- report no rounds number"
        )
        assert sum(pager.rounds_used) == store.submissions - pager.submissions_at_start


# ---- the incumbent's one residency branch ----------------------------------------------------
def test_resident_none_and_an_all_true_mask_are_bit_identical():
    edges = _chain_edges()
    drives = _calm_drives()
    plain = _twin(edges)
    masked = _twin(edges)
    masked.resident = np.ones(N, dtype=bool)
    plain_trains, plain_rings = _run_plain(plain, drives, 30)
    masked_trains, masked_rings = _run_plain(masked, drives, 30)
    assert plain_trains == masked_trains
    assert plain_rings == masked_rings
    for a, b in zip(plain._acc, masked._acc, strict=True):
        assert np.array_equal(a, b)
    assert all(not int(np.abs(p).sum()) for p in masked.pending)
    assert np.array_equal(plain.state.arrays["v"], masked.state.arrays["v"])


def test_compile_bundles_is_the_only_builder_of_bundles_and_by_pre():
    """P2 pin (2.2). Anchored on each module's own `__file__`, never on a literal path."""
    builders = []
    for module in (kernel_tick_module, neuron_pager_module):
        path = Path(module.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owners = {}
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(func):
                    owners[id(node)] = func.name
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            if not targets or node.value is None:
                continue
            fresh = isinstance(node.value, (ast.Dict, ast.DictComp))
            for target in targets:
                name = (
                    target.attr
                    if isinstance(target, ast.Attribute)
                    else getattr(target, "id", None)
                )
                if name in ("bundles", "by_pre") and fresh:
                    builders.append((path.name, owners.get(id(node), "<module>")))
    assert builders, "the census found no builder at all -- the instrument is blind"
    assert set(builders) == {("kernel_tick.py", "compile_bundles")}, builders


# ---- refusals ---------------------------------------------------------------------------------
def test_the_store_refuses_a_block_size_that_is_not_a_positive_multiple_of_128(tmp_path):
    edges = _chain_edges()
    root = Path(tmp_path) / "refuse"
    root.mkdir()
    store = UnitBlockStore(root)
    for bad in (0, -128, 64, 129, 200):
        with pytest.raises(ValueError):
            store.build(*edges, N, bad)


def test_the_store_refuses_a_missing_root_rather_than_defaulting_to_the_live_home(tmp_path):
    with pytest.raises(ValueError):
        UnitBlockStore(Path(tmp_path) / "not-there")


def test_the_pager_refuses_missing_or_non_integer_budget_inputs(tmp_path):
    edges = _chain_edges()
    store, blocks = _store_for(tmp_path, edges, name="refuse-inputs")
    state = KernelState([("net", N)])
    syn = Synapses.empty(N, P_PARTS, FRAC, MAX_DELAY)
    resident = np.zeros(N, dtype=bool)
    kt = KernelTick(state, syn, decay=0.5, v_reset=0.0, refrac_ticks=0, resident=resident)
    full = dict(
        blocks=blocks,
        capacity_blocks=4,
        blocks_per_round=2,
        max_rounds=2,
        max_stale_ticks=1,
        pinned=frozenset({0}),
    )
    for missing in ("capacity_blocks", "blocks_per_round", "max_rounds", "max_stale_ticks"):
        kwargs = {k: v for k, v in full.items() if k != missing}
        with pytest.raises(TypeError):
            NeuronPager(kt, store, **kwargs)
    for field in ("capacity_blocks", "blocks_per_round", "max_rounds", "max_stale_ticks"):
        kwargs = dict(full)
        kwargs[field] = 2.0
        with pytest.raises(ValueError):
            NeuronPager(kt, store, **kwargs)


def test_the_pager_refuses_a_pinned_block_out_of_range_or_larger_than_capacity(tmp_path):
    edges = _chain_edges()
    store, blocks = _store_for(tmp_path, edges, name="refuse-pinned")
    state = KernelState([("net", N)])
    syn = Synapses.empty(N, P_PARTS, FRAC, MAX_DELAY)
    kt = KernelTick(
        state, syn, decay=0.5, v_reset=0.0, refrac_ticks=0, resident=np.zeros(N, dtype=bool)
    )
    base = dict(blocks=blocks, blocks_per_round=2, max_rounds=2, max_stale_ticks=1)
    with pytest.raises(ValueError):
        NeuronPager(kt, store, capacity_blocks=4, pinned=frozenset({N_BLOCKS}), **base)
    with pytest.raises(ValueError):
        NeuronPager(kt, store, capacity_blocks=4, pinned=frozenset({-1}), **base)
    with pytest.raises(ValueError):
        NeuronPager(kt, store, capacity_blocks=2, pinned=frozenset({0, 1, 2}), **base)


def test_a_malformed_unit_block_record_is_refused_whole():
    rec = UnitBlockRecord(
        block_id=2,
        n_units_in_block=BLOCK,
        block_size=BLOCK,
        pre_local=[0, 1],
        post=[300, 301],
        weight=[WEIGHT, WEIGHT],
        delay=[1, 2],
    )
    payload = rec.encode()
    assert payload[0] == TAG, "the record must carry its class tag as the first payload byte"
    back = UnitBlockRecord.decode(payload)
    assert back.block_id == 2
    assert np.array_equal(back.post, np.asarray([300, 301], dtype=np.uint32))
    assert np.array_equal(back.weight, np.asarray([WEIGHT, WEIGHT], dtype=np.float32))
    for bad in (
        payload[:-1],  # length mismatch: the arrays no longer divide evenly
        payload[:5],  # short header
        bytes([0x0B]) + payload[1:],  # foreign tag
        payload[:1] + bytes([9]) + payload[2:],  # unknown record version
    ):
        with pytest.raises(ValueError):
            UnitBlockRecord.decode(bad)


def test_a_fetch_of_an_unknown_block_id_is_refused(tmp_path):
    edges = _chain_edges()
    store, _blocks = _store_for(tmp_path, edges, name="unknown")
    assert set(store.fetch([0, 1])) == {0, 1}
    with pytest.raises(KeyError):
        store.fetch([N_BLOCKS + 5])


def test_a_pinned_block_is_never_paged_out_under_pressure(tmp_path):
    edges = _chain_edges()
    store, blocks = _store_for(tmp_path, edges, name="pinned")
    kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=2,
        blocks_per_round=4,
        max_rounds=2,
        max_stale_ticks=1,
        pinned=frozenset({0, 1}),
    )
    pinned_units = list(range(0, 2 * BLOCK))
    for t in range(45):
        pager.tick(_calm_drives().get(t))
        assert kt.resident[pinned_units].all(), f"a pinned block was paged out at tick {t}"
    assert pager.capacity_trips > 0, "capacity 2 of 8 blocks must actually apply pressure"


# ---- the lake tier -----------------------------------------------------------------------------
def test_the_unit_block_class_is_immortal_and_both_registries_agree():
    label, reclaimable, reason = retention_of(TAG)
    assert label == "IMMORTAL"
    assert reclaimable is False
    assert len(reason) >= 20
    ok, message = registry_consistent()
    assert ok, message


def test_read_record_returns_what_read_segment_yields_at_that_seq(tmp_path):
    edges = _chain_edges()
    store, _blocks = _store_for(tmp_path, edges, name="records")
    segments = sorted((Path(tmp_path) / "records" / "segments").glob("seg-*.lake"))
    assert len(segments) == 1, "one build seals one multi-record segment"
    path = segments[0]
    whole = list(read_segment(path))
    assert len(whole) == N_BLOCKS, "one record per block, including edge-less blocks"
    seq0 = int(path.stem.split("-")[1])
    for offset, payload in enumerate(whole):
        assert read_record(path, seq0 + offset) == payload
    for bad_seq in (seq0 - 1, seq0 + len(whole)):
        with pytest.raises(ValueError):
            read_record(path, bad_seq)


def test_the_lake_digest_is_identical_across_a_full_page_out_page_in_cycle(tmp_path):
    edges = _chain_edges()
    store, blocks = _store_for(tmp_path, edges, name="digest")
    seg_dir = Path(tmp_path) / "digest" / "segments"

    def digest():
        h = hashlib.sha256()
        for seg in sorted(seg_dir.glob("seg-*.lake")):
            h.update(seg.read_bytes())
        return h.hexdigest()

    before = digest()
    _kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=3,
        blocks_per_round=4,
        max_rounds=2,
        max_stale_ticks=1,
        pinned=frozenset({0}),
    )
    _run_paged(pager, _calm_drives(), 45)
    assert pager.page_ins > 0 and pager.evictions > 0, "the cycle must have paged both ways"
    assert digest() == before, "page-out DESTROYS NO RECORD -- the lake bytes must not move"


def test_the_bridge_and_the_rebuild_gate_stay_green_on_one_batch_per_record(tmp_path):
    edges = _chain_edges()
    store, _blocks = _store_for(tmp_path, edges, name="bridge")
    root = Path(tmp_path) / "bridge"
    bridged = list(iter_records(root))
    assert bridged == list(store.iter_payloads()), "the ONE forward reader must see every record"
    assert len(bridged) == N_BLOCKS
    ok, message = assert_rebuild(root, Path(tmp_path) / "probe-index.db")
    assert ok, message


def test_compile_bundles_and_the_constructor_agree_bit_for_bit():
    edges = _chain_edges()
    syn = Synapses(*edges, N, P_PARTS, FRAC, MAX_DELAY)
    bundles, by_pre = compile_bundles(*edges, N, P_PARTS, FRAC, MAX_DELAY)
    assert set(bundles) == set(syn.bundles)
    for key, (posts, q) in bundles.items():
        assert np.array_equal(posts, syn.bundles[key][0])
        assert np.array_equal(q, syn.bundles[key][1])
    assert by_pre == syn.by_pre
    empty = Synapses.empty(N, P_PARTS, FRAC, MAX_DELAY)
    assert empty.bundles == {} and empty.by_pre == {}
    assert empty.max_delay == MAX_DELAY

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

Most fixtures set `refrac_ticks=0` and give every edge a weight above threshold, so a unit that
fires resets to `v_reset` (0.0 here) and the network returns to exact rest between waves. That is
a property of those fixtures, NOT the eviction condition: a paged-out unit's membrane keeps
decaying by the same op the resident twin runs, so a block holding sub-threshold charge is
evictable and comes back holding what the twin holds. The sub-threshold fixture below is the arm
that makes the difference measurable -- it was RED (nothing evictable, capacity exceeded) while
the condition still required `v == 0`.
"""

import ast
import shutil
from pathlib import Path

import numpy as np
import pytest

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
from snn_data_lake.substrate.lake_segments import SEGMENT_DIR, read_record, read_segment
from snn_data_lake.substrate.neuron_pager import (
    NeuronPager,
    RoundsBudgetExceeded,
    RoundWidthExceeded,
    UnitBlockStore,
    segment_digest,
)
from snn_data_lake.substrate.unit_blocks import UnitBlocks

FRAC = 16
BLOCK = 128
N_BLOCKS = 8
N = BLOCK * N_BLOCKS
P_PARTS = 4
MAX_DELAY = 3
WEIGHT = 1.5  # exactly representable in float32; one edge alone clears thr = 1.0
SUB_WEIGHT = 0.25  # exactly representable in float32; one edge alone stays UNDER thr = 1.0
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


def _twin(edges, *, cls=None, max_delay=MAX_DELAY, refrac_ticks=0):
    state = KernelState([("net", N)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses(*edges, N, P_PARTS, FRAC, max_delay)
    tick_cls = cls or _MeteredTick
    return tick_cls(state, syn, decay=0.5, v_reset=0.0, refrac_ticks=refrac_ticks)


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
    refrac_ticks=0,
):
    state = KernelState([("net", N)])
    state.arrays["thr"][:] = 1.0
    syn = Synapses.empty(N, P_PARTS, FRAC, max_delay)
    resident = np.zeros(N, dtype=bool)
    kt = (tick_cls or _MeteredTick)(
        state, syn, decay=0.5, v_reset=0.0, refrac_ticks=refrac_ticks, resident=resident
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


#: THE CALM ARM'S ONE CONFIG, read by both calm tests. Section 2.3 pre-registers that criterion
#: as a CONJUNCTION -- trains and ring-slot key lists byte-identical to the twin AND the run
#: actually paged -- and the first encoding split the two halves across two functions with this
#: config duplicated by hand at both sites. The identity half ALONE survives neutering the
#: mechanism: at capacity_blocks = N_BLOCKS all four of its assertions pass with page_ins=7,
#: evictions=0 and resident fraction 1.000. So the guard is folded into the identity test and
#: neither the config nor the conjunction can drift apart in one edit.
CALM_ARM = dict(
    capacity_blocks=4,
    blocks_per_round=4,
    max_rounds=2,
    max_stale_ticks=1,
    pinned=frozenset({0}),
)


def _assert_actually_paged(pager, kt):
    """The other half of the calm conjunction: an identity that never paged proves nothing."""
    fraction = float(kt.resident.mean())
    assert pager.page_ins > 0, "an identity that never paged proves nothing"
    assert pager.evictions > 0, "an identity that never evicted proves nothing"
    assert 0.0 < fraction < 1.0, f"resident fraction must be strictly inside (0, 1): {fraction}"
    return fraction


def _pulse_drives(n_ticks):
    return {t: _drive(range(0, 8)) for t in range(n_ticks)}


# ---- S-R2: no silent drop --------------------------------------------------------------------
def test_a_calm_paged_run_is_byte_identical_to_the_resident_twin(tmp_path):
    edges = _chain_edges()
    drives = _calm_drives()
    twin = _twin(edges)
    twin_trains, twin_rings = _run_plain(twin, drives, 45)
    store, blocks = _store_for(tmp_path, edges)
    kt, pager = _paged(store, blocks, **CALM_ARM)
    paged_trains, paged_rings, _ = _run_paged(pager, drives, 45)
    assert paged_trains == twin_trains, "a calm paged run must be byte-identical to the twin"
    assert paged_rings == twin_rings, "ring-slot key lists must match the twin tick for tick"
    assert not pager.budget_trips, "the calm arm must not trip the rounds budget"
    assert np.array_equal(kt.delivered, twin.delivered)
    # THE PAGING GUARD, IN THIS TEST. Every assertion above passes on a run with eviction
    # switched off, so the identity alone is not the pre-registered criterion.
    _assert_actually_paged(pager, kt)


def test_the_calm_arm_actually_paged_rather_than_holding_everything_resident(tmp_path):
    """The guard on its own run, over the SAME `CALM_ARM` config the identity test uses."""
    edges = _chain_edges()
    store, blocks = _store_for(tmp_path, edges)
    kt, pager = _paged(store, blocks, **CALM_ARM)
    _run_paged(pager, _calm_drives(), 45)
    _assert_actually_paged(pager, kt)
    # the TWO EXACT residency readings, after a run that evicted: the table sum and the sum over
    # the same arrays reached through `by_pre`. They must agree to the byte with no dangling
    # route -- an orphaned bundle moves only the first, a stale by_pre entry only the second.
    table_bytes = pager.resident_edge_bytes()
    route_bytes, dangling = pager.resident_edge_bytes_via_routes()
    assert (route_bytes, dangling) == (table_bytes, 0), (
        f"the two exact residency instruments disagree: table {table_bytes}, routes "
        f"{route_bytes}, dangling {dangling} -- report no residency number"
    )


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


# ---- the two conservation/eviction defects, each with the input that makes it fail ------------
class _ConservingTick(_MeteredTick):
    """`_MeteredTick` plus the EXTERNAL half of the charge ledger.

    `_MeteredTick` counts bundle charge only, so it is structurally blind to drive that arrives
    through `ext_q`: external drive reaches a unit INSIDE propagate, never through a bundle, and a
    counter differenced around delivery therefore cannot see it. `ext_offered` is every fixed-point
    unit of drive the caller offered; `injected` is what actually entered a membrane (the `acc +
    ext_q` that propagate integrates for the members it integrates); `ext_at_paged_out` is the
    INDEPENDENT counter of drive that arrived while its unit was not resident. Conservation is
    `delivered + ext_offered == injected` once acc and pending have drained.

    The two counters read DIFFERENTLY on the two trees, which is what makes the second one worth
    keeping. While propagate dropped a non-resident member's drive the shortfall equalled
    `ext_at_paged_out` exactly (18874368 both ways), attributing the loss to that port and no
    other. Once the drive is routed into `pending` the shortfall is zero while `ext_at_paged_out`
    stays large -- so it stops being a loss counter and becomes the control that says the arm is
    still deferring fetches at all, which a green conservation assert alone could not say."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = self.state.n_units
        self.ext_offered = np.zeros(n, dtype=np.int64)
        self.ext_at_paged_out = np.zeros(n, dtype=np.int64)
        self.injected = np.zeros(n, dtype=np.int64)

    def _propagate(self, p, ext_q):
        members = self._members[p]
        live = members if self.resident is None else members[self.resident[members]]
        self.injected[live] += self._acc[p][live]
        if ext_q is not None:
            self.ext_offered[members] += ext_q[members]
            self.injected[live] += ext_q[live]
            if self.resident is not None:
                away = members[~self.resident[members]]
                self.ext_at_paged_out[away] += ext_q[away]
        return super()._propagate(p, ext_q)


class _EvictionLog(NeuronPager):
    """The pager plus a record of WHICH block left core on WHICH tick -- `evictions` is a count
    and a count cannot say that the block holding sub-threshold charge was the one demoted."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evicted: list[tuple[int, int]] = []

    def _page_out(self, block):
        self.evicted.append((self.kt.tick_index, block))
        super()._page_out(block)


def _sub_threshold_edges():
    """Block 0's unit b-1 drives ONE unit of block b with a SUB-THRESHOLD weight at delay 1, for
    b in 1..7; every target is terminal. Driving block 0's units one at a time, ten ticks apart,
    therefore makes exactly one block due at a time, and the block it charges cannot fire -- so a
    block sits in core holding charge that the twin's own membrane is decaying."""
    pre, post, weight, delay = [], [], [], []
    for b in range(1, N_BLOCKS):
        pre.append(b - 1)
        post.append(b * BLOCK)
        weight.append(SUB_WEIGHT)
        delay.append(1)
    return pre, post, weight, delay


def _uniform_drive(level):
    ext = np.zeros(N, dtype=np.int64)
    ext[:] = int(level * (1 << FRAC))
    return ext


def test_external_drive_to_a_deferred_paged_out_block_is_conserved(tmp_path):
    """Drive is charge, and a deferred block's drive must survive the deferral.

    `plan()` counts external drive as making a block DUE, but a budget that defers that block
    leaves its units non-resident when propagate runs, and propagate integrates resident members
    only -- so the tick's drive is added nowhere and the next tick's plan sees only the next
    tick's `ext_q`. The arm isolates the port: a sub-threshold drive on EVERY unit every tick,
    nothing ever fires (asserted), so `delivered` is zero and the whole ledger is external."""
    edges = _chain_edges()
    store, blocks = _store_for(tmp_path, edges, name="extdrive")
    kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=1,
        blocks_per_round=1,
        max_rounds=1,
        max_stale_ticks=3,
        pinned=frozenset(),
        tick_cls=_ConservingTick,
    )
    drive = _uniform_drive(0.125)
    _run_paged(pager, {t: drive for t in range(30)}, 40)
    assert int(kt.delivered.sum()) == 0, (
        "the arm must isolate the EXTERNAL port: a fired unit would put bundle charge in the "
        "ledger and the drive would no longer be the only thing measured"
    )
    assert int(kt.ext_at_paged_out.sum()) > 0, (
        "no drive ever arrived at a non-resident unit -- the budget never deferred anything, so "
        "this arm tests nothing"
    )
    leftover = sum(int(np.abs(p).sum()) for p in kt.pending)
    stranded = sum(int(np.abs(a).sum()) for a in kt._acc)
    assert leftover == 0 and stranded == 0, (
        f"the tail is not quiet: pending {leftover}, accumulator {stranded} -- conservation "
        "cannot be read while charge is still in flight"
    )
    shortfall = kt.ext_offered + kt.delivered - kt.injected
    assert int(shortfall.sum()) == 0, (
        f"external drive to a DEFERRED paged-out block was dropped: {int(shortfall.sum())} "
        f"fixed-point units lost across {int((shortfall != 0).sum())} units; the independent "
        f"counter of drive that arrived at non-resident units reads "
        f"{int(kt.ext_at_paged_out.sum())}"
    )
    assert not np.any(shortfall), "conservation must hold per unit, not only in the sum"


def test_a_block_holding_sub_threshold_charge_is_evictable_and_keeps_decaying(tmp_path):
    """A membrane that merely decays toward zero must not pin a block in core.

    A float32 membrane under sustained sub-threshold charge decays toward zero without reaching
    it, so an eviction clause requiring `v == 0` exactly makes eviction rare in the one regime
    that matters. The block charged here holds sub-threshold `v` (asserted at the moment room is
    needed), has no pending, no accumulator charge and no spike within `max_delay` -- it is
    exactly the candidate the ring-safety condition admits. It must be demoted, capacity must
    hold, and the run must stay byte-identical to the resident twin INCLUDING the membrane array:
    a paged-out unit whose membrane froze would come back ahead of the twin."""
    edges = _sub_threshold_edges()
    drives = {0: _drive([0]), 10: _drive([1]), 20: _drive([2]), 30: _drive([3])}
    twin = _twin(edges)
    twin_trains, twin_rings = _run_plain(twin, drives, 40)
    store, blocks = _store_for(tmp_path, edges, name="subthr")
    kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=2,
        blocks_per_round=4,
        max_rounds=2,
        max_stale_ticks=0,
        pinned=frozenset({0}),
        pager_cls=_EvictionLog,
    )
    trains, rings, held = [], [], []
    charge_under_pressure = None
    for t in range(40):
        if t == 11:  # block 2 becomes due here, and the only free slot holds block 1
            charge_under_pressure = float(np.abs(kt.state.arrays["v"][BLOCK : 2 * BLOCK]).max())
        trains.append(pager.tick(drives.get(t)).tobytes())
        rings.append(_rings_of(kt))
        held.append(len(pager._resident_blocks))
    assert charge_under_pressure is not None and 0.0 < charge_under_pressure < 1.0, (
        f"the fixture must hold SUB-THRESHOLD charge when room is needed, not "
        f"{charge_under_pressure}"
    )
    assert 1 in [b for _t, b in pager.evicted], (
        f"the block holding sub-threshold charge was never demoted (evicted {pager.evicted}); a "
        f"membrane that only decays toward zero pinned it in core"
    )
    assert pager.capacity_trips == 0 and max(held) <= 2, (
        f"capacity 2 was exceeded: {pager.capacity_trips} trip(s), held up to {max(held)} blocks "
        "because no resident block was demotable"
    )
    assert trains == twin_trains
    assert rings == twin_rings
    assert np.array_equal(kt.state.arrays["v"], twin.state.arrays["v"]), (
        "a paged-out unit's membrane must keep decaying exactly as the resident twin's does"
    )


REFRAC_MAX_DELAY = 2
REFRAC_TICKS = 5  # > REFRAC_MAX_DELAY, the ONE regime the ring-safety condition does not cover
REFRAC_UNIT = BLOCK  # the first unit of block 1
REFRAC_RELEASE_TICK = 7  # the tick its refractory has expired on and it is at threshold
REFRAC_DRIVE = 4.0  # decays to 2.0 across one tick at decay=0.5, still over thr = 1.0


def _refrac_window_edges():
    """Two edges out of the PINNED block, both at delay 1 and both above threshold.

    Unit 0 drives `REFRAC_UNIT` (block 1) so block 1 pages in on the tick its pending drains and
    its unit fires on the SAME tick the resident twin's does -- the paged run is byte-identical
    up to the eviction, so a divergence can only come from the eviction. Unit 1 drives block 2 so
    block 2 becomes due EXACTLY on the tick block 1's refractory has been released, which is the
    capacity pressure that asks for room. Both targets are terminal, so a dropped spike cannot be
    laundered into a later one."""
    return [0, 1], [REFRAC_UNIT, 2 * BLOCK], [WEIGHT, WEIGHT], [1, 1]


def test_a_unit_at_threshold_with_its_refractory_released_is_not_evicted(tmp_path):
    """A unit about to fire must not be demoted out from under its own spike.

    `KernelTick._propagate` reads `gated = refrac > 0` BEFORE the decrement, so a unit whose
    refractory reaches 0 at tick t is gated at t and fires at t+1 if it is still at threshold.
    The ring-safety condition admits exactly that unit at t+1: its last spike is older than
    `max_delay` (reachable only when `refrac_ticks > max_delay`), its refractory reads 0, and both
    buffers are empty -- while its MEMBRANE, which the condition deliberately does not test,
    stands at or above `thr`. Evicting it before propagate silences a spike the resident twin
    fires, and firing is restricted to resident units, so the spike is not deferred -- it is GONE.

    The fixture puts that unit under capacity pressure on precisely that tick and asserts the
    calm byte-identity S-R2 pre-registers. The disclosed cost of holding it is a capacity trip:
    over capacity for one tick rather than a dropped spike, which is the trade `_make_room`
    already makes everywhere else."""
    edges = _refrac_window_edges()
    drives = {
        0: _drive([0]),
        # unit 1 makes block 2 due next tick; REFRAC_UNIT is charged over threshold while it is
        # still refractory-gated, so it holds that charge into the tick its gate releases
        REFRAC_RELEASE_TICK - 1: _drive([1]) + _drive([REFRAC_UNIT], REFRAC_DRIVE),
    }
    twin = _twin(edges, max_delay=REFRAC_MAX_DELAY, refrac_ticks=REFRAC_TICKS)
    twin_trains, twin_rings = _run_plain(twin, drives, 12)
    store, blocks = _store_for(tmp_path, edges, name="refracwin")
    kt, pager = _paged(
        store,
        blocks,
        capacity_blocks=2,
        blocks_per_round=4,
        max_rounds=2,
        max_stale_ticks=0,
        pinned=frozenset({0}),
        pager_cls=_EvictionLog,
        max_delay=REFRAC_MAX_DELAY,
        refrac_ticks=REFRAC_TICKS,
    )
    arrays = kt.state.arrays
    trains, rings, at_release = [], [], None
    for t in range(12):
        if t == REFRAC_RELEASE_TICK:
            at_release = {
                "refrac": int(arrays["refrac"][REFRAC_UNIT]),
                "v": float(arrays["v"][REFRAC_UNIT]),
                "thr": float(arrays["thr"][REFRAC_UNIT]),
                "last_spike": int(arrays["last_spike"][REFRAC_UNIT]),
                "resident_block": 1 in pager._resident_blocks,
                "pressure": sum(int(np.abs(p[2 * BLOCK : 3 * BLOCK]).sum()) for p in kt.pending),
            }
        trains.append(pager.tick(drives.get(t)).tobytes())
        rings.append(_rings_of(kt))
    # the hole's precondition, asserted rather than assumed -- these hold on BOTH trees, and
    # without them the arm would be testing nothing
    assert at_release is not None and at_release["resident_block"], (
        f"block 1 was not resident when its refractory released: {at_release}"
    )
    assert at_release["refrac"] == 0, (
        f"the refractory had not been released at tick {REFRAC_RELEASE_TICK}: {at_release}"
    )
    assert at_release["v"] >= at_release["thr"], (
        f"the unit was not at threshold when room was needed: {at_release}"
    )
    assert at_release["last_spike"] < REFRAC_RELEASE_TICK - REFRAC_MAX_DELAY, (
        f"the ring-safety clause already excluded this block, so the arm cannot reach the "
        f"membrane clause it exists to test: {at_release}"
    )
    assert at_release["pressure"] > 0, (
        "block 2 was not due, so no room was ever needed and nothing was asked to leave core"
    )
    first_diff = next(
        (i for i, (a, b) in enumerate(zip(trains, twin_trains, strict=True)) if a != b), None
    )
    assert first_diff is None, (
        f"the paged run diverged from the resident twin at tick {first_diff}: the twin fired "
        f"{np.frombuffer(twin_trains[first_diff], dtype=np.int64).tolist()} and the paged run "
        f"fired {np.frombuffer(trains[first_diff], dtype=np.int64).tolist()} -- a unit at "
        f"threshold with its refractory released was demoted before propagate and its spike was "
        f"DROPPED, not deferred (evicted {pager.evicted})"
    )
    assert rings == twin_rings, "ring-slot key lists must match the twin tick for tick"
    assert np.array_equal(arrays["v"], twin.state.arrays["v"]), (
        "a dropped spike leaves the membrane unreset, so the arrays must be compared too"
    )
    assert (REFRAC_RELEASE_TICK, 1) not in pager.evicted, (
        f"block 1 was demoted on the very tick its unit was about to fire: {pager.evicted}"
    )
    assert pager.capacity_trips >= 1, (
        "holding a unit that is about to fire COSTS a capacity trip -- a run that stayed within "
        "capacity did not exercise the refusal at all"
    )


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
    # the width instrument's own control: a CORRECT pager must pass it, or its red means nothing
    assert pager.assert_round_widths() is True
    assert max(pager.round_widths()) <= pager.blocks_per_round


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
    # THE WIDTH RED the rounds pair cannot see: this arm's planner count and the store's
    # submissions AGREE (both 1), so the double-count is green on it. The width is not.
    assert blind.rounds_used == blind.store_rounds_used
    assert blind.round_widths() == [7], "the whole due set went out as one 7-block submission"
    with pytest.raises(RoundWidthExceeded):
        blind.assert_round_widths()
    assert store.blocks_read == sum(store.submission_widths), (
        "blocks_read must equal the recorded widths -- neither may be a write-only counter"
    )


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


def test_the_pager_refuses_a_grain_that_disagrees_with_its_stores_partition(tmp_path):
    """The one input every other check in the constructor is ABOUT was itself unchecked.

    `NeuronPager.__init__` validated capacity_blocks, blocks_per_round, max_rounds,
    max_stale_ticks, the residency mask and every pinned id against `blocks.n_blocks` -- and then
    accepted `blocks` itself unchecked. Under a mismatch `_page_in` compiles edges at the
    RECORD's grain while marking residency at the PAGER's, so it inserts bundles whose pre units
    the mask excludes and nothing raises: a caller silently addresses different units than the
    records describe. The store's own refusal is KEPT (see the block_size test above) -- these
    checks are ADDED, never moved."""
    edges = _chain_edges()
    store, blocks = _store_for(tmp_path, edges, name="grain")
    state = KernelState([("net", N)])
    state.arrays["thr"][:] = 1.0
    kt = KernelTick(
        state,
        Synapses.empty(N, P_PARTS, FRAC, MAX_DELAY),
        decay=0.5,
        v_reset=0.0,
        refrac_ticks=0,
        resident=np.zeros(N, dtype=bool),
    )
    base = dict(
        capacity_blocks=4,
        blocks_per_round=2,
        max_rounds=2,
        max_stale_ticks=1,
        pinned=frozenset({0}),
    )
    with pytest.raises(ValueError, match="addresses"):
        NeuronPager(kt, store, blocks=UnitBlocks(N, BLOCK * 2), **base)
    with pytest.raises(ValueError, match="multiple"):
        NeuronPager(kt, store, blocks=UnitBlocks(N, 64), **base)
    with pytest.raises(ValueError):  # the store's refusal is still the store's
        store.build(*edges, N, 64)
    # ACCEPTED: a REOPENED store has no partition of its own (`blocks is None`) and serves fetch
    # from the durable index. That is the path the pager takes when it joins the live tick at S6,
    # so refusing it would trade a silent defect for a loud one.
    reopened = UnitBlockStore(store.root)
    assert reopened.blocks is None
    pager = NeuronPager(kt, reopened, blocks=blocks, **base)
    assert pager.blocks.n_blocks == N_BLOCKS and pager.blocks.block_size == BLOCK
    reopened.close()


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
    from snn_data_lake.substrate.lake_engine import BlockAdoptRecord, is_block_adoption

    segments = sorted((Path(tmp_path) / "records" / "segments").glob("seg-*.lake"))
    # TWO SEALS SINCE THE ADOPTION FENCE (T067's fix), and the pin is stronger for it: the first
    # seals one 0x0C record per block, the second seals the ADOPT record that makes each of them
    # the block's CURRENT one. An ADOPT record can only name a seq the first seal reported, which
    # is why they cannot share a seal.
    assert len(segments) == 2, "one build seals its records, then the fact that they are adopted"
    path = segments[0]
    whole = list(read_segment(path))
    assert len(whole) == N_BLOCKS, "one record per block, including edge-less blocks"
    seq0 = int(path.stem.split("-")[1])
    for offset, payload in enumerate(whole):
        assert read_record(path, seq0 + offset) == payload
    for bad_seq in (seq0 - 1, seq0 + len(whole)):
        with pytest.raises(ValueError):
            read_record(path, bad_seq)

    fence = list(read_segment(segments[1]))
    assert len(fence) == N_BLOCKS and all(is_block_adoption(p) for p in fence)
    adopted = {}
    for payload in fence:
        record = BlockAdoptRecord.decode(payload)
        assert record.generation == 0 and record.folded_through == 0, "a build folds nothing"
        assert record.container == path.name
        adopted[record.block_id] = record.seq
    assert adopted == {bid: seq0 + bid for bid in range(N_BLOCKS)}


def test_the_lake_digest_is_identical_across_a_full_page_out_page_in_cycle(tmp_path):
    """The identity AND its red, through ONE digest function over a root.

    The identity is true BY CONSTRUCTION -- the pager performs no store write at all, so this arm
    passed unchanged with every write path replaced by raise-on-call, and passed printing
    IDENTICAL over ZERO BYTES when its glob was blinded. A criterion with no reachable red is
    what P8 forbids, so the byte-flip control and the empty-glob refusal live here beside it."""
    edges = _chain_edges()
    store, blocks = _store_for(tmp_path, edges, name="digest")
    before, n_files = segment_digest(store.root)
    assert n_files >= 1, "the digest must be taken over real files, not an empty glob"
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
    assert segment_digest(store.root) == (before, n_files), (
        "page-out DESTROYS NO RECORD -- the lake bytes must not move"
    )

    control = Path(tmp_path) / "digest-control"
    shutil.copytree(store.root / SEGMENT_DIR, control / SEGMENT_DIR)
    assert segment_digest(control) == (before, n_files), "the control copy is not the same bytes"
    victim = sorted((control / SEGMENT_DIR).glob("seg-*.lake"))[0]
    raw = bytearray(victim.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    victim.write_bytes(bytes(raw))
    assert segment_digest(control)[0] != before, (
        "one flipped byte did not move the digest -- the instrument cannot detect a lost record"
    )
    with pytest.raises(ValueError):
        segment_digest(Path(tmp_path) / "nowhere")  # sha256-of-nothing is never a lake identity


def test_the_bridge_and_the_rebuild_gate_stay_green_on_one_batch_per_record(tmp_path):
    edges = _chain_edges()
    store, _blocks = _store_for(tmp_path, edges, name="bridge")
    root = Path(tmp_path) / "bridge"
    bridged = list(iter_records(root))
    from snn_data_lake.substrate.lake_engine import is_block_adoption

    assert bridged == list(store.iter_payloads()), "the ONE forward reader must see every record"
    # ONE 0x0C RECORD AND ONE 0x0D ADOPT RECORD PER BLOCK since the adoption fence: the bridge
    # carries both, because a rebuild that could not see the ADOPT records could not derive a
    # single pointer.
    assert len(bridged) == 2 * N_BLOCKS
    assert sum(is_block_adoption(p) for p in bridged) == N_BLOCKS
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

"""Tests for TimeCellClock (cortex/time_cells.py): a ramping-to-threshold spiking time-cell
population that tiles an interval. Covers the three faculty claims -- (a) decode() tracks true
elapsed steps monotonically, (b) the decoded SD scales up with longer intervals (Weber/scalar
timing emerges from accumulator noise), (c) learn_interval() reduces interval-reproduction error
via reward-modulated STDP -- plus per-instance determinism.

Run: pytest tests/test_time_cells.py
"""

import math
from functools import cache

from snn_data_lake.cortex.time_cells import TimeCellClock


def test_decode_tracks_elapsed_time_monotonically():
    clock = TimeCellClock(n_cells=64, seed=0)
    estimates = [clock.tick()[0] for _ in range(50)]
    assert all(estimates[i + 1] >= estimates[i] for i in range(len(estimates) - 1))
    assert abs(estimates[-1] - 50) < 5.0


def test_decoded_sd_grows_with_duration_weber_law():
    clock = TimeCellClock(n_cells=64, seed=0)
    sd_short = sd_long = 0.0
    for step in range(1, 51):
        _est, sd = clock.tick()
        if step == 10:
            sd_short = sd
        if step == 50:
            sd_long = sd
    assert sd_long > sd_short


def test_learn_interval_reduces_reproduction_error():
    learner = TimeCellClock(n_cells=64, seed=1)
    target = 25.5  # off-grid, between two integer thresholds
    i = min(
        range(learner.n_cells),
        key=lambda idx: abs(learner.thresholds[idx] / learner.drive - target),
    )
    error_pre = abs(learner.thresholds[i] / learner.drive - target)
    for _ in range(30):
        i_cell, _delta = learner.learn_interval(target, reward=1.0)
    error_post = abs(learner.thresholds[i_cell] / learner.drive - target)
    assert error_post < error_pre


def test_same_seed_is_deterministic():
    a = [TimeCellClock(n_cells=32, seed=7).tick() for _ in range(20)]
    b = [TimeCellClock(n_cells=32, seed=7).tick() for _ in range(20)]
    assert a == b


# --------------------------------------------------------------------------------------------
# FR-012: ascending order is an invariant, not an accident. tick() walks the vector in index order
# and stops at the first uncrossed level, so ONE inversion strands every higher cell for the rest of
# the run. Measured before the fix: 9 of 12 seeds inverted somewhere over 4000 ticks.
# --------------------------------------------------------------------------------------------

_CHECKPOINTS = (200, 600, 1200, 4000)
_HORIZON = max(_CHECKPOINTS)
_SEEDS = 12


def _learn_unguarded(clock, target_steps, reward, lr=0.1, a_plus=1.0, a_minus=1.0, tau=10.0):
    """The RETIRED pre-fix rule, kept here as a CONTROL and nowhere else.

    Its only job is to prove the guard in `learn_interval` is load-bearing. Without it
    `test_learn_interval_preserves_ascending_order` is a green light that cannot go red -- it would
    pass identically if the guard were deleted, which is the exact failure Standing Invariant 5
    names. This is a test fixture, not a second live rule."""
    target_steps = float(target_steps)
    i = min(
        range(clock.n_cells),
        key=lambda idx: abs(clock.thresholds[idx] / clock.drive - target_steps),
    )
    t_pre = float(clock.spike_tick[i]) if clock.fired[i] else clock.thresholds[i] / clock.drive
    dt = target_steps - t_pre
    if dt > 0:
        e = a_plus * math.exp(-dt / tau)
    elif dt < 0:
        e = -a_minus * math.exp(dt / tau)
    else:
        e = 0.0
    clock.thresholds[i] = max(0.5, clock.thresholds[i] + lr * reward * e)


def _is_ascending(vec):
    return all(vec[i] < vec[i + 1] for i in range(len(vec) - 1))


@cache
def _sweep(guarded, reset_every, ticks=_HORIZON, teach_every=7):
    """Drive 12 clocks the way brain.py drives one: tick, teach on a periodic 'reward', and -- in
    the reset-driven regime -- restart the interval on a periodic 'salient stimulus'.

    Returns (seeds_inverted, moved_counts). Cached because the same four sweeps answer both the
    invariant question and the coverage question, and 4000 ticks x 12 seeds is not free."""
    inverted, moved = [], []
    for seed in range(_SEEDS):
        clock = TimeCellClock(n_cells=64, seed=seed)
        pristine = list(clock.thresholds)
        est = 0.0
        for step in range(1, ticks + 1):
            if reset_every and step % reset_every == 0:
                clock.reset()
            est, _sd = clock.tick()
            if step % teach_every == 0:
                target = max(1, int(round(est)))
                if guarded:
                    clock.learn_interval(target, reward=1.0)
                else:
                    _learn_unguarded(clock, target, reward=1.0)
            if step in _CHECKPOINTS and not _is_ascending(clock.thresholds):
                inverted.append((seed, step))
        moved.append(sum(1 for a, b in zip(pristine, clock.thresholds, strict=True) if a != b))
    return inverted, moved


def test_learn_interval_preserves_ascending_order():
    """FR-012, with its own control. 12 seeds, 4000 ticks, checked at 200/600/1200/4000.

    The saturating regime is where the defect actually bites: MEASURED 5 of 12 seeds invert under
    the retired rule. The control arm asserts that is still true, so a green invariant arm means the
    guard worked rather than that the regime stopped exercising it."""
    control, _ = _sweep(guarded=False, reset_every=None)
    assert control, (
        "the control arm did not invert -- this test can no longer tell a working guard from a "
        "regime that never provokes one, and proves nothing until the regime is repaired"
    )
    for reset_every in (60, None):
        inverted, _ = _sweep(guarded=True, reset_every=reset_every)
        assert not inverted, f"ascending order broken at {inverted} (reset_every={reset_every})"


def test_learn_interval_still_learns_after_order_fix():
    """Refusal must not buy the invariant by learning nothing (SI-2).

    MEASURED: reset-driven coverage is IDENTICAL either way (36.58 of 64 thresholds move, per-seed
    equal); the saturating regime loses 0.33 of 64 on average, which is the refusals themselves and
    nothing more."""
    _, guarded_reset = _sweep(guarded=True, reset_every=60)
    _, control_reset = _sweep(guarded=False, reset_every=60)
    assert guarded_reset == control_reset, (
        f"reset-driven learning coverage changed: {guarded_reset} vs control {control_reset}"
    )
    _, guarded_sat = _sweep(guarded=True, reset_every=None)
    _, control_sat = _sweep(guarded=False, reset_every=None)
    lost = sum(control_sat) - sum(guarded_sat)
    assert lost <= _SEEDS, (
        f"saturating coverage collapsed: {sum(guarded_sat)} moved vs control {sum(control_sat)} "
        f"-- more than one refused threshold per seed"
    )


def test_degenerate_ladder_control():
    """STANDING INVARIANT 5, and the admission gate for any readout this module ever adopts.

    A ladder of 64 levels spanning 6.3e-5 carries no timing information at all. Run the module's
    own accuracy assertion (`final_err < 5.0`, the one main() makes): the assertion MUST FAIL.
    A readout that passes this control is not reading its population -- it is reading something else
    that happens to correlate with elapsed time -- and is disqualified however good its RMSE looks.
    This control disqualified the spike-tick-anchor candidate during Phase 0, which scored err 0.995
    here where the real ladder scores 0.486."""
    clock = TimeCellClock(n_cells=64, seed=0)
    clock.thresholds = [1.0 + 1e-6 * i for i in range(64)]
    est = 0.0
    for _ in range(50):
        est, _sd = clock.tick()
    final_err = abs(est - 50)
    assert final_err >= 5.0, (
        f"degenerate-ladder control PASSED (err {final_err:.4f} < 5.0) -- the readout is not "
        f"reading its population and is disqualified"
    )


# --------------------------------------------------------------------------------------------
# FR-005 / US2: the readout must read the POPULATION. At HEAD it read the shared ramp `v`, so
# perturbing a threshold changed nothing and RMSE was bit-identical at widths 64/128/256/512
# (0.7276 flat, 16 seeds, horizon 60) -- growth was not merely unmeasured but provably worthless.
# --------------------------------------------------------------------------------------------


def _estimates(clock, ticks):
    return [clock.tick()[0] for _ in range(ticks)]


def test_decode_depends_on_thresholds():
    """SC-002. Perturb one interior threshold; the consumed estimate sequence must move."""
    plain = _estimates(TimeCellClock(n_cells=64, seed=0), 60)
    poked = TimeCellClock(n_cells=64, seed=0)
    poked.thresholds[30] += 0.4
    assert _estimates(poked, 60) != plain


def _rmse_vs_true(n_cells, seeds=16, horizon=60):
    total = 0.0
    for seed in range(seeds):
        clock = TimeCellClock(n_cells=n_cells, seed=seed)
        for step in range(1, horizon + 1):
            est, _sd = clock.tick()
            total += (est - step) ** 2
    return math.sqrt(total / (seeds * horizon))


def test_decode_rmse_falls_with_width():
    """C2, the criterion the old readout failed absolutely. A flat curve means growth cannot
    register at any width."""
    wide, narrow = _rmse_vs_true(512), _rmse_vs_true(64)
    assert wide < narrow, f"RMSE did not fall with width: {narrow:.4f} at 64 vs {wide:.4f} at 512"


def test_decode_divides_by_drive():
    """The latent unit bug: `learn_interval` divides by `drive`, `decode()` did not, so at drive=0.5
    the clock reported 19.34 ramp units for a true elapsed of 40 ticks. Noise is off here so the
    remaining error is the readout's own bracket width, judged against the module's own 5.0 bar."""
    clock = TimeCellClock(n_cells=64, seed=0, drive=0.5, noise_sd=0.0)
    est = 0.0
    for _ in range(40):
        est, _sd = clock.tick()
    assert abs(est - 40) < 5.0, f"drive-scaled elapsed misreported as {est:.4f} (true 40)"


# --------------------------------------------------------------------------------------------
# FR-001 / US1: growth must be non-destructive, refusable, and invisible to the RNG. The learned
# state IS the threshold vector, so "grew" and "kept what it learned" are the same assertion.
# --------------------------------------------------------------------------------------------


def _taught_clock(seed=0, ticks=300):
    clock = TimeCellClock(n_cells=64, seed=seed)
    est = 0.0
    for step in range(1, ticks + 1):
        if step % 60 == 0:
            clock.reset()
        est, _sd = clock.tick()
        if step % 7 == 0:
            clock.learn_interval(max(1, int(round(est))), reward=1.0)
    clock.reset()
    return clock


def test_growth_preserves_learned_thresholds():
    """FR-001, SC-001. Bit-identical, not 'close' -- rounded equality would hide a rebuild."""
    clock = _taught_clock()
    snapshot = list(clock.thresholds)
    for expected_width in (65, 66):
        added = clock.grow("interior")
        assert added is not None
        assert clock.n_cells == expected_width
        survivors = [t for t in clock.thresholds if t in snapshot]
        assert len(survivors) >= len(snapshot)
        for old in snapshot:
            assert old in clock.thresholds, f"learned threshold {old!r} did not survive growth"


def test_growth_is_rng_neutral():
    """What keeps test_same_seed_is_deterministic green once the brain grows itself."""
    clock = _taught_clock()
    before = clock.rng.getstate()
    clock.grow("interior")
    assert clock.rng.getstate() == before


def test_growth_refuses():
    """FR-004. A refusal changes nothing and is counted."""
    clock = _taught_clock()

    mid_interval = _taught_clock()
    mid_interval.tick()  # no longer at a reset boundary
    snapshot = list(mid_interval.thresholds)
    assert mid_interval.grow("interior") is None
    assert mid_interval.thresholds == snapshot
    assert mid_interval.n_cells == 64
    assert mid_interval.refused == 1

    inverted = _taught_clock()
    inverted.thresholds[30], inverted.thresholds[31] = (
        inverted.thresholds[31],
        inverted.thresholds[30],
    )
    snapshot = list(inverted.thresholds)
    assert inverted.grow("interior") is None
    assert inverted.thresholds == snapshot
    assert inverted.refused == 1

    assert clock.grow("no-such-mode") is None
    assert clock.refused == 1
    assert clock.n_cells == 64

    # A prune cannot empty the population because there is no prune: the class exposes no API that
    # removes a cell, so the n_cells >= 1 invariant cannot be violated from outside. Asserted rather
    # than assumed, so adding one later trips this test instead of shipping silently.
    assert not [n for n in dir(clock) if "prune" in n or "shrink" in n]


def test_growth_maintains_length_coherence():
    clock = _taught_clock()
    for _ in range(3):
        clock.grow("interior")
    assert len(clock.thresholds) == len(clock.fired) == len(clock.spike_tick) == clock.n_cells == 67


def test_growth_extend_mode_appends_at_top():
    """The clamp-widening mode. Retained for saturation only; its benefit is never 'earning'."""
    clock = _taught_clock()
    top = clock.thresholds[-1]
    added = clock.grow("extend")
    assert added == 65.0
    assert clock.thresholds[-1] == 65.0 > top
    assert clock.n_cells == 65


def test_growth_failure_is_distinct_from_refusal():
    """FR-013. An exception on the growth path must surface as its own outcome, never as a
    refusal -- a refusal is normal and burying a defect inside a normal outcome hides it."""
    clock = _taught_clock()

    class _Exploding(list):
        def insert(self, *_a, **_k):
            raise RuntimeError("insertion blew up")

    clock.thresholds = _Exploding(clock.thresholds)
    try:
        clock.grow("interior")
    except RuntimeError:
        pass
    else:
        raise AssertionError("the failure was swallowed")
    assert clock.failed == 1
    assert clock.refused == 0
    assert clock.grown == 0


# --------------------------------------------------------------------------------------------
# FR-007 / US4: growth that does not survive a restart is not growth, and a restore that half-
# succeeds is worse than one that refuses.
# --------------------------------------------------------------------------------------------


def _isolated_home(tmp_path, monkeypatch):
    """`home()` re-reads NORD_HOME on every call and caches nothing, so setting the env var is the
    whole isolation. Asserted rather than trusted: a state test that silently wrote to the
    operator's live ~/.snn_data_lake would be worse than no test."""
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    from snn_data_lake.substrate import home

    assert home.home() == str(tmp_path)
    return home


def test_persist_restore_round_trip(tmp_path, monkeypatch):
    """SC-006. Width and every threshold bit-identical across a save/restore."""
    _isolated_home(tmp_path, monkeypatch)
    grown = _taught_clock()
    grown.grow("interior")
    grown.grow("interior")
    grown.save()

    fresh = TimeCellClock(n_cells=64, seed=0)
    restored, why = fresh.restored()
    assert why == "", why
    assert restored is not fresh
    assert restored.n_cells == 66
    assert restored.thresholds == grown.thresholds
    assert len(restored.fired) == len(restored.spike_tick) == 66


def test_restore_refuses_foreign_lineage(tmp_path, monkeypatch):
    """FR-007, FR-013. Four ways a blob is not this clock's state, each refused and RECORDED.

    Lineage matters because `learn_interval` takes `t_pre` from `spike_tick` under a particular
    RNG's noise: a vector learned under another seed is another lineage's state, and adopting it
    would look exactly like a successful restore."""
    home = _isolated_home(tmp_path, monkeypatch)
    good = _taught_clock()
    good.grow("interior")
    good.save()
    blob = home.load_state("time_cells.json")

    cases = {
        "fmt": {**blob, "fmt": "time_cells/v0"},
        "length": {**blob, "n_cells": blob["n_cells"] + 1},
        "seed": {**blob, "seed": 999},
        "drive": {**blob, "drive": 0.5},
        "noise": {**blob, "noise_sd": 0.99},
        "order": {**blob, "thresholds": sorted(blob["thresholds"], reverse=True)},
        "empty": {**blob, "n_cells": 0, "thresholds": []},
    }
    for label, bad in cases.items():
        home.save_state("time_cells.json", bad)
        fresh = TimeCellClock(n_cells=64, seed=0)
        restored, why = fresh.restored()
        assert restored is fresh, f"{label}: a bad blob was adopted"
        assert why, f"{label}: refused without recording a reason"
        assert restored.restore_refused == 1, f"{label}: refusal not counted"
        assert restored.n_cells == 64


def test_grow_if_earned_halts():
    """SC-007, and the gate must be able to go RED.

    Held-out improvement alone cannot halt anything: on a noise-free tiling metric every split of
    the widest gap lowers held-out error, so the first version of this rule grew the live population
    64 -> 104 in 600 ticks with 0 refusals. The binding bar is the accumulator's own first-passage
    dispersion -- a bracket finer than that cannot be resolved however finely it is tiled. This test
    asserts the halt actually arrives, because a bound that is never reached is not a bound."""
    clock = _taught_clock()
    grew = 0
    for _ in range(200):
        if clock.grow_if_earned() is None:
            break
        grew += 1
    else:
        raise AssertionError("growth never halted in 200 attempts -- the bound does not bind")
    assert grew >= 1, "the first insertion should have earned its place"
    assert clock.grown == grew
    assert clock.refused >= 1
    # and it stays halted -- a halt that un-halts on its own is not a halt
    assert clock.grow_if_earned() is None
    assert clock.refused >= 2


def test_would_earn_leaves_no_trace():
    """A candidate that does not earn must cost nothing -- there is no prune to undo a bad one."""
    clock = _taught_clock()
    snapshot, width = list(clock.thresholds), clock.n_cells
    delta = clock.would_earn("interior")
    assert delta is not None
    assert clock.thresholds == snapshot and clock.n_cells == width
    assert clock.grown == 0 and clock.refused == 0


def _live_time_cells(tmp_path, monkeypatch, ticks=300, **env):
    """Tick a real brain under a given environment and return its time-cell state."""
    _isolated_home(tmp_path, monkeypatch)
    for key, val in env.items():
        if val is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, val)
    from snn_data_lake.cortex.brain import make_brain

    brain = make_brain(seed=0, persist=True)
    try:
        for _ in range(ticks):
            brain.tick()
        return brain.state()
    finally:
        brain.close()


def test_growth_is_on_by_default(tmp_path, monkeypatch):
    """CHANGED 2026-08-02: growth is DEFAULT-ON. Pinned so it cannot silently revert to opt-in.

    It was opt-in for one measured reason -- an unconditional RESTORE made brain construction
    non-deterministic in ~42% of trials -- and that reason belongs to the restore, which is still
    opt-in behind its own flag. Growth itself draws no random numbers and happens only at a reset
    boundary the brain reaches on its own."""
    st = _live_time_cells(tmp_path, monkeypatch, NORD_TIME_CELLS_GROW=None)
    assert st["time_cells_width"] > 64, "growth did not run at the default"
    assert st["time_cells_grown"] >= 1
    assert st["time_cells_failed"] == 0


def test_off_switch_leaves_the_population_alone(tmp_path, monkeypatch):
    """FR-016, SC-009. `NORD_TIME_CELLS_GROW=0` must stop growth completely.

    The switch survived the default flip; only its polarity changed, and an off-switch nobody tests
    is not an off-switch. The readout replacement and the learn_interval order fix are deliberately
    NOT behind it -- they are defect fixes owed regardless, and they do change live behaviour."""
    st = _live_time_cells(tmp_path, monkeypatch, NORD_TIME_CELLS_GROW="0")
    assert st["time_cells_width"] == 64, "the off-switch did not stop growth"
    assert st["time_cells_grown"] == 0
    assert st["time_cells_refused"] == 0
    assert st["time_cells_failed"] == 0


def test_restore_stays_opt_in(tmp_path, monkeypatch):
    """The half that broke determinism stays behind its OWN flag. With growth on but
    NORD_TIME_CELLS_RESTORE unset, a state blob on disk must be ignored: construction is a function
    of the seed, not of what the previous process left behind."""
    home = _isolated_home(tmp_path, monkeypatch)
    grown = _taught_clock()
    grown.grow("interior")
    grown.save()
    assert home.load_state("time_cells.json")["n_cells"] == 65

    st = _live_time_cells(
        tmp_path, monkeypatch, ticks=5, NORD_TIME_CELLS_GROW="0", NORD_TIME_CELLS_RESTORE=None
    )
    assert st["time_cells_width"] == 64, "a persisted width was adopted without the restore flag"
    assert st["time_cells_restore_refused"] == 0, "restore ran at all -- it must not have"

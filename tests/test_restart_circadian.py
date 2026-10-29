"""Finding (iii) (launcher D-77, board #6807): a restored brain must resume its clock where an
uninterrupted twin stands, not start a fresh day (`Brain.tick` derives sleep/wake from `_step` and
the clock, which a restart reset). The clock phase rides the named record `circadian_clock.json`
(_T_STATE). Private homes under tmp_path only; never the live lake. Dev seed 15000
(r2-typed-replay-hydration.md 2.6).

Panel lens A BLOCK-1 (#6910): seed 15000's fresh clock starts in subjective night (phase 0.8209), so
a single "asleep" bit after a cut inside the FIRST night cannot tell a restored brain from a fresh
one. The cut is therefore MID-DAY after the first night (a fresh brain is asleep there, the twin is
awake), the restored STATE is asserted (`_step`, clock phase, jitter rng), and a full day period of
ticks is compared with the twin's. A stdlib simulation of CircadianClock (round-2 REPORT) gives a
first divergence at tick 0 on the unfixed code, and none over the period with the fix."""

import pytest

SEED = 15000
WAKE_TICKS_INTO_DAY = 3


def _past_the_first_night(brain):
    """Tick through the first night into the next day, `WAKE_TICKS_INTO_DAY` wake ticks past dawn;
    returns the ticks taken."""
    ticks, slept, woke = 0, False, 0
    while woke < WAKE_TICKS_INTO_DAY:
        ticks += 1
        if ticks > 4 * brain.day_period:
            pytest.fail("no night and dawn within four periods")
        if brain.tick() == "asleep":
            slept = True
        elif slept:
            woke += 1
    return ticks


def _clock_state(brain):
    return brain._step, brain.clock.phase, brain.clock.rng.getstate()


def test_a_restored_brain_resumes_its_twins_clock(tmp_path):
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    with home.throwaway_home(parent=tmp_path):
        home.require_isolated_home()
        twin = make_brain(seed=SEED, persist=True, device="cpu")
        try:
            ticks = _past_the_first_night(twin)
            at_cut = _clock_state(twin)
            period = twin.day_period
            expected = [twin.tick() for _ in range(period)]
        finally:
            twin.close()
    with home.throwaway_home(parent=tmp_path):
        home.require_isolated_home()
        cut = make_brain(seed=SEED, persist=True, device="cpu")
        try:
            for _ in range(ticks):
                cut.tick()
        finally:
            cut.close()
        restored = make_brain(seed=SEED, persist=True, device="cpu")
        try:
            assert _clock_state(restored) == at_cut, "restored clock is not the twin's at the cut"
            assert [restored.tick() for _ in range(period)] == expected, (
                "the restored brain's day diverged from the twin's"
            )
        finally:
            restored.close()

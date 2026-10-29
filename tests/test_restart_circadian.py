"""Finding (iii) (launcher D-77, board #6807): a brain restored mid-night must resume the night, as
an uninterrupted twin does. Today a restoring process starts a fresh clock (`Brain.tick` derives
sleep/wake from `_step`, which a restart resets), so this is a red-first strict xfail on
AssertionError and is NOT fixed here: fix-or-defer is the launcher's call. Private homes under
tmp_path only; never the live lake. Dev seed 15000 (r2-typed-replay-hydration.md 2.6)."""

import pytest

SEED = 15000


def _into_the_night(brain, sleeps=1):
    """Tick until `sleeps` sleep ticks have run; returns the ticks taken."""
    ticks, done = 0, 0
    while done < sleeps:
        ticks += 1
        if ticks > 4 * brain.day_period:
            pytest.fail("no night within four periods")
        if brain.tick() == "asleep":
            done += 1
    return ticks


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="finding (iii): restart forgets circadian phase (D-77)",
)
def test_a_brain_restored_mid_night_resumes_the_night(tmp_path):
    from snn_data_lake.engine.brain import make_brain
    from snn_data_lake.substrate import home

    with home.throwaway_home(parent=tmp_path):
        home.require_isolated_home()
        twin = make_brain(seed=SEED, persist=True, device="cpu")
        try:
            ticks = _into_the_night(twin)
            expected = twin.tick()
        finally:
            twin.close()
    if expected != "asleep":  # a precondition, not the finding: never read as the xfail
        pytest.fail("the twin's night ended after one sleep tick, so the cut is not mid-night")
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
            assert restored.tick() == expected, "the restored brain did not resume its night"
        finally:
            restored.close()

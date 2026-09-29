"""The brain may re-arm itself after an emergency stop -- but ONLY while the breach reads clear.

Owner-authorised 2026-08-03. Two things were granted together and only one of them is safe alone:
the brain may clear its own E-STOP latch, and a local HTTP client may cause arm motion. Granting the
first WITHOUT the breach condition would turn `brainstem/reflex.py`'s persistence latch -- a
consecutive-breach counter that exists precisely so a sustained moderate overload can never be
ignored -- into a no-op, because a brain that can clear unconditionally can clear while still in
breach. That is what a runaway looks like from the inside.

The safety argument is made by CONTROL FLOW, not by a second reading. `AutonomousArm.step()`
early-returns on the breach predicate (`distance < CONTACT`, or non-finite), so reaching the
self-clear at section 1b MEANS this sample was clear. Re-checking the same value would be a second
opinion about one sample, which is not a second opinion.

The other half of the authorisation is observability: every clear is counted, and the count is
exported as `estop_self_clears` beside `arm_stopped`. The pair is the diagnostic: `arm_stopped`
False with a flat count is a calm arm; False with a CLIMBING count is an arm tripping and
re-arming every few ticks. From a status light those are identical, which is why the counter exists.

Run: pytest tests/test_estop_self_clear.py
"""

import os

import pytest

from snn_data_lake.engine.autonomous_loop import CONTACT, AutonomousArm

FAR = CONTACT * 3.0  # comfortably outside contact -- the breach predicate reads clear
NEAR = CONTACT * 0.5  # inside contact -- the breach predicate trips


@pytest.fixture(autouse=True)
def _enable_self_clear(monkeypatch):
    """The capability is OPT-IN and default-off. Ten "the E-STOP freeze must hold" assertions across
    autonomous_loop, brain and live_brain encode "a latched arm stays latched across later ticks" as
    a design invariant, and every one of them ticks a SAFE context afterwards -- so enabling this by
    default would have required rewriting ten safety assertions, which is silencing checks at scale.
    The gates keep their invariant; this suite turns the flag on explicitly."""
    monkeypatch.setenv("NORD_ESTOP_SELF_CLEAR", "1")


def _arm():
    return AutonomousArm(seed=0)


def test_breach_trips_the_latch():
    arm = _arm()
    out = arm.step(0.5, 0.5, 0.5, distance=NEAR)
    assert out["action"] == "EMERGENCY_STOP"
    assert arm.arm.state()["stopped"] is True
    assert arm._estop_clears == 0, "a trip is not a clear"


def test_will_not_clear_while_still_in_breach():
    """The load-bearing case. A latched arm held in breach must STAY latched."""
    arm = _arm()
    for _ in range(6):
        arm.step(0.5, 0.5, 0.5, distance=NEAR)
    assert arm.arm.state()["stopped"] is True, "latch released while the breach was still present"
    assert arm._estop_clears == 0


def test_clears_once_the_breach_ends():
    arm = _arm()
    arm.step(0.5, 0.5, 0.5, distance=NEAR)
    arm.step(0.5, 0.5, 0.5, distance=FAR)
    assert arm.arm.state()["stopped"] is False, "did not re-arm after the breach cleared"
    assert arm._estop_clears == 1


def test_does_not_count_clears_on_an_unlatched_arm():
    """Without this the counter would climb every tick and mean nothing."""
    arm = _arm()
    arm.step(0.5, 0.5, 0.5, distance=NEAR)
    for _ in range(6):
        arm.step(0.5, 0.5, 0.5, distance=FAR)
    assert arm._estop_clears == 1


def test_sensor_fault_trips_and_does_not_self_clear():
    """A non-finite reading is a SENSOR FAULT, not a distance. It must fail closed: reflex.py scores
    it as infinite severity, and a fault must never be a route to re-arming."""
    arm = _arm()
    arm.step(0.5, 0.5, 0.5, distance=float("nan"))
    assert arm.arm.state()["stopped"] is True
    assert arm._estop_clears == 0


def test_a_trip_clear_loop_is_countable_not_silent():
    """The condition the authorisation was granted on: a self-clearing LOOP must be visible."""
    arm = _arm()
    for _ in range(6):
        arm.step(0.5, 0.5, 0.5, distance=NEAR)
        arm.step(0.5, 0.5, 0.5, distance=FAR)
    assert arm._estop_clears == 6, "six trip/clear cycles must read as six, not as a quiet arm"


def test_counter_is_exported_from_brain_state():
    """A count nothing can read is not observability. Exported beside `arm_stopped` on purpose."""
    from snn_data_lake.engine.brain import make_brain

    brain = make_brain(seed=0, persist=False)
    try:
        brain.tick()
        state = brain.state()
        assert "estop_self_clears" in state
        assert isinstance(state["estop_self_clears"], int)
        assert "arm_stopped" in state, "the counter is only meaningful next to the latch state"
    finally:
        brain.close()


def test_the_self_clear_never_re_arms_a_restored_latch():
    """T037a (OD-4): a latch this process did NOT trip -- restored from the sidecar at boot --
    is cleared only by a deliberate reset (the cockpit's HOME, `deliberate=True`). The
    automated self-clear refuses it, counts the refusal, and does not count a clear."""
    import json

    from snn_data_lake.brainstem.so101_arm import ESTOP_LATCH
    from snn_data_lake.substrate.home import home

    with open(os.path.join(home(), ESTOP_LATCH), "w", encoding="utf-8") as f:
        json.dump(
            {
                "token": "another-process",
                "pid": 1,
                "utc": "2026-08-21T00:00:00Z",
                "source": "EMERGENCY_STOP",
            },
            f,
        )
    arm = _arm()
    assert arm.arm.state()["latch_source"] == "restored"
    for _ in range(4):
        arm.step(0.5, 0.5, 0.5, distance=FAR)
    assert arm.arm.state()["stopped"] is True, "the automated self-clear re-armed a restored latch"
    assert arm._estop_clears == 0, "a refused clear was counted as a clear"
    assert arm.arm.state()["latch_refusals"] >= 1, "the refusal was silent"
    refusals = arm.arm.state()["latch_refusals"]
    arm.apply_observed("HOME", deliberate=True)  # the cockpit's reset
    assert arm.arm.state()["stopped"] is False and arm.arm.state()["latch_source"] is None
    arm.step(0.5, 0.5, 0.5, distance=FAR)
    assert arm.arm.state()["latch_refusals"] == refusals, "a cleared arm kept refusing"

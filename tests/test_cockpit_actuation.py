"""A local HTTP client may move the arm -- owner-authorised 2026-08-03. These are its guardrails.

This is the only actuation surface in the repo that is not the brain's own loop, so the properties
below are not style preferences; they are the reason the surface is defensible at all. Each is
asserted rather than described.

  1. ALLOW-LIST ONLY, and refused at the FRONT DOOR. `is_valid_command` is the same wire allow-list
     serial_arm and nl_interface use. A rejected string never reaches the queue, so it can never
     reach the arm -- and the operator gets a 400 instead of a ticket that silently never happens.
  2. THE REQUEST THREAD NEVER TOUCHES THE ARM. Submitting must not move anything; only `pump()` on
     the tick thread does. MuJoCo's `mj_forward` racing the tick's `mj_step` on one `mjData` is
     undefined behaviour -- garbage or a crash, not a stale read.
  3. THE GATE STILL HOLDS. A latched arm accepts the ticket and does not move, and the result record
     SAYS the command was swallowed. "Your command ran" and "your command was ignored" must not look
     the same to whoever sent it.
  4. TRIPPING THE STOP IS PART OF THE GRANT. A client can send EMERGENCY_STOP, which is how the
     authorised "trip the E-STOP" is delivered -- through the existing gate, rather than by making
     `distance` world-sourced. `_safe_ctx` pins `distance` precisely so the reflex stays
     byte-identical, and that pin is untouched here.

Run: pytest tests/test_cockpit_actuation.py
"""

import pytest

from snn_data_lake.cortex.brain import make_brain
from snn_data_lake.substrate.cockpit import Acts

#: Strings that must never reach the arm. `HOME; EMERGENCY_STOP` is a concatenation attempt and
#: `JOINT_GRIPPER_ROTATE(+30deg)` is the token so101_arm's own history records as once accepted by
#: the gate while the vocabulary said it did not exist -- the exact shape of bug this guards.
REJECTED = (
    "rm -rf /",
    "JOINT_GRIPPER_ROTATE(+30deg)",
    "MOVE_ANYWHERE",
    "",
    "HOME; EMERGENCY_STOP",
    "EMERGENCY_STOP\nHOME",
)


@pytest.fixture
def brain():
    b = make_brain(seed=0, persist=False)
    yield b
    b.close()


def test_non_allow_listed_never_reaches_the_queue():
    acts = Acts()
    for bad in REJECTED:
        tid, why = acts.submit(bad)
        assert tid is None, f"{bad!r} was queued; it must be refused at the front door"
        assert why, "a refusal must say why"
    assert acts.pending() == 0, "a refused command must not occupy the queue"


def test_submitting_does_not_move_the_arm(brain):
    """Property 2. If this ever fails, an HTTP thread is touching mjData and the failure mode is
    undefined behaviour rather than a wrong number."""
    arm = brain.brain.arm.arm
    before = list(arm.state()["theta_deg"])
    acts = Acts()
    tid, why = acts.submit("JOINT_SHOULDER_PAN_ROTATE(+5deg)")
    assert tid is not None, why
    after_submit = arm.state()["theta_deg"]
    assert all(abs(a - b) < 1e-9 for a, b in zip(before, after_submit, strict=True)), (
        "the arm moved on SUBMIT -- the request thread must never actuate"
    )
    acts.pump(brain, budget=4)
    after_pump = arm.state()["theta_deg"]
    assert any(abs(a - b) > 1e-9 for a, b in zip(before, after_pump, strict=True)), (
        "the arm did not move after pump on the tick thread"
    )


def test_latched_arm_accepts_the_ticket_and_does_not_move(brain):
    arm = brain.brain.arm.arm
    arm.apply("EMERGENCY_STOP")
    before = list(arm.state()["theta_deg"])
    acts = Acts()
    tid, _ = acts.submit("JOINT_SHOULDER_PAN_ROTATE(+20deg)")
    acts.pump(brain, budget=4)
    after = arm.state()["theta_deg"]
    assert all(abs(a - b) < 1e-9 for a, b in zip(before, after, strict=True)), (
        "a latched arm moved for an HTTP command"
    )
    rec = acts.result(tid)
    assert rec["stopped"] is True
    assert "latched" in rec["note"], "a swallowed command must say it was swallowed"


def test_client_can_trip_the_estop(brain):
    """The authorised half that does NOT require re-plumbing the reflex's sensor channel."""
    arm = brain.brain.arm.arm
    arm.apply("HOME")
    assert arm.state()["stopped"] is False
    acts = Acts()
    tid, _ = acts.submit("EMERGENCY_STOP")
    assert tid is not None, "EMERGENCY_STOP must be an allow-listed command"
    acts.pump(brain, budget=4)
    assert arm.state()["stopped"] is True, "a queued EMERGENCY_STOP must latch the arm"


def test_pump_is_bounded(brain):
    """A client must not be able to starve the tick loop by flooding the queue."""
    acts = Acts()
    for _ in range(10):
        acts.submit("HOLD")
    assert acts.pump(brain, budget=1) == 1, "pump drained more than its budget"
    assert acts.pending() == 9, "the rest must stay queued for later ticks"

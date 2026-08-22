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

import json
import os
import urllib.error
import urllib.request

import pytest

from snn_data_lake.cortex.brain import make_brain
from snn_data_lake.substrate.cockpit import Acts, serve

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


def test_serve_forwards_the_queue_so_the_buttons_actually_reach_it(brain):
    """5. THE WIRE FROM THE PAGE TO THE QUEUE EXISTS -- through serve(), not around it.

    Every other test in this file constructs `Acts()` and calls it directly, which proves the
    mechanism and says NOTHING about production. `serve()` took `acts` and did not forward it to
    `_handler`, so `/api/act` answered 404 "actuation not enabled" for every request the real
    cockpit made -- including EMERGENCY_STOP -- while all four tests above stayed green and
    `learn_daemon.py` correctly passed a queue in. Presence is not participation: this asserts the
    HTTP path a browser button actually takes.
    """
    acts = Acts()
    srv = serve(brain, port=0, acts=acts, block=False)
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/api/act"
        req = urllib.request.Request(
            url, data=json.dumps({"command": "EMERGENCY_STOP"}).encode(), method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
        assert body.get("queued") is True, f"/api/act refused the command: {body}"
        assert body.get("id") is not None, "a queued command must return a ticket"
        assert acts.pending() == 1, "the command reached serve() but never reached the queue"
    finally:
        srv.shutdown()
        srv.server_close()


def test_pump_is_bounded(brain):
    """A client must not be able to starve the tick loop by flooding the queue."""
    acts = Acts()
    for _ in range(10):
        acts.submit("HOLD")
    assert acts.pump(brain, budget=1) == 1, "pump drained more than its budget"
    assert acts.pending() == 9, "the rest must stay queued for later ticks"


def test_a_restored_latch_is_disclosed_and_only_the_cockpit_reset_clears_it():
    """T037a (OD-4). Property 5: a latched arm is never silent. The brain's arm, built after a
    previous process left the sidecar, reports `latch_source == "restored"` -- the snapshot
    copies `state()` wholesale into `body.*`, which is what the cockpit's E-STOP row reads --
    and the cockpit's HOME (a human's deliberate reset, `deliberate=True`) is what clears it."""
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
    b = make_brain(seed=0, persist=False)
    try:
        st = b.brain.arm.arm.state()
        assert st["stopped"] is True and st["latch_source"] == "restored", st
        from snn_data_lake.substrate.brain_telemetry import snapshot, verify_contract

        snap = snapshot(b)
        assert snap["body"]["latch_source"] == "restored", "the cockpit's row reads body.*"
        assert not [k for k in verify_contract(snap) if k.startswith("body.")], verify_contract(
            snap
        )
        acts = Acts()
        tid, why = acts.submit("HOME")
        assert tid is not None, why
        acts.pump(b, budget=4)
        after = b.brain.arm.arm.state()
        assert after["stopped"] is False and after["latch_source"] is None, after
        assert acts.applied == 1
    finally:
        b.close()


def test_the_chat_route_is_the_other_deliberate_reset_and_passes_the_tap():
    """T037a: the NL firewall used the RAW gate (invisible to the efference copy and to the
    latch policy). It now passes apply_observed(deliberate=True): a human typing 'go home' at
    the cockpit chat IS a deliberate reset, and the reply says a restored stop was reset."""
    from snn_data_lake.brainstem.so101_arm import ESTOP_LATCH
    from snn_data_lake.substrate.home import home

    with open(os.path.join(home(), ESTOP_LATCH), "w", encoding="utf-8") as f:
        json.dump({"token": "another-process", "pid": 1, "utc": "x", "source": "EMERGENCY_STOP"}, f)
    b = make_brain(seed=0, persist=False)
    try:
        assert b.brain.arm.arm.state()["latch_source"] == "restored"
        ticks_before = b.brain.arm.efferent.envelope_tick
        # the daemon's transcript ingest is NOT a human: it never actuates (the re-review found
        # prose like 'reset the counter' reaching HOME with deliberate=True through this route)
        ingest = b.respond("Let's reset the counter and re-run the battery.")
        assert "only from a human" in ingest["reply"], ingest
        assert b.brain.arm.arm.state()["stopped"] is True, "ingested text actuated the arm"
        assert b.brain.arm.efferent.envelope_tick == ticks_before, "ingest counted as a passage"
        reply = b.respond("go home", human=True)["reply"]  # the cockpit's chat: a person
        assert "restored from a previous process has been reset" in reply, reply
        assert b.brain.arm.arm.state()["stopped"] is False
        assert b.brain.arm.efferent.envelope_tick == ticks_before + 1, (
            "the chat route skipped the tap"
        )
    finally:
        b.close()

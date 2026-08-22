"""specs/022 T039e: the brain says what its body measures about its own exploration.

Asked three times across two tasks whether it was still finding new places to reach, the live brain
abstained or answered about digits while `/api/all` carried the numbers. Under P4c a sensorimotor
surface that cannot report a signal is dormancy, not cosmetics.

The reply must separate the states T038's rate/motion PAIR was built to separate, because the count
alone cannot: an EMPTY WINDOW is nothing measured yet (not saturation), a flat count with the body
MOVING is re-covered ground, and a flat count with the body STILL is a body that stopped. Each state
below is constructed, and the other states' wording is the control -- a reply that said the same
thing in every state would pass a single-state test and tell the owner nothing.

Run: pytest tests/test_exploration_report.py
"""

import pytest

from snn_data_lake.cortex.live_brain import _IXN_INTENTS, _reply_to_intent

CTX = dict(visual=0.55, temperature=24.0, audio=0.3, distance=0.6)
ASKS = (
    "how much of the space around you have you explored so far?",
    "are you still finding new places to reach with your arm?",
    "what is your coverage?",
)


def _brain(seed=0):
    from snn_data_lake.cortex.brain import make_brain

    return make_brain(seed=seed, persist=False)


def _drive(arm, ticks):
    for _ in range(ticks):
        arm.step(reward_fn=arm.coverage_reward, **CTX)


def test_every_phrasing_of_the_question_reaches_the_report():
    """The three probes that failed in the T038a/T038b polls, plus the routing label."""
    b = _brain()
    try:
        for ask in ASKS:
            reply = b.brain.converse(ask)
            assert "distinct cells" in reply, (ask, reply)
            assert _reply_to_intent(reply) == "EXPLORE", (
                "an exploration answer must not teach the router GREET -- chat() uses this as its "
                "self-supervised label"
            )
        assert "EXPLORE" in _IXN_INTENTS
    finally:
        b.close()


def test_an_empty_window_says_nothing_measured_yet_not_saturated():
    """The state the count alone gets WRONG: a fresh or just-restored body has no coverage ticks,
    and 'rate 0' there means unmeasured, never finished."""
    b = _brain()
    try:
        arm = b.brain.arm
        assert arm.coverage_window() == 0, (
            "precondition: a fresh brain has measured no coverage tick"
        )
        reply = b.brain.converse(ASKS[0])
        assert "not measured a coverage tick yet" in reply, reply
        assert "an empty window is not the same as being finished" in reply, reply
        # the control: it must NOT claim either of the moving states
        assert "still finding new ground" not in reply and "I have stopped" not in reply, reply
    finally:
        b.close()


def test_a_body_still_finding_ground_says_so():
    b = _brain()
    try:
        arm = b.brain.arm
        _drive(arm, 40)
        assert arm.coverage_rate() > 0.0, "precondition: this arm is still finding cells"
        reply = b.brain.converse(ASKS[1])
        assert "still finding new ground" in reply, reply
        assert "new cells per coverage tick" in reply, reply
        assert "I have stopped" not in reply and "re-covering places" not in reply, reply
    finally:
        b.close()


def test_a_pre_covered_workspace_reads_as_re_covered_ground_not_as_a_stall():
    """rate 0 with motion > 0. The count is flat because everywhere it goes is already known --
    the body is working. Under the old bare count this was indistinguishable from a stall."""
    b = _brain()
    try:
        arm = b.brain.arm
        r = 30
        arm._covered = {
            (x, y, z)
            for x in range(-r, r + 1)
            for y in range(-r, r + 1)
            for z in range(-r, r + 1)
            if x * x + y * y + z * z <= r * r
        }
        _drive(arm, 60)
        assert arm.coverage_rate() == 0.0 and arm.coverage_motion() > 0.0, (
            arm.coverage_rate(),
            arm.coverage_motion(),
        )
        reply = b.brain.converse(ASKS[1])
        assert "re-covering places I have been" in reply, reply
        assert "still finding new ground" not in reply and "I have stopped" not in reply, reply
    finally:
        b.close()


def test_a_stopped_body_says_it_stopped_and_does_not_call_it_finished():
    """rate 0 with motion 0 -- an E-STOPped body. The honest reply names the stop AND refuses the
    inference the count invites, that the workspace is exhausted."""
    b = _brain()
    try:
        arm = b.brain.arm
        arm.arm.apply("EMERGENCY_STOP")
        _drive(arm, 60)
        assert arm.coverage_motion() == 0.0, "precondition: an E-STOPped body does not move"
        reply = b.brain.converse(ASKS[2])
        assert "the body did not move on any of my last" in reply, reply
        assert "not the same as having finished" in reply, reply
        assert "re-covering places" not in reply, reply
        # THE STALE-RATE CASE, and it is why motion is read first. A cell found BEFORE the stop
        # sits in the rate's window for up to 200 ticks, so the rate alone would call this body
        # 'still finding new ground' while it has not moved at all.
        if arm.coverage_rate() > 0.0:
            assert "still finding new ground" not in reply, reply
            assert "from before I stopped" in reply, reply
    finally:
        b.close()


def test_the_restored_count_and_the_scale_are_reported_honestly():
    """Two claims the reply must NOT overstate: restored cells are process-lifetime (adopted at
    construction, older than this run), and the scale is a SCALE -- its own docstring records that
    4 of the landing run's 107 cells sat outside it, so it is never a ceiling."""
    b = _brain()
    try:
        arm = b.brain.arm
        arm._covered = {(0, 0, 1), (0, 0, 2), (0, 0, 3)}
        arm._cover_restored = 3
        reply = b.brain.converse(ASKS[0])
        assert "3 of them I adopted from the lake" in reply, reply
        assert "older than this run" in reply, reply
        assert "a scale, not a ceiling" in reply, reply
        assert "%" not in reply, (
            "the reply must not report a fraction of a non-bound as if it were one"
        )
        # and the control: a body with nothing restored says the opposite
        arm._cover_restored = 0
        fresh = b.brain.converse(ASKS[0])
        assert "All of them were reached in this process" in fresh, fresh
        assert "adopted from the lake" not in fresh, fresh
    finally:
        b.close()


def test_the_reply_refuses_rather_than_guessing_when_the_body_cannot_report():
    """A body that cannot answer must produce a refusal, never an invented number -- the failure
    mode this whole task exists to remove is a confident sentence with nothing behind it."""
    b = _brain()
    try:
        arm = b.brain.arm

        def _boom():
            raise RuntimeError("no coverage instrument")

        arm.coverage_window = _boom
        reply = b.brain.converse(ASKS[0])
        assert "I will not guess a number" in reply, reply
        assert "distinct cells" not in reply, reply
    finally:
        b.close()


def test_an_arm_command_is_still_a_command_not_an_exploration_question():
    """The branch sits ahead of the NL firewall, so it must not swallow motion commands."""
    b = _brain()
    try:
        for cmd in ("go home", "close the effector", "retract all"):
            reply = b.brain.converse(cmd)
            assert "distinct cells" not in reply, (cmd, reply)
    finally:
        b.close()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

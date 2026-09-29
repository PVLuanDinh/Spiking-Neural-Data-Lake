"""specs/022 T039i: the brain can say what its insula measures about its own body.

FOUND BY THE POST-TASK POLL, the third time that discipline has caught this shape. Asked
"how does your body feel right now?" on the daemon running T039h, the brain ABSTAINED (route Gf)
and answered about wake phase -- while `/api/all` carried feeling_salience 0.2114 and
interoceptive_attention 0.1, each with a PROVENANCE row. Measured, drawn, unsayable. Under P4c
that is dormancy on a sensorimotor surface, and it is the same defect T039e removed for coverage.

The reply must clear two bars that are easy to miss:
  * it REPORTS rather than RECOMPUTES -- recomputing advances the running baseline a second time,
    so the reported number would be a different measurement from the one the ignition law used;
  * it does not translate a DISTANCE into a mood. `feeling_salience` is deviation from the body's
    own baseline, so a body sitting still at high stress reads LOW once the baseline follows it
    there. "I feel bad" would be an interpretation the measurement does not carry.

Run: pytest tests/test_interoceptive_report.py
"""

import pytest

from snn_data_lake.engine.live_brain import _IXN_INTENTS, _reply_to_intent

ASKS = ("how does your body feel right now?", "what is your interoception?")
MARK = "from its own running baseline"


@pytest.fixture(scope="module")
def brain():
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=0, persist=False)
    for _ in range(60):
        b.tick()
    yield b
    b.close()


def test_a_fresh_body_refuses_to_call_itself_calm():
    """THE STATE A NUMBER GETS WRONG. Before three readings the insula returns 0.0 by design -- a
    baseline two samples old is not a baseline -- and 0.0 there means UNMEASURED, never calm.
    Reporting it as calm would be the confident-sentence-with-nothing-behind-it failure."""
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=0, persist=False)
    try:
        reply = b.brain.converse(ASKS[0])
        assert "not built an interoceptive baseline yet" in reply, reply
        assert "will not call that calm" in reply, reply
        assert MARK not in reply, reply
    finally:
        b.close()


def test_it_reports_the_number_the_ignition_law_actually_used(brain):
    """REPORTS, NEVER RECOMPUTES. `observe` advances a running per-channel baseline, so a reply
    that called it again would report a DIFFERENT measurement than the one that set the threshold
    this tick -- and the two would drift apart silently."""
    reported = brain.brain.insula.last_salience()
    assert reported == pytest.approx(float(brain._feeling_salience), abs=1e-12), (
        f"the surface reports {reported} while the brain measured {brain._feeling_salience}"
    )
    seen_before = brain.brain.insula._seen
    brain.brain.converse(ASKS[0])
    assert brain.brain.insula._seen == seen_before, (
        "answering the question advanced the interoceptive baseline -- the reply is taking a "
        "reading instead of reporting the one already taken"
    )


def test_the_reply_carries_the_measurement_and_its_provenance(brain):
    for ask in ASKS:
        reply = brain.brain.converse(ask)
        assert MARK in reply, (ask, reply)
        assert "stress" in reply and "caution" in reply, reply
        assert _reply_to_intent(reply) == "INTEROCEPT", (
            "an interoceptive answer must not teach the router some other intent -- chat() uses "
            "this as its self-supervised label"
        )
    assert "INTEROCEPT" in _IXN_INTENTS


def test_it_says_the_number_is_a_distance_and_not_a_verdict(brain):
    """The honesty clause. Salience is how UNUSUAL the body is for itself; it carries no valence,
    and it HABITUATES, so a persistent state decays toward zero. A reply that dropped either of
    those would be inviting exactly the reading the number does not support."""
    reply = brain.brain.converse(ASKS[0])
    assert "DISTANCE, not a verdict" in reply, reply
    assert "not whether it is good or bad" in reply, reply
    assert "baseline follows my body there" in reply, reply


def test_the_workspace_amplification_is_reported_beside_it(brain):
    """T039h's top-down term is part of what produced the number, so a reply giving the salience
    without it would present an attended reading as if it were the raw deviation."""
    reply = brain.brain.converse(ASKS[0])
    assert ("amplifying that by" in reply) or ("Nothing in my workspace is amplifying" in reply), (
        reply
    )


def test_an_arm_command_is_still_a_command(brain):
    """The branch sits ahead of the NL firewall, so it must not swallow motion -- and its trigger
    words include `body`, which a command could carry."""
    for cmd in ("go home", "close the effector", "retract all", "move to home"):
        reply = brain.brain.converse(cmd)
        assert MARK not in reply, (cmd, reply)


def test_a_safety_question_still_gets_the_safety_answer(brain):
    """`hurt` and `danger` belong to the safety branch, which sits EARLIER on purpose. A question
    about danger must never be answered with a body-state readout."""
    for q in ("are you safe?", "is it dangerous?", "did that hurt?"):
        reply = brain.brain.converse(q)
        assert MARK not in reply, (q, reply)


def test_the_experience_question_is_NOT_answered_with_a_body_readout(brain):
    """DISCLOSED ORDERING, pinned so it cannot drift. A bare 'how do you feel?' keeps the existing
    consciousness answer -- which neither claims nor denies subjective experience -- because that
    is a question about EXPERIENCE, not about body state. The interoceptive report needs `body` or
    `interoception`. Changing this ordering would quietly replace a careful answer with a number."""
    reply = brain.brain.converse("how do you feel?")
    assert MARK not in reply, reply
    assert "neither claim it nor deny it" in reply, reply


def test_it_refuses_rather_than_guessing_when_the_insula_cannot_report(brain):
    """A body that cannot answer produces a refusal, never an invented number."""
    real = brain.brain.insula
    try:
        brain.brain.insula = None
        reply = brain.brain.converse(ASKS[0])
        assert "I will not guess how my body is" in reply, reply
        assert MARK not in reply, reply
    finally:
        brain.brain.insula = real


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

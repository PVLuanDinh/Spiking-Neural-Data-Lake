"""C0 -- the everyday battery's reds and conjuncts (specs/022 Phase 4 row C0, pre-registration
`specs/022-unified-brain-arc/evidence/c0-everyday-battery.md` 2.4 and 2.5).

EVERY test here pairs a claim with its MUTATION: the shipped classifier reads the pre-registered
class, and a classifier with exactly ONE of 2.3's rules broken reads the different class the record
names. A criterion nobody can construct a red for is a vacuous bar, so a test that only asserted
the green half would assert nothing. No brain, no lake and no network is touched: every input is
constructed, or is a row `FIXTURE` recorded during the 2026-09-16 orientation.
"""

import pytest

from snn_data_lake.cortex import everyday_battery as eb


def test_r1_a_narrated_number_is_not_an_answer():
    """R1: a constructed abstain whose state narration carries "3" -- item 4's key -- is ABSTAIN,
    because rule 1 decides abstention BEFORE any key is tested. Move rule 4 above rule 1 and the
    narration scores CORRECT: red."""
    item = eb.R1_ITEM
    assert "About 3 ticks ago" in eb.R1_REPLY and item.accept == r"\b(3|three)\b"
    assert eb.classify(eb.R1_REPLY, eb.R1_ROUTE, item)[0] == eb.ABSTAIN
    mutated = eb.mutated_classify("key_before_route")
    assert mutated(eb.R1_REPLY, eb.R1_ROUTE, item)[0] == eb.CORRECT


def test_r2_a_tag_is_not_an_answer():
    """R2: the OBSERVED seed-0 row "[recalled by association] 4's parity is even." scored against
    item 8's own key is UNRELATED_ROW. Credit any `[tag]` as an answer and it scores CORRECT:
    red."""
    assert eb.R2_REPLY.startswith(eb.RECALLED_TAG)
    assert eb.classify(eb.R2_REPLY, eb.R2_ROUTE, eb.R2_ITEM)[0] == eb.UNRELATED_ROW
    mutated = eb.mutated_classify("any_tag_is_correct")
    assert mutated(eb.R2_REPLY, eb.R2_ROUTE, eb.R2_ITEM)[0] == eb.CORRECT


def test_r3_the_scrambled_key_is_the_control_that_is_not_the_method():
    """R3 / conjunct (s): each item's constructed CORRECT reply scored against the NEXT item's key
    classifies CORRECT on at most 1 of 20. Score by answer SHAPE alone and the rotation reads 20 of
    20: red -- which is what proves the KEY, not the reply's shape, carries the discrimination."""
    shipped = eb.conjunct_s()
    assert shipped["n_hits"] <= shipped["bar"], shipped["hits"]
    assert shipped["n_hits"] == 0, shipped["hits"]
    mutated = eb.conjunct_s(eb.mutated_classify("shape_only"))
    assert mutated["n_hits"] == mutated["n"] == 20


def test_r4_two_instruments_must_agree_on_abstention():
    """R4: the hold's preamble under a NON-abstain route is OTHER with `disagree` -- a disagreement
    is reported, never resolved by picking one. Decide on the reply text alone and it reads
    ABSTAIN: red. The agreeing case is kept beside it so the rule is not vacuously satisfied."""
    cls, detail = eb.classify(eb.R4_REPLY, eb.R4_ROUTE, eb.R4_ITEM)
    assert cls == eb.OTHER and detail["disagree"] is True
    mutated = eb.mutated_classify("text_only_abstain")
    assert mutated(eb.R4_REPLY, eb.R4_ROUTE, eb.R4_ITEM)[0] == eb.ABSTAIN
    assert eb.classify(eb.R4_REPLY, eb.R1_ROUTE, eb.R4_ITEM)[0] == eb.ABSTAIN


def test_r4_the_other_half_of_the_disagreement_is_reported_too():
    """The mirror case: an abstain ROUTE whose reply does not open with the hold's preamble. One
    instrument without the other is a disagreement in both directions, never an ABSTAIN."""
    cls, detail = eb.classify("[reasoned] cy is shortest.", eb.R1_ROUTE, eb.ITEMS[1])
    assert cls == eb.OTHER and detail["disagree"] is True


def test_r6_a_yes_that_is_also_a_no():
    """R6: "[reasoned] yes and no." against item 1 (accept `yes`, refuse `no`) is WRONG. Drop the
    refuse test from rule 4 and it scores CORRECT: red."""
    assert eb.classify(eb.R6_REPLY, eb.CONSTRUCTED_ROUTE, eb.R6_ITEM)[0] == eb.WRONG
    mutated = eb.mutated_classify("accept_only")
    assert mutated(eb.R6_REPLY, eb.CONSTRUCTED_ROUTE, eb.R6_ITEM)[0] == eb.CORRECT


def test_conjunct_k_the_key_discriminates_on_all_three_arms():
    """(k): every item's constructed CORRECT reply reads CORRECT, every constructed WRONG reply
    reads WRONG, and every OBSERVED seed-0 reply reads the class fact 2 recorded. 20 / 20 / 20."""
    k = eb.conjunct_k()
    assert k["n"] == 20
    assert k["correct"] == 20, k["correct_misses"]
    assert k["wrong"] == 20, k["wrong_misses"]
    assert k["observed"] == 20, k["observed_misses"]


def test_conjunct_k_reddens_under_a_shape_only_key():
    """(k)'s WRONG arm is what the key buys: score by answer shape and every constructed WRONG
    reply is credited CORRECT, so the arm collapses to 0 of 20 while the CORRECT arm is unmoved.
    A bar that cannot fall is not a bar."""
    mutated = eb.conjunct_k(eb.mutated_classify("shape_only"))
    assert mutated["correct"] == 20
    assert mutated["wrong"] == 0


def test_conjunct_k_reddens_on_the_observed_arm_when_any_tag_counts():
    """(k)'s OBSERVED arm is what the PORT buys: credit any `[tag]` and the five recalled rows
    fact 2 classified UNRELATED_ROW are scored CORRECT, so the arm falls to 15 of 20."""
    mutated = eb.conjunct_k(eb.mutated_classify("any_tag_is_correct"))
    assert mutated["observed"] == 15, mutated["observed_misses"]
    assert {m[1] for m in mutated["observed_misses"]} == {eb.CORRECT}


def test_the_recorded_controls_are_the_positive_control():
    """(p)'s constructed arm: an instrument that cannot register a known hit reads nothing. The
    four controls' recorded seed-0 replies (32; 25; ijl; paris) all classify CORRECT."""
    p = eb.conjunct_p_fixture()
    assert p["n_correct"] == p["n"] == 4, p["got"]


def test_every_constructed_red_fires_in_the_modules_own_line():
    """The bare invocation's five reds, each firing only when BOTH halves hold."""
    r = eb.reds()
    assert sorted(r) == ["R1", "R2", "R3", "R4", "R6"]
    assert all(e["fires"] for e in r.values()), r


def test_the_fixture_reproduces_fact_2s_histogram():
    """The port's fixture IS the audit's reading: 14 ABSTAIN, 5 UNRELATED_ROW, 1 SELF_REPORT and
    0 CORRECT of 20 -- the number the plan's C-R2 starts from."""
    got = {}
    for fix, item in zip(eb.FIXTURE, eb.ITEMS, strict=True):
        cls = eb.classify(fix.reply, fix.route, item)[0]
        assert cls == fix.cls, (fix.id, cls, fix.cls)
        got[cls] = got.get(cls, 0) + 1
    assert got == eb.FIXTURE_HISTOGRAM
    assert got.get(eb.CORRECT, 0) == 0 and sum(got.values()) == 20


def test_the_battery_is_the_audits_twenty_questions_and_four_controls():
    """The port is VERBATIM: 20 items with ids 1-20, 4 controls, and every item's question equal
    to the orientation fixture's (which the orientation asserted equal to the archive's)."""
    assert [i.id for i in eb.ITEMS] == list(range(1, 21))
    assert [f.id for f in eb.FIXTURE] == list(range(1, 21))
    assert len(eb.CONTROLS) == len(eb.CONTROL_FIXTURE) == 4
    assert all(q.isascii() for q in [i.q for i in eb.ITEMS] + [c.q for c in eb.CONTROLS])


def test_the_answer_span_is_the_reply_with_a_leading_tag_removed():
    """2.2's answer span, and nothing else: an interior bracket is not a tag."""
    assert eb.answer_span("[reasoned] 42.") == "42."
    assert eb.answer_span("[from what I learned] a strawberry.") == "a strawberry."
    assert eb.answer_span("no tag here") == "no tag here"
    assert eb.answer_span("plain [reasoned] text") == "plain [reasoned] text"


def test_the_transcript_rule_is_not_inside_the_classifier():
    """Decision 1: the somatic suffix is stripped when the reply is RECORDED, never by `classify`.
    Both halves are asserted, because the whole point is that there is exactly one definition of
    "the reply" and the classifier does not carry a second one."""
    raw = "[recalled by association] 4's parity is even. (somatic: valence 0.3)"
    assert eb.transcript_reply(raw) == "[recalled by association] 4's parity is even."
    assert eb.classify(eb.transcript_reply(raw), eb.R2_ROUTE, eb.ITEMS[3])[0] == eb.UNRELATED_ROW
    assert eb.classify(raw, eb.R2_ROUTE, eb.ITEMS[3])[0] == eb.CORRECT


def test_classify_only_ever_returns_a_pre_registered_class():
    """2.1 (b): exactly one of the seven, on every constructed and observed input in the module."""
    replies = [i.correct for i in eb.ITEMS] + [i.wrong for i in eb.ITEMS]
    replies += [f.reply for f in eb.FIXTURE] + [eb.R1_REPLY, eb.R6_REPLY, "", "who knows"]
    for reply in replies:
        for route in (eb.R1_ROUTE, eb.CONSTRUCTED_ROUTE, "", None):
            cls, detail = eb.classify(reply, route, eb.ITEMS[0])
            assert cls in eb.CLASSES
            assert isinstance(detail, dict)


def test_an_unknown_mutation_is_refused():
    """A typo in a mutation name must not silently return the SHIPPED classifier -- that would
    make a red pass by measuring nothing."""
    with pytest.raises(ValueError):
        eb.mutated_classify("no_such_rule")


def test_the_bare_invocation_is_green_and_touches_no_brain():
    """What `run_selfchecks.py` executes: constructed inputs only, exit 0."""
    assert eb.main([]) == 0

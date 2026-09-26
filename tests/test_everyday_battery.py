"""C0 -- the everyday battery's reds and conjuncts (specs/022 Phase 4 row C0, pre-registration
`specs/022-unified-brain-arc/evidence/c0-everyday-battery.md` 2.4 and 2.5).

EVERY test here pairs a claim with its MUTATION: the shipped classifier reads the pre-registered
class, and a classifier with exactly ONE of 2.3's rules broken reads the different class the record
names. A criterion nobody can construct a red for is a vacuous bar, so a test that only asserted
the green half would assert nothing. No brain, no lake and no network is touched: every input is
constructed, or is a row `FIXTURE` recorded during the 2026-09-16 orientation.
"""

import json
from pathlib import Path

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


def test_rule_3_each_bare_fallback_marker_is_fallback():
    """2.3 rule 3: each bare fallback marker classifies FALLBACK against item 4's key.
    Remove rule 3 and each bare marker classifies OTHER: red."""
    item = eb.ITEMS[3]
    for mark in eb.FALLBACK_MARKS:
        assert eb.classify(mark, eb.CONSTRUCTED_ROUTE, item)[0] == eb.FALLBACK


def test_rule_3_fallback_precedes_a_correct_key():
    """2.3 rule 3 precedes rule 4: a fallback marker wins even when the reply satisfies the key.
    Remove rule 3 and marker-plus-key classifies CORRECT: red. The unmarked reply is the control."""
    item = eb.ITEMS[3]
    assert eb.classify(item.correct, eb.CONSTRUCTED_ROUTE, item)[0] == eb.CORRECT
    for mark in eb.FALLBACK_MARKS:
        reply = item.correct + " " + mark
        assert eb.key_hit(eb.answer_span(reply), item)
        assert eb.classify(reply, eb.CONSTRUCTED_ROUTE, item)[0] == eb.FALLBACK


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
    """ITEMS and FIXTURE have ids 1..20; CONTROLS and CONTROL_FIXTURE both have arity 4, and
    the questions are ASCII. These assertions do not check text equality: NO in-tree artifact
    carries the archive's question TEXT, and the archive is outside the repo. The verbatim text
    chain is the 2026-09-16 orientation's, an UNVERIFIABLE-IN-TREE PREMISE."""
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


# ------------------------------------------------------------------------------------------
# 2.9 step 3: the campaign's decision rules and R5, all on constructed input. No brain is run
# here -- `run_seed` and `--c0-campaign` are the only brain-touching paths and neither is called.
# ------------------------------------------------------------------------------------------
def _write_pin(tmp_path, pin, name="pin.json"):
    path = tmp_path / name
    path.write_text(json.dumps(pin, indent=1), encoding="utf-8")
    return path


def test_r5_a_sound_pin_passes_the_re_derivation():
    """R5's CONTROL, and it comes first: a check that refuses everything refuses nothing in
    particular. The constructed pin's 24 rows all re-derive to their stored class."""
    chk = eb.c0_pin_check(eb.constructed_pin())
    assert chk["ok"] is True
    assert chk["checked"] == 24 and chk["disagreements"] == []


def test_r5_a_pin_with_a_flipped_class_refuses(tmp_path):
    """R5 first arm: one row's STORED class edited after the fact. The re-derivation from the
    pinned reply and route disagrees, and the check refuses, naming the row."""
    path = _write_pin(tmp_path, eb.stale_pins()["class_flipped"], "flipped.json")
    chk = eb.c0_pin_check(path)
    assert chk["ok"] is False
    assert len(chk["disagreements"]) == 1
    bad = chk["disagreements"][0]
    assert bad["id"] == 1 and bad["pinned"] == eb.CORRECT and bad["re_derived"] == eb.ABSTAIN


def test_r5_a_pin_whose_reply_re_derives_differently_refuses(tmp_path):
    """R5 second arm, and a DIFFERENT failure: the pin was not edited to lie, the classifier or
    the key moved out from under it. Item 8's recalled row is replaced by a real answer while the
    stored class still reads UNRELATED_ROW."""
    path = _write_pin(tmp_path, eb.stale_pins()["reply_edited"], "edited.json")
    chk = eb.c0_pin_check(path)
    assert chk["ok"] is False
    bad = chk["disagreements"][0]
    assert bad["id"] == 8
    assert bad["pinned"] == eb.UNRELATED_ROW and bad["re_derived"] == eb.CORRECT


def test_a_pin_with_no_rows_is_not_ok(tmp_path):
    """An empty walk and a clean pin must never look alike: zero rows checked is not a pass."""
    path = _write_pin(tmp_path, {"row": "C0", "per_seed": []}, "empty.json")
    chk = eb.c0_pin_check(path)
    assert chk["checked"] == 0 and chk["ok"] is False


def test_r5_fires_as_a_whole_in_the_modules_own_line():
    """The three arms together, as the bare invocation reads them."""
    read = eb.r5()
    assert read["fires"] is True
    assert read["sound"] and read["class_flipped"] and read["reply_edited"]


def test_the_campaign_refuses_to_overwrite_an_existing_pin(tmp_path):
    """The pin is written ONCE. The refusal comes before any brain is constructed, and it leaves
    the existing file byte-identical -- a campaign that half-overwrote a pin would be worse than
    one that refused."""
    path = tmp_path / "c0_everyday_battery.json"
    path.write_text('{"row": "C0", "verdict": "MET"}', encoding="utf-8")
    before = path.read_bytes()
    assert eb.run_c0_campaign(seeds=(0,), pin=path) == 2
    assert path.read_bytes() == before


def test_conjunct_p_refuses_when_a_control_is_not_registered():
    """(p): an instrument that cannot register a known hit reads nothing. The green case is
    asserted beside the red so the conjunct is not vacuously satisfiable."""
    pin = eb.constructed_pin()
    good = eb.conjunct_p(pin["per_seed"])
    assert good["ok"] is True and good["n_correct"] == good["expected"] == 4
    broken = json.loads(json.dumps(pin))
    broken["per_seed"][0]["controls"][2]["cls"] = eb.OTHER
    bad = eb.conjunct_p(broken["per_seed"])
    assert bad["ok"] is False and bad["misses"][0]["id"] == "C3"


def test_conjunct_h_names_a_moved_cell_with_both_replies():
    """(h): a cell that moved is a finding about the port or about determinism, never noise. It
    is reported with the FIXTURE's reply and the campaign's, so a reader can see what changed."""
    pin = eb.constructed_pin()
    assert eb.conjunct_h(pin["per_seed"])["ok"] is True
    moved = json.loads(json.dumps(pin))
    moved["per_seed"][0]["items"][9]["cls"] = eb.CORRECT
    moved["per_seed"][0]["items"][9]["reply"] = "[reasoned] monday."
    h = eb.conjunct_h(moved["per_seed"])
    assert h["ok"] is False and h["n_equal"] == 19
    cell = h["moved"][0]
    assert cell["id"] == 10 and cell["was"] == eb.UNRELATED_ROW and cell["now"] == eb.CORRECT
    assert cell["fixture_reply"] != cell["reply"]


def test_conjunct_c_compares_class_route_and_reply():
    """(c): byte for byte, and on all three fields -- a concurrent run that returned the same
    CLASS from a different reply has still changed the reading."""
    block = eb.constructed_pin()["per_seed"][0]
    assert eb.conjunct_c(block, json.loads(json.dumps(block)))["ok"] is True
    for field, value in (("reply", "something else"), ("route", "Gx (other)"), ("cls", eb.OTHER)):
        other = json.loads(json.dumps(block))
        other["items"][0][field] = value
        c = eb.conjunct_c(block, other)
        assert c["ok"] is False and c["diffs"][0]["field"] == field


def test_the_verdict_rule_is_2_4s():
    """2.4: MET iff (k), (s), (h) and (c) with (p) holding; REFUSED iff (p) fails; NOT MET
    otherwise. A refusal is about the INSTRUMENT and a NOT MET about the port; never merged."""
    k = {"correct": 20, "wrong": 20, "observed": 20, "n": 20}
    s = {"n_hits": 0, "bar": 1}
    ok = {"ok": True}
    assert eb.c0_verdict(k, s, ok, ok, ok) == "MET"
    assert eb.c0_verdict(k, s, ok, ok, {"ok": False}) == "REFUSED"
    assert eb.c0_verdict(k, s, {"ok": False}, ok, ok) == "NOT MET"
    assert eb.c0_verdict(k, s, ok, {"ok": False}, ok) == "NOT MET"
    assert eb.c0_verdict({**k, "wrong": 19}, s, ok, ok, ok) == "NOT MET"
    assert eb.c0_verdict(k, {"n_hits": 2, "bar": 1}, ok, ok, ok) == "NOT MET"
    # (p) failing outranks everything else: no reading is taken at all.
    assert eb.c0_verdict({**k, "wrong": 0}, s, {"ok": False}, ok, {"ok": False}) == "REFUSED"


def test_the_seed_children_never_inherit_a_home(monkeypatch):
    """One process per seed is the ISOLATION, not a speed trick: five brains co-resident in ONE
    process share one process-wide NORD_HOME and could recall each other's ingested questions.
    The pop is defence in depth on top of that -- measured 2026-09-21, a child with NORD_HOME
    inherited still gets its own `throwaway_home()`, so this test pins the child's environment,
    not a reachable failure."""
    monkeypatch.setenv("NORD_HOME", str(Path.home() / ".snn_data_lake"))
    env = eb._child_env()
    assert "NORD_HOME" not in env
    assert str(eb.REPO / "src") in env["PYTHONPATH"] and env["PYTHONUTF8"] == "1"


def test_the_campaigns_tree_state_import_is_pinned_at_the_consumer():
    """cortex/everyday_battery.py::run_c0_campaign reaches across modules for the pin's
    head/dirty (record 3.1, P2). Nothing else executes that line: the bare invocation never
    reaches it and the pin-refusal test returns first, so a rename of the private helper is
    otherwise silent through pytest, run_selfchecks and ruff alike."""
    line = "from snn_data_lake.gates.policy_manifest import _tree_state"
    assert line in Path(eb.__file__).read_text(encoding="utf-8")
    from snn_data_lake.gates.policy_manifest import _tree_state

    t = _tree_state()
    assert set(t) >= {"commit", "dirty"} and isinstance(t["dirty"], bool)

"""specs/022 T102 (R1) -- static-lane tests: the A2 current-schema C8 fixture (r1-bounded-recall.md
A2) and, below it, the A4 generator/oracle core.

Brain-free and lake-free: nothing here builds a Brain or opens a lake. The fixture is judged by the
REAL admission check (`learner_events._refusal`), never by a copy of its rules.
"""

import pytest
from r1_c8_fixture import (
    N_LEGACY_TEACH,
    distractor_events,
    legacy_teach_records,
    refusal_arm_records,
)

from snn_data_lake.instruments import recall_audit as ra
from snn_data_lake.substrate import home as H
from snn_data_lake.substrate import learner_events as LE


def test_distractors_cover_every_supported_kind_exactly_once():
    kinds = [e["kind"] for e in distractor_events()]
    assert sorted(kinds) == sorted(LE.KIND_SPECS)
    assert set(kinds) <= H.LEARNER_EVENT_KINDS


@pytest.mark.parametrize("event", distractor_events(), ids=lambda e: e["kind"])
def test_each_distractor_is_complete_yet_classified_invalid(event):
    # A2: every current schema / identity / encoder field is present, so the ONLY reason to drop
    # the row is valid="0" -- classified invalid, not a critical admission refusal
    assert LE._refusal(event, LE.KIND_SPECS[event["kind"]], False) == "invalid"


def test_distractors_are_excluded_from_recall_by_the_one_predicate():
    assert not any(H.is_autobiographical_episode(e) for e in distractor_events())


def test_legacy_teach_upcasts_and_is_admitted_under_the_current_contract():
    teach = legacy_teach_records()
    assert len(teach) == N_LEGACY_TEACH
    for e in teach:
        up = LE._upcast(e)
        assert up["schema"] == LE.SCHEMA and up["split"] == "train"
        assert LE._refusal(up, LE.KIND_SPECS["teach"], False) is None
    assert not any(H.is_autobiographical_episode(e) for e in teach)


@pytest.mark.parametrize(("reason", "record"), refusal_arm_records())
def test_refusal_arm_rows_are_refused_for_the_named_reason(reason, record):
    spec = LE.KIND_SPECS[record["kind"]]
    assert LE._refusal(LE._upcast(record), spec, False) == reason


def test_refusal_arm_has_the_minimal_old_form_for_every_kind():
    minimal = [r for reason, r in refusal_arm_records() if reason == "missing_slot"]
    assert sorted(r["kind"] for r in minimal) == sorted(LE.KIND_SPECS)
    assert all(set(r) == {"kind", "event_id", "session", "seq"} for r in minimal)


# ---- A4 core, part 1: the generator-side oracle and the scoring table (2.3, I1) ----


def test_oracle_dedups_on_stringified_slots():
    # pack_episode stringifies every value, so {"when": 5} and {"when": "5"} are ONE record
    recs = [
        ra.Record.auto({"who": "self", "when": 5}),
        ra.Record.auto({"who": "self", "when": "5"}),
    ]
    assert len(ra.Oracle(recs).distinct) == 1


def test_oracle_is_presence_checked_on_the_none_trap():
    # 1.4 (a): an episode LACKING `affect` never matches a cue naming affect="None"
    o = ra.Oracle([ra.Record.auto({"who": "self", "when": "u1"})])
    assert o.match({"when": "u1", "affect": "None"}) == []
    assert o.match({"when": "u1"}) == [0]


def test_planted_wrong_episode_scores_false():
    # I1's red: a returned episode outside the cue's match set is FALSE, never VERIFIED
    right = {"who": "self", "what": "reach", "when": "5"}
    wrong = {"who": "self", "what": "wave", "when": "10"}
    o = ra.Oracle([ra.Record.auto(right), ra.Record.auto(wrong)])
    cue = {"when": "5"}
    assert o.score(cue, {"source": "lake", "episode": right}) == "VERIFIED"
    assert o.score(cue, {"source": "lake", "episode": wrong}) == "FALSE"


def test_every_result_shape_has_an_outcome():
    o = ra.Oracle([ra.Record.auto({"who": "self", "when": "1"})])
    cue = {"when": "1"}
    assert o.score(cue, {"source": "refused", "reason": "x"}) == "REFUSED"
    assert o.score(cue, {"source": "error", "error": "boom"}) == "ERROR"
    assert o.score(cue, None) == "ERROR"
    assert o.score(cue, {"episode": {"when": "1"}}) == "ERROR"  # no source = malformed
    assert o.score(cue, {"source": "miss", "episode": {}}) == "MISS"


def test_unknown_source_malformed_episode_and_missing_responses_are_errors():
    o = ra.Oracle([ra.Record.auto({"who": "self", "when": "1"})])
    cue, right = {"when": "1"}, {"who": "self", "when": "1"}
    assert o.score(cue, {"source": "lake", "episode": right}) == "VERIFIED"
    assert o.score(cue, {"source": "bogus", "episode": right}) == "ERROR"
    assert o.score(cue, {"source": "lake", "episode": ["not", "a", "dict"]}) == "ERROR"
    assert o.score(cue, {"source": "lake", "episode": 7}) == "ERROR"
    cues = [ra.Cue("C1", cue)] * 3
    answered = [{"result": {"source": "lake", "episode": right}, "tier": "deep", "seconds": 0.0}]
    rows = ra.score_rows(o, cues, answered)  # two of three responses missing
    assert [r.outcome for r in rows] == ["VERIFIED", "ERROR", "ERROR"]
    assert ra.tally(o, rows)["C1"]["L"] == 2
    assert ra.score_rows(o, cues, [None, "x", 3])[0].outcome == "ERROR"


def test_refusal_on_an_intact_workload_is_a_lost_record():
    # a refused present cue on an intact workload is L = 1; inside a B4 faulted arm it is the
    # permitted alternative and is not counted lost
    o = ra.Oracle([ra.Record.auto({"who": "self", "when": "1"})])
    cue = ra.Cue("C7", {"when": "1"})
    rows = [ra.Row(cue=cue, outcome="VERIFIED", tier="deep") for _ in range(3)]
    rows[0] = ra.Row(cue=cue, outcome="REFUSED", tier=None)
    assert ra.tally(o, rows)["C7"]["L"] == 1
    assert ra.tally(o, rows, faulted=True)["C7"]["L"] == 0


def test_the_table_keeps_the_outcome_by_tier_association():
    # D-a: moving a FALSE answer from the working tier to the deep tier must change the table
    o = ra.Oracle([ra.Record.auto({"who": "self", "when": "1"})])
    c = ra.Cue("C6", {"when": "9"})
    t1 = ra.tally(o, [ra.Row(c, "FALSE", "working"), ra.Row(c, "MISS", "deep")])
    t2 = ra.tally(o, [ra.Row(c, "FALSE", "deep"), ra.Row(c, "MISS", "working")])
    assert t1["C6"]["by_tier"] == {"working": {"FALSE": 1}, "deep": {"MISS": 1}}
    assert t1["C6"]["by_tier"] != t2["C6"]["by_tier"]
    assert t1["C6"]["F"] == t2["C6"]["F"] and t1["C6"]["tiers"] == t2["C6"]["tiers"]


def test_i1_agreement_on_trap_free_classes_fails_on_any_disagreement():
    mk = lambda cls, out: ra.Row(ra.Cue(cls, {}), out, "deep")  # noqa: E731
    ok = [mk("C3", "VERIFIED"), mk("C7", "VERIFIED"), mk("C6_none", "FALSE")]  # traps are not I1's
    assert ra.i1_check(ok)
    assert not ra.i1_check(ok + [mk("C4", "FALSE")])
    assert not ra.i1_check([mk("C6_none", "MISS")])  # no trap-free row at all is not agreement


def test_scope_assertion_refuses_a_disagreeing_predicate():
    # 2.3: the generator's own recallability label must equal the imported predicate
    teach = {"kind": "teach", "who": "self", "when": "1"}
    o = ra.Oracle([ra.Record.auto({"who": "self", "when": "2"}), ra.Record(teach, "teach")])
    o.assert_scope(H.is_autobiographical_episode)
    with pytest.raises(ra.ScopeMismatch):
        o.assert_scope(lambda ep: True)  # calls the teach record recallable

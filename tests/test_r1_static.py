"""specs/022 T102 (R1) -- static-lane tests: the A2 current-schema C8 fixture (r1-bounded-recall.md
A2) and, below it, the A4 generator/oracle core.

Brain-free and lake-free: nothing here builds a Brain or opens a lake. The fixture is judged by the
REAL admission check (`learner_events._refusal`), never by a copy of its rules.
"""

import random
import time

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


# ---- A4 core, part 2: the frozen generator, I6 determinism, the A2 fixture wired in ----

TINY = dict(n_auto=240, boots=3, n_filler=40, filler_bytes=64, capacity=16, counts=ra.TINY_COUNTS)


def _wl(seed=14200):
    return ra.make_workload(seed, teach=legacy_teach_records(), learner=distractor_events(), **TINY)


def test_i6_generator_is_deterministic_and_seed_sensitive():
    a, b, c = _wl(14200), _wl(14200), _wl(14201)
    assert ra.workload_digest(a) == ra.workload_digest(b)
    assert ra.workload_digest(a) != ra.workload_digest(c)


def test_i6_red_a_seed_blind_generator_is_caught():
    # the red: a generator that ignores its seed gives equal digests for different seeds
    blind = lambda seed: _wl(14200)  # noqa: E731
    assert ra.workload_digest(blind(14200)) == ra.workload_digest(blind(14201))
    assert ra.workload_digest(_wl(14200)) != ra.workload_digest(_wl(14201))


def test_actor_distribution_is_the_registered_p_self_point_nine():
    rng = random.Random(14200)
    who = [ra._episode(rng, i)["who"] for i in range(20000)]
    assert abs(who.count("self") / len(who) - 0.9) < 0.01
    assert {"owner", "arm"} <= set(who)


def test_absent_classes_are_absent_and_present_classes_present():
    wl = _wl()
    o = ra.Oracle(wl.records)
    for c in wl.cues:
        n = len(o.match(c.cue))
        if c.cls in ra.ABSENT_CLASSES:
            assert n == 0, (c.cls, c.cue)
        else:
            assert n >= 1, (c.cls, c.cue)
        if c.cls == "C5":
            assert n >= 2 and "when" not in c.cue
    assert {c.cls for c in wl.cues} >= {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"}


def test_resident_classes_sit_in_the_expected_window_and_nonresident_ones_do_not():
    wl = _wl()
    o = ra.Oracle(wl.records)
    window = {ra.episode_key(e) for e in o.expected_window(wl.capacity)}
    for c in wl.cues:
        if c.cls == "C1":
            assert ra.episode_key(c.cue) in window
        if c.cls == "C3":
            assert ra.episode_key(c.cue) not in window


def test_the_a2_fixture_is_in_the_stream_and_the_label_equals_the_one_predicate():
    wl = _wl()
    o = ra.Oracle(wl.records)
    o.assert_scope(H.is_autobiographical_episode)
    kinds = {r.episode["kind"] for r in wl.records if r.episode and "kind" in r.episode}
    assert kinds == set(LE.KIND_SPECS)  # all supported kinds, none silently omitted
    assert sum(1 for r in wl.records if r.label == "teach") == N_LEGACY_TEACH
    assert wl.learner_kinds == tuple(sorted(LE.KIND_SPECS))


def test_c8_cues_are_exact_cues_of_the_non_autobiographical_records_and_expect_miss():
    wl = _wl()
    o = ra.Oracle(wl.records)
    c8 = [c for c in wl.cues if c.cls == "C8"]
    assert len(c8) == ra.TINY_COUNTS["C8"]
    assert all("kind" in c.cue and o.match(c.cue) == [] for c in c8)


# ---- A4 core, part 3: the instrument probes I3 (latency), I4 (memory), I5 (load witness) ----


def test_i3_latency_probe_reads_a_planted_sleep_and_catches_a_span_timed_outside():
    # a 50 ms stub reads >= 50 ms; the red -- a span closed before the call -- reads ~0 and is
    # refused by the probe's own floor check
    _, dt = ra.time_call(lambda: time.sleep(0.05))
    assert dt >= 0.05
    assert not ra.latency_probe_ok(planted=0.05, measured=0.0001)
    assert ra.latency_probe_ok(planted=0.05, measured=dt)


def test_i4_memory_probe_reads_a_known_allocation_within_ten_percent():
    _, peak = ra.traced_peak(lambda: bytearray(10 * 1024 * 1024))
    assert abs(peak - 10 * 1024 * 1024) <= 0.10 * 10 * 1024 * 1024


def test_i5_load_witness_voids_on_drift_and_passes_when_steady():
    # a drift beyond 50 % VOIDS the seed's cost reading (never a pass or a fail)
    assert ra.LOAD_TOLERANCE == 0.5
    assert ra.load_verdict(1.0, 1.2) == "OK"
    assert ra.load_verdict(1.0, 1.6) == "VOID"
    assert ra.load_verdict(1.0, 0.4) == "VOID"


def test_i5_artificial_spinners_void_the_witness():
    # the registered red: a witness read while the host is loaded drifts and must VOID
    base = ra.spin_witness(20000)
    assert base > 0
    assert ra.load_verdict(base, base * 3) == "VOID"
    assert ra.load_verdict(base, base) == "OK"

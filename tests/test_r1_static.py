"""specs/022 T102 (R1) -- static-lane tests: the A2 current-schema C8 fixture (r1-bounded-recall.md
A2) and, below it, the A4 generator/oracle core.

Brain-free and lake-free: nothing here builds a Brain or opens a lake. The fixture is judged by the
REAL admission check (`learner_events._refusal`), never by a copy of its rules.
"""

import hashlib
import inspect
import os
import random
import struct
import time
from pathlib import Path

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
from snn_data_lake.substrate.binlog import BinLog


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


def test_i6_red_a_seed_blind_generator_is_caught(monkeypatch):
    # the red, planted IN the code under test: make_workload's rng ignores its seed, so two seeds
    # give equal digests and the seed-sensitivity check of the test above would FAIL
    import types

    real = random.Random
    monkeypatch.setattr(ra, "random", types.SimpleNamespace(Random=lambda _seed: real(0)))
    blind_a, blind_b = _wl(14200), _wl(14201)
    assert ra.workload_digest(blind_a) == ra.workload_digest(blind_b)


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
        if ra._base_class(c.cls) in ra.ABSENT_CLASSES:
            assert n == 0, (c.cls, c.cue)
        else:
            assert n >= 1, (c.cls, c.cue)
        if c.cls == "C5":
            assert n >= 2 and "when" not in c.cue
    assert {ra._base_class(c.cls) for c in wl.cues} >= {f"C{i}" for i in range(1, 9)}


def test_c6_subclasses_are_emitted_under_their_own_names():
    # 2.2 / enforcer #6673 MINOR 6: the None-trap red and the collision-ghost reading must be
    # separable in the tally, so the generator emits C6_novel / C6_none / C6_ghost / C6_extra
    wl = _wl()
    got = {}
    for c in wl.cues:
        got[c.cls] = got.get(c.cls, 0) + 1
    for sub in ("C6_novel", "C6_none", "C6_ghost", "C6_extra"):
        assert got[sub] == ra.TINY_COUNTS[sub], sub
    assert "C6" not in got
    o = ra.Oracle(wl.records)
    rows = [ra.Row(c, "FALSE" if c.cls == "C6_none" else "MISS", "deep") for c in wl.cues]
    assert ra.tally(o, rows)["C6"]["F"] == ra.TINY_COUNTS["C6_none"]  # base class still folds
    assert all(r.cue.cls == "C6_none" for r in rows if r.outcome == "FALSE")


def test_resident_classes_sit_in_the_expected_window_and_nonresident_ones_do_not():
    # enforcer #6673 MINOR 5: `when` restarts at each boot, so a partial nonresident cue could
    # match a same-index RESIDENT episode; C3/C4 matches must ALL sit outside the window (C1/C2:
    # at least one inside it)
    for seed in (14200, 14201, 14202):
        wl = _wl(seed)
        o = ra.Oracle(wl.records)
        window = {ra.episode_key(e) for e in o.expected_window(wl.capacity)}
        for c in wl.cues:
            hits = {ra.episode_key(o.distinct[i].episode) for i in o.match(c.cue)}
            if c.cls in ("C1", "C2"):
                assert hits & window, (c.cls, c.cue)
            if c.cls in ("C3", "C4"):
                assert hits and not hits & window, (c.cls, c.cue)


def test_w_c_mixes_legacy_and_session_bearing_autobiographical_episodes():
    # 2.2 :123-124 / A2 :806: F3 stamps `session` on durable episodes (engine/brain.py:2495), so
    # W_c carries BOTH schemas, and both stay recallable under the one predicate
    wl = _wl()
    auto = [r.episode for r in wl.records if r.label == "auto"]
    assert any("session" in e for e in auto) and any("session" not in e for e in auto)
    assert all(H.is_autobiographical_episode(e) for e in auto)
    ra.Oracle(wl.records).assert_scope(H.is_autobiographical_episode)


@pytest.mark.parametrize(
    ("shape", "named"),
    [
        (dict(boots=1), "boots"),
        (dict(n_auto=241), "n_auto"),  # 241 % 3: records would be dropped silently
        (dict(n_auto=90), "per_boot"),  # 30 per boot < the 50 cross-boot duplicates
        (dict(capacity=0), "capacity"),
        (dict(capacity=10_000), "capacity"),  # leaves no non-resident episode
    ],
)
def test_impossible_workload_shapes_are_refused_by_name(shape, named):
    # enforcer #6673 NIT 5: a ValueError naming the argument, not an IndexError / sample error
    with pytest.raises(ValueError, match=named):
        ra.make_workload(
            14200, teach=legacy_teach_records(), learner=distractor_events(), **dict(TINY, **shape)
        )


def test_an_empty_window_holds_nothing():
    o = ra.Oracle([ra.Record.auto({"who": "self", "when": "1"})])
    assert o.expected_window(0) == []
    assert len(o.expected_window(1)) == 1


def test_traced_peak_leaves_an_outer_trace_running():
    # enforcer #6673 NIT 6: the I4 probe must not end a trace a caller (the cost driver) started
    import tracemalloc

    tracemalloc.start()
    try:
        ra.traced_peak(lambda: bytearray(1024))
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()
    ra.traced_peak(lambda: bytearray(1024))
    assert not tracemalloc.is_tracing()  # and it still cleans up its own trace


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
    assert dt >= 0.05 - 0.001  # 1 ms slack: sleep and perf_counter are different clocks
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


def test_i5_planted_drift_voids_a_real_witness_reading():
    # r1-bounded-recall.md:504-512 (A-amendment) replaced artificial spinners with PLANTED timings:
    # the base here is a real spin_witness reading, the "after" is a planted 3x drift. Whether
    # spin_witness is sensitive to REAL contention is unproven by this test (owed to the campaign).
    base = ra.spin_witness(20000)
    assert base > 0
    assert ra.load_verdict(base, base * 3) == "VOID"
    assert ra.load_verdict(base, base) == "OK"


# ---- I7: the throwaway-home guard (r1-bounded-recall.md 2.7 I7, 2.14 D-e) -----------------------
def _decoy_home(tmp_path, monkeypatch):
    """A decoy LIVE home and an isolated NORD_HOME elsewhere; the real live path is not touched."""
    live, iso = tmp_path / "live", tmp_path / "iso"
    live.mkdir()
    iso.mkdir()
    monkeypatch.setattr(H, "_default_live_home", lambda: str(live))
    monkeypatch.setenv("NORD_HOME", str(iso))
    return live, iso


def test_i7_the_audit_fails_closed_without_the_shared_guard_and_lets_a_refusal_through(monkeypatch):
    monkeypatch.delattr(H, "require_isolated_home", raising=False)
    with pytest.raises(ra.GuardMissing):
        ra.require_isolation()

    class Refused(ValueError):
        pass

    def refuse(live_home=None):
        raise Refused("unset")

    monkeypatch.setattr(H, "require_isolated_home", refuse, raising=False)
    with pytest.raises(Refused):
        ra.require_isolation()


def test_i7_an_unset_nord_home_is_refused_and_an_isolated_one_is_accepted(tmp_path, monkeypatch):
    _live, iso = _decoy_home(tmp_path, monkeypatch)
    assert ra.require_isolation() == H._resolve(str(iso))  # the positive
    monkeypatch.delenv("NORD_HOME")
    with pytest.raises(H.LiveHomeRefused):  # the default home IS the live lake
        ra.require_isolation()


def test_i7_an_output_root_naming_the_live_home_or_outside_the_active_home_is_refused(
    tmp_path, monkeypatch
):
    live, iso = _decoy_home(tmp_path, monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(H.LiveHomeRefused):
        ra.confine(str(live))
    with pytest.raises(H.LiveHomeRefused, match="outside"):
        ra.confine(str(other))
    assert list(live.iterdir()) == [] and list(other.iterdir()) == []
    inside = iso / "run"
    inside.mkdir()
    assert ra.confine(str(inside)) == H._resolve(str(inside))  # the positive


def test_i7_a_write_to_the_live_home_is_reported_even_when_the_run_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # the registered stamp reads <~>/.snn_data_lake
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    stamped = tmp_path / ".snn_data_lake"
    stamped.mkdir()

    def writer():
        (stamped / "brain_lake.bin").write_bytes(b"x")
        return {}

    def raiser():
        (stamped / "other.bin").write_bytes(b"y")
        raise RuntimeError("boom")

    with pytest.raises(ra.LiveHomeWritten):
        ra.stamped(writer)
    with pytest.raises(ra.LiveHomeWritten):
        ra.stamped(raiser)
    assert ra.stamped(lambda: {"ok": 1}) == {"ok": 1}  # the positive: no write, no refusal


def test_i7_the_stamp_is_home_pys_one_copy_and_sees_nested_and_same_size_rewrites(
    tmp_path, monkeypatch
):
    # enforcer #6734 MAJOR-1: move_gate's private stamp is (size, mtime) of the DEFAULT home's
    # top-level files; home.lake_snapshot / live_lake_untouched is the repo's one stamp (nested
    # files, content digest, any home). RED on the parent: the live home here is a decoy reached
    # through H._default_live_home, the private stamp reads ~/.snn_data_lake instead, so no
    # write below is seen.
    live, _iso = _decoy_home(tmp_path, monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "elsewhere"))
    (live / "segments").mkdir()
    nested = live / "segments" / "seg0.bin"
    nested.write_bytes(b"aaaa")
    top = live / "brain_lake.bin"
    top.write_bytes(b"bbbb")

    def nested_write():
        nested.write_bytes(b"cccccc")
        return {}

    def same_size_rewrite():
        st = top.stat()
        top.write_bytes(b"zzzz")  # same size,
        os.utime(top, ns=(st.st_atime_ns, st.st_mtime_ns))  # mtime put back
        return {}

    def raising_nested_write():
        nested.write_bytes(b"dddddddd")
        raise RuntimeError("boom")

    with pytest.raises(ra.LiveHomeWritten):
        ra.stamped(nested_write)
    with pytest.raises(ra.LiveHomeWritten):
        ra.stamped(same_size_rewrite)
    with pytest.raises(ra.LiveHomeWritten):
        ra.stamped(raising_nested_write)
    assert ra.stamped(lambda: {"ok": 1}) == {"ok": 1}  # the positive: no write, no refusal


# ---- I2: tier attribution (r1-bounded-recall.md 2.7 I2) -----------------------------------------
def test_i2_tier_attribution_reads_the_deep_counter():
    # a lake answer with a deep-tier read is the deep tier; without one it is the window
    assert ra.tier_of({"source": "working", "episode": {"a": "1"}}, deep_calls=0) == "working"
    assert ra.tier_of({"source": "lake", "episode": {"a": "1"}}, deep_calls=1) == "deep"
    assert ra.tier_of({"source": "lake", "episode": {"a": "1"}}, deep_calls=0) == "window"
    assert ra.tier_of({"source": "miss", "episode": {}}, deep_calls=1) is None


def test_i2_attributes_through_the_real_wrappers_and_the_wrong_attribute_reds(
    tmp_path, monkeypatch
):
    # the real `_instrumented` wrappers over the real chain (Brain.recall_episode_durable on a stub
    # that carries only what it reads). The red wraps `save_episode` instead of the deep tier's
    # reader, so the known-deep cue counts 0 deep reads and is mis-attributed `window`.
    iso = tmp_path / "iso"
    iso.mkdir()
    monkeypatch.setenv("NORD_HOME", str(iso))
    assert ra.i2_check(str(iso / "home_ok")) is True
    assert ra.i2_check(str(iso / "home_red"), deep_attr="save_episode") is False


def test_i2_check_confines_its_home_before_creating_or_exporting_anything(tmp_path, monkeypatch):
    # enforcer #6734 MINOR-2: i2_check ran makedirs and set NORD_HOME BEFORE the guard, and
    # checked the PREVIOUS NORD_HOME, not home_dir. RED on the parent: a home_dir outside the
    # active throwaway home is created, then used; one inside the live home is created there,
    # then refused.
    live, iso = _decoy_home(tmp_path, monkeypatch)
    outside = tmp_path / "outside" / "home"
    in_live = live / "i2"
    for bad in (outside, in_live):
        with pytest.raises(H.LiveHomeRefused):
            ra.i2_check(str(bad))
        assert not bad.exists()  # refused before anything was created
        assert os.environ["NORD_HOME"] == str(iso)  # and NORD_HOME was never moved


# ---- S1 / S2: the located reader (r1-bounded-recall.md 2.11 P3; ported from claude/r1) --------
MAGIC = b"SNDLBIN1"

PAYLOADS = [b"alpha", b"bb", b"c" * 300, b"dd", b"\x00\x01\x02"]


def _blog(tmp_path, payloads=PAYLOADS):
    path = str(tmp_path / "lake.bin")
    log = BinLog(path, dedup=False)
    for p in payloads:
        log.append(p)
    return log, path


def _frame_offsets(payloads):
    """INDEPENDENT of the reader under test: summed frame lengths (8 magic + 4 + len each)."""
    out, pos = [], len(MAGIC)
    for p in payloads:
        out.append(pos)
        pos += 4 + len(p)
    return out


def _offsets_agree(located, payloads):
    return [off for off, _p in located] == _frame_offsets(payloads) and [
        p for _o, p in located
    ] == list(payloads)


def test_s1_replay_located_offsets_are_frame_starts_and_read_at_returns_the_payload(tmp_path):
    log, path = _blog(tmp_path)
    located = list(log.replay_located())
    assert _offsets_agree(located, PAYLOADS)
    for off, payload in located:
        assert log.read_at(off) == payload
    # the red: an offset taken at the PAYLOAD (frame + 4) instead of the frame fails the same check
    wrong = [(off + 4, p) for off, p in located]
    assert not _offsets_agree(wrong, PAYLOADS)
    # `start` resumes at a frame boundary: the records after the watermark, offsets unchanged
    tail = list(log.replay_located(start=_frame_offsets(PAYLOADS)[3]))
    assert _offsets_agree(tail, PAYLOADS[3:]) is False  # offsets continue from the middle...
    assert [p for _o, p in tail] == PAYLOADS[3:]
    assert [o for o, _p in tail] == _frame_offsets(PAYLOADS)[3:]  # ...and are the SAME offsets


def test_s1_read_at_refuses_a_non_frame_offset_and_a_torn_frame(tmp_path):
    log, path = _blog(tmp_path)
    with pytest.raises(ValueError):
        log.read_at(0)  # inside the magic
    with open(path, "r+b") as f:  # tear the last frame: keep its head, cut its payload
        f.truncate(_frame_offsets(PAYLOADS)[-1] + 4 + 1)
    assert log.read_at(_frame_offsets(PAYLOADS)[-1]) is None
    assert [p for _o, p in log.replay_located()] == PAYLOADS[:-1]


def _parse(path):
    """INDEPENDENT reference parse of the file bytes (no BinLog): what replay() must yield."""
    data = open(path, "rb").read()
    out, pos = [], len(MAGIC)
    while pos + 4 <= len(data):
        (n,) = struct.unpack_from("<I", data, pos)
        if n == 0 or pos + 4 + n > len(data):
            break
        out.append(data[pos + 4 : pos + 4 + n])
        pos += 4 + n
    return out


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _contract_ok(replay, path):
    """S2's contract: the reference records, in order, with NO write (bytes identical after)."""
    before = _sha(path)
    got = list(replay())
    return got == _parse(path) and _sha(path) == before


def _fixtures(tmp_path):
    """clean; torn tail (partial payload); zero length prefix mid-file; torn head (2 bytes)."""
    paths = {}
    for name in ("clean", "torn", "zero", "head"):
        d = tmp_path / name
        d.mkdir()
        log, path = _blog(d)
        paths[name] = path
        with open(path, "r+b") as f:
            if name == "torn":
                f.truncate(_frame_offsets(PAYLOADS)[-1] + 4 + 1)
            elif name == "zero":
                f.seek(_frame_offsets(PAYLOADS)[2])
                f.write(b"\x00\x00\x00\x00")
            elif name == "head":
                f.truncate(_frame_offsets(PAYLOADS)[-1] + 2)
    return paths


def test_s2_replay_keeps_its_contract_after_delegating_to_the_located_reader(tmp_path):
    paths = _fixtures(tmp_path)
    for name, path in paths.items():
        log = BinLog.__new__(BinLog)  # replay() reads only the file: no constructor, so no repair
        log.path = path
        assert inspect.isgeneratorfunction(BinLog.replay)
        assert _contract_ok(log.replay, path), name
    assert len(_parse(paths["torn"])) == 4 and len(_parse(paths["zero"])) == 2


def test_s2_the_contract_check_rejects_a_dropping_a_continuing_and_a_truncating_reader(tmp_path):
    paths = _fixtures(tmp_path)
    clean, torn = paths["clean"], paths["torn"]

    def dropping():  # drops one record: the digest-moving red
        yield from _parse(clean)[:-1]

    def continuing():  # reads past the torn frame: returns a record the reference refuses
        yield from _parse(torn) + [b"ghost"]

    def truncating():  # repairs the torn tail by truncating the file (a write)
        with open(torn, "r+b") as f:
            f.truncate(_frame_offsets(PAYLOADS)[-1])
        yield from _parse(torn)

    assert not _contract_ok(dropping, clean)
    assert not _contract_ok(continuing, torn)
    assert not _contract_ok(truncating, torn)


# ---- the LakeIndex range / meta / clear API the posting rows need (2.11 P3 (b), 2.21-2.23) ------
def test_lake_index_range_reads_meta_and_clear(tmp_path):
    # P3 (b): the additive range read the posting rows need (key = token hash + frame offset)
    from snn_data_lake.substrate.lake_index import LakeIndex

    idx = LakeIndex(tmp_path / "i.db")
    rows = [(256, f"aa:{o:016x}", "lake", o) for o in (40, 8, 120)] + [
        (256, "ab:" + "0" * 16, "lake", 7)
    ]
    rows += [(1, "aa:" + "0" * 16, "seg", 99)]  # another tag with a colliding key prefix
    idx.put_batch(rows, watermark=200, meta={"anchor": "x"})
    assert [s for _k, s in idx.iter_range(256, "aa:", "aa;")] == [8, 40, 120]  # key order = offset
    assert list(idx.iter_range(256, "aa:", "aa;")) == [(f"aa:{o:016x}", o) for o in (8, 40, 120)]
    assert [s for _k, s in idx.iter_range(256, "aa:", "aa;", reverse=True)] == [120, 40, 8]
    assert idx.count_range(256, "aa:", "aa;") == 3 and idx.count_range(256, "zz:", "zz;") == 0
    assert idx.meta("anchor") == "x" and idx.meta("nope") is None and idx.applied_watermark() == 200
    d = idx.logical_dump_digest()
    idx.put_batch([], meta={"anchor": "y"})  # the anchor is part of the logical state
    assert idx.logical_dump_digest() != d
    idx.clear(256, [])
    assert idx.meta("anchor") == "y"  # rows only: meta belongs to whoever wrote it
    idx.clear(999, ["anchor"])  # a tag with no rows: only the named meta key goes
    assert idx.meta("anchor") is None and idx.applied_watermark() == 200
    assert idx.count_range(256, "", "￿") == 0 and idx.get(1, "aa:" + "0" * 16) == ("seg", 99)


def test_lake_index_keeps_one_range_api_and_one_delete_api():
    # Codex P3 lens 2 MINOR (P2): delete_tag, delete_meta, range_seqs and range_rows duplicated what
    # clear and iter_range do, and only tests called them. One mechanism each.
    from snn_data_lake.substrate.lake_index import LakeIndex

    for gone in ("delete_tag", "delete_meta", "range_seqs", "range_rows"):
        assert not hasattr(LakeIndex, gone), gone
    for kept in ("clear", "iter_range", "count_range", "data_version"):
        assert callable(getattr(LakeIndex, kept)), kept


# ---- S3-S9 + 2.21-2.23: the successor episode index (2.11 P3, 2.19-2.23; ported from claude/r1) -
# ---- the episode index (S3, S4, S9) -----------------------------------------------------------
def _eps(n=60):
    acts = ["rest", "reach", "grasp", "look"]
    return [
        {"who": "self", "what": acts[i % 4], "where": f"p{i % 3}", "when": str(i), "affect": "calm"}
        for i in range(n)
    ]


def _lake(tmp_path, eps):
    from snn_data_lake.substrate import home as H

    path = str(tmp_path / "brain_lake.bin")
    for e in eps:
        assert H.save_episode(e, path=path)
    return H._log(path), path


def _scan(path, cue):
    """The incumbent deep tier's answer: newest-first exact sub-cue over the whole lake."""
    from snn_data_lake.substrate import home as H
    from snn_data_lake.substrate.lake_recall import is_subcue

    return next((e for e in reversed(H.load_episodes(path)) if is_subcue(cue, e)), None)


def _ei(tmp_path, log):
    from snn_data_lake.substrate.episode_index import EpisodeIndex

    return EpisodeIndex(log, tmp_path / "idx.db")


CUES = [
    {"when": "5"},  # unique
    {"what": "rest", "where": "p1"},  # ambiguous: newest wins
    {"who": "self"},  # matches everything: newest
    {"what": "rest", "when": "7"},  # present tokens, no common record
    {"when": "5", "affect": "None"},  # the None trap: the slot exists with another value
    {"what": "novel"},  # absent token
]


def test_the_successor_returns_exactly_the_deep_scan_answer(tmp_path):
    log, path = _lake(tmp_path, _eps())
    ix = _ei(tmp_path, log)
    for cue in CUES:
        got = ix.recall(cue)
        want = _scan(path, cue)
        assert (got["episode"] or None) == want, cue
        assert got["source"] == ("lake" if want else "miss")
    assert ix.recall({})["source"] == "miss" and ix.recall({"session": "1"})["source"] == "miss"


def test_s3_tail_fold_finds_a_record_appended_after_the_first_sync(tmp_path):
    from snn_data_lake.substrate import home as H
    from snn_data_lake.substrate.episode_index import EpisodeIndex

    log, path = _lake(tmp_path, _eps(20))
    ix = _ei(tmp_path, log)
    assert ix.recall({"when": "5"})["source"] == "lake"
    newest = {"who": "owner", "what": "wave", "where": "px", "when": "u9", "affect": "joy"}
    assert H.save_episode(newest, path=path)
    assert ix.recall({"when": "u9"})["episode"] == newest  # folded from the watermark

    class NoFold(EpisodeIndex):  # the red: the tail fold disabled after the first sync
        def sync(self):
            return 0

    frozen = NoFold(log, tmp_path / "frozen.db")
    assert frozen.recall({"when": "u9"})["source"] == "miss"  # loses the newest nonresident record


def test_s4_a_planted_wrong_offset_row_can_never_become_a_false_answer(tmp_path):
    from snn_data_lake.substrate import episode_index as E

    log, path = _lake(tmp_path, _eps(20))
    ix = _ei(tmp_path, log)
    ix.sync()
    offs = [o for o, _p in log.replay_located()]
    # plant: the token row of when=3 now also points at record 9's frame
    tok = E._tok("when", "3")
    ix.idx.put_batch([(E.TAG, E._key(tok, offs[9], E._chk(log.read_at(offs[9]))), "lake", offs[9])])
    got = ix.recall({"when": "3"})
    assert got["episode"]["when"] == "3"  # verification rejected record 9 and found the real one

    class Unverified(E.EpisodeIndex):  # the red: candidates returned unverified
        def recall(self, cue):
            self.sync()
            tok = E._tok("when", cue["when"])
            top = next(s for _k, s in self.idx.iter_range(E.TAG, *E._bounds(tok), reverse=True))
            return {"episode": E.decode(self.log.read_at(top)), "confidence": 1.0, "source": "lake"}

    bad = Unverified(log, tmp_path / "bad.db")
    bad.sync()
    bad.idx.put_batch(
        [(E.TAG, E._key(tok, offs[9], E._chk(log.read_at(offs[9]))), "lake", offs[9])]
    )
    assert bad.recall({"when": "3"})["episode"]["when"] == "9"  # FALSE: the check the real one has


def _ep_payload(ep):
    from snn_data_lake.substrate import home as H
    from snn_data_lake.substrate.binlog import pack_episode

    return H._T_EPISODE + pack_episode(ep)


def test_b1_a_row_pointing_into_a_foreign_record_refuses_never_answers(tmp_path):
    # Codex P3 lens 1 B1 (BLOCKER): a ghost episode framed inside a non-episode record's payload,
    # with one posting redirected (key AND seq) to it, was answered at confidence 1.0 although the
    # framing scan never yields it. The row's frame check no longer matches the bytes: it refuses.
    from snn_data_lake.substrate import episode_index as E

    eps = _eps(4)
    ghost = {"who": "self", "what": "GHOST", "where": "px", "when": "g", "affect": "calm"}
    gp = _ep_payload(ghost)
    filler = b"~" + b"\x00" * 7 + struct.pack("<I", len(gp)) + gp
    log, path = _lake(tmp_path, [])
    for e in eps:
        log.append(_ep_payload(e))
    log.append(filler)
    locs = [o for o, _p in log.replay_located()]
    ghost_off = locs[-1] + 4 + 8
    ix = _ei(tmp_path, log)
    ix.sync()
    tok = E._tok("who", "self")
    victim = next(  # the oldest genuine posting of who=self
        k
        for k, _s in ix.idx.iter_range(E.TAG, *E._bounds(tok))
        if int(k.split(":")[1], 16) == locs[0]
    )
    with ix.idx._db:
        ix.idx._db.execute("DELETE FROM records WHERE tag = ? AND key = ?", (E.TAG, victim))
        ix.idx._db.execute(
            "INSERT INTO records (tag, key, container, seq) VALUES (?, ?, 'lake', ?)",
            (E.TAG, victim.split(":")[0] + f":{ghost_off:016x}:" + victim.split(":")[2], ghost_off),
        )
    with pytest.raises(E.IndexCorrupt):  # the verified instance: only the frame check is left
        ix._recall({"who": "self"})
    got = ix.recall({"who": "self"})  # recall heals once from the lake: the genuine newest episode
    assert got["episode"] == _scan(path, {"who": "self"}) and "GHOST" not in str(got), got


def test_b1_a_row_redirected_to_another_genuine_frame_refuses(tmp_path):
    from snn_data_lake.substrate import episode_index as E

    log, path = _lake(tmp_path, _eps(6))
    ix = _ei(tmp_path, log)
    ix.sync()
    locs = [o for o, _p in log.replay_located()]
    tok = E._tok(
        "what", "rest"
    )  # rest at i = 0 and 4; point the newest (i = 4) posting at record 5's frame
    victim = next(
        k
        for k, _s in ix.idx.iter_range(E.TAG, *E._bounds(tok))
        if int(k.split(":")[1], 16) == locs[4]
    )
    with ix.idx._db:
        ix.idx._db.execute("DELETE FROM records WHERE tag = ? AND key = ?", (E.TAG, victim))
        ix.idx._db.execute(
            "INSERT INTO records (tag, key, container, seq) VALUES (?, ?, 'lake', ?)",
            (E.TAG, f"{tok}:{locs[5]:016x}:" + victim.split(":")[2], locs[5]),
        )
    with pytest.raises(
        E.IndexCorrupt
    ):  # record 5 is not 'rest' and its bytes are not the frame indexed
        ix._recall({"what": "rest"})
    got = ix.recall({"what": "rest"})
    assert got["episode"] == _scan(path, {"what": "rest"}) and got["source"] == "lake"


def test_b1_an_index_of_another_row_format_is_rebuilt_not_trusted(tmp_path):
    import sqlite3

    from snn_data_lake.substrate import episode_index as E

    log, path = _lake(tmp_path, _eps(8))
    ix = _ei(tmp_path, log)
    ix.sync()
    ix.close()
    db = sqlite3.connect(str(tmp_path / "idx.db"))
    with db:  # an index written by the previous row format: no ep_fmt marker
        db.execute("DELETE FROM meta WHERE k = 'ep_fmt'")
        db.execute("UPDATE records SET key = substr(key, 1, 33) WHERE tag = 256")
    db.close()
    ix = E.EpisodeIndex(log, tmp_path / "idx.db")
    got = ix.recall({"when": "5"})
    assert got["source"] == "lake" and got["episode"]["when"] == "5"
    assert ix.idx.meta("ep_fmt") == E.FMT


def test_m3_a_row_preserving_corruption_is_caught_by_the_digest_not_the_count(tmp_path):
    # Codex P3 lens 1 M3: one flipped character of a token key keeps the row count, so the count
    # check passed and a present unique cue answered MISS. The digest over the rows catches it and
    # the index rebuilds from the lake; a count-only variant (the red) misses the same corruption.
    import sqlite3

    from snn_data_lake.substrate import episode_index as E

    log, path = _lake(tmp_path, _eps(12))
    ix = _ei(tmp_path, log)
    ix.sync()
    ix.close()
    db = sqlite3.connect(str(tmp_path / "idx.db"))
    victim = db.execute(  # the newest 'reach' posting: the one a query reads first
        "SELECT key FROM records WHERE tag = ? AND key LIKE ? ORDER BY key DESC LIMIT 1",
        (E.TAG, E._tok("what", "reach") + ":%"),
    ).fetchone()[0]
    flipped = ("0" if victim[0] != "0" else "1") + victim[1:]
    with db:
        db.execute("UPDATE records SET key = ? WHERE tag = ? AND key = ?", (flipped, E.TAG, victim))
    n = db.execute("SELECT COUNT(*) FROM records WHERE tag = ?", (E.TAG,)).fetchone()[0]
    assert n == int(
        sqlite3.connect(str(tmp_path / "idx.db"))
        .execute("SELECT v FROM meta WHERE k = 'ep_rows'")
        .fetchone()[0]
    )  # the count is unchanged: the old check was blind here
    db.close()

    class CountOnly(E.EpisodeIndex):  # the red: the previous row check
        def _rows_ok(self):
            return self.idx.count_range(E.TAG, "", "￿") == int(self.idx.meta("ep_rows") or -1)

    cue = {"what": "reach"}
    want = _scan(path, cue)
    assert CountOnly(log, tmp_path / "idx.db").recall(cue)["episode"] != want  # hidden record
    got = E.EpisodeIndex(log, tmp_path / "idx.db").recall(cue)
    assert got["episode"] == want and got["source"] == "lake"  # rebuilt, then answered


def test_m7_rows_deleted_after_the_first_verification_are_not_trusted_from_the_cache(tmp_path):
    # Codex P3 lens 1 M7: the row check ran once per process; deleting a record's posting rows
    # afterwards (another connection to the index file) hid it until restart. Another connection's
    # commit voids the cached verdict; the rows are verified again, the index rebuilds, the answer
    # is the lake's. The red is the previous behaviour: the cache never voided.
    import sqlite3

    from snn_data_lake.substrate import episode_index as E

    log, path = _lake(tmp_path, _eps(12))
    cue = {"what": "reach"}
    want = _scan(path, cue)
    ix = _ei(tmp_path, log)
    assert ix.recall(cue)["episode"] == want  # verified once; the verdict is cached
    other = sqlite3.connect(str(tmp_path / "idx.db"))
    with other:
        other.execute(
            "DELETE FROM records WHERE tag = ? AND key LIKE ?",
            (E.TAG, E._tok("what", "reach") + ":%"),
        )
    other.close()

    class Cached(E.EpisodeIndex):  # the red: the cached verdict is never voided
        def _guard_cache(self):
            pass

    stale = Cached(log, tmp_path / "idx2.db")
    stale.sync()
    other = sqlite3.connect(str(tmp_path / "idx2.db"))
    with other:
        other.execute(
            "DELETE FROM records WHERE tag = ? AND key LIKE ?",
            (E.TAG, E._tok("what", "reach") + ":%"),
        )
    other.close()
    assert stale.recall(cue)["episode"] != want  # a hidden record: what the guard closes
    got = ix.recall(cue)
    assert got["episode"] == want and got["source"] == "lake"
    # and our own folds do not void it: a late append is folded without a second full verification
    passes = []
    real = ix._rows_ok
    ix._rows_ok = lambda: (passes.append(1), real())[1]
    from snn_data_lake.substrate import home as H

    late = {"who": "owner", "what": "wave", "where": "px", "when": "u9", "affect": "joy"}
    assert H.save_episode(late, path=path)
    assert ix.recall({"when": "u9"})["episode"] == late and passes == []


def test_b2_a_compaction_that_keeps_the_last_frame_in_place_rebuilds_by_file_identity(tmp_path):
    # Codex P3 lens 1 B2 (BLOCKER): [F20, A, F20, Z] rewritten as [A, Fn, Z] moves A but leaves Z,
    # the last folded frame, at its offset with its bytes: the anchor passed, A's cue was lost.
    # BinLog.rewrite replaces the file, so the lake's identity changes and the index rebuilds.
    from snn_data_lake.substrate import episode_index as E

    def filler(n, k=1):  # distinct bytes per k: the log dedups identical payloads
        return b"~" + bytes([k]) * (n - 1)

    a = {"who": "self", "what": "AAA", "where": "p", "when": "uniqueA", "affect": "calm"}
    z = {"who": "self", "what": "ZZZ", "where": "p", "when": "uniqueZ", "affect": "calm"}
    log, path = _lake(tmp_path, [])
    for pl in (filler(20, 1), _ep_payload(a), filler(20, 2), _ep_payload(z)):
        log.append(pl)
    z_off = [o for o, _p in log.replay_located()][-1]
    ix = _ei(tmp_path, log)
    cue = {"when": "uniqueA"}
    assert ix.recall(cue)["episode"] == a and ix.recall({"when": "uniqueZ"})["episode"] == z
    ix.close()
    pad = (
        z_off - 8 - (4 + len(_ep_payload(a))) - 4
    )  # the second filler's size that keeps Z in place
    log.rewrite([_ep_payload(a), filler(pad, 3), _ep_payload(z)])
    assert [o for o, _p in log.replay_located()][-1] == z_off  # the construction holds: Z unmoved
    fresh = E.EpisodeIndex(log, tmp_path / "idx.db")
    assert fresh._anchor_ok() is False  # the identity (not a later heal) flags the compaction
    got = fresh.recall(cue)
    assert got["episode"] == a and got["source"] == "lake"  # rebuilt by identity, then answered

    class NoIdentity(E.EpisodeIndex):  # the red: the last-frame anchor alone
        def _lake_id(self):
            return "1:1"

    log.rewrite(
        [filler(20, 1), _ep_payload(a), filler(20, 2), _ep_payload(z)]
    )  # back to the first layout
    stale = NoIdentity(log, tmp_path / "id2.db")
    stale.sync()
    log.rewrite([_ep_payload(a), filler(pad, 3), _ep_payload(z)])
    assert stale._anchor_ok() is True  # the red: the last-frame anchor alone is blind to it


def test_s7_recall_walks_the_posting_list_lazily_and_closes_its_cursor(tmp_path, monkeypatch):
    # Codex safety lens S7: candidates were fetched as a whole list before verification, so a common
    # token (who=self on every episode) loaded every posting; the newest verified row suffices.
    from snn_data_lake.substrate.lake_index import LakeIndex

    log, path = _lake(tmp_path, _eps(300))
    ix = _ei(tmp_path, log)
    ix.sync()
    pulled, closed = [], []
    real = LakeIndex.iter_range

    def counting(self, *a, **k):
        gen = real(self, *a, **k)
        try:
            for row in gen:
                pulled.append(row)
                yield row
        finally:
            closed.append(1)
            gen.close()

    monkeypatch.setattr(LakeIndex, "iter_range", counting)
    got = ix.recall({"who": "self"})
    assert got["episode"] == _scan(path, {"who": "self"}) and got["source"] == "lake"
    assert len(pulled) == 1 and closed == [1]  # one row fetched of 300; the cursor closed


def test_s8_a_reset_that_fails_midway_leaves_a_state_the_next_call_repairs(tmp_path, monkeypatch):
    # Codex safety lens S8: rebuild() deleted the rows, then the metadata delete raised; _checked
    # stayed True, so the retry trusted an empty index under its old watermark: MISS for a
    # present episode.
    import sqlite3

    log, path = _lake(tmp_path, _eps(12))
    cue = {"when": "5"}
    want = _scan(path, cue)
    for shape in ("atomic", "rows-then-fail"):
        ix = EpisodeIndex_at(tmp_path, log, shape)
        assert ix.recall(cue)["episode"] == want  # warm: verified once, _checked True
        real = ix.idx.clear

        def failing(tag, keys, real=real, ix=ix, shape=shape):
            if shape == "rows-then-fail":  # the old two-step shape: the rows go, the markers stay
                real(tag, [])
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(ix.idx, "clear", failing)
        with pytest.raises(sqlite3.OperationalError):
            ix.rebuild()
        assert ix._checked is False  # the cached verdict is void before anything is touched
        monkeypatch.setattr(ix.idx, "clear", real)
        got = ix.recall(cue)
        assert got["episode"] == want and got["source"] == "lake", shape


def EpisodeIndex_at(tmp_path, log, name):  # noqa: N802 - a tiny factory with its own index file
    from snn_data_lake.substrate.episode_index import EpisodeIndex

    return EpisodeIndex(log, tmp_path / f"{name}.db")


def test_s9_a_self_consistent_wrong_pointer_heals_once_from_the_lake_and_never_touches_it(
    tmp_path, monkeypatch
):
    # Codex safety lens S9: a posting whose key offset and seq name another valid frame passes the
    # count and the anchor; verification rejected the candidate and recall answered MISS. The index
    # now rebuilds ONCE from the lake and answers; a second corruption refuses. Only the index is
    # rebuilt: the lake file is byte-identical and no lake writer is called.
    from snn_data_lake.substrate import binlog as B
    from snn_data_lake.substrate import episode_index as E

    log, path = _lake(tmp_path, _eps(12))
    ix = _ei(tmp_path, log)
    ix.sync()
    locs = [o for o, _p in log.replay_located()]
    tok = E._tok("when", "1")
    victim = next(k for k, _s in ix.idx.iter_range(E.TAG, *E._bounds(tok)))
    moved = f"{victim.split(':')[0]}:{locs[2]:016x}:{victim.split(':')[2]}"
    with ix.idx._db:
        ix.idx._db.execute("DELETE FROM records WHERE tag = ? AND key = ?", (E.TAG, victim))
        ix.idx._db.execute(
            "INSERT INTO records (tag, key, container, seq) VALUES (?, ?, 'lake', ?)",
            (E.TAG, moved, locs[2]),
        )
    before = (Path(path).read_bytes(), Path(path).stat().st_mtime_ns)
    for writer in ("append", "extend", "rewrite"):
        monkeypatch.setattr(
            B.BinLog, writer, lambda *a, **k: (_ for _ in ()).throw(AssertionError("lake write"))
        )
    got = ix.recall({"when": "1"})
    assert got["source"] == "lake" and got["episode"]["when"] == "1", got
    assert (Path(path).read_bytes(), Path(path).stat().st_mtime_ns) == before  # the lake: untouched

    healed = []
    real = ix.rebuild
    monkeypatch.setattr(ix, "rebuild", lambda: (healed.append(1), real())[1])
    monkeypatch.setattr(
        ix,
        "_recall",
        lambda cue: (_ for _ in ()).throw(E.IndexCorrupt("the lake itself disagrees")),
    )
    refused = ix.recall(
        {"when": "1"}
    )  # a corruption a rebuild does not cure refuses, after ONE heal
    assert refused["source"] == "refused" and "disagrees" in refused["reason"] and healed == [1]


def test_p3_a_watermark_that_disagrees_with_its_anchor_rebuilds_instead_of_hiding_appends(tmp_path):
    # Codex P3 lens 2 MAJOR: the anchor validated the previous frame but not that the watermark was
    # that frame's end, so progress corrupted to the new lake end hid every record appended since.
    import sqlite3

    from snn_data_lake.substrate import episode_index as E
    from snn_data_lake.substrate import home as H

    log, path = _lake(tmp_path, _eps(6))
    ix = _ei(tmp_path, log)
    ix.sync()
    ix.close()
    late = {"who": "owner", "what": "wave", "where": "px", "when": "u9", "affect": "joy"}
    assert H.save_episode(late, path=path)
    size = Path(path).stat().st_size
    db = sqlite3.connect(str(tmp_path / "idx.db"))
    with db:
        db.execute("UPDATE meta SET v = ? WHERE k = 'ep_watermark'", (str(size),))
    db.close()

    class NoWatermarkCheck(E.EpisodeIndex):  # the red: the previous anchor check
        def _anchor_ok(self):
            off, sha = self.idx.meta("ep_anchor_off"), self.idx.meta("ep_anchor_sha")
            payload = self.log.read_at(int(off))
            return hashlib.sha256(payload).hexdigest() == sha

    cue = {"when": "u9"}
    assert NoWatermarkCheck(log, tmp_path / "idx.db").recall(cue)["source"] == "miss"  # hidden
    fresh = E.EpisodeIndex(log, tmp_path / "idx.db")
    assert fresh._anchor_ok() is False
    got = fresh.recall(cue)
    assert got["episode"] == late and got["source"] == "lake"


def test_s9_the_builder_constructs_no_second_binlog(tmp_path, monkeypatch):
    from snn_data_lake.substrate import binlog as B

    log, path = _lake(tmp_path, _eps(10))
    built = []
    real = B.BinLog.__init__

    def counting(self, *a, **k):
        built.append(a)
        real(self, *a, **k)

    monkeypatch.setattr(B.BinLog, "__init__", counting)
    ix = _ei(tmp_path, log)
    ix.sync()
    ix.recall({"when": "3"})
    assert built == []  # reads through the instance it was handed
    B.BinLog(path)  # the red: a builder constructing BinLog(path) is caught by the same wrap
    assert len(built) == 1


def test_learner_events_advance_the_watermark_but_are_never_indexed(tmp_path):
    from snn_data_lake.substrate import home as H

    log, path = _lake(tmp_path, _eps(5))
    teach = {
        "kind": "teach",
        "event_id": "1-1",
        "session": "1",
        "seq": "1",
        "source": "k",
        "target": "t",
    }
    assert H.save_episode(teach, path=path)
    ix = _ei(tmp_path, log)
    ix.sync()
    assert ix.recall({"source": "k"})["source"] == "miss"  # not a lived episode
    assert ix._wm() == len(open(path, "rb").read())  # but its frame was folded past


def test_s5_a_compaction_is_detected_and_the_index_rebuilds(tmp_path):
    # BinLog.rewrite moves every offset and keeps stale() False; the anchor catches it
    from snn_data_lake.substrate import episode_index as E

    log, path = _lake(tmp_path, _eps(30))
    ix = _ei(tmp_path, log)
    ix.sync()
    survivors = list(log.replay())[3:]  # a sleep compaction that drops the three oldest records
    log.rewrite(survivors)
    assert ix.recall({"when": "20"})["episode"]["when"] == "20"  # detected -> rebuilt -> found
    assert ix.recall({"when": "1"})["source"] == "miss"  # the dropped record is genuinely gone

    class NoDetect(E.EpisodeIndex):  # the red: detection disabled
        def _anchor_ok(self):
            return True

    (tmp_path / "b").mkdir()
    log2, path2 = _lake(tmp_path / "b", _eps(30))
    blind = NoDetect(log2, tmp_path / "blind.db")
    blind.sync()
    log2.rewrite(list(log2.replay())[3:])
    # stale offsets: present in the lake, unreachable through the blind index (a hidden record)
    assert blind.recall({"when": "20"})["source"] == "miss"
    assert _scan(path2, {"when": "20"}) is not None


def test_s5_a_truncated_lake_also_rebuilds(tmp_path):
    log, path = _lake(tmp_path, _eps(10))
    ix = _ei(tmp_path, log)
    ix.sync()
    with open(path, "r+b") as f:
        f.truncate(60)  # far shorter than the watermark
    assert (
        ix.recall({"when": "9"})["source"] == "miss"
    )  # rebuilt over what remains, no stale answer
    assert ix._wm() <= 60


def test_s6_a_read_failure_refuses_visibly_never_a_miss(tmp_path, monkeypatch):
    from snn_data_lake.substrate import episode_index as E

    log, path = _lake(tmp_path, _eps(10))
    ix = _ei(tmp_path, log)
    ix.sync()

    def boom(*a, **k):
        raise OSError("planted sqlite failure")

    monkeypatch.setattr(ix.idx, "iter_range", boom)
    got = ix.recall({"when": "3"})
    assert got["source"] == "refused" and "planted sqlite failure" in got["reason"]

    class Quiet(E.EpisodeIndex):  # the red: a variant that answers miss on an error is caught
        def recall(self, cue):
            try:
                return super()._recall(cue)
            except Exception:  # noqa: BLE001
                return {"episode": {}, "confidence": 0.0, "source": "miss"}

    quiet = Quiet(log, tmp_path / "q.db")
    monkeypatch.setattr(quiet.idx, "iter_range", boom)
    assert quiet.recall({"when": "3"})["source"] == "miss"  # exactly the hidden record B4 forbids


# ---- S7: rebuild equality through the lake_rebuild_gate (planted-row red, lake-tracking control) -
def _digest(index_path):
    from snn_data_lake.substrate.lake_index import LakeIndex

    idx = LakeIndex(index_path)
    try:
        return idx.logical_dump_digest(), idx.row_count()
    finally:
        idx.close()


def test_s7_rebuild_equality_with_a_planted_row_red_and_a_lake_tracking_control(tmp_path):
    from snn_data_lake.gates import lake_rebuild_gate as G
    from snn_data_lake.substrate import episode_index as E
    from snn_data_lake.substrate import home as H

    _lake(tmp_path, _eps(40))
    ok, msg = G.assert_rebuild(
        tmp_path, tmp_path / "idx.db"
    )  # delete -> rebuild -> equal, both tiers
    assert ok, msg
    clean, rows = _digest(tmp_path / "idx.db")
    assert rows > 40  # pointer rows AND episode posting rows live in the one file
    # the planted-row red: a row added behind the gate's back moves the digest; a rebuild drops it
    from snn_data_lake.substrate.lake_index import LakeIndex

    idx = LakeIndex(tmp_path / "idx.db")
    idx.put_batch([(E.TAG, E._key(E._tok("who", "ghost"), 99, "00000000"), "lake", 99)])
    idx.close()
    assert _digest(tmp_path / "idx.db")[0] != clean
    G._delete_index(tmp_path / "idx.db")
    G._build_all(tmp_path, tmp_path / "idx.db")
    assert _digest(tmp_path / "idx.db")[0] == clean
    # the lake-tracking control: one appended EPISODE must move the rebuilt digest
    late = {"who": "owner", "what": "wave", "where": "px", "when": "u9", "affect": "joy"}
    assert H.save_episode(late, path=str(tmp_path / "brain_lake.bin"))
    G._delete_index(tmp_path / "idx.db")
    G._build_all(tmp_path, tmp_path / "idx.db")
    assert _digest(tmp_path / "idx.db")[0] != clean


# ---- enforcer #6734 MINOR-3/4, NIT-5/6 ----------------------------------------------------------
def test_s6_a_home_with_no_lake_yet_is_a_miss_not_a_refusal(tmp_path):
    # MINOR-3: sync() stat'ed the lake unconditionally, so a home with no brain_lake.bin yet
    # answered "refused" while every other reader of it (replay_located, load_episodes) misses.
    log, _path = _lake(tmp_path, _eps(2))
    os.remove(log.path)
    got = _ei(tmp_path, log).recall({"who": "self"})
    assert got["source"] == "miss", got


def test_s6_a_malformed_lake_id_marker_rebuilds_instead_of_refusing_for_ever(tmp_path):
    # MINOR-4: `was.split(":")[1]` raised IndexError on an ep_lake_id with no ":", which is not
    # IndexCorrupt, so the one index-only heal never ran and recall refused on every call.
    log, _path = _lake(tmp_path, _eps(6))
    ix = _ei(tmp_path, log)
    want = ix.recall({"when": "5"})
    assert want["source"] == "lake"
    ix.idx.put_batch([], meta={"ep_lake_id": "damaged"})
    got = ix.recall({"when": "5"})
    assert got["source"] == "lake" and got["episode"] == want["episode"], got


def test_a_closed_index_reopens_on_the_next_recall(tmp_path):
    # NIT-5: close() left the dead connection in place, so a later recall refused with
    # "Cannot operate on a closed database".
    log, _path = _lake(tmp_path, _eps(4))
    ix = _ei(tmp_path, log)
    want = ix.recall({"when": "3"})
    ix.close()
    assert ix.recall({"when": "3"}) == want


def test_s2_replay_located_refuses_a_start_inside_the_magic_like_read_at(tmp_path):
    # NIT-6: replay_located framed the magic as a length where read_at raised ValueError.
    log, _path = _lake(tmp_path, _eps(2))
    with pytest.raises(ValueError):
        list(log.replay_located(start=3))
    with pytest.raises(ValueError):
        log.read_at(3)

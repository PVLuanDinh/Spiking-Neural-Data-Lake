"""P8 null-scale gate: one reachable red per rule, the clean pass, the dormancy_gate disposition.

specs/022-unified-brain-arc/evidence/p8-null-scale-gate.md, sections 1 and 3 and addenda 1-3.
Every input is built in a temporary tree (or an injected history function for R-EXPIRED), so each
red is shown to fire and the live-tree pass is not vacuous. The inputs are written out by hand and
not taken from the gate's own self-check plants: a test must not share its oracle with the code
it tests.
"""

import pytest

from snn_data_lake.gates import dormancy_gate as dg
from snn_data_lake.gates import null_scale_gate as ng

EV = "specs/s/evidence/"
OTHER = "specs/s/evidence/other.md"
SHA = "a" * 40
STATEMENT = "The method was deleted on its own null result."
STRUCT = "STRUCTURAL: zero call sites, the proposer is empty"


def rec(method="m", units="4", ticks="600", seeds="8", mde="0.5", retest=STRUCT):
    fields = f"units={units} | ticks={ticks} | seeds={seeds} | mde={mde} | retest={retest}"
    return f"NULL-RECORD: {method} | {fields}"


def retest_rec(method="m", axis="units", value="40", result="null", of=EV + "a.md"):
    return f"RETEST-RECORD: {method} | axis={axis} | {axis}={value} | result={result} | of={of}"


def tree(tmp_path, files):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def run(tmp_path, files, **kwargs):
    kwargs.setdefault("grandfathered", frozenset())
    return ng.check(tree(tmp_path, files), **kwargs)


def reds(report, code):
    return [r for r in report.reds if r.code == code]


# ---------------------------------------------------------------- the clean pass


def test_a_clean_tree_passes_and_counts(tmp_path):
    by_ref = rec(retest="REF:" + EV + "b.md#m")
    by_sha = rec(method="n", retest="PENDING since " + SHA)
    files = {
        EV + "a.md": f"# t\n{STATEMENT}\n{by_ref}\n{by_sha}\n",
        EV + "b.md": "# r\n" + retest_rec() + "\n",
    }
    report = run(tmp_path, files, history=lambda sha: [(sha, "feat: x")])
    assert report.reds == []
    assert report.counts["NULL-RECORDs"] == 2
    assert report.counts["RETEST-RECORDs"] == 1
    assert report.counts["statements"] == 1


def test_an_empty_tree_is_clean_and_counts_zero(tmp_path):
    report = run(tmp_path, {EV + "a.md": "# nothing\n"})
    assert report.reds == [] and report.counts["statements"] == 0


# ---------------------------------------------------------------- R-FIELD


@pytest.mark.parametrize(
    "line",
    [
        "NULL-RECORD: m | units=4 | ticks=600 | seeds=8 | mde=0.5",
        "NULL-RECORD: m | units=4 | ticks=600 | seeds=8 | mde=0.5 | wat=1 | retest=" + STRUCT,
        rec(units="abc"),
        rec(units="0"),
        rec(seeds="n/a"),
        rec(mde="unknown"),
        rec(mde="big"),
        rec(retest="STRUCTURAL: too short"),
        rec(retest="PENDING since abc123"),
        rec(retest="REF:no-method-after-the-path"),
        rec(retest="CANDIDATE"),
        rec(retest="NOT-NEEDED: harm measured harm, the bar was cleared"),
        "RETEST-RECORD: m | axis=units | units=40 | result=null",
        "RETEST-RECORD: m | axis=cores | cores=40 | result=null | of=" + EV + "a.md",
        "RETEST-RECORD: m | axis=units | ticks=40 | result=null | of=" + EV + "a.md",
        retest_rec(value="forty"),
        retest_rec(result="maybe"),
    ],
)
def test_a_malformed_record_is_r_field(tmp_path, line):
    report = run(tmp_path, {EV + "a.md": "# t\n" + line + "\n"})
    assert reds(report, "R-FIELD"), line


@pytest.mark.parametrize("units,ticks", [("n/a", "600"), ("4", "n/a"), ("n/a", "n/a")])
def test_n_a_is_allowed_for_units_and_ticks(tmp_path, units, ticks):
    report = run(tmp_path, {EV + "a.md": "# t\n" + rec(units=units, ticks=ticks) + "\n"})
    assert not reds(report, "R-FIELD")


def test_a_record_inside_a_fenced_block_is_an_example_not_a_record(tmp_path):
    report = run(tmp_path, {EV + "a.md": "# t\n```\nNULL-RECORD: <method> | units=<int>\n```\n"})
    assert report.reds == [] and report.counts["NULL-RECORDs"] == 0


def test_records_in_docs_are_read_and_records_outside_the_scope_are_not(tmp_path):
    report = run(
        tmp_path, {"docs/x.md": "NULL-RECORD: broken\n", "notes/y.md": "NULL-RECORD: broken\n"}
    )
    assert [r.where for r in reds(report, "R-FIELD")] == ["docs/x.md:1"]


# ---------------------------------------------------------------- R-UNRECORDED


def test_a_statement_with_no_record_in_its_section_is_r_unrecorded(tmp_path):
    report = run(tmp_path, {EV + "a.md": f"# t\n{STATEMENT}\n"})
    assert [r.where for r in reds(report, "R-UNRECORDED")] == [EV + "a.md:2"]


@pytest.mark.parametrize(
    "text",
    [
        "Retired on the null.",
        "deletion on a null",
        "They were REMOVED ON THEIR OWN NULL.",
        "deleted on its null",
    ],
)
def test_the_statement_regex_reads_each_form(tmp_path, text):
    assert reds(run(tmp_path, {EV + "a.md": f"# t\n{text}\n"}), "R-UNRECORDED")


def test_a_record_in_the_same_section_satisfies_the_statement(tmp_path):
    report = run(tmp_path, {EV + "a.md": f"# t\n{STATEMENT}\n\n{rec()}\n"})
    assert not reds(report, "R-UNRECORDED")


def test_a_record_in_a_deeper_subsection_belongs_to_the_section(tmp_path):
    report = run(tmp_path, {EV + "a.md": f"## t\n{STATEMENT}\n### detail\n{rec()}\n"})
    assert not reds(report, "R-UNRECORDED")


def test_a_record_in_a_sibling_section_does_not(tmp_path):
    report = run(tmp_path, {EV + "a.md": f"## one\n{STATEMENT}\n## two\n{rec()}\n"})
    assert reds(report, "R-UNRECORDED")


def test_a_record_in_a_file_elsewhere_does_not(tmp_path):
    report = run(tmp_path, {EV + "a.md": f"# t\n{STATEMENT}\n", EV + "b.md": f"# t\n{rec()}\n"})
    assert reds(report, "R-UNRECORDED")


def test_a_statement_outside_evidence_files_is_out_of_scope(tmp_path):
    report = run(
        tmp_path, {"docs/x.md": f"# t\n{STATEMENT}\n", "specs/s/plan.md": f"# t\n{STATEMENT}\n"}
    )
    assert report.reds == [] and report.counts["statements"] == 0


def test_a_grandfathered_statement_is_counted_and_not_red(tmp_path):
    key = (EV + "a.md", ng.line_key(STATEMENT))
    report = run(tmp_path, {EV + "a.md": f"# t\n{STATEMENT}\n"}, grandfathered=frozenset({key}))
    assert report.reds == []
    assert report.counts["statements"] == 1 and report.counts["grandfathered"] == 1


def test_a_grandfathered_line_survives_a_move_and_not_a_reword(tmp_path):
    key = (EV + "a.md", ng.line_key(STATEMENT))
    moved = run(
        tmp_path / "moved",
        {EV + "a.md": f"# t\n\n\nfiller\n{STATEMENT}\n"},
        grandfathered=frozenset({key}),
    )
    assert moved.reds == []
    reworded = run(
        tmp_path / "reworded",
        {EV + "a.md": f"# t\n{STATEMENT} Again.\n"},
        grandfathered=frozenset({key}),
    )
    assert {r.code for r in reworded.reds} == {"R-UNRECORDED", "R-STALE-GRANDFATHER"}


# ---------------------------------------------------------------- R-STALE-GRANDFATHER


def test_a_grandfathered_entry_that_matches_no_line_is_stale(tmp_path):
    key = (EV + "a.md", "0" * 12)
    report = run(tmp_path, {EV + "a.md": "# t\nnothing here\n"}, grandfathered=frozenset({key}))
    assert [r.code for r in report.reds] == ["R-STALE-GRANDFATHER"]


# ---------------------------------------------------------------- R-RETEST-REF


def test_a_ref_whose_target_has_no_retest_record_is_r_retest_ref(tmp_path):
    files = {EV + "a.md": f"# t\n{rec(retest='REF:' + EV + 'b.md#m')}\n", EV + "b.md": "# r\n"}
    assert reds(run(tmp_path, files), "R-RETEST-REF")


def test_a_ref_to_a_missing_file_is_r_retest_ref(tmp_path):
    files = {EV + "a.md": f"# t\n{rec(retest='REF:' + EV + 'gone.md#m')}\n"}
    assert reds(run(tmp_path, files), "R-RETEST-REF")


def test_a_retest_that_points_back_elsewhere_is_r_retest_ref(tmp_path):
    files = {
        EV + "a.md": f"# t\n{rec(retest='REF:' + EV + 'b.md#m')}\n",
        EV + "b.md": "# r\n" + retest_rec(of=OTHER) + "\n",
    }
    assert reds(run(tmp_path, files), "R-RETEST-REF")


def test_a_retest_for_another_method_is_r_retest_ref(tmp_path):
    files = {
        EV + "a.md": f"# t\n{rec(retest='REF:' + EV + 'b.md#m')}\n",
        EV + "b.md": "# r\n" + retest_rec(method="other") + "\n",
    }
    assert reds(run(tmp_path, files), "R-RETEST-REF")


def test_a_retest_record_with_no_original_is_r_retest_ref(tmp_path):
    assert reds(run(tmp_path, {EV + "b.md": "# r\n" + retest_rec() + "\n"}), "R-RETEST-REF")


# ---------------------------------------------------------------- R-RETEST-SCALE


def scale_files(value, axis="units", **null):
    return {
        EV + "a.md": f"# t\n{rec(retest='REF:' + EV + 'b.md#m', **null)}\n",
        EV + "b.md": "# r\n" + retest_rec(axis=axis, value=value) + "\n",
    }


def test_a_retest_under_10x_is_r_retest_scale(tmp_path):
    assert reds(run(tmp_path, scale_files("39")), "R-RETEST-SCALE")


def test_a_retest_at_exactly_10x_passes(tmp_path):
    assert run(tmp_path, scale_files("40")).reds == []


def test_the_scale_is_read_on_the_named_axis(tmp_path):
    assert reds(run(tmp_path / "a", scale_files("5999", axis="ticks")), "R-RETEST-SCALE")
    assert run(tmp_path / "b", scale_files("6000", axis="ticks")).reds == []


def test_a_retest_on_an_axis_the_original_marks_n_a_is_r_retest_scale(tmp_path):
    found = reds(run(tmp_path, scale_files("40", units="n/a")), "R-RETEST-SCALE")
    assert found and "n/a" in found[0].message and "cannot be the re-test axis" in found[0].message


# ---------------------------------------- R-EXPIRED: the four points of an arc


def pending(tmp_path):
    return {EV + "a.md": f"# t\n{rec(retest='PENDING since ' + SHA)}\n"}


def history_of(*subjects):
    return lambda sha: [(f"{i:040x}", s) for i, s in enumerate(subjects)]


CLOSING = "chore(record): closing execution record, arc 7"


def test_expired_before_the_record_is_green(tmp_path):
    assert run(tmp_path, pending(tmp_path), history=history_of("feat: a", "feat: b")).reds == []


def test_expired_with_the_record_as_head_is_green(tmp_path):
    assert run(tmp_path, pending(tmp_path), history=history_of("feat: a", CLOSING)).reds == []


def test_expired_one_commit_after_the_record_is_red(tmp_path):
    report = run(
        tmp_path, pending(tmp_path), history=history_of("feat: a", CLOSING, "feat: next arc")
    )
    assert [r.code for r in report.reds] == ["R-EXPIRED"]


def test_expired_with_an_unknown_sha_is_red_never_a_pass(tmp_path):
    report = run(tmp_path, pending(tmp_path), history=lambda sha: None)
    assert [r.code for r in report.reds] == ["R-EXPIRED"]
    assert "history unavailable" in report.reds[0].message


def test_expired_when_the_entry_is_created_at_head_is_green(tmp_path):
    assert run(tmp_path, pending(tmp_path), history=lambda sha: []).reds == []


def test_the_default_history_reads_git_and_a_made_up_sha_is_unavailable():
    assert ng.git_history("b" * 40) is None


# ---------------------------------------- addendum 3: the exclusion is by name


def test_the_two_named_files_may_hold_a_statement_without_a_record(tmp_path):
    files = {ng.PREREG: f"# t\n{STATEMENT}\n", ng.TRIAGE: f"# t\n{STATEMENT}\n"}
    report = run(tmp_path, files)
    assert reds(report, "R-UNRECORDED") == []
    assert report.counts["statements"] == 2


def test_a_statement_in_any_other_evidence_file_still_reds(tmp_path):
    other = "specs/022-unified-brain-arc/evidence/p8-null-scale-gate-copy.md"
    assert reds(run(tmp_path, {other: f"# t\n{STATEMENT}\n"}), "R-UNRECORDED")


def test_the_named_files_are_still_read_for_record_fields(tmp_path):
    assert reds(run(tmp_path, {ng.PREREG: "# t\nNULL-RECORD: broken\n"}), "R-FIELD")


def test_the_exclusion_list_is_exactly_the_two_files_and_is_printed():
    assert ng.UNRECORDED_EXCLUDED == (ng.PREREG, ng.TRIAGE)
    out = ng.render(ng.Report())
    assert ng.PREREG in out and ng.TRIAGE in out


# ---------------------------------------- addenda 1 and 2: triage-only values


def test_unknown_scale_fields_are_allowed_in_the_triage_file_only(tmp_path):
    line = rec(units="unknown", ticks="unknown", seeds="unknown", mde="unknown", retest="CANDIDATE")
    assert run(tmp_path / "t", {ng.TRIAGE: "# t\n" + line + "\n"}).reds == []
    assert reds(run(tmp_path / "e", {OTHER: "# t\n" + line + "\n"}), "R-FIELD")


@pytest.mark.parametrize("field", ["units", "ticks", "seeds", "mde"])
def test_each_unknown_is_triage_only(tmp_path, field):
    line = rec(**{field: "unknown"})
    assert reds(run(tmp_path / "e", {OTHER: "# t\n" + line + "\n"}), "R-FIELD")
    assert run(tmp_path / "t", {ng.TRIAGE: "# t\n" + line + "\n"}).reds == []


def test_not_needed_is_allowed_in_the_triage_file_only(tmp_path):
    line = rec(retest="NOT-NEEDED: harm the harm cleared its pre-registered bar")
    assert run(tmp_path / "t", {ng.TRIAGE: "# t\n" + line + "\n"}).reds == []
    assert reds(run(tmp_path / "e", {OTHER: "# t\n" + line + "\n"}), "R-FIELD")


def test_a_not_needed_with_an_unknown_kind_or_short_reason_is_r_field_even_in_triage(tmp_path):
    for retest in ("NOT-NEEDED: whim some reason that is long enough", "NOT-NEEDED: harm short"):
        assert reds(run(tmp_path, {ng.TRIAGE: "# t\n" + rec(retest=retest) + "\n"}), "R-FIELD")


# ---------------------------------------- STRUCTURAL is printed, never judged


def test_structural_reasons_are_printed_for_review_and_the_limit_is_stated(tmp_path):
    report = run(tmp_path, {EV + "a.md": f"# t\n{rec()}\n"})
    assert report.reds == [] and len(report.structural) == 1
    out = ng.render(report)
    assert "zero call sites" in out and "cannot judge a STRUCTURAL argument" in out


# ---------------------------------------- R-EXEMPT, and dormancy_gate's disposition


GOOD_REASON = (
    f"unwired on a null; NULL-RECORD:{EV}a.md since={SHA}; re-test or delete before the next arc"
)


def alive(sha):
    return [(sha, "feat: x")]


def test_a_reason_naming_record_and_sha_passes():
    assert ng.exempt_problems("m", GOOD_REASON, history=alive) == []


def test_a_reason_with_no_null_record_is_r_exempt():
    assert ng.exempt_problems("m", f"since={SHA}", history=alive)


def test_a_reason_with_no_full_sha_is_r_exempt():
    assert ng.exempt_problems("m", f"NULL-RECORD:{EV}a.md since=abc1234", history=alive)


def test_a_reason_with_an_expired_since_is_r_exempt():
    history = history_of("feat: a", CLOSING, "feat: next")
    assert ng.exempt_problems("m", GOOD_REASON, history=history)


def test_dormancy_gate_reds_a_planted_bad_scale_retest_entry():
    planted = {"some_module": ("scale-retest", "no record, no sha")}
    with pytest.raises(AssertionError, match="some_module"):
        dg._assert_scale_retest(planted, history=alive)


def test_dormancy_gate_passes_a_planted_good_scale_retest_entry():
    dg._assert_scale_retest({"some_module": ("scale-retest", GOOD_REASON)}, history=alive)


def test_dormancy_gate_ignores_other_dispositions():
    dg._assert_scale_retest({"some_module": ("permanent", "no record, no sha")}, history=alive)


def test_dormancy_main_reds_a_bad_scale_retest_row(monkeypatch):
    key = next(iter(dg.EXEMPT))
    monkeypatch.setattr(
        dg, "EXEMPT", {**dg.EXEMPT, key: ("scale-retest", "planted; no record, no sha")}
    )
    with pytest.raises(AssertionError, match=key):
        dg.main()


def test_dormancy_uses_the_gates_check_and_has_no_copy_of_it():
    assert dg._assert_scale_retest.__globals__["null_scale_gate"] is ng


def test_dormancy_selfcheck_plants_in_a_copy_and_reds_when_blind(monkeypatch):
    before = dict(dg.EXEMPT)
    dg._scale_retest_selfcheck()
    assert dg.EXEMPT == before
    monkeypatch.setattr(ng, "exempt_problems", lambda *a, **k: [])
    with pytest.raises(AssertionError, match="did not red"):
        dg._scale_retest_selfcheck()


# ---------------------------------------------------------------- the CLI


def test_the_self_check_passes(capsys):
    assert ng.main(["--self-check"]) == 0
    assert "OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    "blind",
    [
        "expiry_problem",
        "_check_retest_scale",
        "_check_retest_ref",
        "_check_unrecorded",
        "_check_stale",
    ],
)
def test_the_self_check_fails_when_a_check_is_blind(monkeypatch, capsys, blind):
    monkeypatch.setattr(
        ng, blind, (lambda *a, **k: None) if blind == "expiry_problem" else (lambda *a, **k: [])
    )
    assert ng.main(["--self-check"]) == 1
    assert "did not red" in capsys.readouterr().out


def test_the_live_tree_is_green_and_every_statement_is_grandfathered(capsys):
    report = ng.check()
    assert report.reds == []
    assert report.counts["grandfathered"] == len(ng.GRANDFATHERED)
    assert ng.main(["--assert"]) == 0
    assert "statements:" in capsys.readouterr().out


def test_assert_exits_1_on_a_red(tmp_path, capsys):
    root = tree(tmp_path, {EV + "a.md": f"# t\n{STATEMENT}\n"})
    assert ng.main(["--assert"], root=root) == 1
    assert "R-UNRECORDED" in capsys.readouterr().out

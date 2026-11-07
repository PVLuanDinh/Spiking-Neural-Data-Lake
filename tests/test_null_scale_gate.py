"""P8 null-scale gate: one reachable red per rule, the clean pass, the dormancy_gate disposition.

specs/022-unified-brain-arc/evidence/p8-null-scale-gate.md, sections 1 and 3 and addenda 1-3.
Every input is built in a temporary tree (or an injected history function for R-EXPIRED), so each
red is shown to fire and the live-tree pass is not vacuous. The inputs are written out by hand and
not taken from the gate's own self-check plants: a test must not share its oracle with the code
it tests.
"""

import functools
import subprocess
from pathlib import Path

import pytest

from snn_data_lake.gates import dormancy_gate as dg
from snn_data_lake.gates import null_scale_gate as ng

EV = "specs/s/evidence/"
OTHER = "specs/s/evidence/other.md"
SHA = "a" * 40
STATEMENT = "The method was deleted on its own null result."
STRUCT = "STRUCTURAL: zero call sites, the proposer is empty"
# addendum 10: a STRUCTURAL record's paragraph must hold a prose line (R-STRUCTURAL-ARG)
ARGUMENT = "The argument: no call site reaches the method, checked by a positive-controlled grep."


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
    report = run(tmp_path, files, history=alive)
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
        rec(retest="NOT-NEEDED: at-scale measured at the scale the code runs at"),
        rec(retest="NOT-NEEDED: restatement the same null as an earlier record"),
        rec(retest="NOT-NEEDED: harm short reason"),
        rec(retest="NOT-NEEDED: harm a reason that | carries the field separator"),
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
        "deleting on a null",
        "removes on its own null",
        "retiring on the null",
        "retires on their null",
        "removal on a null",
        "deletes on the null",
        "removing on a null",
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
        {EV + "a.md": f"# t\n\n\nfiller\n\n{STATEMENT}\n"},
        grandfathered=frozenset({key}),
    )
    assert moved.reds == []
    reworded = run(
        tmp_path / "reworded",
        {EV + "a.md": f"# t\n{STATEMENT} Again.\n"},
        grandfathered=frozenset({key}),
    )
    assert {r.code for r in reworded.reds} == {"R-UNRECORDED", "R-STALE-GRANDFATHER"}


def test_a_verbatim_copy_of_a_grandfathered_line_in_a_new_section_is_red(tmp_path):
    # review #4483 m1 / #4484 (a): one grandfather key exempts ONE statement, not every copy
    key = (EV + "a.md", ng.line_key(STATEMENT))
    report = run(
        tmp_path,
        {EV + "a.md": f"# old\n{STATEMENT}\n\n# new\n{STATEMENT}\n"},
        grandfathered=frozenset({key}),
    )
    assert [r.code for r in report.reds] == ["R-UNRECORDED"]
    assert report.counts["grandfathered"] == 1


def test_a_reason_that_carries_the_field_separator_is_r_field(tmp_path):
    # review #4483 m2: "STRUCTURAL: <reason> | note=x" must not hide an unknown key in the reason
    reason = "the path has no synapses at any scale whatsoever"
    head = "NULL-RECORD: m | units=n/a | ticks=n/a | seeds=1 | mde=0.1 | retest=STRUCTURAL: "
    bad = run(tmp_path / "bad", {EV + "a.md": f"# t\n{ARGUMENT}\n{head}{reason} | note=x\n"})
    good = run(tmp_path / "good", {EV + "a.md": f"# t\n{ARGUMENT}\n{head}{reason}\n"})
    assert [r.code for r in bad.reds] == ["R-FIELD"]
    assert good.reds == []


# ---------------------------------------- F1: a wrapped statement is one paragraph


def test_a_statement_wrapped_across_lines_is_r_unrecorded_at_the_paragraph_first_line(tmp_path):
    report = run(
        tmp_path, {EV + "a.md": "# t\nintro line\nthe path was a deletion on\n a null result.\n"}
    )
    assert [r.where for r in reds(report, "R-UNRECORDED")] == [EV + "a.md:2"]
    assert report.counts["statements"] == 1


def test_a_wrapped_statement_is_satisfied_by_a_record_in_its_section(tmp_path):
    text = f"# t\nthe path was a deletion on\na null result.\n\n{rec()}\n"
    assert not reds(run(tmp_path, {EV + "a.md": text}), "R-UNRECORDED")


@pytest.mark.parametrize(
    "text",
    [
        "# t\na deletion on\n\na null result.\n",
        "# t\na deletion on\n## sub\na null result.\n",
        "# t\na deletion on\n```\na null result.\n```\n",
    ],
)
def test_a_blank_line_a_heading_or_a_fence_ends_the_paragraph(tmp_path, text):
    report = run(tmp_path, {EV + "a.md": text})
    assert report.reds == [] and report.counts["statements"] == 0


def test_a_wrapped_grandfathered_statement_is_keyed_on_the_joined_paragraph(tmp_path):
    joined = "the path was a deletion on a null result."
    key = (EV + "a.md", ng.line_key(joined))
    text = "# t\nthe path was a deletion on\n a null result.\n"
    report = run(tmp_path, {EV + "a.md": text}, grandfathered=frozenset({key}))
    assert report.reds == [] and report.counts["grandfathered"] == 1


def test_the_live_tree_counts_the_trajectory_audit_wrapped_statement():
    path = "specs/022-unified-brain-arc/evidence/trajectory-audit-2026-09-01.md"
    found = [st for st in ng.statements(ng._scan(ng.REPO)) if st.path == path]
    assert any('`reason_out` branch (P8 "deletion on a null")' in st.text for st in found)


# ---------------- integrator fold: the F1 join must not drop headings or fenced lines


@pytest.mark.parametrize(
    ("text", "where"),
    [
        ("# t\n\n## Results: the rule was deleted on its own null\n\nbody\n", "a.md:3"),
        ("# t\n\nthe rule was deleted on its own null\n=======\n\nbody\n", "a.md:3"),
        ("# t\n\n```\nthe rule was deleted on its own null\n```\n", "a.md:4"),
    ],
)
def test_a_statement_in_a_heading_or_a_fence_is_still_r_unrecorded(tmp_path, text, where):
    report = run(tmp_path, {EV + "a.md": text})
    assert [r.where for r in reds(report, "R-UNRECORDED")] == [EV + where]


def test_the_live_tree_counts_the_t069_heading_statement():
    path = "specs/022-unified-brain-arc/evidence/t069-cerebellar-rule.md"
    found = [st for st in ng.statements(ng._scan(ng.REPO)) if st.path == path]
    assert [st.no for st in found if st.text.startswith("## 3. Results")] == [109]


# ---------------------------------------- F6: setext headings are section boundaries


def test_a_setext_h1_is_a_section_boundary(tmp_path):
    text = f"one\n===\n{STATEMENT}\n\ntwo\n===\n{rec()}\n"
    assert reds(run(tmp_path, {EV + "a.md": text}), "R-UNRECORDED")


def test_a_setext_h2_ends_a_sibling_section_and_not_its_parent(tmp_path):
    sibling = f"top\n===\nsub one\n------\n{STATEMENT}\n\nsub two\n------\n{rec()}\n"
    assert reds(run(tmp_path / "s", {EV + "a.md": sibling}), "R-UNRECORDED")
    parent = f"top\n===\nsub one\n------\n{STATEMENT}\n\n{rec()}\n"
    assert not reds(run(tmp_path / "p", {EV + "a.md": parent}), "R-UNRECORDED")


def test_a_thematic_break_after_a_blank_line_is_not_a_heading(tmp_path):
    text = f"# t\n{STATEMENT}\n\n---\n\n{rec()}\n"
    assert not reds(run(tmp_path, {EV + "a.md": text}), "R-UNRECORDED")


def test_the_limit_line_discloses_the_section_scoping():
    out = ng.render(ng.Report())
    assert "one NULL-RECORD of any method covers every statement in its section" in out
    assert "setext" in out


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


# ---------------------------------------- F5: R-RETEST-CHARGED


def charged_files(result="charges", statement_section="# t"):
    return {
        EV + "a.md": f"{statement_section}\n{STATEMENT}\n\n{rec(retest='REF:' + EV + 'b.md#m')}\n",
        EV + "b.md": "# r\n" + retest_rec(value="40", result=result) + "\n",
    }


def test_a_deletion_whose_retest_charged_is_r_retest_charged(tmp_path):
    report = run(tmp_path, charged_files())
    assert [r.code for r in report.reds] == ["R-RETEST-CHARGED"]
    assert report.reds[0].where == EV + "a.md:2"
    assert "scale-dependent finding; deletion not licensed" in report.reds[0].message


def test_a_deletion_whose_retest_stayed_null_is_green(tmp_path):
    assert run(tmp_path, charged_files(result="null")).reds == []


def test_a_charged_retest_in_another_section_does_not_red_the_statement(tmp_path):
    files = charged_files()
    files[EV + "a.md"] = f"# one\n{STATEMENT}\n# two\n{rec(retest='REF:' + EV + 'b.md#m')}\n"
    assert not reds(run(tmp_path, files), "R-RETEST-CHARGED")


def test_a_charged_retest_reds_a_grandfathered_statement_too(tmp_path):
    key = (EV + "a.md", ng.line_key(STATEMENT))
    report = run(tmp_path, charged_files(), grandfathered=frozenset({key}))
    assert [r.code for r in report.reds] == ["R-RETEST-CHARGED"]


# ---------------------------------------- G2: any charged re-test of the record reds the statement


@pytest.mark.parametrize(
    "retest",
    [
        STRUCT,
        "NOT-NEEDED: harm the pre-registered R4 read turned red against the incumbent",
        "PENDING since " + SHA,
    ],
)
def test_a_charged_retest_reds_the_statement_whatever_the_null_retest_is(tmp_path, retest):
    files = {
        EV + "a.md": f"# t\n{STATEMENT}\n\n{ARGUMENT}\n{rec(retest=retest)}\n",
        EV + "b.md": "# r\n" + retest_rec(result="charges") + "\n",
    }
    assert [r.code for r in run(tmp_path, files, history=alive).reds] == ["R-RETEST-CHARGED"]


def test_a_charged_retest_in_a_third_file_reds_a_ref_to_a_null_retest(tmp_path):
    files = charged_files(result="null")
    files[EV + "c.md"] = "# r\n" + retest_rec(result="charges") + "\n"
    assert [r.code for r in run(tmp_path, files).reds] == ["R-RETEST-CHARGED"]


def test_a_charged_retest_of_another_method_or_file_does_not_red_the_statement(tmp_path):
    other = f"{EV}z.md"
    files = {
        EV + "a.md": f"# t\n{STATEMENT}\n\n{rec()}\n{rec(method='n')}\n",
        other: f"# z\n{rec(method='n')}\n",
        EV + "b.md": "# r\n" + retest_rec(method="n", of=other, result="charges") + "\n",
    }
    assert not reds(run(tmp_path, files), "R-RETEST-CHARGED")


# ---------------------------------------- R-EXPIRED: the four points of an arc


def pending(tmp_path):
    return {EV + "a.md": f"# t\n{rec(retest='PENDING since ' + SHA)}\n"}


CLOSING = "chore(record): closing execution record, arc 7"
FIRST = f"{0:040x}"


def test_expired_before_the_record_is_green(tmp_path):
    assert run(tmp_path, pending(tmp_path), history=lambda sha: ng.History([])).reds == []


def test_expired_with_an_unknown_sha_is_red_never_a_pass(tmp_path):
    report = run(tmp_path, pending(tmp_path), history=lambda sha: None)
    assert [r.code for r in report.reds] == ["R-EXPIRED"]
    assert "history unavailable" in report.reds[0].message


# ---------------------------------------- F4: only non-merge commits after the record count


def test_real_git_a_no_ff_merge_on_top_of_the_closing_record_is_green(tmp_path):
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n"})
    repo.commit("feat: add the null", record_file(base))
    main = repo.git("rev-parse", "--abbrev-ref", "HEAD")
    repo.git("checkout", "-q", "-b", "pr")
    close_arc(repo)
    repo.git("checkout", "-q", main)
    repo.git("merge", "-q", "--no-ff", "-m", "Merge pull request #7", "pr")
    assert repo.git("rev-list", "--parents", "-n1", "HEAD").count(" ") == 2
    assert real_check(repo).reds == []
    repo.commit("feat: next arc", {"README.md": "y\n"})
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


# ---------------------------------------- F8: a closing record is a commit that touches the record

OTHER_SUBJECTS = [
    "chore(record): batch 4 closing record",
    "G1(c) closing record",
    "docs(evidence): closing record",
]


@pytest.mark.parametrize("subject", OTHER_SUBJECTS)
def test_real_git_a_closing_record_is_the_commit_that_touches_the_execution_record(
    tmp_path, subject
):
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n"})
    repo.commit("feat: add the null", record_file(base))
    close_arc(repo, subject=subject)
    assert real_check(repo).reds == []
    repo.commit("feat: next arc", {"README.md": "y\n"})
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


def test_real_git_the_old_subject_without_touching_the_execution_record_is_green(tmp_path):
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n"})
    repo.commit("feat: add the null", record_file(base))
    repo.commit(CLOSING, {"README.md": "z\n"})
    repo.commit("feat: next arc", {"README.md": "y\n"})
    assert real_check(repo).reds == []


def test_the_default_history_reads_git_and_a_made_up_sha_is_unavailable():
    assert ng.git_history("b" * 40) is None


# ---------------------------------------- F2: expiry anchors to the commit that ADDED the record


class Repo:
    """A throwaway git repo; every commit is real, so the default history function runs unmocked."""

    def __init__(self, root):
        self.root = root
        self.git("init", "-q")

    def git(self, *argv):
        flags = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false"]
        out = subprocess.run(
            ["git", *flags, *argv], cwd=self.root, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()

    def write(self, files):
        for rel, text in files.items():
            tree(self.root, {rel: text})

    def commit(self, message, files):
        self.write(files)
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)
        return self.git("rev-parse", "HEAD")


CERT = "tools/policy_certification.json"


def record_file(sha):
    return {EV + "a.md": f"# t\n{rec(retest='PENDING since ' + sha)}\n"}


def real_check(repo):
    history = functools.partial(ng.git_history, repo=repo.root)
    return ng.check(repo.root, grandfathered=frozenset(), history=history)


def close_arc(repo, n=1, subject=CLOSING):
    return repo.commit(subject, {CERT: f"{n}\n"})


def test_real_git_an_honest_entry_is_green_until_a_commit_follows_its_arcs_record(tmp_path):
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n"})
    repo.commit("feat: add the null", record_file(base))
    assert real_check(repo).reds == []
    close_arc(repo)
    assert real_check(repo).reds == []
    repo.commit("feat: next arc", {"README.md": "y\n"})
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


# ---------------------------------------- G1: expiry without a creation-commit lookup


def pending_file(sha, prefix="NULL-RECORD: ", path=EV + "a.md"):
    return {
        path: f"# t\n{rec(retest='PENDING since ' + sha).replace('NULL-RECORD: ', prefix, 1)}\n"
    }


def armed_repo(tmp_path, files=None):
    """base, then a commit that adds the PENDING record; returns (repo, base)."""
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n"})
    repo.commit("feat: add the null", files or record_file(base))
    return repo, base


def expires_after_close(repo):
    assert real_check(repo).reds == []
    close_arc(repo)
    assert real_check(repo).reds == []
    repo.commit("feat: next arc", {"README.md": "y\n"})
    return [r.code for r in real_check(repo).reds]


@pytest.mark.parametrize("prefix", ["NULL-RECORD:", "NULL-RECORD:  ", "NULL-RECORD:\t"])
def test_real_git_a_record_spelled_off_the_canonical_spacing_still_expires(tmp_path, prefix):
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n"})
    repo.commit("feat: add the null", pending_file(base, prefix))
    assert expires_after_close(repo) == ["R-EXPIRED"]


def test_real_git_a_record_moved_to_another_file_still_expires(tmp_path):
    repo, base = armed_repo(tmp_path)
    close_arc(repo)
    repo.git("mv", EV + "a.md", EV + "b.md")
    repo.git("commit", "-q", "-m", "feat: next arc moves the record")
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


def test_real_git_a_record_moved_with_the_rest_of_the_file_still_expires(tmp_path):
    repo, base = armed_repo(tmp_path)
    close_arc(repo)
    repo.commit(
        "feat: next arc", {EV + "a.md": "# t\n", EV + "b.md": record_file(base)[EV + "a.md"]}
    )
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


def test_real_git_an_older_record_of_the_same_method_does_not_red_a_fresh_one(tmp_path):
    repo = Repo(tmp_path)
    repo.commit("feat: an old null", {EV + "a.md": f"# t\n{rec()}\n"})
    repo.commit("feat: the old null is removed", {EV + "a.md": "# t\n"})
    base = repo.commit("feat: base", {"README.md": "x\n"})
    repo.commit("feat: a fresh PENDING null", record_file(base))
    assert real_check(repo).reds == []


def test_real_git_a_triage_candidate_switched_to_pending_is_green(tmp_path):
    candidate = rec(
        units="unknown", ticks="unknown", seeds="unknown", mde="unknown", retest="CANDIDATE"
    )
    repo = Repo(tmp_path)
    repo.commit("docs: the triage list", {ng.TRIAGE: f"# t\n{candidate}\n"})
    base = repo.commit("feat: base", {"README.md": "x\n"})
    repo.commit("docs: the owner picks it", pending_file(base, path=ng.TRIAGE))
    assert real_check(repo).reds == []


def test_real_git_a_since_moved_past_its_own_pending_record_is_a_forward_edit(tmp_path):
    repo, base = armed_repo(tmp_path)
    record = repo.git("rev-parse", "HEAD")
    repo.commit("fix: re-point the sha", record_file(record))
    report = real_check(repo)
    assert [r.code for r in report.reds] == ["R-EXPIRED"]
    assert "forward edit" in report.reds[0].message


def test_real_git_a_since_moved_past_an_expired_arc_is_red(tmp_path):
    # review #4519 M2: moving `since` past the closing record must not reset the clock
    repo, base = armed_repo(tmp_path)
    closing = close_arc(repo)
    repo.commit("feat: next arc", {"README.md": "y\n"})
    repo.commit("fix: re-point the sha", record_file(closing))
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


def test_real_git_a_mid_arc_recertification_is_not_a_closing_record(tmp_path):
    repo, base = armed_repo(tmp_path)
    repo.commit("chore(record): execution record -- battery GREEN on becb96b7", {CERT: "1\n"})
    repo.commit(
        "chore(record): re-certify at d07d4fd6 so the arc battery's record assert can open",
        {CERT: "2\n"},
    )
    repo.commit("feat: more of the same arc", {"README.md": "y\n"})
    assert real_check(repo).reds == []
    repo.commit("chore(record): closing execution record -- arc 7", {CERT: "3\n"})
    assert real_check(repo).reds == []
    repo.commit("feat: next arc", {"README.md": "z\n"})
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


def test_real_git_a_closing_subject_that_does_not_touch_the_record_is_not_a_closing_record(
    tmp_path,
):
    repo, base = armed_repo(tmp_path)
    repo.commit(CLOSING, {"README.md": "z\n"})
    repo.commit("feat: next arc", {"README.md": "y\n"})
    assert real_check(repo).reds == []


@pytest.mark.parametrize(
    "subject", ["chore(record): Closing Execution Record -- arc 7", "G1 CLOSING"]
)
def test_real_git_the_closing_subject_is_matched_without_case(tmp_path, subject):
    repo, base = armed_repo(tmp_path)
    close_arc(repo, subject=subject)
    repo.commit("feat: next arc", {"README.md": "y\n"})
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


def test_real_git_a_parallel_commit_on_main_does_not_expire_the_arcs_own_entry(tmp_path):
    repo, base = armed_repo(tmp_path)
    main = repo.git("rev-parse", "--abbrev-ref", "HEAD")
    repo.git("checkout", "-q", "-b", "pr")
    close_arc(repo)
    repo.git("checkout", "-q", main)
    repo.commit("feat: another lane lands on main", {"other.md": "o\n"})
    repo.git("checkout", "-q", "pr")
    repo.git("merge", "-q", "--no-ff", "-m", "merge main into the arc", main)
    assert real_check(repo).reds == []
    repo.commit("feat: next arc", {"README.md": "y\n"})
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


def test_real_git_main_closing_record_merged_into_the_arc_then_work_expires_it(tmp_path):
    repo, base = armed_repo(tmp_path)
    main = repo.git("rev-parse", "--abbrev-ref", "HEAD")
    repo.git("checkout", "-q", "-b", "arc", base)
    repo.commit("feat: arc work", {"arc.md": "a\n"})
    repo.git("checkout", "-q", main)
    close_arc(repo)
    repo.git("checkout", "-q", "arc")
    repo.git("merge", "-q", "--no-ff", "-m", "merge main into arc", main)
    assert real_check(repo).reds == []
    repo.commit("feat: arc keeps working", {"arc.md": "b\n"})
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


def test_the_history_function_is_called_with_the_sha_alone(tmp_path):
    seen = []

    def history(sha):
        seen.append(sha)
        return ng.History([])

    run(tmp_path, pending(tmp_path), history=history)
    assert seen == [SHA]


def test_a_closing_record_in_the_history_expires_and_none_is_green(tmp_path):
    expired = run(tmp_path, pending(tmp_path), history=lambda sha: ng.History([FIRST]))
    assert [r.code for r in expired.reds] == ["R-EXPIRED"] and FIRST[:12] in expired.reds[0].message
    assert (
        run(tmp_path / "ok", pending(tmp_path / "ok"), history=lambda sha: ng.History([])).reds
        == []
    )


def test_a_pending_record_of_the_method_in_the_tree_of_since_is_a_forward_edit(tmp_path):
    line = rec(retest="PENDING since " + SHA)
    history = lambda sha: ng.History([], [(EV + "a.md", 2, line)])  # noqa: E731
    report = run(tmp_path, pending(tmp_path), history=history)
    assert [r.code for r in report.reds] == ["R-EXPIRED"] and "forward edit" in report.reds[
        0
    ].message


@pytest.mark.parametrize(
    "line",
    [
        rec(retest="STRUCTURAL: zero call sites, the proposer is empty"),
        rec(method="other", retest="PENDING since " + SHA),
        rec(units="unknown", ticks="unknown", seeds="unknown", mde="unknown", retest="CANDIDATE"),
        "NULL-RECORD: m | not a record",
    ],
)
def test_other_records_in_the_tree_of_since_are_not_a_forward_edit(tmp_path, line):
    history = lambda sha: ng.History([], [(ng.TRIAGE, 2, line)])  # noqa: E731
    assert run(tmp_path, pending(tmp_path), history=history).reds == []


def dormancy_file(*modules):
    rows = "".join(
        f'    "{m}": (\n        "scale-retest",\n        "x",\n    ),\n' for m in modules
    )
    return {ng.DORMANCY_GATE: f"EXEMPT = {{\n{rows}}}\n"}


def exempt_real(repo, base):
    reason = f"unwired on a null; NULL-RECORD:{EV}a.md since={base}; re-test before the next arc"
    history = functools.partial(ng.git_history, repo=repo.root)
    return ng.exempt_problems("m", reason, history=history, root=repo.root)


def test_real_git_an_exempt_row_is_green_in_its_arc_and_expires_after_the_closing_record(tmp_path):
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n", **dormancy_file()})
    repo.commit("feat: the row and its record", {**record_file(base), **dormancy_file("m")})
    assert exempt_real(repo, base) == []
    close_arc(repo)
    assert exempt_real(repo, base) == []
    repo.commit("feat: next arc", {"README.md": "y\n"})
    assert any("expired" in p for p in exempt_real(repo, base))


def test_real_git_an_exempt_since_moved_past_its_own_row_is_a_forward_edit(tmp_path):
    repo = Repo(tmp_path)
    repo.commit("feat: base", {"README.md": "x\n", **dormancy_file()})
    row = repo.commit("feat: the row and its record", {**dormancy_file("m"), **record_file(SHA)})
    repo.commit("fix: re-point the sha", record_file(row))
    assert any("forward edit" in p for p in exempt_real(repo, row))


def structural_file(sha):
    reason = "STRUCTURAL: the path has no synapses at any scale whatsoever"
    return {EV + "a.md": f"# t\n{rec(retest=reason)}\n"}


@pytest.mark.parametrize(
    "parked",
    [
        {EV + "a.md": "# t\n"},  # removed
        "structural",  # flipped to a non-PENDING retest
        {EV + "a.md": "# t\n", "notes/a.md": "parked here\n"},  # moved out of scope
    ],
)
def test_real_git_a_since_laundered_through_one_commit_after_the_closing_record_is_red(
    tmp_path, parked
):
    # re-panel 2 #4627 M1: the record leaves the tree for a commit, then returns PENDING since it
    repo, base = armed_repo(tmp_path)
    close_arc(repo)
    x1 = repo.commit(
        "feat: next arc parks it", structural_file(base) if parked == "structural" else parked
    )
    repo.commit("feat: next arc restores it", record_file(x1))
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


def test_real_git_an_exempt_row_laundered_through_one_commit_is_red(tmp_path):
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n", **dormancy_file()})
    repo.commit("feat: the row and its record", {**record_file(base), **dormancy_file("m")})
    close_arc(repo)
    x1 = repo.commit("feat: next arc drops the row", dormancy_file())
    repo.commit("feat: next arc restores it", {**record_file(x1), **dormancy_file("m")})
    assert any("expired" in p for p in exempt_real(repo, x1))


def test_real_git_a_candidate_at_the_closing_record_may_turn_pending_in_the_next_arc(tmp_path):
    candidate = rec(
        units="unknown", ticks="unknown", seeds="unknown", mde="unknown", retest="CANDIDATE"
    )
    repo = Repo(tmp_path)
    repo.commit("docs: the triage list", {ng.TRIAGE: f"# t\n{candidate}\n"})
    closing = close_arc(repo)
    repo.commit("docs: the owner picks it", pending_file(closing, path=ng.TRIAGE))
    assert real_check(repo).reds == []


def test_real_git_a_subject_that_only_contains_the_letters_closing_is_not_a_closing_record(
    tmp_path,
):
    # re-panel 2 #4627 m2: a word, not a substring ("disclosing")
    repo, base = armed_repo(tmp_path)
    repo.commit("chore(record): re-certify, disclosing the CRLF note", {CERT: "1\n"})
    repo.commit("feat: same arc", {"README.md": "y\n"})
    assert real_check(repo).reds == []


def test_backtick_emphasis_inside_the_phrase_does_not_hide_a_statement(tmp_path):
    # re-panel 2 #4627 m9
    report = run(tmp_path, {EV + "a.md": "# t\nthe rule was deleted on its `own` null\n"})
    assert [r.code for r in report.reds] == ["R-UNRECORDED"]


def test_real_git_a_second_row_for_another_module_is_not_a_forward_edit(tmp_path):
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n", **dormancy_file("other")})
    repo.commit(
        "feat: the row and its record", {**record_file(base), **dormancy_file("other", "m")}
    )
    assert exempt_real(repo, base) == []


@pytest.fixture
def real_history():
    """This repository's own history; skips on a clone too shallow to hold the arc."""
    if ng.git_history("4be0d691ec38fd20160dd3eba9b15d71981a911c") is None:
        pytest.skip("the arc 4be0d691..fbb1f38e is not in this checkout's history")


FBB = "fbb1f38eadb1a063fd15d7904dfcb858cfdb0bf9"


def repo_git(*argv):
    return subprocess.run(
        ["git", *argv], cwd=ng.REPO, capture_output=True, text=True, check=True
    ).stdout.split()


def test_real_history_the_arc_4be0d691_is_green_at_its_closing_record_and_red_after(real_history):
    base = "4be0d691ec38fd20160dd3eba9b15d71981a911c"
    at = lambda head: ng.expiry_problem(base, functools.partial(ng.git_history, head=head), "m")  # noqa: E731
    assert at(FBB) is None
    after = repo_git("rev-list", "--reverse", "--no-merges", "--ancestry-path", f"{FBB}..HEAD")[0]
    assert "expired" in at(after)


def test_real_history_a_triage_candidate_switched_to_pending_later_is_green(real_history):
    later = repo_git("rev-parse", "HEAD")[0]
    method = "categorical coupling (counts-matrix relabelling prior on the readout)"
    assert ng.expiry_problem(later, ng.git_history, method) is None


def test_real_history_an_exempt_row_switched_to_scale_retest_is_green(real_history):
    # Judged at the P8 arc's own closing record, not at HEAD: at HEAD the verdict expires as soon
    # as any commit lands on top of 20e1e742 (measured 2026-10-23 on the OG1 stack), so a
    # live-HEAD form reds every later arc.
    cb = "cb294e571e7a48b5fc66c26b3b39de84c2725fa0"
    close = "20e1e742f6d1d9deb19dee0b443e6737592a1200"
    at_close = functools.partial(ng.git_history, head=close)
    assert ng.expiry_problem(cb, at_close, module="feed_readout") is None
    # ... and red one non-merge commit past it (engine-arc panel A-m1: the module path's red
    # half). Every tree holding this test is past 20e1e742, so the descendant exists, as in the
    # 4be0d691 sibling.
    after = repo_git("rev-list", "--reverse", "--no-merges", "--ancestry-path", f"{close}..HEAD")[0]
    at_after = functools.partial(ng.git_history, head=after)
    assert "expired" in ng.expiry_problem(cb, at_after, module="feed_readout")


# ---------------------------------------- G3: block quotes, G6: emphasis


@pytest.mark.parametrize(
    "text",
    [
        "> The method was deleted on its\n> own null.",
        ">The method was deleted on its\n>own null.",
        "> > The method was deleted on its\n> > own null.",
        "  > The method was deleted on its\n  > own null.",
        "The method was deleted\n> on its own null.",
    ],
)
def test_a_block_quoted_statement_wrapped_across_lines_is_r_unrecorded(tmp_path, text):
    report = run(tmp_path, {EV + "a.md": f"# t\n{text}\n"})
    assert [(r.code, r.where) for r in report.reds] == [("R-UNRECORDED", EV + "a.md:2")]


def test_a_block_quoted_statement_is_satisfied_by_a_record_in_its_section(tmp_path):
    text = f"# t\n> The method was deleted on its\n> own null.\n\n{ARGUMENT}\n{rec()}\n"
    assert run(tmp_path, {EV + "a.md": text}).reds == []


@pytest.mark.parametrize(
    "text",
    [
        "The method was deleted on *its own* null.",
        "The method was deleted on _its own_ null.",
        "The method was **deleted** on its own null.",
        "The method was deleted on __its__ own null.",
        "The method was deleted on *its*\n*own* null.",
    ],
)
def test_emphasis_inside_the_phrase_does_not_hide_a_statement(tmp_path, text):
    report = run(tmp_path, {EV + "a.md": f"# t\n{text}\n"})
    assert [r.code for r in report.reds] == ["R-UNRECORDED"]


def test_the_grandfather_key_is_computed_on_the_text_with_its_emphasis(tmp_path):
    text = "The method was deleted on *its own* null."
    key = (EV + "a.md", ng.line_key(text))
    report = run(tmp_path, {EV + "a.md": f"# t\n{text}\n"}, grandfathered=frozenset({key}))
    assert report.reds == [] and report.counts["grandfathered"] == 1


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
    line = ARGUMENT + "\n" + rec(**{field: "unknown"})
    assert reds(run(tmp_path / "e", {OTHER: "# t\n" + line + "\n"}), "R-FIELD")
    assert run(tmp_path / "t", {ng.TRIAGE: "# t\n" + line + "\n"}).reds == []


HARM = "NOT-NEEDED: harm the harm cleared its pre-registered bar"


@pytest.mark.parametrize("path", [OTHER, "docs/x.md", ng.TRIAGE])
def test_a_measured_harm_needs_no_retest_in_every_in_scope_file(tmp_path, path):
    # owner answer 3: "a measured harm is enough to delete; no scale re-test required"
    assert run(tmp_path, {path: "# t\n" + rec(retest=HARM) + "\n"}).reds == []


def test_a_harm_record_satisfies_the_statement_in_its_section(tmp_path):
    text = f"# t\n{STATEMENT}\n\n{rec(retest=HARM)}\n"
    assert run(tmp_path, {OTHER: text}).reds == []


@pytest.mark.parametrize("kind", ["at-scale", "restatement"])
def test_the_other_not_needed_kinds_stay_triage_only(tmp_path, kind):
    line = rec(retest=f"NOT-NEEDED: {kind} a reason that is long enough to count")
    assert run(tmp_path / "t", {ng.TRIAGE: "# t\n" + line + "\n"}).reds == []
    assert reds(run(tmp_path / "e", {OTHER: "# t\n" + line + "\n"}), "R-FIELD")


def test_candidate_stays_triage_only(tmp_path):
    line = rec(retest="CANDIDATE")
    assert run(tmp_path / "t", {ng.TRIAGE: "# t\n" + line + "\n"}).reds == []
    assert reds(run(tmp_path / "e", {OTHER: "# t\n" + line + "\n"}), "R-FIELD")


def test_a_not_needed_with_an_unknown_kind_or_short_reason_is_r_field_even_in_triage(tmp_path):
    for retest in ("NOT-NEEDED: whim some reason that is long enough", "NOT-NEEDED: harm short"):
        assert reds(run(tmp_path, {ng.TRIAGE: "# t\n" + rec(retest=retest) + "\n"}), "R-FIELD")


# ---------------------------------------- STRUCTURAL is printed, never judged


def test_structural_reasons_are_printed_for_review_and_the_limit_is_stated(tmp_path):
    report = run(tmp_path, {EV + "a.md": f"# t\n{ARGUMENT}\n{rec()}\n"})
    assert report.reds == [] and len(report.structural) == 1
    out = ng.render(report)
    assert "zero call sites" in out and "cannot judge a STRUCTURAL argument" in out


# ---------------------------------------- G4: the seeds axis is not a re-test axis


def test_axis_seeds_is_r_field(tmp_path):
    files = {
        EV + "a.md": f"# t\n{rec(retest='REF:' + EV + 'b.md#m')}\n",
        EV + "b.md": "# r\n" + retest_rec(axis="seeds", value="80") + "\n",
    }
    assert "R-FIELD" in run(tmp_path, files).codes()


def test_the_re_test_axes_are_units_and_ticks():
    assert ng.AXES == ("units", "ticks")


# ---------------------------------------- G5: a harm claim is printed for review beside STRUCTURAL

HARM = "NOT-NEEDED: harm the pre-registered R4 read turned red against the incumbent"


def test_every_harm_record_is_printed_for_review_and_the_limit_says_the_bar_is_unjudged(tmp_path):
    report = run(
        tmp_path, {EV + "a.md": f"# t\n{rec(retest=HARM)}\n{rec(method='n')}\n{ARGUMENT}\n"}
    )
    assert report.reds == [] and len(report.harm) == 1
    out = ng.render(report)
    assert f"HARM (printed for review): {EV}a.md:2 m: {HARM}" in out
    assert "pre-registered bar" in out.split("LIMIT:")[1]


# ---------------------------------------- R-EXEMPT, and dormancy_gate's disposition


GOOD_REASON = (
    f"unwired on a null; NULL-RECORD:{EV}a.md since={SHA}; re-test or delete before the next arc"
)


def alive(sha):
    return ng.History([])


@pytest.fixture
def home(tmp_path):
    """A tree whose evidence file holds the PENDING NULL-RECORD that GOOD_REASON names."""
    return tree(tmp_path, {EV + "a.md": f"# t\n{rec(retest='PENDING since ' + SHA)}\n"})


def test_a_reason_naming_record_and_sha_passes(home):
    assert ng.exempt_problems("m", GOOD_REASON, history=alive, root=home) == []


def test_a_reason_with_no_null_record_is_r_exempt():
    assert ng.exempt_problems("m", f"since={SHA}", history=alive)


def test_a_reason_with_no_full_sha_is_r_exempt():
    assert ng.exempt_problems("m", f"NULL-RECORD:{EV}a.md since=abc1234", history=alive)


def exempt_with(root, reason_path):
    reason = (
        f"unwired on a null; NULL-RECORD:{reason_path} since={SHA}; re-test before the next arc"
    )
    return ng.exempt_problems("m", reason, history=alive, root=root)


def test_a_named_record_file_that_does_not_exist_is_r_exempt(tmp_path):
    assert exempt_with(tmp_path, EV + "missing.md")


def test_a_named_record_file_with_no_pending_null_record_is_r_exempt(tmp_path):
    for n, retest in enumerate(
        ("STRUCTURAL: zero call sites, never executed", "REF:" + EV + "b.md#m")
    ):
        root = tree(tmp_path / str(n), {EV + "a.md": f"# t\n{rec(retest=retest)}\n"})
        assert exempt_with(root, EV + "a.md")
    assert exempt_with(tree(tmp_path / "none", {EV + "a.md": "# t\n"}), EV + "a.md")


def test_a_named_record_file_outside_the_scope_is_r_exempt(tmp_path):
    pending_record = f"# t\n{rec(retest='PENDING since ' + SHA)}\n"
    root = tree(tmp_path, {"notes/y.md": pending_record, "docs/sub/z.md": pending_record})
    assert exempt_with(root, "notes/y.md") and exempt_with(root, "docs/sub/z.md")


def test_a_named_record_file_inside_docs_with_a_pending_record_passes(tmp_path):
    root = tree(tmp_path, {"docs/x.md": f"{rec(retest='PENDING since ' + SHA)}\n"})
    assert exempt_with(root, "docs/x.md") == []


def test_a_named_record_whose_pending_since_differs_from_the_entrys_is_r_exempt(tmp_path):
    root = tree(tmp_path, {EV + "a.md": f"# t\n{rec(retest='PENDING since ' + 'b' * 40)}\n"})
    problems = exempt_with(root, EV + "a.md")
    assert problems and "PENDING since " + SHA in problems[0]


def test_a_named_file_with_a_second_pending_record_at_the_entrys_since_passes(tmp_path):
    other = rec(method="n", retest="PENDING since " + "b" * 40)
    root = tree(tmp_path, {EV + "a.md": f"# t\n{other}\n{rec(retest='PENDING since ' + SHA)}\n"})
    assert exempt_with(root, EV + "a.md") == []


def test_the_default_root_is_the_repo_and_a_phantom_path_is_r_exempt():
    for path in ("specs/nope/evidence/missing.md", "docs/POLICIES.md"):
        assert exempt_with(ng.REPO, path), path


def test_a_reason_with_an_expired_since_is_r_exempt(home):
    assert ng.exempt_problems("m", GOOD_REASON, history=lambda sha: ng.History([FIRST]), root=home)


def test_dormancy_gate_reds_a_planted_bad_scale_retest_entry():
    planted = {"some_module": ("scale-retest", "no record, no sha")}
    with pytest.raises(AssertionError, match="some_module"):
        dg._assert_scale_retest(planted, history=alive)


def test_dormancy_gate_passes_a_planted_good_scale_retest_entry(home):
    planted = {"some_module": ("scale-retest", GOOD_REASON)}
    dg._assert_scale_retest(planted, history=alive, root=home)


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


# ---------------------------------------- F9: provenance of the constants


def test_scale_factor_and_mde_carry_their_provenance():
    source = Path(ng.__file__).read_text(encoding="utf-8")
    assert ng.SCALE_FACTOR == 10
    assert '"Scale factor: 10x minimum on the scaling axis."' in source
    assert "one-sided alpha .05, power .8" in source


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
        "_check_charged",
        "_check_stale",
        "_check_structural_argument",
        "_mde_problem",  # reviewer-3 #9172 C1: --self-check itself must see the mde=n/a branch
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


# ---------------------------------------- addendum 10 (D-402): mde=n/a only for a STRUCTURAL null


def field_reds(report):
    return [r.message for r in reds(report, "R-FIELD")]


def test_mde_n_a_on_a_structural_record_with_an_argument_is_green(tmp_path):
    # (a)
    report = run(tmp_path, {EV + "a.md": f"# t\n{ARGUMENT}\n{rec(mde='n/a')}\n"})
    assert report.reds == [] and report.counts["NULL-RECORDs"] == 1


@pytest.mark.parametrize(
    "retest",
    [
        "REF:" + EV + "b.md#m",  # (b)
        "PENDING since " + SHA,  # (c)
        "NOT-NEEDED: harm the harm cleared its pre-registered bar",  # (d)
    ],
)
def test_mde_n_a_with_any_other_retest_is_r_field_naming_the_structural_condition(tmp_path, retest):
    files = {
        EV + "a.md": f"# t\n{ARGUMENT}\n{rec(mde='n/a', retest=retest)}\n",
        EV + "b.md": "# r\n" + retest_rec() + "\n",
    }
    messages = field_reds(run(tmp_path, files, history=alive))
    assert messages and all("mde" in m and "STRUCTURAL" in m for m in messages), messages


def test_mde_n_a_with_candidate_in_the_triage_file_is_r_field(tmp_path):
    report = run(tmp_path, {ng.TRIAGE: f"# t\n{ARGUMENT}\n{rec(mde='n/a', retest='CANDIDATE')}\n"})
    assert any("STRUCTURAL" in m for m in field_reds(report))


def test_mde_n_a_on_a_structural_record_in_the_triage_file_is_green(tmp_path):
    report = run(tmp_path, {ng.TRIAGE: f"# t\n{ARGUMENT}\n{rec(mde='n/a')}\n"})
    assert report.reds == []


@pytest.mark.parametrize(
    "text",
    [
        f"# t\n{rec()}\n",  # (e) alone under its heading: a heading is not an argument
        f"# t\n{ARGUMENT}\n\n{rec(mde='n/a')}\n",  # prose in ANOTHER paragraph does not count
        f"# t\n{rec()}\n{rec(method='n')}\n\n{ARGUMENT}\n",  # two records, no prose between them
        f"# t\n{rec()}\n```\n{ARGUMENT}\n```\n",  # a fence ends the paragraph
        # panel-3 #9345: a --- / === right under a record is a setext underline, which made the record a
        # "heading" and hid it from the paragraph reader; a record line is never part of a heading
        f"# t\n{rec()}\n---\n",
        f"# t\n{rec()}\n===\n",
        f"# t\n- {rec()}\n---\ntext after the break\n",
        # reviewer-3 C4 #9375: a record plus ONE non-prose line (break, markup, table rule) is still no argument
        f"# t\n{rec()}\n***\n",
        f"# t\n{rec()}\n___\n",
        f"# t\n{rec()}\n* * *\n",
        f"# t\n{rec()}\n- - -\n",
        f"# t\n{rec()}\n    ---\n",
        f"# t\n{rec()}\n<!-- note -->\n",
        f"# t\n{rec()}\n|---|\n",
        f"# t\n{rec()}\n<br>\n",
        f"# t\n{rec()}\n>\n",
    ],
)
def test_a_structural_record_with_no_argument_in_its_paragraph_is_r_structural_arg(tmp_path, text):
    report = run(tmp_path, {EV + "a.md": text})
    assert reds(report, "R-STRUCTURAL-ARG"), text
    assert not reds(report, "R-FIELD")


def test_r_structural_arg_names_each_bare_record_line(tmp_path):
    text = f"# t\n{rec()}\n{rec(method='n')}\n"
    where = {r.where for r in reds(run(tmp_path, {EV + "a.md": text}), "R-STRUCTURAL-ARG")}
    assert where == {EV + "a.md:2", EV + "a.md:3"}


@pytest.mark.parametrize(
    "text",
    [
        f"# t\n{ARGUMENT}\n{rec()}\n",
        f"# t\n{rec()}\n{ARGUMENT}\n",
        f"# t\n- {rec()}\n  {ARGUMENT}\n",
    ],
)
def test_a_structural_record_with_prose_in_its_paragraph_is_green(tmp_path, text):
    assert run(tmp_path, {EV + "a.md": text}).reds == []


def test_a_non_structural_record_needs_no_argument_paragraph(tmp_path):
    assert run(tmp_path, {EV + "a.md": f"# t\n{rec(retest=HARM)}\n"}).reds == []


def test_a_numeric_mde_on_a_structural_record_stays_legal(tmp_path):
    assert run(tmp_path, {EV + "a.md": f"# t\n{ARGUMENT}\n{rec(mde='0.5')}\n"}).reds == []


def test_mde_unknown_outside_the_triage_file_is_still_r_field(tmp_path):
    # (f) on a STRUCTURAL record too: n/a widened, unknown did not
    report = run(tmp_path, {OTHER: f"# t\n{ARGUMENT}\n{rec(mde='unknown')}\n"})
    assert reds(report, "R-FIELD")


@pytest.mark.parametrize("mde", ["0.5", "n/a"])
def test_seeds_n_a_is_still_r_field(tmp_path, mde):
    # (g)
    report = run(tmp_path, {EV + "a.md": f"# t\n{ARGUMENT}\n{rec(seeds='n/a', mde=mde)}\n"})
    assert any("seeds" in m for m in field_reds(report))


def test_the_self_check_plants_r_structural_arg_and_a_green_mde_n_a_record():
    assert "R-STRUCTURAL-ARG" in ng._planted()
    assert "R-FIELD/mde-n/a-without-STRUCTURAL" in ng._planted()
    source = Path(ng.__file__).read_text(encoding="utf-8")
    assert "mde=n/a" in source

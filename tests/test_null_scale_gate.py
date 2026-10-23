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
    bad = run(tmp_path / "bad", {EV + "a.md": f"# t\n{head}{reason} | note=x\n"})
    good = run(tmp_path / "good", {EV + "a.md": f"# t\n{head}{reason}\n"})
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


# ---------------------------------------- R-EXPIRED: the four points of an arc


def pending(tmp_path):
    return {EV + "a.md": f"# t\n{rec(retest='PENDING since ' + SHA)}\n"}


def history_of(*subjects, created=None):
    """Rows are subjects (CLOSING models a commit that touches the execution record) or Commits."""
    commits = [
        s if isinstance(s, ng.Commit) else ng.Commit(f"{i:040x}", s, closing=s == CLOSING)
        for i, s in enumerate(subjects)
    ]
    return lambda sha, added=None: ng.History(commits, created)


CLOSING = "chore(record): closing execution record, arc 7"
FIRST = f"{0:040x}"


def test_expired_before_the_record_is_green(tmp_path):
    assert run(tmp_path, pending(tmp_path), history=history_of("feat: a", "feat: b")).reds == []


def test_expired_with_the_record_as_head_is_green(tmp_path):
    assert run(tmp_path, pending(tmp_path), history=history_of("feat: a", CLOSING)).reds == []


def test_expired_one_commit_after_the_record_is_red(tmp_path):
    report = run(
        tmp_path,
        pending(tmp_path),
        history=history_of("feat: a", CLOSING, "feat: next arc", created=FIRST),
    )
    assert [r.code for r in report.reds] == ["R-EXPIRED"]


def test_expired_with_an_unknown_sha_is_red_never_a_pass(tmp_path):
    report = run(tmp_path, pending(tmp_path), history=lambda sha, added=None: None)
    assert [r.code for r in report.reds] == ["R-EXPIRED"]
    assert "history unavailable" in report.reds[0].message


def test_expired_when_the_entry_is_created_at_head_is_green(tmp_path):
    assert run(tmp_path, pending(tmp_path), history=history_of()).reds == []


# ---------------------------------------- F4: only non-merge commits after the record count


def test_a_merge_commit_after_the_closing_record_does_not_expire_the_entry(tmp_path):
    merge = ng.Commit(f"{9:040x}", "Merge pull request #7 from x/arc", merge=True)
    commits = [
        ng.Commit(f"{i:040x}", s, closing=s == CLOSING) for i, s in enumerate(("feat: a", CLOSING))
    ]
    history = lambda sha, added=None: ng.History([*commits, merge], FIRST)  # noqa: E731
    assert run(tmp_path, pending(tmp_path), history=history).reds == []


def test_a_non_merge_commit_after_a_merge_after_the_closing_record_still_expires(tmp_path):
    merge = ng.Commit(f"{8:040x}", "Merge branch x", merge=True)
    last = ng.Commit(f"{9:040x}", "feat: next arc")
    commits = [
        ng.Commit(f"{i:040x}", s, closing=s == CLOSING) for i, s in enumerate(("feat: a", CLOSING))
    ]
    history = lambda sha, added=None: ng.History([*commits, merge, last], FIRST)  # noqa: E731
    assert [r.code for r in run(tmp_path, pending(tmp_path), history=history).reds] == ["R-EXPIRED"]


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
    "record(022): battery GREEN on 89f3af14",
]


@pytest.mark.parametrize("subject", OTHER_SUBJECTS)
def test_a_closing_record_with_a_non_standard_subject_expires_the_entry(tmp_path, subject):
    row = ng.Commit(f"{1:040x}", subject, closing=True)
    history = history_of("feat: a", row, "feat: next arc", created=FIRST)
    assert [r.code for r in run(tmp_path, pending(tmp_path), history=history).reds] == ["R-EXPIRED"]


def test_the_standard_subject_without_a_change_to_the_record_is_not_a_closing_record(tmp_path):
    row = ng.Commit(f"{1:040x}", CLOSING, closing=False)
    history = history_of("feat: a", row, "feat: next arc", created=FIRST)
    assert run(tmp_path, pending(tmp_path), history=history).reds == []


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


def test_real_git_an_expired_sha_edited_forward_stays_expired(tmp_path):
    # review #4519 M2: moving `since` past the closing record must not reset the clock
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n"})
    repo.commit("feat: add the null", record_file(base))
    closing = close_arc(repo)
    repo.commit("feat: next arc", {"README.md": "y\n"})
    repo.commit("fix: re-point the sha", record_file(closing))
    report = real_check(repo)
    assert [r.code for r in report.reds] == ["R-EXPIRED"]
    assert "ancestor" in report.reds[0].message


def test_real_git_a_record_not_yet_committed_is_created_at_head(tmp_path):
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n"})
    close_arc(repo)
    repo.commit("feat: next arc", {"README.md": "y\n"})
    repo.write(record_file(base))
    assert real_check(repo).reds == []


def test_real_git_a_sha_that_is_not_an_ancestor_of_the_creation_commit_is_red(tmp_path):
    repo = Repo(tmp_path)
    base = repo.commit("feat: base", {"README.md": "x\n"})
    repo.git("checkout", "-q", "-b", "side")
    side = repo.commit("feat: side work", {"side.md": "s\n"})
    repo.git("checkout", "-q", "-")
    repo.commit("feat: add the null", record_file(base))
    repo.git("merge", "-q", "--no-edit", "side")
    assert real_check(repo).reds == []
    repo.write(record_file(side))
    repo.git("commit", "-q", "-am", "fix: point at the side commit")
    assert [r.code for r in real_check(repo).reds] == ["R-EXPIRED"]


def test_expiry_reads_the_creation_commit_through_the_injected_history(tmp_path):
    seen = []

    def history(sha, added=None):
        seen.append((sha, added))
        return ng.History([ng.Commit("1" * 40, "feat: a")])

    run(tmp_path, pending(tmp_path), history=history)
    assert seen == [(SHA, (EV + "a.md", "NULL-RECORD: m | "))]


def test_an_entry_created_before_the_closing_record_that_is_head_past_is_red(tmp_path):
    created = f"{1:040x}"
    history = history_of("feat: a", "feat: b", CLOSING, "feat: next", created=created)
    assert [r.code for r in run(tmp_path, pending(tmp_path), history=history).reds] == ["R-EXPIRED"]


def test_an_entry_created_after_the_closing_record_rides_its_own_arc(tmp_path):
    created = f"{3:040x}"
    history = history_of("feat: a", CLOSING, "feat: next", "feat: add the null", created=created)
    assert run(tmp_path, pending(tmp_path), history=history).reds == []


def test_a_creation_commit_outside_the_written_shas_range_is_red(tmp_path):
    history = history_of("feat: a", created="f" * 40)
    report = run(tmp_path, pending(tmp_path), history=history)
    assert [r.code for r in report.reds] == ["R-EXPIRED"] and "ancestor" in report.reds[0].message


def test_a_creation_commit_the_sha_is_not_an_ancestor_of_is_red(tmp_path):
    commits = [ng.Commit(f"{1:040x}", "feat: a")]
    history = lambda sha, added=None: ng.History(commits, f"{1:040x}", anchored=False)  # noqa: E731
    report = run(tmp_path, pending(tmp_path), history=history)
    assert [r.code for r in report.reds] == ["R-EXPIRED"]


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
    report = run(tmp_path, {EV + "a.md": f"# t\n{rec()}\n"})
    assert report.reds == [] and len(report.structural) == 1
    out = ng.render(report)
    assert "zero call sites" in out and "cannot judge a STRUCTURAL argument" in out


# ---------------------------------------- R-EXEMPT, and dormancy_gate's disposition


GOOD_REASON = (
    f"unwired on a null; NULL-RECORD:{EV}a.md since={SHA}; re-test or delete before the next arc"
)


def alive(sha, added=None):
    return ng.History([ng.Commit(sha, "feat: x")])


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


def test_exempt_expiry_reads_the_creation_of_the_exempt_row(home):
    seen = []

    def history(sha, added=None):
        seen.append(added)
        return alive(sha)

    ng.exempt_problems("m", GOOD_REASON, history=history, root=home)
    assert seen == [(ng.DORMANCY_GATE, '"m": (')]


def test_exempt_since_edited_forward_past_the_creation_is_r_exempt(home):
    history = history_of("feat: a", created="f" * 40)
    assert any(
        "ancestor" in p for p in ng.exempt_problems("m", GOOD_REASON, history=history, root=home)
    )


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


def test_the_default_root_is_the_repo_and_a_phantom_path_is_r_exempt():
    for path in ("specs/nope/evidence/missing.md", "docs/POLICIES.md"):
        assert exempt_with(ng.REPO, path), path


def test_a_reason_with_an_expired_since_is_r_exempt(home):
    history = history_of("feat: a", CLOSING, "feat: next", created=FIRST)
    assert ng.exempt_problems("m", GOOD_REASON, history=history, root=home)


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

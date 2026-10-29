"""P1 sample-size gate: one reachable red per rule, the clean pass, the closed grandfather list.

docs/POLICIES.md, "P1 amendment", Gate specification. Every input is built in a temporary tree,
so each red is shown to fire and a pass is not vacuous. The inputs are written out by hand and not
taken from the gate's own self-check plants: a test must not share its oracle with its code.
"""

import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from snn_data_lake.gates import sample_size_gate as sg

EV = "specs/s/evidence/"
A = EV + "a.md"
FIELD_LINES = {
    "measured SD": "Measured SD: 0.31 (the paired SD of the prior run, n = 80).",
    "effect": "Claimed effect: DH = 0.035.",
    "target": "Fixed-sample target: 485 seeds at power .8, one-sided alpha .05.",
    "looks": "Looks: 11 blocks of 64 seeds.",
    "spending": "Alpha spending: Lan-DeMets O'Brien-Fleming, one boundary per look.",
    "futility": "Futility: non-binding, opens at the look where the mde is below the effect.",
    "cap": "Hard cap: 704 seeds, power .8 at the cap.",
}
ONE_SHOT_REASON = "One-shot run: a single block of 8 seeds, power stated before data."
SEQUENTIAL = tuple(FIELD_LINES)
ONE_SHOT_FIELDS = ("measured SD", "effect", "target")


def prereg(omit=(), heading="## 4. Sample size (P1)", one_shot=False):
    keep = ONE_SHOT_FIELDS if one_shot else SEQUENTIAL
    body = [FIELD_LINES[f] for f in keep if f not in omit]
    if one_shot and "one-shot" not in omit:
        body.append(ONE_SHOT_REASON)
    return "\n".join(
        [
            "# X -- pre-registration",
            "",
            "## 2. Pre-registration",
            "text",
            "",
            heading,
            "",
            *body,
            "",
        ]
    )


def tree(tmp_path, files):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def run(tmp_path, files, **kwargs):
    kwargs.setdefault("grandfathered", frozenset())
    kwargs.setdefault("owes_at_freeze", {})
    kwargs.setdefault("not_prereg", frozenset())
    kwargs.setdefault("today", date(2026, 10, 29))
    return sg.check(tree(tmp_path, files), **kwargs)


def reds(report, code):
    return [r for r in report.reds if r.code == code]


def test_complete_sequential_section_passes(tmp_path):
    assert run(tmp_path, {A: prereg()}).reds == []


def test_complete_one_shot_section_passes(tmp_path):
    assert run(tmp_path, {A: prereg(one_shot=True)}).reds == []


@pytest.mark.parametrize("missing", SEQUENTIAL)
def test_each_missing_field_reds(tmp_path, missing):
    found = reds(run(tmp_path, {A: prereg(omit=(missing,))}), "R-FIELD")
    assert len(found) == 1 and found[0].where == A and missing in found[0].message


def test_one_shot_needs_its_reason_and_the_fixed_fields(tmp_path):
    assert reds(run(tmp_path, {A: prereg(one_shot=True, omit=("one-shot",))}), "R-FIELD")
    assert reds(run(tmp_path, {A: prereg(one_shot=True, omit=("target",))}), "R-FIELD")


def test_one_shot_does_not_owe_the_sequential_fields(tmp_path):
    text = prereg(one_shot=True)
    assert not any(w in text.lower() for w in ("futility", "cap", "spending"))
    assert run(tmp_path, {A: text}).reds == []


def test_no_section_reds(tmp_path):
    text = "# X -- pre-registration\n\n## 2. Pre-registration\nseeds 8, habitual.\n"
    found = reds(run(tmp_path, {A: text}), "R-NO-SECTION")
    assert [r.where for r in found] == [A]


def test_fields_in_prose_outside_the_section_do_not_count(tmp_path):
    text = prereg(heading="## 4. Seeds") + "\n".join(FIELD_LINES.values())
    assert reds(run(tmp_path, {A: text}), "R-NO-SECTION")


def test_fields_union_over_the_files_sample_size_sections(tmp_path):
    first = "\n".join(["## Sample size (P1)", FIELD_LINES["measured SD"], FIELD_LINES["effect"]])
    rest = [FIELD_LINES[f] for f in SEQUENTIAL[2:]]
    second = "\n".join(["## Addendum 2", "### Sample size (P1)", *rest])
    text = "# X -- pre-registration\n\n" + first + "\n\n" + second + "\n"
    assert run(tmp_path, {A: text}).reds == []


def test_section_heading_inside_a_fence_is_not_a_heading(tmp_path):
    text = "# X -- pre-registration\n\n```\n## Sample size (P1)\n" + "\n".join(FIELD_LINES.values())
    assert reds(run(tmp_path, {A: text + "\n```\n"}), "R-NO-SECTION")


def test_file_that_never_mentions_pre_registration_is_not_checked(tmp_path):
    assert run(tmp_path, {A: "# A results record\n\nseeds 8.\n"}).reds == []


def test_grandfathered_name_passes_without_the_section(tmp_path):
    files = {A: "# X -- pre-registration\n\nseeds 8, habitual.\n"}
    assert run(tmp_path, files, grandfathered=frozenset({A})).reds == []


def test_new_file_not_on_the_list_reds_even_when_a_neighbour_is_grandfathered(tmp_path):
    old = "# Old -- pre-registration\n\nseeds 8.\n"
    files = {A: old, EV + "new.md": old}
    found = run(tmp_path, files, grandfathered=frozenset({A})).reds
    assert [(r.code, r.where) for r in found] == [("R-NO-SECTION", EV + "new.md")]


def test_verbatim_copy_of_a_grandfathered_file_under_a_new_name_reds(tmp_path):
    old = "# Old -- pre-registration\n\n## 2. Pre-registration\nseeds 8.\n"
    files = {A: old, EV + "a-copy.md": old}
    report = run(tmp_path, files, grandfathered=frozenset({A}))
    assert [r.where for r in reds(report, "R-NO-SECTION")] == [EV + "a-copy.md"]


def test_stale_grandfather_entry_reds(tmp_path):
    report = run(tmp_path, {A: prereg()}, grandfathered=frozenset({EV + "gone.md"}))
    assert [r.where for r in reds(report, "R-STALE-GRANDFATHER")] == [EV + "gone.md"]


def test_grandfather_entry_for_a_file_that_is_not_a_pre_registration_is_stale(tmp_path):
    files = {A: "# A results record\n\nseeds 8.\n"}
    report = run(tmp_path, files, grandfathered=frozenset({A}))
    assert reds(report, "R-STALE-GRANDFATHER")


LATER = {A: "2026-11-05"}
DRAFT = "# G -- pre-registration DRAFT\n\n## 2. Pre-registration (DRAFT)\nseeds 8.\n"


def test_owes_at_freeze_is_deferred_and_counted_until_its_dated_expiry(tmp_path):
    report = run(tmp_path, {A: DRAFT}, owes_at_freeze=LATER)
    assert report.reds == [] and report.deferred == [A]


def test_deferral_reds_once_its_expiry_has_passed(tmp_path):
    report = run(tmp_path, {A: DRAFT}, owes_at_freeze=LATER, today=date(2026, 11, 6))
    assert [r.code for r in report.reds] == ["R-DEFERRAL-EXPIRED"] and report.deferred == []


def test_deferral_does_not_depend_on_the_word_draft(tmp_path):
    # the author-maintained word neither buys a deferral nor ends one: a listed, unexpired file with
    # no DRAFT anywhere is deferred, the same as one that says DRAFT
    quiet = "# G -- pre-registration\n\n## 2. Pre-registration\nseeds 8.\n"
    assert run(tmp_path, {A: quiet}, owes_at_freeze=LATER).deferred == [A]


@pytest.mark.parametrize(
    "line",
    [
        "## 2. Pre-registration (DRAFT; frozen at `6bc97430`)",
        "Status: FROZEN",
        "The freeze commit 6bc97430 froze this text.",
        "Frozen in 1d99ff46 by the orchestrator.",
    ],
)
def test_deferred_file_that_says_it_froze_reds_even_with_draft_in_its_heading(tmp_path, line):
    text = DRAFT + line + "\n"
    report = run(tmp_path, {A: text}, owes_at_freeze=LATER)
    assert [r.code for r in report.reds] == ["R-FROZEN-DEFERRED"] and report.deferred == []


def test_a_deferred_draft_that_merely_discusses_freezing_stays_deferred(tmp_path):
    text = (
        DRAFT
        + "The orchestrator freezes it after review; a frozen random expansion is the control.\n"
    )
    assert run(tmp_path, {A: text}, owes_at_freeze=LATER).deferred == [A]


def test_a_new_draft_file_is_not_deferred_unless_named(tmp_path):
    assert reds(run(tmp_path, {A: DRAFT}), "R-NO-SECTION")


def test_stale_owes_at_freeze_entry_reds(tmp_path):
    report = run(tmp_path, {}, owes_at_freeze=LATER)
    assert [r.where for r in reds(report, "R-STALE-GRANDFATHER")] == [A]


def test_declared_not_a_pre_registration_is_skipped_but_must_exist(tmp_path):
    text = "A retroactive record, not a pre-registration.\n"
    assert run(tmp_path, {A: text}, not_prereg=frozenset({A})).reds == []
    assert reds(run(tmp_path, {}, not_prereg=frozenset({A})), "R-STALE-GRANDFATHER")


def test_counts_name_every_class(tmp_path):
    files = {
        A: prereg(),
        EV + "b.md": "# B -- pre-registration\n",
        EV + "c.md": "no mention\n",
        EV + "d.md": "# D -- pre-registration\n",
    }
    report = run(tmp_path, files, grandfathered=frozenset({EV + "b.md"}))
    assert report.counts["pre-registrations found"] == 3
    assert report.counts["grandfathered"] == 1
    assert report.counts["owing the section now"] == 1
    assert report.counts["carry the section"] == 1


def test_main_exit_codes_and_census(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sg, "GRANDFATHERED", frozenset())
    monkeypatch.setattr(sg, "OWES_AT_FREEZE", {})
    monkeypatch.setattr(sg, "NOT_PREREGISTRATIONS", frozenset())
    root = tree(tmp_path, {A: "# X -- pre-registration\n\nseeds 8.\n"})
    assert sg.main(["--assert"], root=root) == 1
    assert sg.main(["--census"], root=root) == 0
    assert f'"{A}"' in capsys.readouterr().out
    green = tree(tmp_path / "ok", {A: prereg()})
    assert sg.main(["--assert"], root=green) == 0


def test_self_check_plants_every_red():
    assert sg.self_check() == 0


def test_cli_runs_as_a_module_on_the_live_tree():
    out = subprocess.run(
        [sys.executable, "-m", "snn_data_lake.gates.sample_size_gate", "--census"],
        cwd=Path(sg.REPO),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0 and "specs/" in out.stdout


# --- round 2 (policy-enforcer #6562): scope is R1 union R2, not evidence/*.md alone ---
OUT = "research/r/session.md"
HEADED = "# Record\n\n### 3.1 Design as it stands (pre-registration plus A0)\n\nseeds 8.\n"


def test_heading_pre_registration_outside_evidence_reds(tmp_path):
    report = run(tmp_path, {OUT: HEADED})
    assert [(r.code, r.where) for r in report.reds] == [("R-NO-SECTION", OUT)]


def test_heading_pre_registration_outside_evidence_passes_with_the_section(tmp_path):
    text = HEADED + "\n" + "\n".join(["## Sample size (P1)", *FIELD_LINES.values()]) + "\n"
    assert run(tmp_path, {OUT: text}).reds == []


def test_body_mention_outside_evidence_is_a_reference_not_a_pre_registration(tmp_path):
    assert run(tmp_path, {OUT: "# Notes\n\nsee the pre-registration for seeds.\n"}).reds == []


def test_heading_inside_a_fence_outside_evidence_is_not_a_heading(tmp_path):
    text = "# Notes\n\n```\n### 3.1 pre-registration\n```\n"
    assert run(tmp_path, {OUT: text}).reds == []


def test_outside_evidence_file_declared_not_a_pre_registration_is_skipped(tmp_path):
    assert run(tmp_path, {OUT: HEADED}, not_prereg=frozenset({OUT})).reds == []


@pytest.mark.parametrize("hidden", [".venv/x.md", "node_modules/p/x.md", "graphify-out/x.md"])
def test_vendored_and_generated_trees_are_not_scanned(tmp_path, hidden):
    assert run(tmp_path, {hidden: HEADED}).reds == []


def test_grandfathered_entry_outside_the_scan_is_stale(tmp_path):
    report = run(tmp_path, {OUT: "# Notes\n"}, grandfathered=frozenset({OUT}))
    assert reds(report, "R-STALE-GRANDFATHER")


R1_REFERENCE_FILES = (
    "docs/AGENT_COLD_START.md",
    "specs/011-h01-neurogenesis-prior/decisions.md",
    "specs/022-unified-brain-arc/quickstart.md",
    "research/cloud-replies/REPLY-2-FROM-CLOUD-SESSION-2026-10-28.md",
    "research/representational-drift/SESSION-RECORD-2026-10-27.md",
)


@pytest.mark.parametrize("name", R1_REFERENCE_FILES)
def test_r1_reference_files_are_declared_not_pre_registrations_not_grandfathered(name):
    assert name in sg.NOT_PREREGISTRATIONS and name not in sg.GRANDFATHERED


def test_every_grandfathered_name_is_an_evidence_file_the_gate_reads():
    assert all("/evidence/" in n for n in sg.GRANDFATHERED)


def test_docstring_does_not_claim_r1_is_a_subset_of_r2():
    assert "subset" not in (sg.__doc__ or "")


def test_g2_is_grandfathered_g3_and_g5_are_deferred_by_name():
    ev = "specs/022-unified-brain-arc/evidence/"
    assert ev + "g2-taught-problems.md" in sg.GRANDFATHERED
    assert set(sg.OWES_AT_FREEZE) == {ev + "g3-growth-loop.md", ev + "g5-readout-variants.md"}
    assert not {ev + "g3-growth-loop.md", ev + "g5-readout-variants.md"} & sg.GRANDFATHERED


# --- MINOR-1: the one-shot waiver needs a stated, un-negated reason ---
THREE = "\n".join(FIELD_LINES[f] for f in ONE_SHOT_FIELDS)


@pytest.mark.parametrize(
    "line",
    [
        "This is not a one-shot design.",
        "Never one-shot: the count is open.",
        "Design: one-shot.",
    ],
)
def test_one_shot_without_a_reason_or_negated_does_not_waive_the_sequential_fields(tmp_path, line):
    text = "# X -- pre-registration\n\n## Sample size (P1)\n" + THREE + "\n" + line + "\n"
    found = reds(run(tmp_path, {A: text}), "R-FIELD")
    assert len(found) == 1 and "one-shot reason" in found[0].message
    assert "looks" in found[0].message and "cap" in found[0].message


def test_one_shot_reason_may_use_the_word_no(tmp_path):
    line = "One-shot because there is no interim analysis: a single block of 8 seeds."
    text = "# X -- pre-registration\n\n## Sample size (P1)\n" + THREE + "\n" + line + "\n"
    assert run(tmp_path, {A: text}).reds == []


def test_negated_one_shot_with_every_sequential_field_still_passes(tmp_path):
    text = prereg() + "\nThis is not a one-shot design.\n"
    assert run(tmp_path, {A: text}).reds == []


def test_r1_heading_rule_is_the_lists_exact_one_pre_registered_headings_are_not_matched(tmp_path):
    text = "# Notes\n\n## 3. Pre-registered criteria, baseline vs after\n\nseeds 8.\n"
    assert run(tmp_path, {OUT: text}).reds == []


# --- round 3 (enforcer #6592) ---
LIST = EV + "p1-sample-size-grandfather.md"
LIST_TEXT = (
    "# list\n\n## 3. Grandfathered\n\n| 1 | `{other}` | R2 |\n\n"
    "## 4. Owes a section at freeze (1; NOT grandfathered)\n\n"
    "| 11 | `{owed}` | R1 |\n\n## 5. Notes\n"
)


def test_a_file_the_list_file_keeps_owing_cannot_be_grandfathered_by_the_gate(tmp_path):
    owed = EV + "g.md"
    files = {
        LIST: LIST_TEXT.format(owed=owed, other=A),
        owed: "# G -- pre-registration\n\nseeds 8.\n",
    }
    report = run(tmp_path, files, grandfathered=frozenset({owed}))
    found = reds(report, "R-LIST-CONFLICT")
    assert len(found) == 1 and found[0].where == owed


def test_list_file_section_4_name_deferred_or_carrying_the_section_is_no_conflict(tmp_path):
    owed = EV + "g.md"
    files = {LIST: LIST_TEXT.format(owed=owed, other=A), owed: prereg()}
    assert run(tmp_path, files).reds == []
    files[owed] = "# G -- pre-registration\n\nseeds 8.\n"
    report = run(tmp_path, files, owes_at_freeze={owed: "2999-01-01"})
    assert report.reds == [] and report.deferred == [owed]


def test_a_grandfathered_name_in_section_3_of_the_list_file_is_no_conflict(tmp_path):
    files = {
        LIST: LIST_TEXT.format(owed=EV + "g.md", other=A),
        A: "# X -- pre-registration\n\nseeds 8.\n",
    }
    assert run(tmp_path, files, grandfathered=frozenset({A})).reds == []


def test_the_live_gate_and_the_live_list_file_agree_on_what_is_still_owing():
    text = (sg.REPO / sg.LIST_FILE).read_text(encoding="utf-8")
    owed = sg._owed_by_list(text)
    ev = "specs/022-unified-brain-arc/evidence/"
    assert ev + "g3-growth-loop.md" in owed and ev + "g5-readout-variants.md" in owed
    assert not owed & sg.GRANDFATHERED


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@needs_git
def test_a_git_checkout_reads_tracked_markdown_only_not_untracked_litter(tmp_path):
    bare = "# T -- pre-registration\n\nseeds 8.\n"
    tree(tmp_path, {A: prereg(), EV + "litter.md": bare, "nested/copy/docs/litter.md": bare})
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "--", A)
    report = sg.check(tmp_path, frozenset(), {}, frozenset(), today=date(2026, 10, 29))
    assert report.reds == [] and report.counts["pre-registrations found"] == 1


@needs_git
def test_a_git_checkout_still_reds_a_tracked_file_without_the_section(tmp_path):
    bare = "# T -- pre-registration\n\nseeds 8.\n"
    tree(tmp_path, {A: bare})
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "--", A)
    report = sg.check(tmp_path, frozenset(), {}, frozenset(), today=date(2026, 10, 29))
    assert [r.where for r in reds(report, "R-NO-SECTION")] == [A]


def test_a_positive_one_shot_claim_with_a_colon_tail_is_not_read_as_negated(tmp_path):
    line = "A one-shot design, not sequential: the pilot only estimates SD."
    text = "# X -- pre-registration\n\n## Sample size (P1)\n" + THREE + "\n" + line + "\n"
    assert run(tmp_path, {A: text}).reds == []


def test_one_shot_is_not_a_design_still_reads_as_negated(tmp_path):
    line = "A one-shot is not a design we use for this row."
    text = "# X -- pre-registration\n\n## Sample size (P1)\n" + THREE + "\n" + line + "\n"
    assert len(reds(run(tmp_path, {A: text}), "R-FIELD")) == 1


def test_the_expiry_is_lane_set_and_names_its_ratification_in_the_pinned_form():
    src = Path(sg.__file__).read_text(encoding="utf-8")
    assert "integrator-set bound" not in src
    assert re.search(
        r"integrator to ratify|ratified by the integrator \d{4}-\d{2}-\d{2}, D-73", src
    )


def test_the_freeze_dates_in_the_comments_are_author_dates():
    src = Path(sg.__file__).read_text(encoding="utf-8")
    assert "froze 2026-10-06" not in src and "2026-10-03" in src


# --- round 3b (D-73): "frozen" is keyed on the file text, never on a commit subject ---


def test_the_gate_never_reads_a_commit_subject():
    src = Path(sg.__file__).read_text(encoding="utf-8")
    assert "git log" not in src and '"log"' not in src and "'log'" not in src


def test_the_marker_rule_is_named_in_the_docstring():
    assert "freeze marker" in (sg.__doc__ or "") and "commit subject" in (sg.__doc__ or "")


@pytest.mark.parametrize(
    "line",
    [
        "Freeze commit 6bc97430 (2026-10-06): section 2 frozen as written.",
        "Status: FROZEN 2026-10-06, 2.1-2.13 as written.",
        "Frozen at 6bc97430.",
    ],
)
def test_a_dated_freeze_marker_in_the_text_ends_the_deferral(tmp_path, line):
    text = "# G -- DRAFT pre-registration\n\n" + line + "\n"
    report = run(tmp_path, {A: text}, owes_at_freeze={A: "2999-01-01"})
    assert [r.code for r in report.reds] == ["R-FROZEN-DEFERRED"]


# --- round 4 (enforcer #6613) ---
LIVE_LIST = (sg.REPO / sg.LIST_FILE).read_text(encoding="utf-8")
RENEWAL = "renewable only by a dated integrator commit"


def test_grandfathering_rests_on_execution_not_on_a_commit_reference():
    src = Path(sg.__file__).read_text(encoding="utf-8")
    doc = sg.__doc__ or ""
    assert "froze in 6bc97430" not in src and "frozen in 6bc97430" not in LIVE_LIST
    assert "executed before the landing" in doc and "history, not a freeze" in doc


def test_the_renewal_rule_is_written_in_the_comment_the_docstring_and_the_list_file():
    src = Path(sg.__file__).read_text(encoding="utf-8")
    assert RENEWAL in (sg.__doc__ or "") and src.count(RENEWAL) >= 2 and RENEWAL in LIVE_LIST
    assert "cannot see who moved a date" in (sg.__doc__ or "")


def test_list_file_section_4_counts_its_two_rows_and_says_g5_parts_a_to_d_ran():
    assert "These three rows" not in LIVE_LIST and "(G2, G3, G5)" not in LIVE_LIST
    assert "Parts A-D" in LIVE_LIST


def test_scan_line_names_a_directory_walk(tmp_path):
    report = run(tmp_path, {A: prereg()})
    assert report.scan == "walk" and "scan: walk" in sg.render(report)


@needs_git
def test_scan_line_names_git_ls_files_in_a_git_checkout(tmp_path):
    tree(tmp_path, {A: prereg()})
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "--", A)
    report = sg.check(tmp_path, frozenset(), {}, frozenset(), today=date(2026, 10, 29))
    assert report.scan == "git ls-files" and "scan: git ls-files" in sg.render(report)


@needs_git
def test_a_failing_git_in_a_git_checkout_is_printed_not_swallowed(tmp_path):
    tree(tmp_path, {A: prereg()})
    (tmp_path / ".git").write_text("gitdir: /nonexistent/gitdir\n", encoding="utf-8")
    report = sg.check(tmp_path, frozenset(), {}, frozenset(), today=date(2026, 10, 29))
    assert report.scan.startswith("walk, git failed: ")
    assert "scan: walk, git failed: " in sg.render(report)


def test_a_grandfathered_name_the_list_file_does_not_list_is_a_conflict(tmp_path):
    files = {
        LIST: LIST_TEXT.format(owed=EV + "g.md", other=EV + "o.md"),
        A: "# X -- pre-registration\n\nseeds 8.\n",
    }
    found = reds(run(tmp_path, files, grandfathered=frozenset({A})), "R-LIST-CONFLICT")
    assert [r.where for r in found] == [A] and "does not list" in found[0].message


def test_a_name_in_section_4b_of_the_list_file_may_be_grandfathered(tmp_path):
    text = LIST_TEXT.format(owed=EV + "g.md", other=EV + "o.md") + (
        "\n### 4b. Executed rows (1; grandfathered)\n\n| 10 | `" + A + "` | R1 | ran |\n"
    )
    files = {LIST: text, A: "# X -- pre-registration\n\nseeds 8.\n"}
    assert run(tmp_path, files, grandfathered=frozenset({A})).reds == []


def test_prose_naming_a_file_in_section_3_does_not_list_it(tmp_path):
    text = LIST_TEXT.format(owed=EV + "g.md", other=EV + "o.md").replace(
        "## 3. Grandfathered\n", "## 3. Grandfathered\n\nNot `" + A + "` here.\n"
    )
    files = {LIST: text, A: "# X -- pre-registration\n\nseeds 8.\n"}
    found = reds(run(tmp_path, files, grandfathered=frozenset({A})), "R-LIST-CONFLICT")
    assert [r.where for r in found] == [A]


def test_the_live_grandfathered_names_are_a_subset_of_list_file_sections_3_and_4b():
    assert sg.GRANDFATHERED <= sg._listed_by_list(LIVE_LIST)


def test_expired_deferral_of_a_file_that_carries_the_section_says_delete_the_entry(tmp_path):
    report = run(tmp_path, {A: prereg()}, owes_at_freeze=LATER, today=date(2026, 11, 6))
    found = reds(report, "R-DEFERRAL-EXPIRED")
    assert len(found) == 1 and "delete the OWES_AT_FREEZE entry" in found[0].message
    assert "write the section" not in found[0].message


def test_expired_deferral_of_a_file_without_the_section_says_write_it(tmp_path):
    report = run(tmp_path, {A: DRAFT}, owes_at_freeze=LATER, today=date(2026, 11, 6))
    assert "write the section" in reds(report, "R-DEFERRAL-EXPIRED")[0].message


# ROUND 5 (enforcer #6646)
EXECUTION = "shown by execution (a dated read in its own text)"
NOT_EVIDENCE = "a commit reference alone is not evidence of a freeze"


def test_grandfather_basis_has_one_wording_in_docstring_comments_and_list_file():
    src = Path(sg.__file__).read_text(encoding="utf-8")
    assert (sg.__doc__ or "").count(EXECUTION) >= 2 and src.count(EXECUTION) >= 4
    assert NOT_EVIDENCE in (sg.__doc__ or "") and NOT_EVIDENCE in LIVE_LIST
    assert EXECUTION in LIVE_LIST


def test_self_check_goes_red_when_the_list_file_reader_goes_blind(monkeypatch):
    monkeypatch.setattr(sg, "_listed_by_list", lambda text: frozenset({A}))
    assert sg.self_check() == 1


def test_self_check_goes_red_when_the_section_4_reader_goes_blind(monkeypatch):
    monkeypatch.setattr(sg, "_owed_by_list", lambda text: frozenset())
    assert sg.self_check() == 1


def test_limit_line_names_the_unseen_date_move(tmp_path):
    out = sg.render(run(tmp_path, {A: prereg()}))
    assert "who moved" in out.split("LIMIT:")[1].split("\n")[0]


def test_grandfathered_comment_count_is_the_live_count():
    src = Path(sg.__file__).read_text(encoding="utf-8")
    assert f"({len(sg.GRANDFATHERED)} evidence files)" in src


def test_docstring_does_not_overclaim_that_the_two_sources_cannot_disagree():
    doc = sg.__doc__ or ""
    assert "cannot disagree" not in doc
    assert "unless the same commit edits the integrator-ruled list file" in doc


def test_a_sub_heading_inside_section_3_keeps_its_rows_listed():
    text = (
        "## 3. List\n\n| 1 | `a.md` |\n\n### 3a. More\n\n| 2 | `b.md` |\n\n"
        "## 4. Owes\n\n| 3 | `c.md` |\n"
    )
    assert sg._listed_by_list(text) == {"a.md", "b.md"}
    assert sg._owed_by_list(text) == {"c.md"}

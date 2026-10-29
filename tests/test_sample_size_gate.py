"""P1 sample-size gate: one reachable red per rule, the clean pass, the closed grandfather list.

docs/POLICIES.md, "P1 amendment", Gate specification. Every input is built in a temporary tree,
so each red is shown to fire and a pass is not vacuous. The inputs are written out by hand and not
taken from the gate's own self-check plants: a test must not share its oracle with its code.
"""

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


def test_pre_landing_frozen_drafts_g2_g3_are_grandfathered_and_g5_part_e_is_deferred():
    ev = "specs/022-unified-brain-arc/evidence/"
    assert {ev + "g2-taught-problems.md", ev + "g3-growth-loop.md"} <= sg.GRANDFATHERED
    assert set(sg.OWES_AT_FREEZE) == {ev + "g5-readout-variants.md"}
    assert ev + "g5-readout-variants.md" not in sg.GRANDFATHERED


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

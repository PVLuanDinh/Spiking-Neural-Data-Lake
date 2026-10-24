"""P5b gate: one reachable red per code, the clean pass, the real-tree count.

docs/POLICIES.md P5b Gate specification. Every input is built by hand in a temporary git tree, so
each red is shown to fire and the real-tree pass is not vacuous. The trees are not taken from the
gate's own self-check plants: a test must not share its oracle with the code it tests.
"""

from pathlib import Path

import pytest

from snn_data_lake.gates import native_dormancy
from snn_data_lake.gates import p5b_gate as pg

EV = "specs/s/evidence/p.md"
MOD = "c/src/lib.rs"
CITE = f"//! P5b-PREREG: {EV}\n"
LABELS = ("hot", "python-bound", "coarse-boundary", "charged", "python-fix-first", "parity-oracle")
WHY = f"before the gate; see {pg.RETRO}"


def record(modules=MOD, drop=None, **over):
    rows = {k: f"P5b-{k} {LABELS[k - 1]}: MET -- {k * 7} ms over 5 seeds" for k in range(1, 7)}
    rows.update({int(k): v for k, v in over.items()})
    rows.pop(drop, None)
    return "\n".join([*rows.values(), f"P5b-MODULES: {modules}", ""])


def same(root, rels):
    return {rel: "h" for rel in rels}


def run(tmp_path, files, grandfathered=None, cutoff=lambda root: {}, blobs=same):
    native_dormancy._write(tmp_path, files)
    return pg.check(tmp_path, grandfathered=grandfathered or {}, cutoff=cutoff, blobs=blobs)


def tree(src="//! x\n" + CITE, rec=None, **extra):
    return {f"native/{MOD}": src, EV: record() if rec is None else rec, **extra}


def test_clean_tree_is_green_and_counted(tmp_path):
    report = run(tmp_path, tree(**{"native/c/tests/t.rs": "// t\n"}))
    assert report.reds == []
    assert report.counts == {
        "modules": 1,
        "cited": 1,
        "grandfathered": 0,
        "test files": 1,
        "reds": 0,
    }
    assert pg.count_line(report) == "modules 1, cited 1, grandfathered 0, test files 1, reds 0"


def test_native_absent_passes_and_says_so(tmp_path):
    report = run(tmp_path, {"README.md": "x\n"})
    assert report.reds == [] and report.absent
    assert pg.count_line(report) == "native/ ABSENT"


def test_native_present_with_zero_modules_reds(tmp_path):
    assert "R-ZERO" in run(tmp_path, {"native/c/README": ""}).codes()


def test_uncited_module_reds(tmp_path):
    assert run(tmp_path, tree(src="//! x\n")).codes() == {"R-UNCITED"}


def test_two_citations_red(tmp_path):
    assert "R-DUP-CITE" in run(tmp_path, tree(src=CITE + CITE)).codes()


def test_citation_outside_the_leading_block_is_not_a_citation(tmp_path):
    assert run(tmp_path, tree(src="//! x\nfn f() {}\n" + CITE)).codes() == {"R-UNCITED"}


def test_cited_file_missing_reds(tmp_path):
    report = run(tmp_path, {f"native/{MOD}": CITE})
    assert report.codes() == {"R-MISSING"}


@pytest.mark.parametrize("k", range(1, 7))
def test_each_missing_criterion_reds_once(tmp_path, k):
    reds = run(tmp_path, tree(rec=record(drop=k))).reds
    assert [(r.code, LABELS[k - 1] in r.message) for r in reds] == [("R-CRITERION", True)]


def test_duplicate_criterion_reds(tmp_path):
    rec = record() + "P5b-3 coarse-boundary: MET -- 9 calls\n"
    assert run(tmp_path, tree(rec=rec)).codes() == {"R-CRITERION"}


def test_wrong_label_does_not_count_as_the_criterion(tmp_path):
    rec = record(**{"4": "P5b-4 cheap: MET -- 3 ms"})
    assert run(tmp_path, tree(rec=rec)).codes() == {"R-CRITERION"}


def test_evidence_without_a_digit_reds(tmp_path):
    rec = record(**{"5": "P5b-5 python-fix-first: MET -- tried it, no gain"})
    assert run(tmp_path, tree(rec=rec)).codes() == {"R-NONUMBER"}


@pytest.mark.parametrize("verdict", ["UNMET", "PENDING", "met"])
def test_verdict_other_than_met_reds(tmp_path, verdict):
    rec = record(**{"6": f"P5b-6 parity-oracle: {verdict} -- 12 vectors"})
    assert run(tmp_path, tree(rec=rec)).codes() == {"R-UNMET"}


def test_record_not_naming_the_module_reds(tmp_path):
    assert run(tmp_path, tree(rec=record(modules="c/src/other.rs"))).codes() == {"R-UNLISTED"}


def test_record_naming_several_modules_serves_each(tmp_path):
    files = tree(rec=record(modules=f"{MOD}, c/src/b.rs"), **{"native/c/src/b.rs": CITE})
    assert run(tmp_path, files).reds == []


def test_grandfathered_module_is_green_and_counted(tmp_path):
    files = tree(src="//! x\n", **{pg.RETRO: f"{MOD}\n"})
    report = run(tmp_path, files, {MOD: WHY}, cutoff=lambda root: {MOD: "h"})
    assert report.reds == [] and report.counts["grandfathered"] == 1 and report.counts["cited"] == 0


def test_grandfather_entry_for_a_missing_module_reds(tmp_path):
    gone = "c/src/gone.rs"
    files = tree(src="//! x\n", **{pg.RETRO: f"{MOD}\n{gone}\n"})
    report = run(tmp_path, files, {MOD: WHY, gone: WHY}, cutoff=lambda root: {MOD: "h", gone: "h"})
    assert [(r.code, r.where) for r in report.reds] == [("R-STALE-GRANDFATHER", gone)]


def test_grandfathered_module_that_also_cites_reds(tmp_path):
    files = tree(**{pg.RETRO: f"{MOD}\n"})
    report = run(tmp_path, files, {MOD: WHY}, cutoff=lambda root: {MOD: "h"})
    assert report.codes() == {"R-STALE-GRANDFATHER"}


def test_grandfather_reason_naming_a_missing_path_reds(tmp_path):
    report = run(tmp_path, tree(src="//! x\n"), {MOD: WHY}, cutoff=lambda root: {MOD: "h"})
    assert report.codes() == {"R-MISSING"}


def test_grandfather_entry_added_after_the_cutoff_reds(tmp_path):
    files = tree(src="//! x\n", **{pg.RETRO: f"{MOD}\n"})
    report = run(tmp_path, files, {MOD: WHY}, cutoff=lambda root: {})
    assert report.codes() == {"R-GRANDFATHER-OPEN"}


def test_grandfather_record_that_never_names_the_module_reds(tmp_path):
    files = tree(src="//! x\n", **{pg.RETRO: "c/src/other.rs\n"})
    report = run(tmp_path, files, {MOD: WHY}, cutoff=lambda root: {MOD: "h"})
    assert [(r.code, r.where) for r in report.reds] == [("R-STALE-GRANDFATHER", MOD)]


def test_grandfather_record_may_name_a_module_by_file_name_under_its_directory(tmp_path):
    files = tree(src="//! x\n", **{pg.RETRO: "`c/src/*`: lib.rs\n"})
    assert run(tmp_path, files, {MOD: WHY}, cutoff=lambda root: {MOD: "h"}).reds == []


def test_grandfathered_module_edited_since_the_cutoff_reds(tmp_path):
    files = tree(src="//! x\nmod hot { fn k() {} }\n", **{pg.RETRO: f"{MOD}\n"})
    report = run(
        tmp_path,
        files,
        {MOD: WHY},
        cutoff=lambda root: {MOD: "h"},
        blobs=lambda root, rels: {rel: "changed" for rel in rels},
    )
    assert report.codes() == {"R-GRANDFATHER-DRIFT"}


def test_unhashable_grandfathered_module_fails_closed(tmp_path):
    files = tree(src="//! x\n", **{pg.RETRO: f"{MOD}\n"})
    report = run(
        tmp_path, files, {MOD: WHY}, cutoff=lambda root: {MOD: "h"}, blobs=lambda root, rels: None
    )
    assert report.codes() == {"R-HISTORY"}


def test_rs_file_outside_src_is_a_module(tmp_path):
    report = run(tmp_path, tree(**{"native/c/kernels/hot.rs": "//! x\n"}))
    assert [(r.code, r.where) for r in report.reds] == [("R-UNCITED", "c/kernels/hot.rs")]
    assert report.counts["modules"] == 2


@pytest.mark.parametrize("sub", ["tests", "benches", "examples"])
def test_tests_benches_examples_are_not_modules(tmp_path, sub):
    report = run(tmp_path, tree(**{f"native/c/{sub}/t.rs": "// t\n"}))
    assert report.reds == [] and report.counts["modules"] == 1 and report.counts["test files"] == 1


def test_build_script_is_excluded_by_name_and_counted(tmp_path):
    report = run(tmp_path, tree(**{"native/c/build.rs": "fn main() {}\n"}))
    assert report.reds == [] and report.counts["build.rs"] == 1
    assert pg.count_line(report).endswith("test files 0, build.rs 1, reds 0")


def test_a_src_file_named_build_rs_is_still_a_module(tmp_path):
    assert run(tmp_path, tree(**{"native/c/src/build.rs": "//! x\n"})).codes() == {"R-UNCITED"}


def test_module_citing_itself_with_a_block_comment_reds(tmp_path):
    body = "/*\n" + record().replace("P5b-", " * P5b-") + "*/\n"
    src = f"//! P5b-PREREG: native/{MOD}\n{body}"
    assert run(tmp_path, {f"native/{MOD}": src}).codes() == {"R-RECORD-PLACE"}


@pytest.mark.parametrize(
    "cited",
    ["docs/p.md", "specs/p.txt", "specs/../docs/p.md", "/specs/p.md", "specs\\p.md", "native/p.md"],
)
def test_record_must_be_a_specs_markdown_document(tmp_path, cited):
    files = {f"native/{MOD}": f"//! P5b-PREREG: {cited}\n"}
    assert run(tmp_path, files).codes() == {"R-RECORD-PLACE"}


def test_git_ignored_record_reds(tmp_path):
    files = tree(**{".gitignore": f"{EV}\n"})
    assert run(tmp_path, files).codes() == {"R-RECORD-PLACE"}


@pytest.mark.parametrize(
    "evidence", ["see #3847", "per 2026-10-17", "at 7bc376d4faa0", "in notes/t018.md", "x.rs"]
)
def test_a_citation_is_not_a_number(tmp_path, evidence):
    rec = record(**{"1": f"P5b-1 hot: MET -- {evidence}"})
    assert run(tmp_path, tree(rec=rec)).codes() == {"R-NONUMBER"}


def test_a_number_beside_a_citation_still_counts(tmp_path):
    rec = record(**{"1": "P5b-1 hot: MET -- see #3847: 12 ms, 0.8 power"})
    assert run(tmp_path, tree(rec=rec)).reds == []


def test_unreadable_history_fails_closed(tmp_path):
    assert run(tmp_path, tree(), cutoff=lambda root: None).codes() == {"R-HISTORY"}


def test_cutoff_modules_is_none_where_the_cutoff_commit_is_absent(tmp_path):
    native_dormancy._write(tmp_path, tree())
    assert pg.cutoff_modules(tmp_path) is None


def test_real_grandfathered_modules_match_their_cutoff_blobs():
    then = pg.cutoff_modules(Path(pg.REPO))
    assert then is not None
    now = pg.current_blobs(Path(pg.REPO), sorted(pg.GRANDFATHERED))
    assert now == {rel: then[rel] for rel in pg.GRANDFATHERED}


def test_census_lists_every_rs_with_a_status(tmp_path):
    files = tree(
        **{"native/c/tests/t.rs": "// t\n", "native/c/src/g.rs": "//! x\n", "native/c/build.rs": ""}
    )
    native_dormancy._write(tmp_path, files)
    assert dict(pg.census(tmp_path, {"c/src/g.rs": WHY})) == {
        "c/build.rs": "OUT-OF-SCOPE",
        "c/src/g.rs": "GRANDFATHERED",
        "c/src/lib.rs": "CITED",
        "c/tests/t.rs": "TEST-FILE",
    }


def test_main_exit_codes(tmp_path, capsys):
    native_dormancy._write(tmp_path, tree(src="//! x\n"))
    assert pg.main(["--assert"], root=tmp_path) == 1
    assert "RED R-UNCITED" in capsys.readouterr().out


def test_self_check_passes():
    assert pg.self_check() == 0


def test_real_tree_counts_eleven_grandfathered_and_no_reds():
    report = pg.check(Path(pg.REPO))
    assert report.counts == {
        "modules": 11,
        "cited": 0,
        "grandfathered": 11,
        "test files": 1,
        "reds": 0,
    }, pg.render(report)
    assert report.reds == []
    assert len(pg.GRANDFATHERED) == 11

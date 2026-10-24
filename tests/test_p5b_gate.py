"""P5b gate: one reachable red per code, the clean pass, the real-tree count.

docs/POLICIES.md P5b (its Mechanization line). A module is any native `.rs` except a crate root's
build.rs and its tests/benches/examples; a record is a specs/**.md file naming each grandfathered
module by full path. Every input is built by hand in a temporary git tree, so each red is shown to
fire and the real-tree pass is not vacuous. The trees are not taken from the gate's own self-check
plants: a test must not share its oracle with the code it tests.
"""

import subprocess
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


@pytest.mark.parametrize(
    "record_text", ["`c/src/*`: lib.rs\n", f"x{MOD}\n", f"{MOD}x\n", "lib.rs\n"]
)
def test_grandfather_record_naming_a_module_by_file_name_or_inside_another_token_reds(
    tmp_path, record_text
):
    files = tree(src="//! x\n", **{pg.RETRO: record_text})
    report = run(tmp_path, files, {MOD: WHY}, cutoff=lambda root: {MOD: "h"})
    assert [(r.code, r.where) for r in report.reds] == [("R-STALE-GRANDFATHER", MOD)]


@pytest.mark.parametrize(
    "record_text", [f"`{MOD}`\n", f"| {MOD} |\n", f"a, {MOD}, b\n", f"{MOD}", f"x {MOD} y\n"]
)
def test_grandfather_record_naming_a_module_as_a_whole_token_is_green(tmp_path, record_text):
    files = tree(src="//! x\n", **{pg.RETRO: record_text})
    assert run(tmp_path, files, {MOD: WHY}, cutoff=lambda root: {MOD: "h"}).reds == []


def test_record_that_drops_one_lib_rs_while_another_stays_reds(tmp_path):
    other = "d/src/lib.rs"
    files = tree(src="//! x\n", **{pg.RETRO: f"`{other}`\n", f"native/{other}": "//! x\n"})
    report = run(
        tmp_path,
        files,
        {MOD: WHY, other: WHY},
        cutoff=lambda root: {MOD: "h", other: "h"},
    )
    assert [(r.code, r.where) for r in report.reds] == [("R-STALE-GRANDFATHER", MOD)]


def test_the_pfc_reason_says_it_landed_under_p5b_and_the_others_do_not():
    pfc = pg.GRANDFATHERED["sndl-core/src/prefrontal_executive.rs"]
    assert "5978bbcc" in pfc and "c3079685" in pfc and "p5b-pfc-retro-bench.md" in pfc
    others = [why for rel, why in pg.GRANDFATHERED.items() if "prefrontal" not in rel]
    assert len(others) == 10 and set(others) == {pg._WHY}


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


def test_a_tests_directory_inside_src_is_a_module(tmp_path):
    report = run(tmp_path, tree(**{"native/c/src/tests/t.rs": "//! x\n"}))
    assert [(r.code, r.where) for r in report.reds] == [("R-UNCITED", "c/src/tests/t.rs")]
    assert report.counts["modules"] == 2 and report.counts["test files"] == 0


def test_only_a_crate_roots_own_build_rs_is_excluded(tmp_path):
    report = run(tmp_path, tree(**{"native/c/kernels/build.rs": "//! x\n"}))
    assert report.codes() == {"R-UNCITED"} and "build.rs" not in report.counts


@pytest.mark.parametrize(
    ("head", "path"),
    [
        ("[lib]", "build.rs"),
        ("[lib]", "tests/lib.rs"),
        ("[lib]", "examples/lib.rs"),
        ("[lib]", "benches/lib.rs"),
        ("[[bin]]", "tests/main.rs"),
        ("[[example]]", "build.rs"),
    ],
)
def test_a_cargo_target_resolving_to_an_excluded_file_reds(tmp_path, head, path):
    files = tree(
        **{
            "native/c/Cargo.toml": f'{head}\npath = "{path}"\n',
            f"native/c/{path}": "#[pyfunction]\nfn f() {}\n",
        }
    )
    assert "R-TARGET-EXCLUDED" in run(tmp_path, files).codes()


def test_planted_lib_path_build_rs_is_the_one_new_red(tmp_path):
    files = tree(
        **{"native/c/Cargo.toml": '[lib]\npath = "build.rs"\n', "native/c/build.rs": "fn f() {}\n"}
    )
    reds = run(tmp_path, files).reds
    assert [(r.code, r.where) for r in reds] == [("R-TARGET-EXCLUDED", "c/build.rs")]


def test_an_example_target_in_its_own_directory_and_the_default_lib_are_green(tmp_path):
    files = tree(
        **{
            "native/c/Cargo.toml": (
                '[lib]\npath = "src/lib.rs"\n[[example]]\nname = "e"\npath = "examples/e.rs"\n'
            ),
            "native/c/examples/e.rs": "fn main() {}\n",
        }
    )
    assert run(tmp_path, files).reds == []


def test_an_unreadable_cargo_manifest_fails_closed(tmp_path):
    report = run(tmp_path, tree(**{"native/c/Cargo.toml": "[lib\n"}))
    assert report.codes() == {"R-RUSTFACTS"}


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


def _gone(tmp_path, files, rel):
    native_dormancy._write(tmp_path, files)
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    (tmp_path / rel).unlink()


def test_a_listed_rs_gone_from_disk_reds_instead_of_crashing(tmp_path):
    _gone(tmp_path, tree(), f"native/{MOD}")
    report = pg.check(tmp_path, grandfathered={}, cutoff=lambda root: {}, blobs=same)
    assert [(r.code, r.where) for r in report.reds] == [("R-RUSTFACTS", MOD)]


def test_census_marks_a_listed_rs_gone_from_disk_instead_of_crashing(tmp_path):
    _gone(tmp_path, tree(), f"native/{MOD}")
    assert pg.census(tmp_path, {}) == [(MOD, "R-RUSTFACTS unreadable")]


def test_census_survives_rust_facts_unavailable(tmp_path, monkeypatch):
    native_dormancy._write(tmp_path, tree())

    def boom(root, sub):
        raise pg.rust_facts.RustFactsUnavailable("no git")

    monkeypatch.setattr(pg.rust_facts, "rust_files", boom)
    assert pg.census(tmp_path, {}) == [("native", "R-RUSTFACTS no git")]


def test_limit_line_names_the_id_hole_the_ratio_rule_and_the_record_timing(tmp_path):
    text = pg.render(run(tmp_path, tree()))
    for needle in ("T018", "B2", "PFC-NEUTER-1023", "ratios", "predates its module"):
        assert needle in text


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

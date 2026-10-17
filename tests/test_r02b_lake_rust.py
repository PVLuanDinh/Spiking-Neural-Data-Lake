"""The Rust half of immortal_retention (L5, DISPOSITION row 9) and move_gate (G8, row 10): the fixture
RED arms, their GREEN twins, fail-closed, and the real tree. Every red goes through the gate's own entry
point (`all_findings`, `by_file`, `static`/`diff`, `copy_tree`, `c2d`) with the gate's own fixture injected."""

import subprocess

from snn_data_lake.gates import immortal_retention as l5
from snn_data_lake.gates import move_gate as mg
from snn_data_lake.gates import rust_facts

PAIRS, PKGS = {("oldpkg", "mod"): "newpkg"}, {"gonepkg"}


def _tree(root, sources, under=()):
    """A constructed tree under `root` (a git repo: rust_facts reads git's view)."""
    for rel, text in sources.items():
        p = root.joinpath(*under, *rel.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


# ---- L5: immortal_retention ------------------------------------------------------------------------
def _l5_lines(monkeypatch, tree):
    monkeypatch.setattr(l5, "REPO", tree)
    return [f for f in l5.all_findings() if f.startswith("native/")]


def test_l5_red_arm_is_exactly_the_four_preregistered_reclaimers(tmp_path, monkeypatch):
    _tree(tmp_path, l5._rust_fixture(red=True), under=("native",))
    assert _l5_lines(monkeypatch, tmp_path) == [
        f"native/sndl-core/src/lake.rs:{line} fn {fn} reclaims a lake file without a retention read"
        for line, fn in ((6, "compact_lake"), (11, "evict"), (15, "truncate_log"), (19, "reopen_empty"))
    ]


def test_l5_green_twin_is_zero(tmp_path, monkeypatch):
    _tree(tmp_path, l5._rust_fixture(red=False), under=("native",))
    assert _l5_lines(monkeypatch, tmp_path) == []


def test_l5_rust_facts_unavailable_with_rs_present_is_one_red(tmp_path, monkeypatch):
    _tree(tmp_path, l5._rust_fixture(red=False), under=("native",))
    monkeypatch.setattr(rust_facts.shutil, "which", lambda _name: None)
    lines = _l5_lines(monkeypatch, tmp_path)
    assert len(lines) == 1 and "RustFactsUnavailable" in lines[0], lines


def test_l5_real_tree_prints_absent_and_python_counts_hold():
    out = l5.report()
    assert "native/ ABSENT (0 .rs files)" in out
    assert "reclaimers of DURABLE records that do not consult the label: 0" in out
    assert out.rstrip().endswith("findings: 0")


# ---- G8: move_gate ----------------------------------------------------------------------------------
def _g8(root, red=True, **kw):
    return _tree(root, mg._native_fixture(red, **kw))


def test_g8a_module_file_resolves_the_extension_and_an_absent_one_stays_red(tmp_path):
    tree = _g8(tmp_path)
    regs = {"run_selfchecks": ["snn_data_lake._native"], "ci.m_targets": ["snn_data_lake._native"]}
    assert mg.by_file(tree, regs) == ["ci.m_targets: snn_data_lake._native has no __main__"]
    (tree / mg._NATIVE_EXT).unlink()
    assert mg.by_file(tree, regs) == [
        "ci.m_targets: snn_data_lake._native",
        "run_selfchecks: snn_data_lake._native",
    ]


def test_g8b_census_records_the_lock_hash_and_the_artifact_and_a_lock_change_diffs(tmp_path):
    a = _g8(tmp_path / "a")
    b = _g8(tmp_path / "b", lock="# Cargo.lock\nversion = 4\n[[package]]\n")
    ra, rb = (mg.static(t, mg.tables(t)).get("native") for t in (a, b))
    assert ra and ra["status"] == "native/ PRESENT (1 .rs files)", ra
    assert ra["extension"] == [mg._NATIVE_EXT]
    assert list(ra["cargo_lock"]) == ["native/sndl-core/Cargo.lock"]
    assert ra["cargo_lock"] != rb["cargo_lock"]
    pre, post = mg._mini(), mg._mini()
    pre["static"]["native"], post["static"]["native"] = ra, rb
    probs = mg.diff(pre, post, {})[1]
    assert len(probs) == 2 and all("cargo_lock.native/sndl-core/Cargo.lock" in p for p in probs), probs
    assert mg.diff(pre, pre, {})[1] == []


def test_g8c_copy_tree_carries_the_gitignored_extension(tmp_path):
    tree, dest = _g8(tmp_path / "repo"), tmp_path / "copy"
    mg.copy_tree(tree, dest, False)
    assert (dest / mg._NATIVE_EXT).is_file()
    assert (dest / ".gitignore").is_file() and (dest / "a.py").is_file()
    assert not (dest / "src" / "snn_data_lake" / "junk.pyc").exists()


def test_g8d_c2d_reads_rs_string_literals(tmp_path):
    found, n, unparsed = mg.c2d(_g8(tmp_path), PAIRS, PKGS)
    assert (n, unparsed) == (2, [])
    assert found == [
        ("native/sndl-python/src/lib.rs", 3, "rs-string", "oldpkg.mod"),
        ("native/sndl-python/src/lib.rs", 4, "rs-string", "gonepkg"),
    ]


def test_g8d_green_twin_is_zero(tmp_path):
    assert mg.c2d(_g8(tmp_path, red=False), PAIRS, PKGS)[0] == []


def test_g8d_rust_facts_unavailable_with_rs_present_is_red(tmp_path, monkeypatch):
    tree = _g8(tmp_path, red=False)
    monkeypatch.setattr(rust_facts.shutil, "which", lambda _name: None)
    found, _n, unparsed = mg.c2d(tree, PAIRS, PKGS)
    assert found == [] and len(unparsed) == 1 and "UNAVAILABLE" in unparsed[0], unparsed
    assert mg.main(["--c2d", "--root", str(tree)]) == 1


def test_g8_real_tree_c2d_unchanged_and_census_absent():
    found, _n, unparsed = mg.c2d(mg.REPO)
    assert (found, unparsed) == ([], [])
    rec = mg.static(mg.REPO, mg.tables(mg.REPO)).get("native")
    assert rec and rec["status"] == "native/ ABSENT (0 .rs files)" and rec["cargo_lock"] == {}, rec

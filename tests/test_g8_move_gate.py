"""G8 S1a and S1b (specs/022 evidence g8-reorganization.md 2.4-2.6): the frozen LEGACY_* label
tables, the package_of()/module_path() join the region-keyed gates read through, move_gate's diff
engine and its registration (S1a); the legacy alias layer, its strict tripwire, the re-pointed
checkpoint provider and C8/C9 by file (S1b); C2(d)'s scanner, the census mode and the reds'
basename-only names (S1d). The tree-level reds (C1, F5, C2(b), F3(c), F2, C2(c), C5 and C2(d)'s
positive control) run through `python -m snn_data_lake.gates.move_gate --self-check --mover MOVER`,
per stage.
"""

import ast
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys

import pytest

import snn_data_lake as S
from snn_data_lake import brain_atlas
from snn_data_lake.gates import move_gate as mg

# The S0-head REGIONS / TORCH_TIER / TRAIN_TIER (393411b), digested: LEGACY_* carry their keys and
# members byte-identical and in order, and are never edited after S1a (C6). Re-derived at S1a from
# the S0 head's own tables through a git-archive export, and equal.
S0_TABLES_SHA256 = "4e3ab7cdfec8ff0f51287a2859c0875ce990fd50bdf6cddd63c6f64404982394"


def _digest(*tables):
    return hashlib.sha256(json.dumps(list(tables), sort_keys=True).encode()).hexdigest()


def test_legacy_tables_are_the_frozen_s0_tables():
    assert _digest(S.LEGACY_REGIONS, S.LEGACY_TORCH_TIER, S.LEGACY_TRAIN_TIER) == S0_TABLES_SHA256
    # the red: one planted member moves the digest
    planted = {**S.LEGACY_REGIONS, "cortex": [*S.LEGACY_REGIONS["cortex"], "planted"]}
    assert _digest(planted, S.LEGACY_TORCH_TIER, S.LEGACY_TRAIN_TIER) != S0_TABLES_SHA256
    assert "gates" not in S.LEGACY_REGIONS  # a G8 package is never a pre-G8 label (decision 4)


def test_module_path_is_the_live_join_and_resolves_by_file():
    assert S.module_path("move_gate") == "snn_data_lake.gates.move_gate"
    assert S.package_of("move_gate") == "gates"
    assert S.package_of("no_such_module") is None
    with pytest.raises(KeyError):
        S.module_path("no_such_module")
    src = os.path.dirname(S.__file__)
    for table in (S.REGIONS, S.TORCH_TIER, S.TRAIN_TIER):
        for pkg, mods in table.items():
            for m in mods:
                assert S.package_of(m) == pkg
                path = os.path.join(src, *S.module_path(m).split(".")[1:]) + ".py"
                assert os.path.isfile(path), f"{m}: module_path names no file ({path})"


def test_region_of_answers_the_frozen_label():
    for table in (S.LEGACY_REGIONS, S.LEGACY_TORCH_TIER, S.LEGACY_TRAIN_TIER):
        for region, mods in table.items():
            for m in mods:
                assert brain_atlas.region_of(m) == region
    assert brain_atlas.region_of("move_gate") is None  # code layout, not anatomy


def test_label_consumers_take_paths_from_module_path(monkeypatch):
    """F5 at unit grain: the file and the reachability of a labelled module come from
    module_path(); point it at a package that holds nothing and both censuses lose `insula`."""
    from snn_data_lake.substrate import atlas_node_gate, region_charge_gate

    assert atlas_node_gate.node_references()["insula"]
    assert "insula" in region_charge_gate.reachable_regions()
    monkeypatch.setattr(S, "module_path", lambda m: f"snn_data_lake.g8_nowhere.{m}")
    assert atlas_node_gate.node_references()["insula"] == []
    assert "insula" not in region_charge_gate.reachable_regions()


def test_selfcheck_cache_anchor_is_line_anchored():
    """F17: `LEGACY_REGIONS = {` contains the bare `REGIONS = {` substring."""
    text = 'LEGACY_REGIONS = {\n    "a": ["x"],\n}\nREGIONS = {\n    "a": ["x"],\n}\n'
    naive = text.index("REGIONS = {")
    anchored = re.search(r"(?m)^REGIONS = \{", text).start()
    assert naive < text.index("\nREGIONS") < anchored  # the substring lands inside LEGACY_*
    from snn_data_lake.substrate import selfcheck_cache

    assert selfcheck_cache._EXPECTED_ROSTERS["root"][3:] == tuple("LEGACY_" + t for t in mg.TABLES)


def test_move_gate_constructed_reds_fire(monkeypatch):
    assert mg.constructed_reds() == []
    monkeypatch.setattr(mg, "compare", lambda obs, declared: [])
    assert mg.constructed_reds(), "a blinded comparison must red the bare self-check"


def test_move_gate_diff_is_exact_both_ways():
    a, b = mg._mini(("x 1",)), mg._mini(("x 1", "y 2"))
    obs, probs = mg.diff(a, b, {})
    assert obs == {"gates": {"g": {"+": ["y 2"]}}} and probs
    assert mg.diff(a, b, obs)[1] == []
    assert mg.diff(a, a, obs)[1]  # declared but absent


def test_move_gate_is_registered_and_imports_nothing_it_measures():
    from snn_data_lake.substrate import dormancy_gate, import_manifest

    assert S.REGIONS["gates"] == ["move_gate"]
    assert dormancy_gate.EXEMPT["move_gate"][0] == "permanent"
    assert "gates" in import_manifest.GOVERNED
    spec = importlib.util.find_spec("snn_data_lake.gates.move_gate")
    with open(spec.origin, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    # its subprocess probes are strings; the module itself imports nothing of the package
    assert not {m for m in imported if m.startswith("snn_data_lake")}, "stdlib only"


# ---- S1b ----------------------------------------------------------------------------------------
def test_legacy_layer_is_empty_before_any_move_and_the_provider_returns_it():
    from snn_data_lake.cortex import interaction_learning as ixn

    assert S._LEGACY == {} and S._LEGACY_PKGS == frozenset()
    assert ixn._g0c_legacy_modules() is S._LEGACY  # the ONE table, read at call time (P2)
    assert len(S._MOVED) == 286 and "claustrum" not in S._MOVED


@pytest.fixture
def planted_legacy(monkeypatch):
    """A legacy package and path planted in-process: `g8probe` stands for an emptied package whose
    module `region_charge_gate` now lives in `substrate` (nothing on disk changes)."""
    monkeypatch.setattr(S, "_LEGACY_PKGS", S._LEGACY_PKGS | {"g8probe"})
    monkeypatch.setitem(S._LEGACY, "g8probe.region_charge_gate", "substrate.region_charge_gate")
    yield
    for name in ("snn_data_lake.g8probe.region_charge_gate", "snn_data_lake.g8probe"):
        sys.modules.pop(name, None)
    if hasattr(S, "g8probe"):
        delattr(S, "g8probe")  # the import system bound the synthetic package on its parent


def test_legacy_path_is_the_module_and_the_package_is_synthetic(planted_legacy, monkeypatch):
    monkeypatch.delenv("SNN_ALIAS_STRICT", raising=False)
    pkg = importlib.import_module("snn_data_lake.g8probe")
    assert list(pkg.__path__) == [] and "G8" in pkg.__doc__
    old = importlib.import_module("snn_data_lake.g8probe.region_charge_gate")
    new = importlib.import_module("snn_data_lake.substrate.region_charge_gate")
    assert old is new  # identity is the contract: one object, two names
    spec = S._AliasFinder().find_spec("snn_data_lake.g8probe.region_charge_gate")
    assert spec.has_location and spec.origin == new.__spec__.origin  # runpy sets __file__ from it
    code = spec.loader.get_code("snn_data_lake.g8probe.region_charge_gate")
    assert code.co_filename == new.__spec__.origin  # compiled under the target's own name
    flat = S._AliasFinder().find_spec("snn_data_lake.region_charge_gate")
    assert not hasattr(flat.loader, "get_code")  # the flat alias serves imports only (F2)


def test_strict_raises_on_legacy_names_never_on_a_flat_alias(planted_legacy, monkeypatch):
    monkeypatch.setenv("SNN_ALIAS_STRICT", "1")
    for name in ("snn_data_lake.g8probe", "snn_data_lake.g8probe.region_charge_gate"):
        with pytest.raises(ImportError, match="SNN_ALIAS_STRICT"):
            importlib.import_module(name)
    assert S._AliasFinder().find_spec("snn_data_lake.signal_loop") is not None


def test_registry_names_resolve_by_file_and_a_stale_one_reds():
    names = mg.names(mg.REPO, mg.tables(mg.REPO))
    assert mg.by_file(mg.REPO, names) == []
    stale = "snn_data_lake.insula.gone"
    names["manifest_gate.REGISTRANTS"] = [*names["manifest_gate.REGISTRANTS"], stale]
    assert mg.by_file(mg.REPO, names) == [f"manifest_gate.REGISTRANTS: {stale}"]


def test_worktree_export_drops_a_renamed_source_and_the_directory_it_empties(tmp_path):
    """A `git mv` leaves no file at its source: the WORKTREE export must not either (M1, the
    first stage that renames, found the export keeping every moved module at its old path), and a
    directory the move empties is gone (the F4 shadow). The unrenamed file is the control."""
    import subprocess

    repo = tmp_path / "repo"
    (repo / "old").mkdir(parents=True)
    (repo / "old" / "mod.txt").write_text("m\n", encoding="utf-8")
    (repo / "keep.txt").write_text("k\n", encoding="utf-8")
    ident = ["-c", "user.email=g8@example.invalid", "-c", "user.name=g8"]
    for args in (["init", "-q"], ["add", "-A"], [*ident, "commit", "-q", "-m", "base"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "new").mkdir()
    subprocess.run(
        ["git", "mv", "old/mod.txt", "new/mod.txt"], cwd=repo, check=True, capture_output=True
    )
    dest = mg.export("WORKTREE", tmp_path / "export", repo)
    assert (dest / "new" / "mod.txt").is_file() and (dest / "keep.txt").is_file()
    assert not (dest / "old" / "mod.txt").exists(), "the rename's source survived the export"
    assert not (dest / "old").exists(), "the directory the rename emptied survived the export"


# ---- S1d ----------------------------------------------------------------------------------------
def test_c2d_scanner_reds_when_a_detector_is_blinded(monkeypatch):
    """The bare self-check's C2(d) red: the constructed source names every form; blind one
    detector and the self-check reds."""
    assert mg.constructed_reds() == []
    monkeypatch.setattr(mg, "_split_hits", lambda operands, pairs, pkgs: [])
    assert any(f.startswith("C2(d)") for f in mg.constructed_reds())


def test_c2d_reads_the_trees_own_legacy_names_and_finds_none_in_live_code():
    """The AST twin of the running alias layer: the tree's legacy pairs and packages equal
    `_LEGACY` and `_LEGACY_PKGS`, and no live `.py` spells one of them."""
    tabs = mg.tables(mg.REPO)
    running = {tuple(k.split(".")): v.split(".")[0] for k, v in S._LEGACY.items()}
    assert mg.legacy_pairs(tabs) == running
    assert mg.legacy_pkgs(tabs) == set(S._LEGACY_PKGS)
    found, n, unparsed = mg.c2d(mg.REPO)
    assert n > 500 and unparsed == [] and found == []


def test_diff_refuses_two_censuses_of_different_modes(tmp_path, capsys):
    """A checkout census and an export census differ in `root` and the manifest's freshness line
    (measured at 09b68a4): `--diff` refuses a mixed pair, and a census with no mode."""
    files = {}
    for name, payload in (("declared", {}), ("expected", {"modules": {}})):
        files[name] = tmp_path / f"{name}.json"
        files[name].write_text(json.dumps(payload), encoding="utf-8")
    for pre_mode, post_mode, rc in (
        ("export", "export", 0),
        ("export", "checkout", 2),
        (None, "export", 2),
    ):
        for side, mode in (("pre", pre_mode), ("post", post_mode)):
            c = mg._mini()
            if mode:
                c["mode"] = mode
            files[side] = tmp_path / f"{side}.json"
            files[side].write_text(json.dumps(c), encoding="utf-8")
        argv = [
            "--diff",
            str(files["pre"]),
            str(files["post"]),
            "--declared",
            str(files["declared"]),
        ]
        assert mg.main([*argv, "--expected", str(files["expected"])]) == rc
    assert "REFUSED" in capsys.readouterr().out


def test_the_reds_name_modules_by_basename_through_the_tables():
    """S1d: each module a tree red plants is a registered basename whose package the red reads
    from the copy's tables, so no stage re-points a red."""
    for base in (mg.F5_BASE, mg.F2_BASE, mg.C2B_BASE, mg.C5_BASE, mg.FLAT_BASE, *mg.REPOINT):
        assert "." not in base and "/" not in base
        assert S.module_path(base).endswith("." + base)
    for base in mg.C2C_TARGETS:
        assert S.module_path(base)  # every C2(c) target is a registered module

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
    from snn_data_lake.gates import atlas_node_gate, region_charge_gate

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
    from snn_data_lake.gates import selfcheck_cache

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
    from snn_data_lake.gates import dormancy_gate, import_manifest

    assert "move_gate" in S.REGIONS["gates"]  # beside the gates M1 moved in
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
def test_legacy_layer_holds_exactly_the_moved_modules_and_the_provider_returns_it():
    """Each `_LEGACY` pair is a module whose live package differs from its frozen label (none
    before the first move, every moved module after); `_LEGACY_PKGS` stays empty until a package
    empties (M11a: `claustrum`, the first; M11b: seven more; M11c: two more; M11d: four more;
    M11e: `cortex`, the last)."""
    from snn_data_lake.readout import interaction_learning as ixn

    label = {
        m: k
        for t in (S.LEGACY_REGIONS, S.LEGACY_TORCH_TIER, S.LEGACY_TRAIN_TIER)
        for k, ms in t.items()
        for m in ms
    }
    moved = {m for m in label if S.package_of(m) not in (None, label[m])}
    assert S._LEGACY == {
        f"{label[m]}.{m}": S.module_path(m)[len("snn_data_lake.") :] for m in moved
    }
    assert (
        S._LEGACY_PKGS
        == frozenset(  # G8 M11a-M11e: 15 packages (1+7+2+4+1) emptied into faculties
            {
                "amygdala",
                "basal_ganglia",
                "cerebellum",
                "claustrum",
                "cortex",
                "epithalamus",
                "hippocampus",
                "hypothalamus",
                "insula",
                "medulla",
                "midbrain",
                "neuromodulation",
                "olfactory",
                "pons",
                "thalamus",
            }
        )
    )
    assert ixn._g0c_legacy_modules() is S._LEGACY  # the ONE table, read at call time (P2)
    # 286 -> 287 on 2026-10-02: gates.docs_only_battery joined REGIONS (its flat alias is derived)
    # 287 -> 286 on 2026-10-02: sensorimotor.spiking_lm_serve left REGIONS (retired)
    # 286 -> 292 on 2026-10-03: G1(c)'s six new instruments modules joined REGIONS
    # (carried_state_campaign, inseparable_mapping, carried_state_driver, carried_state_stages,
    # carried_state_controls, carried_state_report)
    # 306 -> 310 on 2026-10-10: the integrator's single re-pin at batch-3 integration, for the four
    # governed modules the G4/G7 ranges added (h01_distance_kernel, h01_partb_tangential,
    # unit_organization, wiring_prior). Measured, not assumed: origin/main imports 306 (and its pin
    # says 306), the stacked head imports 310, the set difference is exactly those four names and
    # nothing was removed. No row bumps this pin.
    # 310 -> 311 on 2026-10-11: the integrator's single re-pin at the G6-3 landing, for the
    # one governed module the G6-3 range added (sleep_mask_apply). Measured, not assumed:
    # origin/main imports 306, the batch-3 head imported 310, this head imports 311, and the
    # set difference is exactly that one name with nothing removed. _MOVED is a comprehension
    # over REGIONS taking every module whose name differs from its region, so sleep_mask_apply
    # under 'instruments' enters it. No row bumps this pin. G5-3's three Part D registrations
    # will require a further re-pin at their landing.
    # 311 -> 314 on 2026-10-11: G5 Part D adds exactly g5_part_d_adaptors,
    # g5_part_d_cells and g5_part_d_reading under instruments; no existing alias
    # is removed. This is the integrator's separately reviewed exact count pin.
    # 314 -> 315 on 2026-10-12: G6 (owner answer #1038) registers exactly one new module,
    # instruments/eligible_circuit_census, under 'instruments'; no existing alias is removed
    # and no name collides with a _LEGACY_PKGS package key, so it enters _MOVED. Measured, not
    # assumed: all six literal tables were re-read and _MOVED re-derived from its own
    # comprehension at __init__.py -- raw REGIONS members 315 -> 316, less the ONE unchanged
    # claustrum legacy-package shadow, gives _MOVED 314 -> 315. The claustrum clause is
    # unchanged and still holds.
    # 2026-10-13/G9: posed_problem adds one readout entry. Re-derived from
    # actual tables: REGIONS=318, less the claustrum package shadow -> 317.
    # 2026-10-13/COMPOSE: actual REGIONS=320, _MOVED=319 after the two G3
    # growth entries batch_reconciler/feed_readout; claustrum shadow unchanged.
    # 2026-10-16/R2: re-derived from the live tables against fcdfa7f6's parent:
    # REGIONS=321, less the unchanged claustrum shadow -> _MOVED=320.
    # Exactly instruments.replay_hydration was added by fcdfa7f6; nothing removed.
    # 2026-10-17/R0.2 (plan section 10, Rust-aware gates): measured by diffing _MOVED between
    # origin/main 06744945 (320) and claude/r02-gates-1017 (323): added exactly gates.rust_facts,
    # gates.native_dormancy, gates.native_ratchet; nothing removed; claustrum shadow unchanged.
    assert len(S._MOVED) == 323 and "claustrum" not in S._MOVED


@pytest.fixture
def planted_legacy(monkeypatch):
    """A legacy package and path planted in-process: `g8probe` stands for an emptied package whose
    module `region_charge_gate` now lives in `substrate` (nothing on disk changes)."""
    monkeypatch.setattr(S, "_LEGACY_PKGS", S._LEGACY_PKGS | {"g8probe"})
    monkeypatch.setitem(S._LEGACY, "g8probe.region_charge_gate", "gates.region_charge_gate")
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
    new = importlib.import_module("snn_data_lake.gates.region_charge_gate")
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
    stale = "snn_data_lake.faculties.gone"
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


def test_c2d_resolves_relative_imports_from_the_source_package(tmp_path, monkeypatch):
    source = tmp_path / "src" / "snn_data_lake" / "faculties" / "probe.py"
    source.parent.mkdir(parents=True)
    monkeypatch.setattr(mg, "live_py", lambda root: ["src/snn_data_lake/faculties/probe.py"])
    old_pkg, old_mod = "plasticity", "deep_stdp"
    pairs = {(old_pkg, old_mod): "growth"}
    source.write_text(f"from ..{old_pkg}.{old_mod} import main\n", encoding="utf-8")
    found, n, unparsed = mg.c2d(tmp_path, pairs, set())
    assert (n, unparsed) == (1, [])
    assert found == [("src/snn_data_lake/faculties/probe.py", 1, "import", f"{old_pkg}.{old_mod}")]

    source.write_text(f"from ..{old_pkg} import {old_mod}\n", encoding="utf-8")
    assert mg.c2d(tmp_path, pairs, set())[0] == found

    source.write_text(f"from ..growth import {old_mod}\n", encoding="utf-8")
    assert mg.c2d(tmp_path, pairs, set())[0] == []


def test_c2d_unparsed_file_is_a_cli_failure(tmp_path, monkeypatch):
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n", encoding="utf-8")
    monkeypatch.setattr(mg, "live_py", lambda root: ["broken.py"])
    assert mg.c2d(tmp_path, {}, set()) == ([], 1, ["broken.py"])
    monkeypatch.setattr(mg, "tables", lambda root: {})
    monkeypatch.setattr(mg, "legacy_pairs", lambda tabs: {})
    monkeypatch.setattr(mg, "legacy_pkgs", lambda tabs: set())
    assert mg.main(["--c2d", "--root", str(tmp_path)]) == 1


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


def test_c2c_ignores_brian_measurements_but_keeps_result_differences():
    first = (
        'r={"timeout_elapsed_s": 0.511, "wall_s": 3.31, '
        '"peak_working_set_bytes": 634777600, "requested_cells": 2000001}'
    )
    later = (
        'r={"timeout_elapsed_s": 0.527, "wall_s": 3.54, '
        '"peak_working_set_bytes": 634818560, "requested_cells": 2000001}'
    )
    changed = later.replace('"requested_cells": 2000001', '"requested_cells": 2000002')
    assert mg._c2c_line("brian2_backend", first, []) == mg._c2c_line("brian2_backend", later, [])
    assert mg._c2c_line("brian2_backend", first, []) != mg._c2c_line("brian2_backend", changed, [])
    assert mg._c2c_line("other_gate", first, []) != mg._c2c_line("other_gate", later, [])
    stderr_a = r'"stderr": "C:\\Temp\\brian_debug_nn_172eo.log\\nRuntimeError: GIL enabled"'
    stderr_b = r'"stderr": "C:\\Temp\\brian_debug_uv_9f13a.log\\nRuntimeError: GIL enabled"'
    assert mg._c2c_line("brian2_backend", stderr_a, []) == mg._c2c_line(
        "brian2_backend", stderr_b, []
    )
    assert mg._c2c_line("brian2_backend", stderr_a, []) != mg._c2c_line(
        "brian2_backend", stderr_b.replace("GIL enabled", "different error"), []
    )


def test_unordered_gate_report_rows_keep_every_element_visible():
    immortal_a = (
        "excluded, scratch-home deletions (cannot reach the durable lake): 2 "
        "[('a/x', 'f'), ('b/y', 'g')]"
    )
    immortal_b = (
        "excluded, scratch-home deletions (cannot reach the durable lake): 2 "
        "[('b/y', 'g'), ('a/x', 'f')]"
    )
    assert mg._gate_lines("immortal_retention", [immortal_a], {}) == mg._gate_lines(
        "immortal_retention", [immortal_b], {}
    )
    assert mg._gate_lines("immortal_retention", [immortal_a], {}) != mg._gate_lines(
        "immortal_retention", [immortal_b.replace("b/y", "b/z")], {}
    )
    method_a = "  UNCERTAIN x [name-matched in: a.x, b.y, c.z]"
    method_b = "  UNCERTAIN x [name-matched in: c.z, a.x, b.y]"
    assert mg._gate_lines("method_dormancy_gate", [method_a], {}) == mg._gate_lines(
        "method_dormancy_gate", [method_b], {}
    )
    assert mg._gate_lines("method_dormancy_gate", [method_a], {}) != mg._gate_lines(
        "method_dormancy_gate", [method_b.replace("c.z", "c.q")], {}
    )


def test_geometry_seed_pin_stays_on_frozen_label_after_move():
    from snn_data_lake.infra import brain_geometry

    assert S.module_path("rank_order_coding") == "snn_data_lake.sensorimotor.rank_order_coding"
    assert S._LABEL["rank_order_coding"] == "cortex"
    assert brain_geometry._SEED_PIN_KEY == f"{S._LABEL['rank_order_coding']}/rank_order_coding"
    assert brain_geometry.layout()["modules"][0]["key"] == brain_geometry._SEED_PIN_KEY

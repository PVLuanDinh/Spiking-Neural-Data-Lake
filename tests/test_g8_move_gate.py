"""G8 S1a and S1b (specs/022 evidence g8-reorganization.md 2.4-2.6): the frozen LEGACY_* label
tables, the package_of()/module_path() join the region-keyed gates read through, move_gate's diff
engine and its registration (S1a); the legacy alias layer, its strict tripwire, the re-pointed
checkpoint provider and C8/C9 by file (S1b). The tree-level reds (C1, F5, C2(b), F3(c), F2, C5) run
through `python -m snn_data_lake.gates.move_gate --self-check --mover MOVER`, per stage.
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

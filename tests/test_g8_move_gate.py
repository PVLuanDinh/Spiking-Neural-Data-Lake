"""G8 S1a (specs/022 evidence g8-reorganization.md 2.4, 2.6): the frozen LEGACY_* label tables,
the package_of()/module_path() join the region-keyed gates read through, move_gate's diff engine
and its registration. The tree-level reds (C1's planted EXEMPT-row deletion, F5's join red) run
through `python -m snn_data_lake.gates.move_gate --self-check --mover MOVER`, per stage.
"""

import hashlib
import importlib.util
import json
import os
import re

import pytest

import snn_data_lake as S
from snn_data_lake import brain_atlas
from snn_data_lake.gates import move_gate as mg

# The S0-head REGIONS / TORCH_TIER / TRAIN_TIER (393411b), digested: LEGACY_* are byte-identical
# copies of them and are never edited after S1a (C6). Re-derived at S1a from the S0 head's own
# tables through a git-archive export, and equal.
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
        text = f.read()
    assert not re.search(r"^\s*(from|import)\s+snn_data_lake", text, re.M), "stdlib only"

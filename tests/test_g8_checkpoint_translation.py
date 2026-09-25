"""G8 S0 / R-L0: a checkpoint whose class paths moved restores through the legacy map, or refuses.

specs/022-unified-brain-arc/evidence/g8-reorganization.md, 2.4 (S0) and 2.7 (R-L0). The G0(c)
codec stores each package object's class as "<module>.<qualname>" (`_encode_state`), and G8 moves
modules, so a checkpoint written before a move names its classes by their OLD module.
`interaction_learning._merge_state` maps a stored class path's MODULE through the one legacy map,
`_g0c_legacy_modules()` (empty until G8's S1b re-points it at the alias layer's table), before the
lazy-class lookup and before the class comparison. Read side only: the lake is never rewritten, the
schema stays g0c.v1, a save writes the CURRENT path, and a name the map cannot resolve refuses.

R-L0, the mechanism before anything moves. A real brain (`make_brain(seed=100, persist=True)`)
taught with the frozen `_g0c_schedule` and flushed writes the checkpoint in its own process and
temporary home. A second process rewrites the `__cls__` strings of the three classes G8 moves
(`SpikingInteractionLearner`, `RSTDPNeuron`, `LIFNeuron`) to planted legacy names and saves the
result back with the real `save_state`. Each restore arm then runs in its own fresh process over a
fresh COPY of that home, asserts `home()` is that copy before it constructs anything, and patches
`Brain._report_ixn_refusal` to RAISE, because the construction swallows a refused restore (F8).

PASS: with the planted map injected, the restore succeeds, the `_g0c_walk` digest equals the
writer's and the answers on the frozen G0 probe set are equal. The controls must each refuse: an
EMPTY map ("checkpoint class mismatch": the translation is what makes the PASS arm pass); an
unmapped unknown class (the planted map without `LIFNeuron`'s row, so its planted name is unknown);
a name mapped to a missing target (`LIFNeuron`'s row pointed at a module that does not import).
The last two differ from the PASS arm's map in that one row only, which attributes each refusal.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from snn_data_lake.basal_ganglia.r_stdp import RSTDPNeuron
from snn_data_lake.cortex import interaction_learning as ixn
from snn_data_lake.cortex.lif_neuron import LIFNeuron

pytestmark = pytest.mark.slow

_SRC = Path(__file__).resolve().parents[1] / "src"
_PKG = "snn_data_lake."
_LEGACY_ROOT = "r_l0_legacy"  # a planted package: no module lives there
_MISSING = "r_l0_missing.lif_neuron"  # a planted target that does not import


def _rel(cls):
    """A class's module relative to the package, the form the legacy map is keyed in."""
    return cls.__module__[len(_PKG) :]


# The three classes G8 moves (evidence 1.7), read from the live classes, so a moved class is
# followed rather than planted from a stale string: class -> its planted legacy module.
_MOVING = {
    cls: f"{_LEGACY_ROOT}.{_rel(cls)}"
    for cls in (ixn.SpikingInteractionLearner, RSTDPNeuron, LIFNeuron)
}
_PLANTED_MAP = {legacy: _rel(cls) for cls, legacy in _MOVING.items()}
_RENAMES = {
    f"{cls.__module__}.{cls.__qualname__}": f"{_PKG}{legacy}.{cls.__qualname__}"
    for cls, legacy in _MOVING.items()
}
_LIF = _MOVING[LIFNeuron]
_ARMS = {
    "pass": _PLANTED_MAP,
    "empty_map": {},
    "unmapped_unknown": {k: v for k, v in _PLANTED_MAP.items() if k != _LIF},
    "missing_target": {**_PLANTED_MAP, _LIF: _MISSING},
}

# Every child opens by proving it can only touch its own temporary home, before any construction.
_GUARD = r"""
import json, os, sys
home = os.path.realpath(sys.argv[1])
if os.path.realpath(os.environ.get("NORD_HOME", "")) != home:
    raise SystemExit("NORD_HOME is not this arm's temporary home")
from snn_data_lake.substrate.home import home as lake_home, load_state, save_state
if os.path.realpath(lake_home()) != home:
    raise SystemExit("home() resolves outside this arm's temporary home: " + lake_home())


def census(tree):
    out = {}

    def walk(node):
        if isinstance(node, dict):
            name = node.get("__cls__")
            if isinstance(name, str):
                out[name] = out.get(name, 0) + 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(tree)
    return out


def emit(result):
    print("R_L0 " + json.dumps(result), flush=True)
"""

_BRAIN = r"""
import hashlib
from unittest.mock import patch
from snn_data_lake.cortex import interaction_learning as ixn
from snn_data_lake.cortex.brain import Brain, make_brain


class Refused(Exception):
    pass


def refuse(what, exc):
    raise Refused(f"{what}: {type(exc).__name__}: {exc}") from exc


def digest(learner):
    walk = json.dumps(ixn._g0c_walk(learner), sort_keys=True, ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(walk.encode("ascii")).hexdigest()


def answers(learner):
    return [learner.route_text(u) for u, _ in ixn._conv_flat(ixn._CONV_TEST)]
"""

_WRITER = (
    _GUARD
    + _BRAIN
    + r"""
with patch.object(Brain, "_report_ixn_refusal", staticmethod(refuse)):
    brain = make_brain(seed=100, persist=True)
    try:
        learner = brain.brain._ixn
        fresh = digest(learner)
        ixn._g0c_schedule(brain, 100)
        flushed = brain._flush_interaction_learner()
        taught, probe = digest(learner), answers(learner)
        result = {
            "fresh_digest": fresh,
            "digest": taught,
            "answers": probe,
            "flushed": flushed,
            "probes_pure": digest(learner) == taught,
            "checkpoint_is_live": load_state("interaction_learner.json") == learner.state_dict(),
        }
    finally:
        brain.close()
emit(result)
"""
)

_PLANT = (
    _GUARD
    + r"""
renames = json.loads(sys.argv[2])
checkpoint = load_state("interaction_learner.json")
before = census(checkpoint)


def plant(node):
    if isinstance(node, dict):
        if isinstance(node.get("__cls__"), str) and node["__cls__"] in renames:
            node["__cls__"] = renames[node["__cls__"]]
        for value in node.values():
            plant(value)
    elif isinstance(node, list):
        for value in node:
            plant(value)


plant(checkpoint)
emit(
    {
        "before": before,
        "after": census(checkpoint),
        "schema": checkpoint["schema"],
        "written": save_state("interaction_learner.json", checkpoint),
        "read_back": load_state("interaction_learner.json") == checkpoint,
    }
)
"""
)

_RESTORE = (
    _GUARD
    + _BRAIN
    + r"""
legacy = json.loads(sys.argv[2])
stored = json.dumps(load_state("interaction_learner.json"), sort_keys=True)
result = {"refused": None}
with (
    patch.object(ixn, "_g0c_legacy_modules", lambda: dict(legacy)),
    patch.object(Brain, "_report_ixn_refusal", staticmethod(refuse)),
):
    try:
        brain = make_brain(seed=100, persist=True)
    except Refused as exc:
        result["refused"] = str(exc)
    else:
        try:
            learner = brain.brain._ixn
            result["digest"], result["answers"] = digest(learner), answers(learner)
            now = json.dumps(load_state("interaction_learner.json"), sort_keys=True)
            result["lake_unchanged_by_restore"] = now == stored
        finally:
            brain.close()
        saved = load_state("interaction_learner.json")
        result["saved_schema"] = saved["schema"]
        result["saved_census"] = census(saved)
emit(result)
"""
)


def _run(script, home, *args):
    env = dict(os.environ, PYTHONUTF8="1", NORD_HOME=str(home))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(_SRC), env.get("PYTHONPATH", "")) if p)
    proc = subprocess.run(
        [sys.executable, "-c", script, str(home), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("R_L0 "):
            return json.loads(line[len("R_L0 ") :])
    raise AssertionError(f"no R-L0 reading (rc={proc.returncode})\n{proc.stderr[-4000:]}")


@pytest.fixture(scope="module")
def r_l0(tmp_path_factory):
    """The writer, the plant, then every restore arm, each in its own fresh process."""
    root = tmp_path_factory.mktemp("r_l0")
    written = root / "written"
    written.mkdir()
    writer = _run(_WRITER, written)
    plant = _run(_PLANT, written, json.dumps(_RENAMES))
    arms = {}
    for arm, legacy in _ARMS.items():
        home = root / f"arm_{arm}"
        shutil.copytree(written, home)  # a fresh COPY: a restore then a flush writes its home
        arms[arm] = _run(_RESTORE, home, json.dumps(legacy))
    print("R-L0 readings " + json.dumps({"writer": writer, "plant": plant, "arms": arms}))
    return writer, plant, arms


def test_the_writer_and_the_plant_are_not_vacuous(r_l0):
    writer, plant, _ = r_l0
    # The lake's newest checkpoint IS the taught learner, read back. (`flushed` is only a reading:
    # False means unchanged, because the schedule's last completed night already saved this state.)
    assert writer["checkpoint_is_live"] is True and writer["probes_pure"] is True
    assert writer["digest"] != writer["fresh_digest"]  # the schedule taught the learner
    for current, planted in _RENAMES.items():
        assert plant["before"].get(current, 0) >= 1, current
        assert current not in plant["after"], current
        assert plant["after"][planted] == plant["before"][current], planted
    assert plant["schema"] == "g0c.v1"
    assert plant["written"] is True and plant["read_back"] is True


def test_r_l0_planted_legacy_names_restore_through_the_injected_map(r_l0):
    writer, _, arms = r_l0
    arm = arms["pass"]
    assert arm["refused"] is None
    assert arm["digest"] == writer["digest"]
    assert arm["answers"] == writer["answers"]


@pytest.mark.parametrize(
    ("arm", "message"),
    [
        ("empty_map", "checkpoint class mismatch"),
        ("unmapped_unknown", "checkpoint class mismatch"),
        ("missing_target", f"{_PKG}{_MISSING} does not import"),
    ],
)
def test_r_l0_controls_refuse(r_l0, arm, message):
    refused = r_l0[2][arm]["refused"]
    assert refused is not None, arm
    assert refused.startswith("restore refused; learner stays fresh: ValueError: "), refused
    assert message in refused, refused


def test_s0_is_read_side_only(r_l0):
    _, plant, arms = r_l0
    arm = arms["pass"]
    assert arm["lake_unchanged_by_restore"] is True
    assert arm["saved_schema"] == "g0c.v1"
    assert arm["saved_census"] == plant["before"]  # the save writes the CURRENT class paths


class _Nested:
    """Planted as a nested class for the dotted-qualname case; src/ holds none today."""


def test_the_provider_is_empty_at_s0():
    assert ixn._g0c_legacy_modules() == {}


def test_translation_maps_the_longest_legacy_module_and_keeps_a_dotted_qualname(monkeypatch):
    lif = f"{LIFNeuron.__module__}.LIFNeuron"
    assert ixn._g0c_translate(lif, {}) == lif
    assert ixn._g0c_translate(lif, {_rel(LIFNeuron)[:-1]: _MISSING}) == lif  # "." bounds a module
    assert ixn._g0c_translate("builtins.object", {"builtins": _MISSING}) == "builtins.object"
    legacy = {_LEGACY_ROOT: _MISSING, _LIF: _rel(LIFNeuron)}
    assert ixn._g0c_translate(f"{_PKG}{_LIF}.LIFNeuron", legacy) == lif
    monkeypatch.setattr(ixn.SpikingInteractionLearner, "Nested", _Nested, raising=False)
    old = f"{_PKG}{_LEGACY_ROOT}.il.SpikingInteractionLearner.Nested"
    new = f"{ixn.__name__}.SpikingInteractionLearner.Nested"
    legacy = {f"{_LEGACY_ROOT}.il": _rel(ixn.SpikingInteractionLearner)}
    assert ixn._g0c_translate(old, legacy) == new


def test_translation_refuses_a_target_that_does_not_resolve():
    legacy = {f"{_LEGACY_ROOT}.il": _rel(ixn.SpikingInteractionLearner)}
    with pytest.raises(ValueError, match="NoSuchClass"):
        ixn._g0c_translate(f"{_PKG}{_LEGACY_ROOT}.il.NoSuchClass", legacy)
    with pytest.raises(ValueError, match="does not import"):
        ixn._g0c_translate(
            f"{_PKG}{_LEGACY_ROOT}.il.SpikingInteractionLearner", {f"{_LEGACY_ROOT}.il": _MISSING}
        )

"""specs/022 G3-T1: `Grower` + `PerceptProblem` wrap the percept engine (B1) byte-identically.

Step 1 of a merge, not a rewrite (lane brief T1-BRIEF.md, amended by G3-BUILD.BRIEF.md): the
Grower is on the live path and nothing is deleted. Each test is a red the wrapper can fail:
R1 value drift, R3 admission path severed, R4 a captured bound method, R5 the `Brain.neuro`
alias, R6 a dormant Grower. R2 (the T0 five-arm comparison) waits for g3-t0's baseline.
The pins are imported from tests/test_candidate_lifecycle.py, never re-measured. CPU, 1 thread.
"""

import json
import types
import zlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_candidate_lifecycle import ONLINE, STREAM, TRAJ  # noqa: E402

from snn_data_lake.growth import candidate_lifecycle  # noqa: E402
from snn_data_lake.growth.grower import Grower, PerceptProblem  # noqa: E402
from snn_data_lake.growth.neurogenesis_engine import (  # noqa: E402
    NeurogenesisSelfImprove,
    SyntheticPerceptStream,
    growth_fingerprint,
)
from snn_data_lake.substrate.home import throwaway_home  # noqa: E402


def _traj(engine):
    return zlib.crc32(json.dumps(engine.trajectory, default=str).encode())


def _grower(seed, **kw):
    torch.set_num_threads(1)
    return Grower(PerceptProblem(NeurogenesisSelfImprove(seed=seed, device="cpu", **kw)))


def _drive(seed, ticks):
    g = _grower(seed)
    SyntheticPerceptStream(seed=seed).drive(g, ticks)
    return g


@pytest.mark.parametrize("seed,ticks", sorted(STREAM))
def test_percept_via_grower_is_byte_identical(seed, ticks):
    eng = _drive(seed, ticks).percept.engine
    assert (growth_fingerprint(eng), int(eng.grown_units)) == STREAM[(seed, ticks)]
    assert _traj(eng) == TRAJ[(seed, ticks)]


@pytest.mark.parametrize("seed,name", sorted(ONLINE))
def test_online_engines_via_grower_are_byte_identical(seed, name):
    from snn_data_lake.readout.online_readout import ClosingGate, OnlineReadout, _chain, _feed

    torch.set_num_threads(1)
    readout = None if name == "batch" else OnlineReadout("delta")
    gate = None if name == "delta_open" else ClosingGate(3)
    g = Grower(
        PerceptProblem(
            NeurogenesisSelfImprove(
                seed=seed, device="cpu", refit_every=1, online_readout=readout, growth_gate=gate
            )
        )
    )
    _feed(g, _chain(seed, 600, flip_every=100))
    eng = g.percept.engine
    assert (growth_fingerprint(eng), _traj(eng), int(eng.grown_units)) == ONLINE[(seed, name)]


_ROOT = Path(__file__).resolve().parents[1]
_T0_JSON = _ROOT / "specs/022-unified-brain-arc/evidence/g3-t0-baseline.json"
_CENSUS = ("total", "reachable", "exempt", "dormant")
_KEY = "snap.derived.coverage_{}"
_needs_t0 = pytest.mark.skipif(
    not _T0_JSON.exists(),
    reason="R2 needs g3-t0's committed g3-t0-baseline.json (thread G3-T0B); absent in this tree",
)


def _git(root, *args):
    import subprocess

    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _t0_base(root=_ROOT, rel=None):
    """The LAST commit touching T0's capture in HEAD's history (an in-place re-capture moves it;
    closing review A-2). The capture's `base_commit` names the lane commit it ran on, which a
    restack leaves off every pushed ref (re-panel A-M1: 78af459c is in no clone); this one is in
    any full clone's lineage, and its src tree equals the captured one (T0 adds no src module;
    git diff 78af459c 4bc2a726 -- src is empty). A shallow clone refuses: its boundary commit
    reads as touching every file (closing review A-1)."""
    rel = rel or _T0_JSON.relative_to(_ROOT).as_posix()
    assert _git(root, "rev-parse", "--is-shallow-repository") == "false", (
        "R2's census anchor needs full history (shallow clone)"
    )
    sha = _git(root, "log", "--format=%H", "-1", "--", rel)
    assert sha, f"no commit touches {rel} in HEAD's history"
    return sha


def test_t0_anchor_is_the_last_capture_commit_and_refuses_a_shallow_clone(tmp_path):
    """_t0_base on a scratch repo: a re-capture moves the anchor; a shallow clone refuses."""
    repo, rel = tmp_path / "r", "cap.json"
    repo.mkdir()
    _git(repo, "init", "-q")
    for body in ("1", "2"):
        (repo / rel).write_text(body, encoding="utf-8")
        _git(repo, "add", rel)
        _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", body)
    assert _t0_base(repo, rel) == _git(repo, "rev-parse", "HEAD")
    shallow = tmp_path / "s"
    _git(tmp_path, "clone", "-q", "--depth", "1", repo.as_uri(), str(shallow))
    with pytest.raises(AssertionError, match="shallow"):
        _t0_base(shallow, rel)


def _census_delta(base):
    """How far the module census ("structural census of modules on disk") must move from T0's
    capture: each brain module ADDED since `base` (git), classified as live_coverage() classifies
    it. Derived, never a literal. A deleted module or a reclassified old one is not derived, so it
    reds the comparison."""
    import subprocess

    from snn_data_lake.gates import dormancy_gate as dg

    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=A", base, "--", "src/snn_data_lake"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    added = {dg._mpath(str(_ROOT / p)) for p in out} & dg.BRAIN_MODULES
    reach = dg.reachable()
    exempt = {dg._CANON[m] for m in dg.EXEMPT if m in dg._CANON}
    return {
        "total": len(added),
        "reachable": len(added & reach),
        "exempt": len(added & exempt),
        "dormant": len(added - reach - exempt),
    }


def _census_of(view):
    return {k: view[_KEY.format(k)] for k in _CENSUS}


def _census_drift(t0, new, delta):
    """{key: (T0, derived delta, new)} for each census key `new` does not reach from `t0`."""
    return {k: (t0[k], delta[k], new[k]) for k in _CENSUS if new[k] != t0[k] + delta[k]}


def _t0():
    return json.loads(_T0_JSON.read_text(encoding="utf-8"))


def _live_drift():
    from snn_data_lake.infra.region_wiring import live_coverage

    old = _t0()
    return _census_drift(
        _census_of(old["arms"][0]["view"]), live_coverage(), _census_delta(_t0_base())
    )


@_needs_t0
def test_live_census_moves_from_t0_by_the_derived_delta():
    assert not _live_drift()


@_needs_t0
def test_registered_module_outside_the_derivation_reds_the_census(monkeypatch):
    from snn_data_lake.gates import dormancy_gate as dg

    monkeypatch.setattr(dg, "BRAIN_MODULES", dg.BRAIN_MODULES | {"snn_data_lake.planted_module"})
    assert set(_live_drift()) == {"total", "dormant"}


@_needs_t0
def test_coverage_drift_without_a_registry_change_reds_the_census(monkeypatch):
    from snn_data_lake.gates import dormancy_gate as dg

    real = dg.reachable
    monkeypatch.setattr(dg, "reachable", lambda: real() - {"snn_data_lake.engine.brain"})
    assert set(_live_drift()) == {"reachable", "dormant"}


@pytest.mark.slow
@_needs_t0
def test_brain_five_arm_matches_t0(tmp_path):
    """Re-run T0's driver on the wrapped tree; every arm equals the committed capture, with only
    T0's own wall-clock key (snap.t) and wall_s excluded. The red arms must still MOVE the same
    keys as in T0 (equal diff lists, not merely non-empty)."""
    from tests import test_g3_t0_baseline as t0

    old = _t0()
    new = t0.capture(tmp_path / "t1.json")
    assert [a["perception_tier"] for a in new["arms"]] == [
        a["perception_tier"] for a in old["arms"]
    ]
    census = tuple(_KEY.format(k) for k in _CENSUS)
    delta = _census_delta(_t0_base())
    bad = {}
    for o, n in zip(old["arms"], new["arms"], strict=True):
        assert (o["seed"], o["arm"]) == (n["seed"], n["arm"])
        assert not _census_drift(_census_of(o["view"]), _census_of(n["view"]), delta)
        drift = [k for k in t0._diff(n["view"], o["view"], t0.WALL_CLOCK) if k not in census]
        rest = {k: (o[k], n[k]) for k in o if k not in ("view", "wall_s") and o[k] != n[k]}
        if drift or rest:
            bad[(o["seed"], o["arm"])] = (drift, rest)
    assert not bad, bad
    v_old, v_new = t0.verdicts(old), t0.verdicts(new)
    assert v_old == v_new


def test_reversed_admission_tie_break_changes_grower_fingerprint(monkeypatch):
    original = candidate_lifecycle.select_block

    def reversed_order(*args, **kwargs):
        kwargs["reverse"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate_lifecycle, "select_block", reversed_order)
    changed = [
        s
        for s in (9000, 9001, 9002, 9003)
        if growth_fingerprint(_drive(s, 200).percept.engine) != STREAM[(s, 200)][0]
    ]
    assert changed, "the Grower's fingerprint did not move: admission is severed from the wrapper"


def test_grower_tick_honours_instance_patch():
    g = _grower(9000)
    stream = SyntheticPerceptStream(seed=9000)
    for _ in range(40):
        g.observe(*stream.next())
    g.percept.engine.tick = lambda: None  # the growth_off arm, patched AFTER construction
    for _ in range(50):
        g.tick()
    assert g.percept.engine.ticks == 0 and g.percept.engine.grown_units == 0


def test_grower_observe_honours_instance_patch():
    g = _grower(9000)
    seen = []
    g.percept.engine.observe = lambda f, c: seen.append(c)
    g.observe([0.0], 3)
    assert seen == [3]


def _brain():
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=0, persist=False)
    if b.neuro is None:
        pytest.skip("neurogenesis tier absent (torch)")
    return b


def test_brain_neuro_is_the_grower_engine():
    torch.set_num_threads(1)
    with throwaway_home():
        b = _brain()
        try:
            assert b.neuro is b.grower.percept.engine
            b.neuro.seed = 1
            assert b.grower.percept.engine.seed == 1
        finally:
            b.close()


def test_brain_without_torch_has_no_grower(monkeypatch):
    from snn_data_lake.engine.brain import Brain, make_brain

    monkeypatch.setattr(Brain, "_make_neuro", lambda self, seed, device: None)
    with throwaway_home():
        b = make_brain(seed=0, persist=False)
        try:
            assert b.neuro is None and b.grower is None
            b.tick()
        finally:
            b.close()


def test_neuro_property_is_none_before_grower_is_bound():
    from snn_data_lake.engine.brain import Brain

    assert Brain.neuro.fget(types.SimpleNamespace()) is None


@pytest.mark.slow
def test_brain_tick_drives_the_grower(monkeypatch):
    torch.set_num_threads(1)
    calls = {"observe": 0, "tick": 0}
    real_o, real_t = Grower.observe, Grower.tick

    def observe(self, *a, **k):
        calls["observe"] += 1
        return real_o(self, *a, **k)

    def tick(self, *a, **k):
        calls["tick"] += 1
        return real_t(self, *a, **k)

    monkeypatch.setattr(Grower, "observe", observe)
    monkeypatch.setattr(Grower, "tick", tick)
    with throwaway_home():
        b = _brain()
        try:
            for _ in range(300):
                b.tick()
            assert calls["observe"] > 0 and calls["tick"] > 0
            grew = b.state()["grown_units"]
        finally:
            b.close()
    # the growth_off control, expressed on the Grower: a neutered Grower.tick grows nothing
    monkeypatch.setattr(Grower, "tick", lambda self, *a, **k: None)
    with throwaway_home():
        b = _brain()
        try:
            for _ in range(300):
                b.tick()
            assert grew > 0 and b.state()["grown_units"] == 0
        finally:
            b.close()

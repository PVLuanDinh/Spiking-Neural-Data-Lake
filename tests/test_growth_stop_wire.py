"""specs/022 T047: the growth gate is consulted at the REAL refit, a closed gate stops
GROWTH without stopping MEASUREMENT, and the gate's evidence SURVIVES a restart.

The wire is NeurogenesisSelfImprove(growth_gate=...) -> _refit consults the ONE-ARG
callable with the engine's own trajectory (the system under test reaches the criterion
through its parameter); Brain._make_neuro passes Brain._growth_gate_verdict. These tests
pin (1) a closed gate freezes the unit count while the trajectory keeps appending (the
evidence stream the gate re-opens on must keep flowing), (2) an open gate is
behaviourally identical to no gate (A/A -- the incumbent fixed schedule IS the control
arm), (3) the brain constructs its engine with the gate attached and surfaces verdicts
on state(), and (4) the trajectory tail rides growth_state_bytes so a relaunched daemon
resumes the same verdict evidence (the panel's P4b find: the substrate persisted while
the evidence governing its growth did not)."""

import pytest

torch = pytest.importorskip("torch")

from snn_data_lake.cortex.neurogenesis_engine import NeurogenesisSelfImprove  # noqa: E402


def _run(engine, ticks=200):
    for _ in range(ticks):
        engine.tick()
    return engine


def test_closed_gate_stops_growth_not_measurement():
    calls = {"n": 0}

    def gate(trajectory):
        calls["n"] += 1
        return False  # closed from the first consult

    eng = _run(NeurogenesisSelfImprove(seed=7, refit_every=20, device="cpu", growth_gate=gate))
    # the FIRST refit grows (an engine that has never grown ignores a closed gate: the
    # criterion has no evidence about a substrate that does not exist); every later
    # refit is measure-only, so units freeze at the first refit's count.
    assert eng.trajectory, "no refits ran"
    units = [u for _t, u, _a in eng.trajectory]
    assert units[0] > 0, "the first refit must grow (no substrate = nothing to gate)"
    assert all(u == units[0] for u in units[1:]), (
        f"a closed gate must freeze growth at the first refit's units: {units}"
    )
    assert len(eng.trajectory) >= 5, (
        f"measurement stopped with the gate: only {len(eng.trajectory)} trajectory rows "
        "-- a gate that stops the evidence stream welds itself shut"
    )
    assert calls["n"] >= len(eng.trajectory) - 1, "the gate was not consulted per refit"


def test_open_gate_is_identical_to_no_gate():
    """A/A: gate returning True must not perturb the incumbent behaviour in any way --
    same units, same accs, same trajectory ticks."""
    a = _run(NeurogenesisSelfImprove(seed=7, refit_every=20, device="cpu"))
    b = _run(
        NeurogenesisSelfImprove(seed=7, refit_every=20, device="cpu", growth_gate=lambda _t: True)
    )
    assert a.trajectory == b.trajectory, "an open gate changed the trajectory (A/A must be exact)"
    assert a.grown_units == b.grown_units


def test_trajectory_tail_survives_the_state_blob():
    """P4b: the gate's evidence rides growth_state_bytes -- a fresh engine restoring the
    blob resumes with the SAME trailing trajectory, so the growth gate is not amnesiac
    across the daemon's routine relaunches (pre-fix: restore left trajectory == [],
    resetting the gate to its 9-refit insufficient-history horizon every restart)."""
    eng = _run(NeurogenesisSelfImprove(seed=7, refit_every=20, device="cpu"), ticks=400)
    assert eng.trajectory and eng.substrate_is_retained()
    blob = eng.growth_state_bytes()
    assert blob is not None
    fresh = NeurogenesisSelfImprove(seed=7, refit_every=20, device="cpu")
    assert fresh.trajectory == []
    assert fresh.restore_growth_state(blob)
    tail = eng._traj_tail(eng.trajectory)
    assert fresh.trajectory == tail, "the restored trajectory tail differs from what was persisted"
    # and a PRE-T047 blob (no trajectory arrays) still restores, with an empty trajectory
    import io

    import numpy as np

    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        stripped = {k: z[k] for k in z.files if not k.startswith("traj_")}
    buf = io.BytesIO()
    np.savez(buf, **stripped)
    older = NeurogenesisSelfImprove(seed=7, refit_every=20, device="cpu")
    assert older.restore_growth_state(buf.getvalue())
    assert older.trajectory == [], (
        "a pre-T047 blob must restore exactly as before (empty trajectory)"
    )


def test_brain_engine_carries_the_gate_and_surfaces_verdicts():
    """The anatomy pin: Brain._make_neuro constructs the engine with the brain's verdict
    callable attached -- the wire exists at birth -- and a verdict reached through the
    brain's own callable lands on state()['growth_gate'] (P4c: the cockpit renders what
    the brain measures)."""
    from snn_data_lake.cortex.brain import make_brain

    b = make_brain(seed=0)
    if b.neuro is None:
        pytest.skip("torch tier absent in this brain build")
    assert getattr(b.neuro, "_growth_gate", None) is not None, "the engine has no growth gate"
    assert b.neuro._growth_gate.__func__ is type(b)._growth_gate_verdict, (
        "the engine's gate is not the brain's verdict callable"
    )
    # drive the verdict through the brain's own callable with a real-shaped trajectory
    assert b._growth_gate_verdict([(20, 3, 0.5), (40, 4, 0.55)]) is True
    gg = b.state()["growth_gate"]
    assert "cortex" in gg and gg["cortex"]["grow"] is True and gg["cortex"]["reason"]
    # a poisoned row is COUNTED and skipped, never raised through the tick path
    before = b._growth_feed_refusals
    assert b._growth_gate_verdict([(20, 3, 0.5), (40, 4, 0.55), (60, 5, float("nan"))]) is True
    assert b._growth_feed_refusals == before + 1
    assert b.state()["growth_gate"].get("feed_refusals") == before + 1

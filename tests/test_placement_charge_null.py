"""Placement is WRITE-ONLY on the live path at HEAD -- a pinned null (specs/022 audit s8 rank 1).

`cortex/placement.place` gives every grown unit a (position, layer, sign). This test pins the
measured fact that nothing ON A LIVE PATH reads that record: a brain whose placement is replaced
by constant tuples is value-identical to a live one on every key of `state()` and the telemetry
snapshot but the wall clock, over enough ticks for the neurogenesis engine to refit and grow. Two
positive controls must move the same differencer -- growth switched off (a different stage) and
the engine's seed shifted (the SAME stage as placement, inside `_refit`) -- so a green here is a
real null and not a blind instrument. The one built reader of the record,
`cortex/genomic_bottleneck` (FR-033: unit tags -> a grown unit's initial fan-in), is
dormancy-EXEMPT `wire` and unwired; the growth blob and its restore path are OUT OF THIS VIEW
(placement is serialized there by design, and read back only to rebuild the record itself).

WHY THIS IS PINNED RATHER THAN FIXED. The composed growth->placement wire
(`cortex/placement.compose_growers` driven from the sleep tick, giving the hippocampal recruits
of `hippocampus/growing_learner` a placement) was BUILT and MEASURED the same way: 0 differing
keys beyond the nondeterminism baseline, i.e. it charged nothing, so P3 ("a wire that does not
charge is reverted") reverted it. The record is `specs/022-unified-brain-arc/evidence/
s1-placement-null.md`. The day this test goes RED -- placement has gained a live reader -- retire
the DISCLOSURE line in `substrate/growth_coverage_gate.report` and reopen the wire with evidence.

The nondeterminism baseline is measured in the same run (two live brains) and pinned by IDENTITY
(`snap.t`, the wall clock), never by a count.
"""

from __future__ import annotations

import pytest

from snn_data_lake.substrate.home import throwaway_home

TICKS = 300  # >= 1 engine refit at the default cadence (measured: grown_units > 0 at seeds 0-2)


def _flatten(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(_flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def _run(arm):
    """One brain per arm under a throwaway home (the real lake and the daemon are never
    touched); returns the flattened state()+telemetry view, the engine's unit count and its
    layer census."""
    import snn_data_lake.cortex.placement as placement
    from snn_data_lake.cortex.brain import make_brain
    from snn_data_lake.substrate import brain_telemetry

    real_place = placement.place
    try:
        if arm == "neuter_place":

            def constant_place(n_units, seed=0, region="cortex"):
                return [(0.5, "Layer 2", 1)] * n_units

            placement.place = constant_place
        with throwaway_home():
            b = make_brain(seed=0, persist=False)
            if b.neuro is None:
                pytest.skip("neurogenesis tier absent (torch)")
            if arm == "growth_off":
                b.neuro.tick = lambda: None
            if arm == "engine_seed_shift":
                b.neuro.seed = 1  # the fit seed AND the placement seed, inside _refit
            for _ in range(TICKS):
                b.tick()
            view = {
                **_flatten(b.state(), "state."),
                **_flatten(brain_telemetry.snapshot(b), "snap."),
            }
            grown = b.neuro.grown_units
            layers = b.neuro.layer_census()
            try:
                b.close()
            except Exception:
                pass
            return view, grown, layers
    finally:
        placement.place = real_place


def _diff(a, b, baseline):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k not in baseline)


def test_placement_is_write_only_on_the_live_path():
    live_a, grown_a, layers_a = _run("live")
    live_b, _, _ = _run("live")
    baseline = set(_diff(live_a, live_b, set()))
    assert grown_a > 0, f"the engine grew nothing in {TICKS} ticks -- nothing to differ on"
    assert baseline <= {"snap.t"}, (
        f"the nondeterminism baseline is no longer just the wall clock: {sorted(baseline)}"
    )

    # POSITIVE CONTROL 1 (a different stage): growth switched off must move the differencer.
    off, grown_off, _ = _run("growth_off")
    moved = _diff(live_a, off, baseline)
    assert grown_off == 0 and "state.grown_units" in moved, (
        f"the differencer is blind: growth_off moved {moved} (grown {grown_off})"
    )

    # POSITIVE CONTROL 2 (the SAME stage): the engine's seed shifted inside _refit -- where
    # placement is drawn -- must move held-out accuracy, so a placement-only null is read
    # against a control that discriminates AT that stage.
    shifted, _, _ = _run("engine_seed_shift")
    assert "state.predict_accuracy" in _diff(live_a, shifted, baseline), (
        "the differencer cannot see a same-stage change (engine seed shift left accuracy equal)"
    )

    # THE NULL: placement replaced by constant tuples changes NO downstream key.
    neutered, _, layers_n = _run("neuter_place")
    assert layers_n != layers_a, "the neuter did not reach the engine (layer census unchanged)"
    inert = _diff(live_a, neutered, baseline)
    assert inert == [], (
        "PLACEMENT HAS GAINED A LIVE READER: these keys now depend on it -> "
        f"{inert}. Retire the DISCLOSURE line in substrate/growth_coverage_gate.report, reopen "
        "the growth->placement wire (evidence/s1-placement-null.md) and move the growth-coverage "
        "pin with a measured charge beside it."
    )

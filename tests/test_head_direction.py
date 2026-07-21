"""Tests for HeadDirection + GoalVector (hippocampus/head_direction.py): a spiking LIF
head-direction ring attractor that integrates angular velocity with a landmark drift reset,
plus goal-vector cells that population-code the direction and distance to a remembered goal.

Includes the honest FAILING controls (drift without a landmark, out-running the bump, bearing
undefined at the goal, distance silent beyond range) -- these assert that the limits are REAL,
not tuned away.

Run: pytest tests/test_head_direction.py
"""

import math

import numpy as np
import pytest

from snn_data_lake.hippocampus.head_direction import GoalVector, HeadDirection, _wrap_pi

# --------------------------------------------------------------------------------------
# HeadDirection -- the spiking ring attractor
# --------------------------------------------------------------------------------------


def _settled(n=36, seed=0, heading=0.0, settle=30):
    hd = HeadDirection(n=n, seed=seed)
    hd.reset(heading)
    for _ in range(settle):
        hd.step(0.0)
    return hd


def test_bump_persists_as_one_localized_attractor():
    hd = _settled(settle=60)
    assert abs(math.degrees(_wrap_pi(hd.decode()))) < 2.0, (
        "bump must hold its heading with no input"
    )
    assert hd.bump_width() < 60.0, "activity must stay one localized bump, not spread over the ring"
    assert hd.bump_amplitude() > 0.5, "the spiking bump must stay alive"
    assert hd.certainty() > 0.5, "a localized bump implies a concentrated population vector"


def test_heading_integrates_a_known_angular_velocity_sequence():
    hd = _settled()
    start = hd.decode()
    seq = [0.04] * 20 + [-0.03] * 10 + [0.05] * 12
    target = start
    for av in seq:
        hd.step(av)
        target += av
    err = abs(math.degrees(_wrap_pi(hd.decode() - target)))
    assert err < 5.0, (
        f"decoded heading must track the angular-velocity integral (err {err:.2f} deg)"
    )


def test_full_turn_returns_to_start():
    hd = _settled()
    start = hd.decode()
    for _ in range(126):
        hd.step(2.0 * math.pi / 126.0)
    err = abs(math.degrees(_wrap_pi(hd.decode() - start)))
    assert err < 5.0, f"a full 360 turn must come back to the start (err {err:.2f} deg)"


def test_negative_angular_velocity_turns_the_other_way():
    hd = _settled()
    start = hd.decode()
    for _ in range(20):
        hd.step(-0.04)
    assert _wrap_pi(hd.decode() - start) < 0.0, "negative angular velocity must decrease heading"


def test_reset_clamps_the_bump_exactly():
    hd = HeadDirection(n=36, seed=0)
    for target in (0.0, 1.0, -2.0, math.pi / 3):
        hd.reset(target)
        assert abs(_wrap_pi(hd.decode() - target)) < 1e-6, "reset must clamp the bump exactly"


def test_biased_velocity_drifts_without_a_landmark_and_the_landmark_corrects_it():
    """The honest control: the ring CANNOT detect its own drift. Biased/noisy vestibular input
    makes the integral wander, and only a supplied landmark fixes it."""
    rng = np.random.default_rng(0)
    hd = _settled()
    true_h = hd.decode()
    for _ in range(120):
        av_true = 0.03
        hd.step(av_true * 1.25 + float(rng.normal(0.0, 0.006)))  # biased + noisy
        true_h += av_true
    drift_before = abs(math.degrees(_wrap_pi(hd.decode() - true_h)))
    # CONTROL MUST FAIL: no landmark => real, large drift
    assert drift_before > 20.0, (
        f"control is not failing: biased input should drift badly, got {drift_before:.1f} deg"
    )

    for _ in range(6):
        hd.landmark(true_h, strength=0.5)
        hd.step(0.0)
    drift_after = abs(math.degrees(_wrap_pi(hd.decode() - true_h)))
    assert drift_after < 5.0, f"the landmark must correct the drift (got {drift_after:.1f} deg)"
    assert drift_after < drift_before / 4.0, "the landmark must cut drift by at least 4x"


def test_landmark_strength_zero_is_a_no_op():
    hd = _settled(heading=0.5)
    before = hd.decode()
    hd.landmark(2.5, strength=0.0)
    assert abs(_wrap_pi(hd.decode() - before)) < 1e-9, "strength=0 must not move the bump"


def test_fast_turning_out_runs_the_bump_and_integrates_sub_linearly():
    """Honest CEILING, kept failing: past the calibrated linear regime the bump cannot keep up."""

    def shortfall(av, steps=25):
        hd = _settled()
        start = hd.decode()
        for _ in range(steps):
            hd.step(av)
        err = math.degrees(_wrap_pi(hd.decode() - (start + av * steps)))
        return abs(err) / abs(math.degrees(av * steps)) * 100.0

    assert shortfall(0.04) < 10.0, "av=0.04 must stay inside the linear regime"
    fast = shortfall(0.14)
    assert fast > 10.0, (
        f"control is not failing: av=0.14 should out-run the bump and fall short, got {fast:.1f}"
    )


def test_head_direction_is_deterministic():
    a, b = HeadDirection(n=36, seed=0), HeadDirection(n=36, seed=0)
    a.reset(0.2)
    b.reset(0.2)
    assert all(a.step(0.03) == b.step(0.03) for _ in range(30))


# --------------------------------------------------------------------------------------
# GoalVector -- spiking population code for the place-minus-goal vector
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pos,true_bearing,true_dist",
    [
        ((0.0, 0.0), 0.0, 1.0),  # goal due east
        ((1.0, -1.0), math.pi / 2, 1.0),  # due north
        ((2.0, 0.0), math.pi, 1.0),  # due west
        ((0.0, -1.0), math.pi / 4, math.sqrt(2)),  # north-east
        ((1.5, 0.5), -math.pi * 3 / 4, math.hypot(0.5, 0.5)),  # south-west
    ],
)
def test_goal_vector_points_correctly_from_several_positions(pos, true_bearing, true_dist):
    gv = GoalVector(seed=0)
    gv.set_goal((1.0, 0.0))
    r = gv.vector(pos, 0.0)
    berr = abs(math.degrees(_wrap_pi(r["bearing"] - true_bearing)))
    assert berr < 8.0, f"bearing err {berr:.2f} deg from {pos}"
    assert abs(r["distance"] - true_dist) < 0.25, f"distance {r['distance']:.3f} vs {true_dist:.3f}"
    assert r["in_range"] and r["certainty"] > 0.5


def test_heading_error_shrinks_as_the_agent_turns_toward_the_goal():
    gv = GoalVector(seed=0)
    gv.set_goal((1.0, 0.0))  # due east of the origin
    headings = [math.pi * 0.9, math.pi * 0.6, math.pi * 0.3, math.pi * 0.1, 0.0]
    errs = [abs(gv.vector((0.0, 0.0), h)["heading_error"]) for h in headings]
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), (
        f"heading_error must shrink monotonically as the agent turns toward the goal: {errs}"
    )
    assert errs[-1] < math.radians(8.0), "facing the goal must give ~zero heading error"


def test_approach_drive_is_signed_by_heading_error():
    gv = GoalVector(seed=0)
    gv.set_goal((1.0, 0.0))
    facing = gv.vector((0.0, 0.0), 0.0)["approach"]
    away = gv.vector((0.0, 0.0), math.pi)["approach"]
    assert facing > 0.5, "facing the goal must give a positive approach drive"
    assert away < -0.5, "facing away must give a negative approach drive"


def test_bearing_is_undefined_at_the_goal():
    """Honest gap, kept: standing on the goal there is no direction to encode."""
    gv = GoalVector(seed=0)
    gv.set_goal((1.0, 0.0))
    r = gv.vector((1.0, 0.0), 0.0)
    assert r["certainty"] < 0.05, "the population must fall silent at the goal"
    assert not r["in_range"], "distance is meaningless at the goal"
    assert sum(gv.population()["direction"]) == 0, "no direction cell may spike at the goal"


def test_distance_goes_silent_beyond_range_rather_than_saturating():
    """Honest ceiling, kept: past max_distance the distance cells stop firing and distance reads
    0.0 -- it does NOT saturate at max_distance. Bearing survives, so certainty separates this
    case from 'at the goal'."""
    gv = GoalVector(seed=0)
    gv.set_goal((8.0, 0.0))
    r = gv.vector((0.0, 0.0), 0.0)
    assert r["distance"] < 0.01, (
        f"out-of-range failure is SILENCE, not saturation; got {r['distance']:.3f}"
    )
    assert not r["in_range"], "in_range must flag that distance is meaningless"
    # bearing still works out of range -- this is what distinguishes it from standing on the goal
    assert abs(math.degrees(_wrap_pi(r["bearing"]))) < 8.0
    assert r["certainty"] > 0.5, "certainty stays high out of range (vs ~0 at the goal)"


def test_distance_decode_is_biased_inward_at_the_range_edge():
    """Honest bias, kept: population edge effect pulls the estimate in."""
    gv = GoalVector(seed=0)
    gv.set_goal((2.0, 0.0))  # exactly max_distance
    assert 2.0 - gv.vector((0.0, 0.0), 0.0)["distance"] > 0.02


def test_goal_vector_requires_a_goal():
    with pytest.raises(ValueError):
        GoalVector(seed=0).vector((0.0, 0.0), 0.0)


def test_goal_vector_is_deterministic_and_seed_matters():
    a, b = GoalVector(seed=0), GoalVector(seed=0)
    a.set_goal((0.3, 0.7))
    b.set_goal((0.3, 0.7))
    assert a.vector((0.1, 0.2), 0.4) == b.vector((0.1, 0.2), 0.4)
    c = GoalVector(seed=1)
    c.set_goal((0.3, 0.7))
    assert c.vector((0.1, 0.2), 0.4) != a.vector((0.1, 0.2), 0.4), "seed must change the population"


def test_population_exposes_the_raw_spike_counts():
    gv = GoalVector(seed=0)
    gv.set_goal((1.0, 0.0))
    gv.vector((0.0, 0.0), 0.0)
    pop = gv.population()
    assert len(pop["direction"]) == gv.n_dir and len(pop["distance"]) == gv.n_dist
    assert sum(pop["direction"]) > 0 and sum(pop["distance"]) > 0
    assert all(0 <= c <= gv.ticks for c in pop["direction"]), (
        "counts are bounded by the tick budget"
    )

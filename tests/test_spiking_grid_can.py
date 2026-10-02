"""Tests for SpikingGridCAN (faculties/spiking_grid_can.py): a 2-D continuous-attractor SPIKING
grid network whose moving LIF bump path-integrates velocity, plus its landmark (place->grid) reset.

Run: pytest tests/test_spiking_grid_can.py
"""

import math

from snn_data_lake.faculties.spiking_grid_can import (
    SpikingGridCAN,
    _circ_diff,
    _graded_probe,
    _run_circuits,
    _settled,
)


def test_path_integration_tracks_velocity_integral():
    can = SpikingGridCAN(n=16, seed=0)
    can.reset((0.15, 0.15))
    for _ in range(25):
        can.step((0.0, 0.0))  # let the clamped bump relax onto the sheet
    sx, sy = can.decode()
    tx, ty = sx, sy
    for vx, vy in [(0.03, 0.0)] * 8 + [(0.0, 0.025)] * 8:
        can.step((vx, vy))
        tx += vx
        ty += vy
    dx, dy = can.decode()
    err = math.hypot(_circ_diff(dx, tx, can.spacing), _circ_diff(dy, ty, can.spacing))
    assert err < 0.06, f"decoded position must track the velocity integral (err {err:.4f})"
    assert can.bump_amplitude() > 0.05, "the spiking bump must stay alive during integration"


def test_landmark_reset_reclamps_the_bump():
    can = SpikingGridCAN(n=16, seed=0)
    can.reset((0.7, 0.2))
    x, y = can.decode()
    assert abs(_circ_diff(x, 0.7, can.spacing)) < 1e-6
    assert abs(_circ_diff(y, 0.2, can.spacing)) < 1e-6


def test_deterministic_trajectory():
    a, b = SpikingGridCAN(n=16, seed=0), SpikingGridCAN(n=16, seed=0)
    a.reset((0.1, 0.1))
    b.reset((0.1, 0.1))
    assert all(a.step((0.02, 0.015)) == b.step((0.02, 0.015)) for _ in range(20))


# ---------------------------------------------------------------------------
# landmark phase reset for path-integration drift
# ---------------------------------------------------------------------------


def test_dead_reckoning_drifts_monotonically():
    """Without landmarks the decoded position DRIFTS, and the drift GROWS with path length."""
    errs, _worst, min_amp = _run_circuits(_settled(), 4, use_landmarks=False)
    assert errs == sorted(errs), f"drift must grow monotonically with path length (got {errs})"
    assert errs[-1] > 0.3, f"drift must be substantial, not negligible (got {errs[-1]:.4f})"
    assert min_amp > 0.5, "the drifting bump must still be alive -- a lost estimate, not a dead net"


def test_landmark_reset_keeps_error_bounded():
    """The SAME trajectory with periodic landmark sightings stays bounded, far below free drift."""
    free, _fw, _fa = _run_circuits(_settled(), 4, use_landmarks=False)
    lm, lm_worst, lm_amp = _run_circuits(_settled(), 8, use_landmarks=True)
    assert lm[3] < free[3] / 5.0, (
        f"landmark reset must cut the error ({lm[3]:.4f} vs {free[3]:.4f})"
    )
    assert max(lm) < 0.25, f"error must stay bounded over 8 circuits (got {max(lm):.4f})"
    assert lm_worst < 0.25 and lm_amp > 0.5, "bounded per-tick; the bump survives every correction"


def test_correction_is_graded_and_preserves_the_bump():
    """gain 0 is a no-op, gain 1 a full re-clamp, in between the residual shrinks monotonically."""
    residuals, amps, corrs = [], [], []
    for g in (0.0, 0.25, 0.5, 0.9, 1.0):
        corr, _gap_before, gap_after, amp = _graded_probe(g)
        residuals.append(gap_after)
        amps.append(amp)
        corrs.append(corr)
    assert residuals == sorted(residuals, reverse=True), f"not monotone in gain: {residuals}"
    assert corrs[0] == 0.0, "gain 0 must apply no correction at all"
    assert residuals[-1] < 1e-9, "gain 1 must land exactly on the stored phase"
    assert min(amps) > 0.5, f"a graded correction must not kill the bump (min amp {min(amps):.3f})"


def test_first_sighting_learns_the_phase_and_applies_no_correction():
    can = _settled()
    assert can.known_landmarks() == []
    assert can.stored_phase("home") is None
    here = can.decode()
    assert can.observe_landmark("home") == 0.0, (
        "the first sighting only imprints; it must not correct"
    )
    assert can.known_landmarks() == ["home"]
    anchor = can.stored_phase("home")
    gap = math.hypot(
        _circ_diff(anchor[0], here[0], can.spacing), _circ_diff(anchor[1], here[1], can.spacing)
    )
    assert gap < 1e-9, f"the Hebbian imprint must anchor the CURRENT grid phase (gap {gap:.6f})"
    # an explicit true_pos imprints that place instead
    can.observe_landmark("beacon", true_pos=(0.25, 0.75))
    bx, by = can.stored_phase("beacon")
    assert abs(_circ_diff(bx, 0.25, can.spacing)) < 1e-9
    assert abs(_circ_diff(by, 0.75, can.spacing)) < 1e-9


def test_drift_reports_error_against_ground_truth():
    can = _settled()
    x, y = can.decode()
    assert can.drift((x, y)) < 1e-9, "zero error against its own decode"
    assert abs(can.drift((x + 0.1, y)) - 0.1) < 1e-9, "drift is the torus distance to ground truth"
    # wraps the short way round the period rather than reporting 0.9
    assert abs(can.drift((x + 0.9, y)) - 0.1) < 1e-9


def test_raw_mixture_extinguishes_the_bump_at_large_mismatch():
    """KEPT FAILING CONTROL: cue combination is only safe while the mismatch is a modest
    fraction of a period. With the capture rule disabled a large mismatch really does kill the
    bump -- documented, not tuned away. The capture rule survives the identical mismatch."""
    _c, gap_before, _g, dead_amp = _graded_probe(0.6, walk=12, capture_threshold=None, settle=20)
    assert gap_before > 0.25, "this control needs a genuinely large anchor/bump mismatch"
    assert dead_amp < 0.01, (
        f"the raw mixture is expected to extinguish the bump (amp {dead_amp:.3f})"
    )
    _c2, _gb2, resid, live_amp = _graded_probe(0.6, walk=12, capture_threshold=0.2, settle=20)
    assert live_amp > 0.5, f"the cue-capture rule must keep the bump alive (amp {live_amp:.3f})"
    assert resid < 0.05, f"and land near the anchor (residual {resid:.4f})"


def test_landmarks_survive_reset_and_the_landmark_path_is_deterministic():
    can = _settled()
    can.observe_landmark("home")
    before = can.stored_phase("home")
    can.reset((0.1, 0.9))  # re-clamping the bump must not wipe long-term landmark memory
    assert can.known_landmarks() == ["home"]
    assert can.stored_phase("home") == before
    assert (
        _run_circuits(_settled(), 3, use_landmarks=True)[0]
        == _run_circuits(_settled(), 3, use_landmarks=True)[0]
    )

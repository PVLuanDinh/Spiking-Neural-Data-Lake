"""Tests for SpikingGridCAN (hippocampus/spiking_grid_can.py): a 2-D continuous-attractor SPIKING
grid network whose moving LIF bump path-integrates velocity, plus its landmark (place->grid) reset.

Run: pytest tests/test_spiking_grid_can.py
"""

import math

from snn_data_lake.hippocampus.spiking_grid_can import SpikingGridCAN, _circ_diff


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

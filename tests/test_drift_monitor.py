"""
Edge tests for drift_monitor: the online drift/novelty watch that warns before recognition silently
collapses on a rig move (roadmap D3). Safety-relevant — a missed drift warning means the arm keeps
acting on a degraded classifier. Drift in EITHER direction must trip, and a stable stream must stay
quiet (false alarms erode trust in the warning).

Run: pytest tests/test_drift_monitor.py
"""

import random

from snn_data_lake.infra.drift_monitor import DriftMonitor

DIM = 8


def _windows(center, spread, n, seed):
    rng = random.Random(seed)
    return [
        [min(1.0, max(0.0, center + rng.uniform(-spread, spread))) for _ in range(DIM)]
        for _ in range(n)
    ]


def _ref():
    return _windows(0.5, 0.1, 60, seed=1)


def test_stable_stream_does_not_warn():
    mon = DriftMonitor(_ref())
    for w in _windows(0.5, 0.1, 40, seed=2):  # same distribution as the reference
        mon.push(w)
    psi, nov, warn = mon.assess()
    assert not warn, f"stable stream must stay quiet (psi={psi:.3f}, nov={nov:.2f})"
    assert psi < mon.psi_warn


def test_forward_drift_warns():
    mon = DriftMonitor(_ref())
    for w in _windows(0.85, 0.1, 40, seed=3):  # rig moved up -> distribution shifts
        mon.push(w)
    psi, _, warn = mon.assess()
    assert warn, f"upward drift must warn (psi={psi:.3f})"


def test_reverse_drift_warns():
    mon = DriftMonitor(_ref())
    for w in _windows(0.15, 0.1, 40, seed=4):  # rig moved down -> also drift
        _ = mon.push(w)
    _, _, warn = mon.assess()
    assert warn, "downward (reverse) drift must warn too"


def test_novelty_flag_boundary():
    mon = DriftMonitor(_ref())
    assert mon.is_novel([0.95] * DIM) is True  # far from the ~0.5 centroid
    assert mon.is_novel(mon.centroid) is False  # the centroid itself is not novel


def test_recent_buffer_is_capped():
    mon = DriftMonitor(_ref(), window=30)
    for w in _windows(0.5, 0.1, 100, seed=5):
        mon.push(w)
    assert len(mon.recent) == 30, "recent buffer must stay bounded to the window size"


def test_psi_near_zero_on_identical_distribution():
    ref = _ref()
    mon = DriftMonitor(ref)
    for w in ref:  # feed the reference back as the live stream
        mon.push(w)
    assert mon.psi() < 0.05, "PSI should be ~0 when the live distribution matches the reference"

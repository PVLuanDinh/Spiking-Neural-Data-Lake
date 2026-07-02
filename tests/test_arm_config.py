"""
Boundary tests for arm_config: dbin/present/bins_from_capture cut points and the
uncalibrated-IR gate. These are the recognition-calibration edges a rig re-mount hits, so
an off-by-one here silently mis-bins distance into the wrong Paradigm-B channel.

Run: pytest tests/test_arm_config.py
"""

from snn_data_lake import arm_config as ac


def test_dbin_boundaries():
    assert ac.dbin(0.0) == ac.NEAR
    assert ac.dbin(ac.NEAR_MAX - 1e-9) == ac.NEAR
    assert ac.dbin(ac.NEAR_MAX) == ac.MID  # d < NEAR_MAX is strict -> 0.30 is MID
    assert ac.dbin(ac.MID_MAX - 1e-9) == ac.MID
    assert ac.dbin(ac.MID_MAX) == ac.FAR  # 0.60 is FAR
    assert ac.dbin(ac.IN_RANGE_MAX - 1e-9) == ac.FAR
    assert ac.dbin(ac.IN_RANGE_MAX) is None  # >= in-range -> idle (no object)
    assert ac.dbin(99.0) is None


def test_present_distance_and_ir_gate():
    assert ac.present(ac.IN_RANGE_MAX - 1e-9) is True
    assert ac.present(ac.IN_RANGE_MAX) is False  # boundary: not present at exactly in-range
    # IR disabled by default -> the ir value is ignored either way
    assert ac.present(0.5, ir=0) is True
    assert ac.present(0.5, ir=99999) is True


def test_present_ir_confirms_only_when_calibrated(monkeypatch):
    monkeypatch.setattr(ac, "IR_PRESENT", (">", 500))
    assert ac.present(0.5, ir=600) is True  # in range AND IR agrees
    assert ac.present(0.5, ir=400) is False  # in range but IR rejects (distance ghost)
    assert ac.present(2.5, ir=600) is False  # out of range regardless of IR
    monkeypatch.setattr(ac, "IR_PRESENT", ("<", 500))
    assert ac.present(0.5, ir=400) is True  # "<" polarity honoured


def test_bins_from_capture_defaults_when_too_few():
    assert ac.bins_from_capture([0.1, 0.2]) == (ac.NEAR_MAX, ac.MID_MAX)


def test_bins_from_capture_percentiles_and_range_filter():
    caps = [0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 5.0]  # 5.0 is out of range -> must be dropped
    near, mid = ac.bins_from_capture(caps)
    assert 0.0 < near <= mid < ac.IN_RANGE_MAX  # monotone, in range, derived from real data

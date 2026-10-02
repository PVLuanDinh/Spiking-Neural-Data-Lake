"""Tests for faculties.region_coherence -- cross-region Kuramoto coherence + integration proxy."""

import math

import numpy as np

from snn_data_lake.faculties.region_coherence import RegionCoherence


def _sine_window(length, phase, freq=1.0):
    n = np.arange(length)
    return list(np.sin(2.0 * np.pi * freq * n / length + phase))


def _sync(length=32, k=6):
    return {f"r{i}": _sine_window(length, 0.0) for i in range(k)}


def _desync(length=32, k=6):
    return {f"r{i}": _sine_window(length, 2.0 * math.pi * i / k) for i in range(k)}


def test_update_returns_required_keys():
    out = RegionCoherence(seed=0).update(_sync())
    for key in ("coherence", "phase_order", "n_regions"):
        assert key in out
    assert out["n_regions"] == 6
    assert 0.0 <= out["coherence"] <= 1.0
    assert 0.0 <= out["phase_order"] <= 1.0


def test_synchronized_high_phase_order():
    out = RegionCoherence(seed=0).update(_sync())
    assert out["phase_order"] > 0.95


def test_desynchronized_low_phase_order():
    out = RegionCoherence(seed=0).update(_desync())
    assert out["phase_order"] < 0.20


def test_sync_dominates_desync():
    sync = RegionCoherence(seed=0).update(_sync())
    desync = RegionCoherence(seed=0).update(_desync())
    assert sync["phase_order"] > desync["phase_order"]
    assert sync["coherence"] > desync["coherence"]


def test_silent_bank_not_spuriously_coherent():
    # A silent/constant bank has no oscillation to lock; it must not read as synchronized
    # (crc32-scattered surrogate phases).
    silent = {f"r{i}": [0.0] * 32 for i in range(6)}
    out = RegionCoherence(seed=0).update(silent)
    sync = RegionCoherence(seed=0).update(_sync())
    assert out["phase_order"] < sync["phase_order"]


def test_coherence_property_tracks_last_update():
    rc = RegionCoherence(seed=0)
    out = rc.update(_sync())
    assert rc.coherence == out["coherence"]
    assert rc.phase_order == out["phase_order"]
    assert rc.n_regions == 6


def test_ignition_bias_monotonic_in_coherence():
    base = 0.55
    sweep = [0.0, 0.25, 0.5, 0.75, 1.0]
    # exercise the LIVE calibrated gain, not a re-typed copy (see RegionCoherence.IGNITION_GAIN)
    biases = [RegionCoherence._adjust(base, c, RegionCoherence.IGNITION_GAIN) for c in sweep]
    assert all(b2 < b1 for b1, b2 in zip(biases, biases[1:], strict=False))
    assert biases[0] == base  # coherence 0 -> unchanged


def test_ignition_bias_higher_coherence_lowers_threshold():
    base = 0.55
    rc = RegionCoherence(seed=0)
    rc.update(_sync())
    hi_c, hi_bias = rc.coherence, rc.ignition_bias(base)
    rc.update(_desync())
    lo_c, lo_bias = rc.coherence, rc.ignition_bias(base)
    assert hi_c > lo_c
    assert hi_bias < lo_bias
    assert 0.0 < hi_bias <= base
    assert 0.0 < lo_bias <= base


def test_determinism_same_seed_same_input():
    a = RegionCoherence(seed=0).update(_sync())
    b = RegionCoherence(seed=0).update(_sync())
    assert a == b
    c = RegionCoherence(seed=0).update(_desync())
    d = RegionCoherence(seed=0).update(_desync())
    assert c == d


def test_scalar_rate_rolling_path():
    # One scalar rate per region per tick accumulates an internal rolling window and yields a
    # coherence in [0,1]; in-phase scalar oscillations should read as coherent.
    rc = RegionCoherence(seed=0)
    out = None
    for t in range(40):
        v = math.sin(2.0 * math.pi * t / 16.0)
        out = rc.update({f"r{i}": v for i in range(5)})
    assert out is not None
    assert 0.0 <= out["coherence"] <= 1.0
    assert out["n_regions"] == 5
    # identical in-phase scalar streams -> high synchrony.
    assert out["phase_order"] > 0.8


def test_empty_and_single_region_are_safe():
    rc = RegionCoherence(seed=0)
    empty = rc.update({})
    assert empty["n_regions"] == 0
    assert empty["coherence"] == 0.0
    one = rc.update({"solo": _sine_window(32, 0.0)})
    assert one["n_regions"] == 1
    assert 0.0 <= one["coherence"] <= 1.0


def test_selfcheck_main_runs():
    from snn_data_lake.faculties import region_coherence

    region_coherence.main()  # asserts internally; must not raise

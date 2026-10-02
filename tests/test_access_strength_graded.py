"""Regression tests for ReentrantWorkspace.access_strength (faculties/reentrant_workspace.py).

The measure is documented as GRADED. It previously was not: a single pool's sustained firing RATE is
capped at one spike per micro-step, and every drive the brain produces sits far above threshold, so
over 300 live wake ticks it took exactly ONE distinct value (1.000). These tests pin the property
that the docstring claims.

Run: pytest tests/test_access_strength_graded.py
"""

from snn_data_lake.faculties.reentrant_workspace import ReentrantWorkspace


def _strength_at(drive):
    """access_strength for an ignited workspace whose winner is driven at `drive`."""
    ws = ReentrantWorkspace(seed=0)
    ws._last = {"ignited": True, "winner": "x", "availability": 1.0}
    ws._last_drive = drive
    return ws.access_strength()


def test_dark_tick_is_exactly_zero():
    ws = ReentrantWorkspace(seed=0)
    ws._last = {"ignited": False, "winner": None, "availability": 0.0}
    assert ws.access_strength() == 0.0


def test_graded_across_the_measured_operating_band():
    """The live winner drive (base + re-entrant sustain) was measured at 0.871..1.899 over 360
    ignited ticks. The readout must produce many distinct levels across that band, not one."""
    vals = [_strength_at(0.871 + (1.899 - 0.871) * i / 40.0) for i in range(41)]
    assert len(set(vals)) >= 8, f"readout must be graded over the operating band (got {set(vals)})"
    assert min(vals) < max(vals)


def test_monotone_in_drive():
    """Stronger support for the ignited content must never read as weaker access."""
    xs = [0.0 + 1.3 * i / 30.0 for i in range(31)]
    vals = [_strength_at(x) for x in xs]
    assert vals == sorted(vals), f"access strength must be monotone in drive: {vals}"


def test_bounded_and_saturates_only_past_the_population():
    assert _strength_at(0.0) == 0.0
    assert _strength_at(50.0) == 1.0, "far above every unit threshold -> full access"
    assert all(0.0 <= _strength_at(d) <= 1.0 for d in (0.3, 0.9, 1.4, 2.5))


def test_does_not_pin_at_one_over_the_live_band():
    """The exact failure being regressed: pinned at 1.000 on essentially every live tick."""
    band = [0.871 + (1.899 - 0.871) * i / 20.0 for i in range(21)]
    pinned = sum(1 for d in band if _strength_at(d) >= 1.0)
    assert pinned <= len(band) // 4, f"{pinned}/{len(band)} of the live band saturates"


def test_deterministic():
    assert _strength_at(1.688) == _strength_at(1.688)

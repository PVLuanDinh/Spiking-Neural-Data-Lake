"""
Edge tests for the Reflex fast-path: firing exactly at threshold, a severe breach in one
step, reverse-sign (negative) danger, worst-action selection, and out-of-range channels
ignored. Safety-critical: this preempts cognition on a collision, so a missed fire drives
the arm into the obstacle and a spurious fire freezes it.

Run: pytest tests/test_reflex.py
"""

from snn_data_lake.reflex import Reflex, reflex_guard


def _row(ch6=500.0, ch7=0.0):
    """8-channel raw sample; ch6=motor current, ch7=force/torque (the two rule channels)."""
    r = [500.0] * 8
    r[6], r[7] = ch6, ch7
    return r


def test_safe_signal_never_fires():
    r = Reflex()
    assert all(r.step(_row()) is None for _ in range(50))


def test_fires_exactly_at_threshold_in_one_step():
    # clean arithmetic (no float-boundary fragility): range=10, over=5 -> sev=0.5 == threshold.
    # membrane v = 0*decay + 0.5 == threshold on step 1 -> fires (>= is inclusive).
    r = Reflex(rules={0: (0.0, 10.0, "STOP")}, threshold=0.5)
    assert r.step([15.0]) == "STOP"


def test_just_below_threshold_does_not_fire_on_first_step():
    r = Reflex(rules={0: (0.0, 10.0, "STOP")}, threshold=0.5)
    assert r.step([14.9]) is None  # over=4.9 -> sev=0.49 < 0.5


def test_severe_breach_fires_in_one_step():
    assert Reflex().step(_row(ch6=5000.0)) == "STOP"  # far past 800 -> instant


def test_reverse_sign_danger_is_caught():
    assert Reflex().step(_row(ch7=-500.0)) == "WITHDRAW"  # ch7 below -300 (negative overload)
    assert Reflex().step(_row(ch6=-1000.0)) == "STOP"  # ch6 below -50 (reverse current)


def test_worst_channel_action_wins():
    # both channels breach; ch6 severity dominates -> its action (STOP) fires, not WITHDRAW
    assert Reflex().step(_row(ch6=100000.0, ch7=-400.0)) == "STOP"


def test_channels_beyond_row_length_are_ignored():
    r = Reflex()
    assert all(r.step([0.0, 0.0, 0.0]) is None for _ in range(10))  # no ch6/ch7 present


def test_guard_surfaces_action_as_side_channel():
    fired = []
    rows = [_row()] * 3 + [_row(ch6=1500.0)]
    passed = list(reflex_guard(rows, Reflex(), lambda act, seq: fired.append((seq, act))))
    assert len(passed) == len(rows)  # every row passes through unchanged
    assert fired and fired[0] == (4, "STOP")  # tripped on the 4th sample (1-indexed)

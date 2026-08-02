"""Tests for the held-out growth instrument (tools/timing_heldout.py).

These test the INSTRUMENT, not the brain. The two things that can silently ruin a held-out verdict
are a split that is not really held out and a readout that cannot see the population it is judging,
and both have already happened in this repo -- so both are asserted here rather than assumed.

Run: pytest tests/test_timing_heldout.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.timing_heldout import (  # noqa: E402
    TimeCellClock,
    eligible,
    interval_universe,
    readout_blind_ramp,
    readout_geometric_mid,
    teach_clock,
    teach_residual_gap,
)
from tools.timing_heldout import main as heldout_main  # noqa: E402


def test_split_is_disjoint_and_covers_every_gap():
    """FR-009. A literal trailing-20% split leaves only 4 of 63 gaps able to move the metric; the
    gap-major grid gives all 63. Disjointness is the other half -- a split that is not disjoint is
    not held out, however it is described."""
    thresholds = TimeCellClock(n_cells=64, seed=0).thresholds
    teach, test, gaps_hit = interval_universe(thresholds)
    assert len(teach) + len(test) == 630
    assert len(teach) == 504 and len(test) == 126
    assert not (set(teach) & set(test))
    assert gaps_hit == 63


def test_instrument_refuses_blind_readout():
    """FR-005. A readout that ignores the thresholds must produce NOT-ELIGIBLE, not a null. Exit 2
    is not a soft 0: a probe that cannot ask the question must not answer it."""
    ok, why = eligible(readout_blind_ramp, 64)
    assert not ok and why


def test_instrument_admits_the_live_readout():
    """The gate must be able to say yes, or it is not a gate but a wall."""
    ok, why = eligible(readout_geometric_mid, 64)
    assert ok, why


def test_null_control_gap_is_measurable():
    """The control's zero has to be earned. On its first run this instrument picked the
    lowest-teach-residual gap without checking whether any TEST interval fell inside it -- MEASURED
    across seeds 0/1/2, none did -- so the null arm tied 12 of 12 because nothing measured it. A
    control whose zero comes from the instrument's own blindness launders the main arm's result."""
    probe = TimeCellClock(n_cells=64, seed=0)
    teach, test, _ = interval_universe(probe.thresholds)
    for seed in range(3):
        clock = teach_clock(seed, 64, teach)
        j = teach_residual_gap(clock, teach, test, readout_geometric_mid)
        assert j is not None
        lo, hi = clock.thresholds[j], clock.thresholds[j + 1]
        assert any(lo <= tau < hi for tau in test), f"null gap {j} holds no test interval"


def test_cli_refuses_a_dispersion_it_cannot_compute():
    """FR-002: no value computed from no data. One seed has no dispersion, so there is no statistic
    to report and the instrument must refuse rather than print a confident single number."""
    assert heldout_main(["--seeds", "1"]) == 2

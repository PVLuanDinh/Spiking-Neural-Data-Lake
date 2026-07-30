"""Tests for `tools/coverage_score.py`'s observable fingerprint.

`_fingerprint` used to require the literal token `ticks`, so every probe that reported `N/M turns`
fingerprinted as the EMPTY tuple -- which made `rerun` an exit-code-only check for those artifacts
and made W3/W5 structurally unable to fire on them (both short-circuit on the falsy empty tuple).
The regex was widened on 2026-07-30 to accept any bare `N/M` counter.

These tests police the widening rather than the counting. Accepting more shapes is only safe if the
NON-counter shapes stay rejected: a date, a path, or a decimal that leaked into a fingerprint would
poison every artifact's comparison at once, and `rerun` would start refusing honest reproductions
for a reason no reader could see. So the rejection cases are the load-bearing half of this file.
Deliberately NOT asserted: the fingerprint of any specific tracked artifact -- pinning those would
make a probe's honest output change look like an instrument regression.
"""

import importlib.util
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SPEC = importlib.util.spec_from_file_location(
    "coverage_score", os.path.join(_ROOT, "tools", "coverage_score.py")
)
cs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cs)


# Shapes a probe really prints. All five occur verbatim in docs/coverage/artifacts/.
@pytest.mark.parametrize(
    "line,want",
    [
        ("act/hold decision changed by the signal       : 0/60 ticks", [("0", "60")]),
        ("reply changed by neutering the faculty        : 2/2 turns", [("2", "2")]),
        ("gate `da` changed by the neuter               : 15/31 calls", [("15", "31")]),
        ("wake ticks on the wired run (A)               : 31/60", [("31", "60")]),
        ("control random  :  51.1%  (856/1675)", [("856", "1675")]),
        ("split=evaluation lib=off limit=None : 12/400 = 3.00%  (227 s)", [("12", "400")]),
        ("0/60 ticks act, 0/60 traj", [("0", "60"), ("0", "60")]),
    ],
)
def test_real_counters_are_captured(line, want):
    assert cs.FLIPS.findall(line) == want


# The whole reason `(\d+)/(\d+)` was refused for a session: these must contribute NOTHING.
@pytest.mark.parametrize(
    "line",
    [
        "recorded_at: 2026/07/30T11:00:00Z",
        "2026/07/30",
        "src/snn_data_lake/cortex/readout.py:12",
        "docs/coverage/artifacts/gf_raven_dist3.txt",
        "rate 1.5/2 of baseline",
        "accuracy 12/400.5 percent",
        "path a/12/34/b",
        "C:\\logs\\7\\8 done",
    ],
)
def test_dates_paths_and_decimals_are_rejected(line):
    assert cs.FLIPS.findall(line) == []


def test_a_turns_probe_no_longer_fingerprints_empty():
    """The exact hole this widening closed: `rerun` compared () to () for every `turns` probe."""
    stdout = (
        "wired replies carrying the faculty marker     : 2/3\n"
        "reply changed by neutering the faculty        : 2/3 turns\n"
    )
    assert cs._fingerprint(stdout) == (("2", "3"), ("2", "3"))


def test_fingerprint_of_a_transcript_with_no_counter_is_still_empty():
    """An empty fingerprint is still legal -- it means `rerun` degrades to an exit-code check, which
    is the honest state for a probe that reports no counter at all, not a bug to paper over."""
    assert cs._fingerprint("unknown drive source 'letter_analogy'\nNOT ELIGIBLE\n") == ()

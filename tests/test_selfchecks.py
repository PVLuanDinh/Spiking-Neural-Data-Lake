"""
Pins run_selfchecks' latch-attribution and gate-home helpers.

The per-module self-checks (CORE + TORCH_GATED) are NOT re-run here any more (owner 2026-10-19):
the battery's `coverage run run_selfchecks.py --battery` runs them once, in-process (the
cross-module leak detector) under coverage, plus the other tiers as subprocesses; its coverage is
combined with pytest's and the 70% bar is checked on the combined report. An in-process copy here
re-ran all 331 checks for coverage alone (P2: one mechanism).
"""

import os
import time

import pytest

from run_selfchecks import (
    _HOME,
    _gate_home,
    _sweep_latch,
    latch_in_window,
)


def test_latch_attribution_refuses_the_smear(tmp_path):
    """Drive the classify-and-remove unit (_sweep_latch), where the pre-fix defect lived as
    attribution-by-EXISTENCE (chip task_100cb4d5: batteries named innocent checks for
    straggler latches), against a REAL latch file: an out-of-slot mtime must land in the
    UNATTRIBUTED stray bucket, an in-slot mtime attributes, an absent latch is a no-op, and
    the file is removed either way. Honest scope: the TWO production call seams (run_all's and
    run_all_in_process's per-check calls) are not executed here -- the in-process seam was
    run-verified end-to-end 2026-08-31 (sentinel-home --in-process run, 266/266, the one real
    latch-leaver attributed in-slot); the predicate's inclusive boundaries are pinned
    separately below so the convention cannot drift silently."""
    latch = tmp_path / "estop.latch"
    w0 = time.time()
    w1 = w0 + 5.0
    left, stray = [], []
    # the smear: written 30s before this slot began -> stray, unattributed, removed
    latch.write_text("stop")
    os.utime(latch, (w0 - 30.0, w0 - 30.0))
    _sweep_latch(str(latch), "innocent.check", w0, w1, left, stray)
    assert left == [] and len(stray) == 1
    assert stray[0].startswith("innocent.check@-") and stray[0].endswith("s")
    assert not latch.exists()
    # written after the awaited child exited -> writer unknown -> stray
    latch.write_text("stop")
    os.utime(latch, (w1 + 2.0, w1 + 2.0))
    _sweep_latch(str(latch), "innocent.check", w0, w1, left, stray)
    assert left == [] and len(stray) == 2
    # the check's own in-slot write -> attributed by name
    latch.write_text("stop")
    os.utime(latch, (w0 + 1.0, w0 + 1.0))
    _sweep_latch(str(latch), "stopping.check", w0, w1, left, stray)
    assert left == ["stopping.check"] and len(stray) == 2
    assert not latch.exists()
    # no latch -> no-op on both buckets
    _sweep_latch(str(latch), "clean.check", w0, w1, left, stray)
    assert left == ["stopping.check"] and len(stray) == 2
    # predicate boundary convention: inclusive at both ends, exclusive just outside
    assert latch_in_window(w0, w0, w1) and latch_in_window(w1, w0, w1)
    assert not latch_in_window(w0 - 0.001, w0, w1)


def test_gate_home_isolates_and_restores(monkeypatch):
    """_gate_home (chip task_f266d626): run_selfchecks' OWN in-process runners had no
    NORD_HOME isolation, so a check run through them hit the operator's real ~/.snn_data_lake
    (THIS pytest runner was never exposed -- conftest's autouse _hermetic_home has isolated it
    since 2026-07-31). Inside the context, home() must resolve to the runner's shared isolated
    _HOME (home() reads the env at USE time -- probed 2026-08-31, set-after-import redirects);
    on exit the caller's own value must be RESTORED, never bare-popped (substrate.home's
    recorded lesson), and an unset caller must come back unset. Honest scope: the wiring seams
    (run_one_in_process's per-call gate, run_all_in_process's loop span, the dispatch's
    process-scope set) are not executed here; the loop span was run-verified end-to-end
    2026-08-31."""
    from snn_data_lake.substrate.home import home

    # caller HAD a value -> restored exactly
    monkeypatch.setenv("NORD_HOME", "caller-sentinel")
    with _gate_home():
        assert os.environ["NORD_HOME"] == _HOME
        assert os.path.normcase(home()) == os.path.normcase(_HOME)
    assert os.environ["NORD_HOME"] == "caller-sentinel"
    # caller had NO value -> comes back unset
    monkeypatch.delenv("NORD_HOME", raising=False)
    with _gate_home():
        assert os.environ["NORD_HOME"] == _HOME
    assert "NORD_HOME" not in os.environ
    # an exception inside still restores
    monkeypatch.setenv("NORD_HOME", "caller-sentinel-2")
    with pytest.raises(RuntimeError):
        with _gate_home():
            raise RuntimeError("boom")
    assert os.environ["NORD_HOME"] == "caller-sentinel-2"


def test_latch_report_lines_shape():
    """_latch_report_lines claims ONE formatter so the two runners cannot drift into
    different claims about the same signal; prose alone does not enforce that. Pin the
    shape: empty buckets emit nothing; attributed emits one line carrying the names and the
    mtime-in-slot wording; strays emit one line carrying UNATTRIBUTED and the offset entries;
    both emit both, attributed first."""
    from run_selfchecks import _latch_report_lines

    assert _latch_report_lines([], []) == []
    left = _latch_report_lines(["a.check"], [])
    assert len(left) == 1 and "left by 1 check(s)" in left[0] and "a.check" in left[0]
    assert "mtime-in-slot" in left[0]
    stray = _latch_report_lines([], ["b.check@-3.2s"])
    assert len(stray) == 1 and "UNATTRIBUTED" in stray[0] and "b.check@-3.2s" in stray[0]
    both = _latch_report_lines(["a.check"], ["b.check@-3.2s"])
    assert len(both) == 2 and "left by" in both[0] and "UNATTRIBUTED" in both[1]

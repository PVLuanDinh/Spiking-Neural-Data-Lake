"""specs/022 T038 (FR-027): the coverage ledger's two measured properties.

1. `_covered` is LAKE-PERSISTED and comes back after a restart -- the P4b red, proven the way P4b
   requires: a REAL brain (make_brain, persist=True) covers cells, its sleep tick flushes the
   ledger, and the process ENDS WITHOUT close() (a kill); a fresh process adopts the ledger.
   Restore is OPT-IN (NORD_TIME_CELLS_RESTORE=1, the daemon's switch) AND persist-gated, so the
   gates keep cold-starting from the seed; a lineage mismatch is REFUSED, never merged.
2. Saturation is READABLE: a rate (new cells per coverage tick) and its MOTION companion beside
   the count. The review's constructed pair: a pre-covered workspace with the body still moving
   and an E-STOPped body both read rate 0.0 -- only the pair tells them apart.

Run: pytest tests/test_coverage_persistence.py
"""

import json
import os
import subprocess
import sys

import pytest

from snn_data_lake.engine import autonomous_loop as al
from snn_data_lake.engine.autonomous_loop import COVERAGE_STATE, AutonomousArm
from snn_data_lake.substrate import home

CTX = dict(visual=0.55, temperature=24.0, audio=0.3, distance=0.6)

# The child is the REAL composition, not a bare arm calling save_coverage() by hand (the first
# draft did that, and its 'killed-process red' never touched the brain's flush path -- the review
# neutered Brain._flush_coverage and the test stayed green). It ticks until the SLEEP-TICK flush
# has written the ledger, then dies without close().
_CHILD = r"""
import json, os, sys
os.environ["NORD_HOME"] = sys.argv[1]
os.environ.pop("NORD_TIME_CELLS_RESTORE", None)
from snn_data_lake.engine.autonomous_loop import COVERAGE_STATE
from snn_data_lake.engine.brain import make_brain
from snn_data_lake.substrate import home
b = make_brain(seed=0, persist=True)
flushed_at = None
for t in range(1, int(sys.argv[2]) + 1):
    b.tick()
    if home.load_state(COVERAGE_STATE, default=None) is not None:
        flushed_at = t
        break
print("RESULT " + json.dumps({"cells": len(b.brain.arm._covered), "flushed_at": flushed_at}))
sys.stdout.flush()
os._exit(0)  # deliberately NO close(): the kill the red is about
"""


def _cover(ticks, seed=0, arm=None):
    a = arm or AutonomousArm(seed=seed)
    for _ in range(ticks):
        a.step(reward_fn=a.coverage_reward, **CTX)
    return a


def _run_child(max_ticks=400):
    env = dict(os.environ)
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, os.environ["NORD_HOME"], str(max_ticks)],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    raise AssertionError(f"child produced no RESULT.\nstdout:{proc.stdout}\nstderr:{proc.stderr}")


def test_the_ledger_survives_a_killed_brain_and_is_adopted_wholesale():
    """The P4b red: a real brain covers cells, its sleep tick flushes, the process dies without
    close(); a NEW process adopts. Goes red when the sleep-tick flush is removed."""
    child = _run_child()
    assert child["flushed_at"] is not None, f"the sleep tick never flushed the ledger: {child}"
    assert child["cells"] > 1, child
    blob = home.load_state(COVERAGE_STATE, default=None)
    assert isinstance(blob, dict) and 1 < len(blob["cells"]) <= child["cells"], blob
    fresh = AutonomousArm(seed=0)
    assert len(fresh._covered) == 0, "an AutonomousArm must cold-start unless told to restore"
    n, why = fresh.restore_coverage(blob)
    assert n == len(blob["cells"]) and why.startswith("restored"), why
    assert len(fresh._covered) == n and fresh._cover_new == n
    # and it KEEPS COUNTING from there, not from zero
    fresh.step(reward_fn=fresh.coverage_reward, **CTX)
    assert len(fresh._covered) >= n


def test_a_lineage_mismatch_is_refused_never_merged():
    a = _cover(30)
    blob = a.coverage_state()
    assert set(blob) == {"fmt", "cell_m", "reach_mm", "cells"}, "only what the restore adopts"
    fresh = AutonomousArm(seed=0)
    bad = [
        ({**blob, "cell_m": blob["cell_m"] * 2.0}, "grid"),  # same indices, different places
        ({**blob, "reach_mm": blob["reach_mm"] + 1.0}, "body"),
        ({**blob, "fmt": 99}, "fmt"),
        ({**blob, "cells": [["x", 0, 0]]}, "malformed"),
        ({**blob, "cells": "notalist"}, "malformed"),
        ({**blob, "cells": [[1.9, 0, 0]]}, "non-integer"),  # int(1.9) would be another cell
        ({**blob, "cell_m": None}, "malformed"),  # the first draft RAISED on these three
        ({**blob, "reach_mm": "x"}, "malformed"),
        ({**blob, "cell_m": "abc"}, "malformed"),
    ]
    for blob_bad, word in bad:
        n, why = fresh.restore_coverage(blob_bad)
        assert n == 0 and why.startswith("refused") and word in why, (word, why)
        assert len(fresh._covered) == 0 and fresh._cover_restored == 0, (
            "a refused restore must not mutate"
        )
    n, why = fresh.restore_coverage(None)
    assert n == 0 and why == "nothing persisted"
    # the good blob still adopts -- the refusals above are not a broken adopter
    n, why = fresh.restore_coverage(blob)
    assert n == len(a._covered) and why.startswith("restored")


def test_the_brain_restores_only_under_the_switch_and_persist_and_flushes_on_close(monkeypatch):
    """The wiring: a brain adopts the ledger ONLY under NORD_TIME_CELLS_RESTORE=1 AND persist=True
    (read and write share one gate), and close() -- the second durable-save point -- flushes."""
    from snn_data_lake.engine.brain import make_brain

    seed = _cover(60)
    assert seed.save_coverage()
    n_lake = len(home.load_state(COVERAGE_STATE)["cells"])
    monkeypatch.delenv("NORD_TIME_CELLS_RESTORE", raising=False)
    cold = make_brain(seed=0, persist=True)
    try:
        assert len(cold.brain.arm._covered) == 0, "a brain restored coverage without the switch"
        assert cold.brain.arm.coverage_restore_note == "disabled"
    finally:
        cold.close()
    monkeypatch.setenv("NORD_TIME_CELLS_RESTORE", "1")
    gate = make_brain(seed=0, persist=False)
    try:
        assert len(gate.brain.arm._covered) == 0, "a NON-persistent brain read the lake"
        assert gate.brain.arm.coverage_restore_note == "disabled"
    finally:
        gate.close()
    warm = make_brain(seed=0, persist=True)
    try:
        assert len(warm.brain.arm._covered) == n_lake, warm.brain.arm.coverage_restore_note
        assert warm.brain.arm.coverage_restore_note.startswith("restored")
        assert warm.brain.arm.coverage_window() == 0, "a restore must not fake a measured window"
        for _ in range(25):
            warm.brain.arm.step(reward_fn=warm.brain.arm.coverage_reward, **CTX)
        grown = len(warm.brain.arm._covered)
    finally:
        warm.close()  # the second durable-save point
    blob = home.load_state(COVERAGE_STATE, default=None)
    assert isinstance(blob, dict) and len(blob["cells"]) == grown, (
        "close() did not flush the ledger"
    )


def test_the_rate_and_its_motion_companion_separate_exhausted_from_stopped():
    """The constructed pair. A: every cell the body can reach is pre-covered, so nothing is ever
    new while the body keeps moving -> rate 0, motion > 0. B: an E-STOPped body -> rate 0,
    motion 0. C: a fresh body early in its run -> both > 0. The count alone cannot tell A from
    B, and the rate alone cannot either (the review measured both at 0.0000)."""
    r = 30  # the landing run reaches 22.3 cells from the origin; pre-cover well past it
    everywhere = {
        (x, y, z)
        for x in range(-r, r + 1)
        for y in range(-r, r + 1)
        for z in range(-r, r + 1)
        if x * x + y * y + z * z <= r * r
    }
    a = AutonomousArm(seed=0)
    a._covered = set(everywhere)
    _cover(150, arm=a)
    assert a.coverage_window() == 150
    assert a.coverage_rate(100) == 0.0, "a pre-covered workspace yielded a new cell"
    assert a.coverage_motion(100) > 0.0, "the body stopped moving on retrodden ground"
    b = AutonomousArm(seed=0)
    b.arm.apply("EMERGENCY_STOP")
    _cover(150, arm=b)
    assert b.coverage_rate(100) == 0.0 and b.coverage_motion(100) == 0.0, "a stopped body moved"
    c = _cover(50)
    assert c.coverage_rate() > 0.0 and c.coverage_motion() > 0.0 and c.coverage_window() == 50
    # the scale is the exact lattice count, derived from the resolution, and named a scale
    assert al._lattice_ball_cells(2) == 33 and al._lattice_ball_cells(4) == 257
    assert AutonomousArm.coverage_scale_cells() == al._lattice_ball_cells(al._COVER_CELLS_PER_REACH)


def test_the_safety_reply_reports_what_the_latch_measures():
    """The post-T037a poll's revelation: the brain must report what the gate measures -- a
    restored stop, and durability read from the persist-error counter, never asserted."""
    from snn_data_lake.brainstem.so101_arm import ESTOP_LATCH
    from snn_data_lake.engine.brain import make_brain

    with open(os.path.join(home.home(), ESTOP_LATCH), "w", encoding="utf-8") as f:
        json.dump({"token": "another-process", "pid": 1, "utc": "x", "source": "EMERGENCY_STOP"}, f)
    b = make_brain(seed=0, persist=False)
    try:
        reply = b.brain.converse("is it safe?")
        assert (
            "restored from a previous process" in reply and "survives a daemon restart" in reply
        ), reply
        b.brain.arm.arm.latch_persist_errors = 2  # the gate counted a sidecar it could not write
        reply = b.brain.converse("is it safe?")
        assert "would NOT survive a restart" in reply and "2 persist errors" in reply, reply
        b.brain.arm.arm.latch_persist_errors = 0
        b.brain.converse("go home", human=True)
        reply = b.brain.converse("is it safe?")
        assert "free to move" in reply and "survives a daemon restart" in reply, reply
    finally:
        b.close()


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

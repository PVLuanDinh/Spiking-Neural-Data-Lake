"""
Run every CORE and TORCH_GATED module's __main__ self-check in-process under pytest.
run_selfchecks.py runs the same asserts as isolated subprocesses (the zero-dep smoke gate) —
that gives no coverage signal and truncates tracebacks (A1). Running them in-process makes
coverage.py see the product code and surfaces the full traceback on failure. Reuses
run_selfchecks' own lists so they never drift from the canonical runner.

TORCH_GATED added 2026-07-30, and the reason is worth keeping. This file imported CORE alone
while `run_selfchecks.MODULES` is `CORE + TORCH_GATED + NENGO_GATED + RESEARCH`, so six
modules' self-checks — including `cortex.brain`, `growth.ng_snn` and
`growth.neurogenesis_engine` — were imported but never RUN in-process. Their `main()` bodies
measured 0/126, 0/55 and 0/57. The whole torch growth tier was therefore reached only as a
side effect of `infra.brain_telemetry`'s self-check, whose tick budget is data-dependent:
it landed on either side of the engine's `refit_every=20` from run to run, covering 0, 5 or
115 statements by accident and swinging the reported total 0.32 points on byte-identical
source. Running these on purpose is the fix; the incidental path stops being the only driver.

NOT included, deliberately: NENGO_GATED (optional dep, self-skips when absent) and RESEARCH
(benchmarks needing the repo root on PYTHONPATH — not product code). Cost: ~70s per run.

Each module is run in a fresh temp cwd (its ./data writes stay isolated) with argv trimmed to
the no-arg self-check path.

Run: pytest tests/test_selfchecks.py
"""

import os
import runpy
import sys
import time

import pytest

from run_selfchecks import (
    _HOME,
    _TORCH,
    CORE,
    TORCH_GATED,
    _gate_home,
    _sweep_latch,
    latch_in_window,
)

# `_TORCH` is run_selfchecks' own probe, so a broken torch install skips here exactly as it
# skips there, rather than turning into six confusing import errors.
_IN_PROCESS = CORE + (TORCH_GATED if _TORCH else [])


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


@pytest.mark.parametrize("mod", _IN_PROCESS)
def test_core_selfcheck_in_process(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate any ./data writes per module
    monkeypatch.setattr(sys, "argv", [mod])  # no-arg -> each module's self-check path
    try:
        runpy.run_module(f"snn_data_lake.{mod}", run_name="__main__")
    except SystemExit as e:  # a module may sys.exit(0) on success
        assert e.code in (0, None), f"{mod} self-check exited non-zero: {e.code}"

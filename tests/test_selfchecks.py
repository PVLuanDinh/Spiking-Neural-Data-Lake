"""
Run every CORE and TORCH_GATED module's __main__ self-check in-process under pytest.
run_selfchecks.py runs the same asserts as isolated subprocesses (the zero-dep smoke gate) —
that gives no coverage signal and truncates tracebacks (A1). Running them in-process makes
coverage.py see the product code and surfaces the full traceback on failure. Reuses
run_selfchecks' own lists so they never drift from the canonical runner.

TORCH_GATED added 2026-07-30, and the reason is worth keeping. This file imported CORE alone
while `run_selfchecks.MODULES` is `CORE + TORCH_GATED + NENGO_GATED + RESEARCH`, so six
modules' self-checks — including `cortex.brain`, `hippocampus.ng_snn` and
`cortex.neurogenesis_engine` — were imported but never RUN in-process. Their `main()` bodies
measured 0/126, 0/55 and 0/57. The whole torch growth tier was therefore reached only as a
side effect of `substrate.brain_telemetry`'s self-check, whose tick budget is data-dependent:
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

from run_selfchecks import _TORCH, CORE, TORCH_GATED, _sweep_latch, latch_in_window

# `_TORCH` is run_selfchecks' own probe, so a broken torch install skips here exactly as it
# skips there, rather than turning into six confusing import errors.
_IN_PROCESS = CORE + (TORCH_GATED if _TORCH else [])


def test_latch_attribution_refuses_the_smear(tmp_path):
    """Drive the classify-and-remove unit (_sweep_latch), where the pre-fix defect lived as
    attribution-by-EXISTENCE (chip task_100cb4d5: batteries named innocent checks for
    straggler latches), against a REAL latch file: an out-of-slot mtime must land in the
    UNATTRIBUTED stray bucket, an in-slot mtime attributes, an absent latch is a no-op, and
    the file is removed either way. Honest scope: run_all's single call into _sweep_latch is
    the one line this does not execute; the predicate's inclusive boundaries are pinned
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


@pytest.mark.parametrize("mod", _IN_PROCESS)
def test_core_selfcheck_in_process(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate any ./data writes per module
    monkeypatch.setattr(sys, "argv", [mod])  # no-arg -> each module's self-check path
    try:
        runpy.run_module(f"snn_data_lake.{mod}", run_name="__main__")
    except SystemExit as e:  # a module may sys.exit(0) on success
        assert e.code in (0, None), f"{mod} self-check exited non-zero: {e.code}"

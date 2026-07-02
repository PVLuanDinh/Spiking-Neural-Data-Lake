"""
Run every CORE module's __main__ self-check in-process under pytest. run_selfchecks.py runs
the same asserts as isolated subprocesses (the zero-dep smoke gate) — that gives no coverage
signal and truncates tracebacks (A1). Running them in-process makes coverage.py see the
product code and surfaces the full traceback on failure. Reuses run_selfchecks.CORE so the
list never drifts from the canonical runner.

Each module is run in a fresh temp cwd (its ./data writes stay isolated) with argv trimmed to
the no-arg self-check path.

Run: pytest tests/test_selfchecks.py
"""

import runpy
import sys

import pytest

from run_selfchecks import CORE


@pytest.mark.parametrize("mod", CORE)
def test_core_selfcheck_in_process(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate any ./data writes per module
    monkeypatch.setattr(sys, "argv", [mod])  # no-arg -> each module's self-check path
    try:
        runpy.run_module(f"snn_data_lake.{mod}", run_name="__main__")
    except SystemExit as e:  # a module may sys.exit(0) on success
        assert e.code in (0, None), f"{mod} self-check exited non-zero: {e.code}"

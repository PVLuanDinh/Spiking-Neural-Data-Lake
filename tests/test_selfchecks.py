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

import runpy
import sys

import pytest

from run_selfchecks import _TORCH, CORE, TORCH_GATED

# `_TORCH` is run_selfchecks' own probe, so a broken torch install skips here exactly as it
# skips there, rather than turning into six confusing import errors.
_IN_PROCESS = CORE + (TORCH_GATED if _TORCH else [])


@pytest.mark.parametrize("mod", _IN_PROCESS)
def test_core_selfcheck_in_process(mod, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # isolate any ./data writes per module
    monkeypatch.setattr(sys, "argv", [mod])  # no-arg -> each module's self-check path
    try:
        runpy.run_module(f"snn_data_lake.{mod}", run_name="__main__")
    except SystemExit as e:  # a module may sys.exit(0) on success
        assert e.code in (0, None), f"{mod} self-check exited non-zero: {e.code}"

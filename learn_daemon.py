"""
SNDL LEARNING DAEMON -- launcher shim. The daemon itself lives in the governed package at
`src/snn_data_lake/engine/learn_daemon.py`.

WHY THIS FILE IS A SHIM (P4c, owner directive 2026-08-04): the daemon is not a launcher that
operates the brain from outside -- it IS the brain's drive, owning the wake/sleep cycle, the
tick loop and the brain's continuity. So it belongs inside the governed package, where the
three P3/P4 gates, the dormancy reasoning and `run_selfchecks.py` all reach it. Until
2026-08-04 it was 370 lines sitting here at the repo root, in no REGION, in neither self-check
list, with no self-check and not one assert: the most load-bearing process in the project and
the least verified file in it.

Moving it would have broken `python learn_daemon.py` -- the command documented in CLAUDE.md,
README.md, docs/RUNNING.md, learn_daemon.bat's restart loop and the PyInstaller build -- so
this shim keeps that entrypoint working unchanged. It deliberately contains NO logic: anything
that lived here would be code outside the gates again, the exact defect P4c names.

  python learn_daemon.py     -> the always-on brain; cockpit at http://127.0.0.1:8137/

The module's own `__main__` is its SELF-CHECK, like every other module in the package (that is
what run_selfchecks.py executes); `--serve` there starts the drive loop, as this shim does.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from snn_data_lake.engine.learn_daemon import main  # noqa: E402

if __name__ == "__main__":
    main()

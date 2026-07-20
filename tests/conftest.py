"""Pytest bootstrap: run the IN-PROCESS suite with the torch tier OFF.

The in-process self-check runner (test_selfchecks.py) does `from run_selfchecks import CORE`, whose
module-level `_torch_ok()` would `import torch` (a DEFAULT dependency now that the torch tier is on
by default) and load torch+numpy into the SHARED pytest process. That trips every core module's
pure-stdlib purity assert (`numpy`/`torch` must not be in sys.modules) -- which is valid only in a
clean process. Setting NORD_TORCH=0 HERE, before any test imports run_selfchecks, short-circuits
`_torch_ok()` BEFORE the torch import, so the pytest process stays numpy/torch-free and the purity
asserts hold.

The torch tier itself is validated separately: run_selfchecks runs each torch-gated module as an
ISOLATED SUBPROCESS with torch ON (per-module purity still holds), and CI adds a torch-self-checks
job. This pytest run measures coverage of the pure-stdlib path. A developer can force the torch tier
on for pytest with `NORD_TORCH=1 pytest` (the strict purity asserts will then trip in-process, by
design). `setdefault` leaves any explicit value untouched.
"""

import os

os.environ.setdefault("NORD_TORCH", "0")

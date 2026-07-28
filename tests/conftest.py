"""Pytest bootstrap: run the IN-PROCESS suite with the torch tier OFF.

RATIONALE CORRECTED 2026-07-28. This docstring used to justify the setting by "every core module's
pure-stdlib purity assert (`numpy`/`torch` must not be in sys.modules)". Those asserts no longer
exist -- all 59 were stripped across 46 files on 2026-07-20 when the zero-dependency rule was
removed project-wide, and `run_selfchecks.py`'s own docstring says so. The setting stayed; only
its stated reason had died, which is the drift that makes a reader trust the next explanation
less.

The setting is KEPT, because two live reasons survive:

  1. COST. `test_selfchecks.py` does `from run_selfchecks import CORE`, whose module-level
     `_torch_ok()` would `import torch` into the shared pytest process -- seconds of import and
     hundreds of MB, paid once per run, for a suite that does not exercise the torch tier.
  2. WHAT THE COVERAGE NUMBER MEANS. With the tier off, `pytest` measures the path that runs
     everywhere, on any machine, with no GPU. That is the number worth gating on.

`NORD_TORCH=0` is still behaviourally live regardless of the purity question: `run_selfchecks`
skips the torch-gated modules under it (275 checks with torch, 267 without), and `brain_atlas`
gates its torch-tier back-compat imports on the same flag.

The torch tier is validated separately -- `run_selfchecks` runs each torch-gated module as an
ISOLATED SUBPROCESS with torch ON, and CI adds a dedicated torch-self-checks job. Force it on for
pytest with `NORD_TORCH=1 pytest`. `setdefault` leaves any explicit value untouched.
"""

import os

os.environ.setdefault("NORD_TORCH", "0")

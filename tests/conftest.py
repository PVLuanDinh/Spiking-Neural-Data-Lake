"""Pytest bootstrap.

EMPTIED 2026-07-29. This file used to do exactly one thing:

    os.environ.setdefault("NORD_TORCH", "0")

which ran the whole in-process suite with the torch tier OFF. Its docstring had already been
corrected once, on 2026-07-28, when the purity asserts it originally cited turned out to have been
stripped on 2026-07-20 with the zero-dependency rule -- the setting outlived its own stated reason
by a week. Now the setting goes too, because the two reasons that replaced it do not hold either:

  1. "COST -- `test_selfchecks.py` does `from run_selfchecks import CORE`, whose module-level
     `_torch_ok()` would import torch into the shared pytest process." True, and now the point.
     torch is an unconditional core dependency; a suite that avoids importing it measures a
     configuration no user has.
  2. "WHAT THE COVERAGE NUMBER MEANS -- with the tier off, pytest measures the path that runs
     everywhere, on any machine, with no GPU." Reads well, and is backwards. It made the coverage
     gate authoritative over a path the product does not ship, and left the torch-gated modules
     measured only as isolated subprocesses. Deterministic-on-any-machine is worth having, but it
     comes from seeding (`random.Random(seed)`, `zlib.crc32`), not from refusing an import.

Consequences, accepted deliberately: pytest is slower (torch import, once per run) and the coverage
percentage moves, because previously-skipped torch-gated code now executes in-process. A moved
coverage number here is the gate becoming honest, not a regression.

Kept as a file rather than deleted so that this reasoning is discoverable at the place the setting
used to be, and so `git log tests/conftest.py` still leads here.
"""

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

RE-POPULATED 2026-07-31 with an isolation fixture, for a defect the in-process suite created and
then hid. `tests/test_selfchecks.py` runs every module's `__main__` in ONE interpreter, and
`substrate/home.py::main()` used to `os.environ.pop("NORD_HOME")` in its `finally` -- dropping the
CALLER's value, not just its own. Every module after it in the list therefore fell back to
`~/.snn_data_lake` and read and WROTE the operator's live brain on every pytest run, while whoever
launched the run believed they had set an isolated home. Worse, `Brain.__init__` restored the grown
substrate from that lake unconditionally, so with a daemon appending to it concurrently, two seed=0
Brains built two seconds apart differed ~42% of the time and `engine.brain`'s claim 4 ("two seed=0
Brains must be byte-identical") failed intermittently -- a real determinism defect that subprocess
isolation had been under-sampling rather than preventing.

`home.py` and `brain.py` are both fixed at the source. This fixture is the belt-and-braces half: the
test suite must be incapable of touching the durable home even if something strips the variable
again.
FUNCTION-scoped on purpose -- a session fixture would be set once and lost the moment any module
mutates the environment mid-suite, which is the exact failure being defended against.
"""

import os
import warnings

import pytest


@pytest.fixture(scope="module", autouse=True)
def _hermetic_module_home(tmp_path_factory):
    """A module's OWN NORD_HOME for module-scoped fixtures, set up BEFORE the function-scoped
    _hermetic_home below. Without it a `scope="module"` fixture that builds a Brain found NORD_HOME
    unset and home() refused (pytest -n, 2026-10-19: 18 interoceptive errors); serially it was
    masked by a NORD_HOME an earlier test leaked into the process. Autouse fixtures run first
    within their scope, so module fixtures see this; _hermetic_home still overrides it per test."""
    mp = pytest.MonkeyPatch()
    mp.setenv("NORD_HOME", str(tmp_path_factory.mktemp("module_home") / "nord_home"))
    yield
    mp.undo()


@pytest.fixture(autouse=True)
def _hermetic_home(tmp_path, monkeypatch):
    """Every test gets its own NORD_HOME. Re-applied per test, never once.

    NORD_HOME only -- deliberately NOT NORD_PERSIST=0. Isolation is what this defect needs, and
    turning persistence off would break the self-checks that exist to PROVE persistence
    (`sleep.nightly_skills` asserts a reload restores competence exactly). Pointed at a private
    directory those still pass, and still cannot reach the operator's lake."""
    monkeypatch.setenv("NORD_HOME", str(tmp_path / "nord_home"))
    os.makedirs(tmp_path / "nord_home", exist_ok=True)


# Restart tests (R2/R3/G9) spawn real child processes and refuse to run without a bounded
# controller deadline. A battery stage that invokes pytest directly never created one, so the
# promoted restart tests died with KeyError in an ordinary run (panel 04febad0, finding C1).
# The deadline is generated ONCE, when the pytest session starts, so it cannot already be
# expired by earlier battery stages. An expired caller value is discarded for the own 6 h bound
# (with a warning); a future one is respected even if it is very near, never loosened, and the
# wait stays bounded (no unlimited child wait is introduced).
STAGE_DEADLINE_S = 6 * 3600


def bounded_stage_deadline(now_ns, caller=None, budget_s=STAGE_DEADLINE_S):
    own = now_ns + int(budget_s) * 10**9
    try:
        caller_ns = int(caller) if caller not in (None, "") else None
    except ValueError:
        caller_ns = None
    if caller_ns is not None and caller_ns <= now_ns:
        warnings.warn(
            f"SNDL_R2_CHILD_DEADLINE_NS={caller_ns} is already expired; discarded for the own "
            f"{int(budget_s) // 3600} h stage bound",
            stacklevel=2,
        )
        return str(own)
    if caller_ns is None:
        return str(own)
    return str(min(own, caller_ns))


def pytest_configure(config):
    import time

    os.environ["SNDL_R2_CHILD_DEADLINE_NS"] = bounded_stage_deadline(
        time.time_ns(), os.environ.get("SNDL_R2_CHILD_DEADLINE_NS")
    )


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    """Under `-n --dist loadgroup`, keep every test that uses a module-scoped fixture on ONE worker.

    xdist otherwise spreads a module's tests across workers and each worker rebuilds that module's
    `scope="module"` fixture (carried_state's driver setup alone is minutes). Serial runs ignore the
    mark. A test that already carries an explicit `xdist_group` keeps its own group."""
    for item in items:
        if item.get_closest_marker("xdist_group") is not None:
            continue
        defs = getattr(getattr(item, "_fixtureinfo", None), "name2fixturedefs", {})
        if any(fd[-1].scope == "module" for fd in defs.values()):
            item.add_marker(pytest.mark.xdist_group(item.module.__name__))

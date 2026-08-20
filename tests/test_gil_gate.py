"""The GIL hard-assert (specs/022 T016, FR-006) -- reds and greens through REAL interpreters.

The guard is ONE mechanism with two tiers (snn_data_lake.require_free_threaded): plain import
keeps the narrow NORD_ALLOW_GIL escape; the brain's construction path and the daemon boot are
STRICT and ignore it. Every arm here spawns a genuine subprocess under PYTHON_GIL=0/1 on the
free-threaded build -- the runtime dimension the pre-T016 guard could not see (its build-flag
check reads Py_GIL_DISABLED=1 even while PYTHON_GIL=1 runs the GIL).
"""

import os
import subprocess
import sys
import sysconfig
import tempfile

import pytest

pytestmark = pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="needs the free-threaded build to toggle the GIL at runtime",
)


def _run(code, gil, allow_escape=False, timeout=600):
    env = dict(os.environ)
    env["PYTHON_GIL"] = str(gil)
    env["NORD_HOME"] = tempfile.mkdtemp(prefix="gil_gate_")
    if allow_escape:
        env["NORD_ALLOW_GIL"] = "1"
    else:
        env.pop("NORD_ALLOW_GIL", None)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout
    )


def test_strict_red_ignores_the_escape():
    """The T016 differentiator: strict refuses under PYTHON_GIL=1 even WITH NORD_ALLOW_GIL=1."""
    proc = _run(
        "from snn_data_lake import require_free_threaded;"
        "require_free_threaded('red-arm', strict=True)",
        gil=1,
        allow_escape=True,
    )
    assert proc.returncode != 0, "strict tier passed a GIL-enabled interpreter"
    assert (
        "refuses" not in proc.stdout
        and "FR-006" in (proc.stderr + proc.stdout)
        or ("GIL" in proc.stderr)
    ), f"the refusal message is missing:\n{proc.stderr[-500:]}"


def test_construction_red_under_gil():
    """FR-006 verbatim: brain CONSTRUCTION under a GIL-enabled interpreter fails, before any
    state is touched. The escape is set on purpose -- it must not help construction."""
    proc = _run(
        "from snn_data_lake.cortex.brain import make_brain;make_brain(seed=0, persist=False)",
        gil=1,
        allow_escape=True,
    )
    assert proc.returncode != 0, "make_brain constructed under a GIL-enabled interpreter"
    assert "make_brain" in proc.stderr and "GIL" in proc.stderr, (
        f"construction died for some other reason:\n{proc.stderr[-500:]}"
    )


def test_import_red_without_escape():
    """The import tier still refuses a runtime-GIL'd interpreter when no escape is set."""
    proc = _run("import snn_data_lake", gil=1, allow_escape=False)
    assert proc.returncode != 0, "package import passed a runtime-GIL'd interpreter"
    assert "GIL" in proc.stderr


def test_import_escape_still_works_for_bystanders():
    """The contributor path survives: NORD_ALLOW_GIL=1 downgrades the IMPORT refusal to a
    warning (and only the import -- the strict arms above prove construction stays closed)."""
    proc = _run("import snn_data_lake; print('imported')", gil=1, allow_escape=True)
    assert proc.returncode == 0, f"the escape hatch broke:\n{proc.stderr[-500:]}"
    assert "imported" in proc.stdout
    assert "WARNING" in proc.stderr, "the escape must warn loudly, never pass silently"


def test_green_control():
    """PYTHON_GIL=0: strict passes and the message machinery stays out of the way -- the
    control that separates 'the gate fires on the GIL' from 'the gate fires on everything'."""
    proc = _run(
        "from snn_data_lake import require_free_threaded;"
        "require_free_threaded('green-arm', strict=True);"
        "print('green')",
        gil=0,
    )
    assert proc.returncode == 0 and "green" in proc.stdout, (
        f"the green control failed -- the gate reds unconditionally:\n{proc.stderr[-500:]}"
    )

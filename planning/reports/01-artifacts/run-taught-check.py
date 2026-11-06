"""Bounded stage-01 source self-check; run with the workspace's .venv-01 Python."""

import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile


def main():
    root = Path(__file__).resolve().parents[3]
    repo = root / "repo"
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("Use the workspace's CPython 3.14t interpreter")
    if sysconfig.get_config_var("Py_GIL_DISABLED") != 1 or sys._is_gil_enabled():
        raise RuntimeError("The free-threaded build with the GIL off is required")

    child_env = os.environ.copy()
    for key in ("NORD_ALLOW_GIL", "PYTHON_GIL"):
        child_env.pop(key, None)
    child_env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUTF8="1",
        PYTHONPATH=str(repo / "src"),
    )
    child_code = (
        "import runpy, sys; "
        "runpy.run_module('snn_data_lake.instruments.taught_problems', run_name='__main__'); "
        "print('GIL enabled after self-check:', sys._is_gil_enabled()); "
        "assert sys._is_gil_enabled() is False"
    )
    with tempfile.TemporaryDirectory(prefix="sndl-01-check-") as task_home:
        child_env["NORD_HOME"] = task_home
        result = subprocess.run(
            [sys.executable, "-B", "-c", child_code],
            cwd=repo,
            env=child_env,
            timeout=60,
            check=True,
        )
        home_empty = not any(Path(task_home).iterdir())
        print("isolated_home_empty:", home_empty, flush=True)
        if not home_empty:
            raise RuntimeError("Unexpected writes in the isolated diagnostic home")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())

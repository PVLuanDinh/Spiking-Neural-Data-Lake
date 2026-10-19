"""Collection-time tool imports must preserve the pytest caller's environment."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("home_present", [False, True])
@pytest.mark.parametrize("persist", [None, "1"])
def test_lang_coverage_collection_preserves_environment(tmp_path, home_present, persist):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ("NORD_HOME", "NORD_PERSIST"):
        env.pop(key, None)
    if home_present:
        env["NORD_HOME"] = str(tmp_path / "caller_home")
    if persist is not None:
        env["NORD_PERSIST"] = persist
    env["TMP"] = env["TEMP"] = str(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, runpy; "
            "keys = ('NORD_HOME', 'NORD_PERSIST'); "
            "before = {key: os.environ.get(key) for key in keys}; "
            "runpy.run_path('tests/test_lang_coverage.py'); "
            "after = {key: os.environ.get(key) for key in keys}; "
            "assert after == before, (before, after)",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr

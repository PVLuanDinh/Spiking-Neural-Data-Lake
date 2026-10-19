"""H1 (2026-09-02): the tools-gate stage of run_selfchecks has a reachable red.

`run_selfchecks.run_tools_gates` runs each tools/ gate as a subprocess and returns the failed
names. A gate that exits non-zero must come back failed (the red); one that exits zero must not
(the positive control), so a stage that swallowed exit codes could not pass this file.
"""

import os
import pathlib

import pytest

import run_selfchecks as rs

ROOT = pathlib.Path(rs.__file__).resolve().parent


def test_tools_stage_red_and_control(tmp_path):
    red = tmp_path / "red_gate.py"
    red.write_text("raise SystemExit(1)\n", encoding="utf-8")
    green = tmp_path / "green_gate.py"
    green.write_text("print('ok')\n", encoding="utf-8")
    lines = []
    failed = rs.run_tools_gates([[str(red)], [str(green), "--assert"]], emit=lines.append)
    assert failed == ["tools/red_gate.py"], (failed, lines)
    assert any(line.startswith("FAIL  tools/red_gate.py") for line in lines), lines
    assert any(line.startswith("PASS  tools/green_gate.py") for line in lines), lines
    assert "1/2 tools gates passed" in lines[-1], lines


def test_default_gate_list_names_the_six_tools_gates():
    """Six tools, seven invocations: red_reachability runs --assert and (R0.2b) --self-check."""
    names = [argv[0] for argv in rs.TOOLS_GATES]
    assert names == [
        "tools/p4c_ratchet.py",
        "tools/p6_gate.py",
        "tools/dependency_ledger.py",
        "tools/red_reachability.py",
        "tools/red_reachability.py",
        "tools/citation_guard.py",
        "tools/doc_command_guard.py",
    ]
    for name in names:
        assert (ROOT / name).exists(), name


def test_battery_partition_runs_every_module_exactly_once():
    # Owner 2026-10-19 option A + Codex closing report MAJOR 1: the battery's in-process run must
    # not drop RESEARCH / NENGO_GATED / TRAIN_GATED -- they go to the subprocess half.
    import run_selfchecks as rs

    in_proc, rest = rs.battery_partition()
    assert set(in_proc).isdisjoint(rest)
    assert sorted(in_proc + rest) == sorted(rs.MODULES)
    assert set(rs.RESEARCH) <= set(rest) and rs.RESEARCH  # red if the subprocess half is dropped


def test_in_process_exit_code_matches_a_real_interpreter(monkeypatch):
    # Codex re-review MAJOR 1: SystemExit(0.0) exits 1 in a real interpreter, so it must FAIL here.
    import runpy

    import run_selfchecks as rs

    for code, want in (
        (None, True),
        (0, True),
        (False, True),
        (0.0, False),
        (1, False),
        ("x", False),
    ):

        def boom(*a, code=code, **k):
            raise SystemExit(code)

        monkeypatch.setattr(runpy, "run_module", boom)
        assert rs.run_one_in_process("brain_atlas")[0] is want, code


def test_in_process_run_arms_the_live_home_tripwire(monkeypatch, tmp_path):
    # Codex re-review MAJOR 2: a check that pops NORD_HOME is refused, never handed the live lake.
    import run_selfchecks as rs
    from snn_data_lake.substrate import home as H

    monkeypatch.setattr(rs, "_HOME", str(tmp_path))  # never the runner's shared home
    seen = []

    def probe(mod, home=None):
        seen.append(os.environ.get("SNDL_LIVE_HOME_GUARD"))
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.delenv("NORD_HOME", raising=False)
        with pytest.raises(H.LiveHomeRefused):
            H.home()
        return True, ""

    monkeypatch.setattr(rs, "run_one_in_process", probe)
    monkeypatch.setattr(rs, "CORE", ["brain_atlas"])
    monkeypatch.setattr(rs, "TORCH_GATED", [])
    assert rs.run_all_in_process(emit=lambda *_: None)
    assert seen == ["1"]  # red if the runner stops arming SNDL_LIVE_HOME_GUARD
    assert os.environ.get("SNDL_LIVE_HOME_GUARD") is None  # restored after the run


def test_each_runner_process_gets_its_own_home_and_ignores_any_pin(tmp_path):
    # Codex MAJOR 1: concurrent runners never share the E-STOP latch's home (red on the old fixed
    # %TEMP%/snn_selfchecks_home: both processes printed the same path). Opus MAJOR 1: no env var
    # can point the runner at another home -- a pin at the live lake would let the runner os.remove
    # the operator's real latch (red on the SNDL_SELFCHECK_HOME version: the pin was honoured).
    import subprocess
    import sys

    fake_live = tmp_path / "fake live" / ".snn_data_lake"
    fake_live.mkdir(parents=True)
    latch = fake_live / "estop.latch"
    latch.write_text("planted")
    probe = "import run_selfchecks as r, os; print(r._HOME); print(os.path.isdir(r._HOME))"
    env = {**os.environ, "SNDL_SELFCHECK_HOME": str(fake_live)}

    def home():
        out = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout.splitlines()
        return out[0], out[1]

    (a, a_live), (b, _) = home(), home()
    assert a != b and a_live == "True"
    assert os.path.normcase(a) != os.path.normcase(str(fake_live)), "the pin must be ignored"
    assert not os.path.isdir(a), "a home the process made must be removed at its exit"
    assert latch.read_text() == "planted"


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to delete a file held open")
def test_home_cleanup_reports_what_it_could_not_remove(tmp_path, capsys):
    # Opus MINOR 3: a straggler holding a handle used to leak the home silently
    d = tmp_path / "home"
    d.mkdir()
    with open(d / "held.bin", "wb") as held:
        held.write(b"x")
        rs._remove_home(str(d))
    assert "could not remove" in capsys.readouterr().err

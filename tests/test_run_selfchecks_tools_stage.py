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


# --- b3-selfchecks (2026-10-20): the guard reaches children, is re-armed per check, CPython's exit
# rule, the latch survives, cleanup reports, the timing-deferral partition. Each test names its red.

_PROBE_POP_NORD_HOME = (
    "import os, sys\n"
    "os.environ.pop('NORD_HOME', None)\n"
    "from snn_data_lake.substrate import home as H\n"
    "try:\n"
    "    H.home()\n"
    "except H.LiveHomeRefused:\n"
    "    print('REFUSED')\n"
    "    sys.exit(0)\n"
    "print('REACHED', H.home())\n"
    "sys.exit(3)\n"
)


def _fake_user(tmp_path, monkeypatch):
    """A fake USERPROFILE/HOME; its ~/.snn_data_lake (the live lake) holds a planted latch."""
    fake = tmp_path / "fake user"
    live = fake / ".snn_data_lake"
    live.mkdir(parents=True)
    (live / "estop.latch").write_text("planted")
    for key in ("USERPROFILE", "HOME"):
        monkeypatch.setenv(key, str(fake))
        monkeypatch.setitem(rs._ENV, key, str(fake))  # children are built from _ENV, not os.environ
    assert os.path.expanduser("~") == str(fake)  # the fake IS what home() resolves as live
    return fake, live


def _probe_env(monkeypatch):
    """rs._ENV as every child gets it, importable from any cwd, outside a pytest phase."""
    monkeypatch.setitem(rs._ENV, "PYTHONPATH", str(ROOT / "src"))
    monkeypatch.delitem(rs._ENV, "PYTEST_CURRENT_TEST", raising=False)
    return rs._ENV


def test_subprocess_env_arms_the_live_home_tripwire(tmp_path, monkeypatch):
    # bfix3 MAJOR 1. RED on the old _ENV (no SNDL_LIVE_HOME_GUARD): the probe pops NORD_HOME, home()
    # falls back to the fake live home and the probe prints REACHED / exits 3.
    import subprocess
    import sys

    _fake_user(tmp_path, monkeypatch)
    env = _probe_env(monkeypatch)
    r = subprocess.run(
        [sys.executable, "-c", _PROBE_POP_NORD_HOME],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert (r.returncode, r.stdout.strip()) == (0, "REFUSED"), (r.stdout, r.stderr[-500:])


def test_tools_gates_run_with_the_live_home_tripwire(tmp_path, monkeypatch):
    # bfix3 MAJOR 1, the TOOLS_GATES half. RED on the old _ENV: the gate reaches the fake live home.
    _fake_user(tmp_path, monkeypatch)
    _probe_env(monkeypatch)
    gate = tmp_path / "pop_home_gate.py"
    gate.write_text(_PROBE_POP_NORD_HOME, encoding="utf-8")
    lines = []
    assert rs.run_tools_gates([[str(gate)]], emit=lines.append) == [], lines


def test_run_all_hands_every_module_the_guarded_env(monkeypatch):
    # bfix3 MAJOR 1, the subprocess-module half: run_all passes env=_ENV, the guard is in it.
    import subprocess

    seen = []

    def fake_run(cmd, **kw):
        seen.append(kw.get("env"))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(rs.subprocess, "run", fake_run)
    monkeypatch.setattr(rs, "run_tools_gates", lambda *a, **k: [])
    assert rs.run_all(modules=["brain_atlas"])
    assert len(seen) == 1 and seen[0]["SNDL_LIVE_HOME_GUARD"] == "1", seen
    assert seen[0]["NORD_HOME"] == rs._HOME


def test_each_in_process_check_is_rearmed(monkeypatch, tmp_path):
    # bfix3 MAJOR 2. RED on the old loop (armed once before the loop): check 1 pops the guard and
    # NORD_HOME and checks 2 and 3 then start with neither.
    monkeypatch.setattr(rs, "_HOME", str(tmp_path))
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    seen = []

    def disarming_check(mod, home=None):
        seen.append((os.environ.get("SNDL_LIVE_HOME_GUARD"), os.environ.get("NORD_HOME")))
        os.environ.pop("SNDL_LIVE_HOME_GUARD", None)
        os.environ.pop("NORD_HOME", None)
        return True, ""

    monkeypatch.setattr(rs, "run_one_in_process", disarming_check)
    monkeypatch.setattr(rs, "CORE", ["a", "b", "c"])
    monkeypatch.setattr(rs, "TORCH_GATED", [])
    assert rs.run_all_in_process(emit=lambda *_: None)
    assert seen == [("1", str(tmp_path))] * 3, seen

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


def test_in_process_exit_code_follows_cpython_for_int_subclasses(monkeypatch):
    # bfix3 MINOR 4: the expected value is what a REAL interpreter exits with, not our guess of it.
    # RED on `type(code) in (int, bool)`: an IntEnum zero exits 0 in CPython and failed here.
    import runpy
    import subprocess
    import sys

    prelude = "import enum\nimport numpy as np\nclass Z(enum.IntEnum):\n    ZERO = 0\n    ONE = 1\n"
    exprs = ["None", "0", "1", "False", "True", "0.0", "'x'", "Z.ZERO", "Z.ONE", "np.int64(0)"]
    real = {}
    for e in exprs:
        r = subprocess.run(
            [sys.executable, "-c", f"{prelude}raise SystemExit({e})"],
            capture_output=True,
            timeout=60,
        )
        real[e] = r.returncode == 0
    assert real["Z.ZERO"] is True and real["True"] is False and real["np.int64(0)"] is False

    import enum

    import numpy as np

    class Z(enum.IntEnum):
        ZERO = 0
        ONE = 1

    values = {"Z.ZERO": Z.ZERO, "Z.ONE": Z.ONE, "np.int64(0)": np.int64(0)}
    values |= {e: eval(e) for e in exprs if e not in values}
    for e in exprs:

        def boom(*a, code=values[e], **k):
            raise SystemExit(code)

        monkeypatch.setattr(runpy, "run_module", boom)
        assert rs.run_one_in_process("brain_atlas")[0] is real[e], e


def test_in_process_run_never_clears_a_latch_outside_its_own_home(monkeypatch, tmp_path):
    # Opus final M1. CORE/TORCH patched empty, so nothing can write a latch: whatever the pre-clear
    # removes is the pre-clear's own doing. RED on a runner that pre-clears a pinned/live path
    # (latch = ~/.snn_data_lake/estop.latch): the planted latch is gone.
    from snn_data_lake.brainstem.so101_arm import ESTOP_LATCH

    _, live = _fake_user(tmp_path, monkeypatch)
    planted = live / ESTOP_LATCH
    mine = tmp_path / "runner home"
    mine.mkdir()
    stale = mine / ESTOP_LATCH
    stale.write_text("stale")
    monkeypatch.setattr(rs, "_HOME", str(mine))
    monkeypatch.setattr(rs, "CORE", [])
    monkeypatch.setattr(rs, "TORCH_GATED", [])
    lines = []
    assert rs.run_all_in_process(emit=lines.append)
    assert planted.exists() and planted.read_text() == "planted", lines
    # positive control: the sweep is live (a vacuous run would leave BOTH latches alone)
    assert not stale.exists(), lines
    assert any("from a previous run cleared" in line for line in lines), lines


def _rmtree_leaves_one(monkeypatch):
    """shutil.rmtree that really removes `path`, then reports one undeletable leftover via onexc."""
    import shutil

    real = shutil.rmtree

    def fake(path, onexc=None, **kw):
        real(path)
        onexc(os.unlink, os.path.join(path, "held.bin"), PermissionError("held"))

    monkeypatch.setattr(shutil, "rmtree", fake)


def test_home_cleanup_is_silent_when_the_home_is_already_gone(tmp_path, capsys):
    # Opus final NOTE. RED on the old onexc: FileNotFoundError was reported as a leftover.
    rs._remove_home(str(tmp_path / "already gone"))
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("stderr", ["closed", None])
def test_home_cleanup_reports_when_stderr_is_closed(tmp_path, capfd, monkeypatch, stderr):
    # Opus final NOTE. RED on the old sys.stderr.write: ValueError/AttributeError, no report.
    import io
    import sys

    d = tmp_path / "home"
    d.mkdir()
    _rmtree_leaves_one(monkeypatch)
    dead = io.StringIO()
    dead.close()
    monkeypatch.setattr(sys, "stderr", dead if stderr == "closed" else None)
    rs._remove_home(str(d))
    assert "could not remove" in capfd.readouterr().err


def test_home_cleanup_falls_back_to_a_file_when_fd2_is_dead_too(tmp_path, monkeypatch):
    import io
    import sys

    d = tmp_path / "home"
    d.mkdir()
    _rmtree_leaves_one(monkeypatch)
    dead = io.StringIO()
    dead.close()
    monkeypatch.setattr(sys, "stderr", dead)

    def no_fd2(*a, **k):
        raise OSError("fd 2 closed")

    monkeypatch.setattr(os, "write", no_fd2)
    rs._remove_home(str(d))
    monkeypatch.undo()  # os.write back before reading
    assert "could not remove" in (tmp_path / "snn_selfchecks_unremoved.log").read_text()


def test_in_process_scratch_cleanup_reports_leftovers(monkeypatch):
    # Opus final NOTE. RED on rmtree(ignore_errors=True): a scratch dir that would not go is silent.
    _rmtree_leaves_one(monkeypatch)
    monkeypatch.setattr(rs, "run_one_in_process", lambda mod, home=None: (True, ""))
    monkeypatch.setattr(rs, "CORE", ["brain_atlas"])
    monkeypatch.setattr(rs, "TORCH_GATED", [])
    lines = []
    assert rs.run_all_in_process(emit=lines.append)
    assert any("could not remove" in line and "brain_atlas" in line for line in lines), lines


def test_reaper_removes_only_old_selfcheck_homes(tmp_path):
    # Opus final NOTE. Never younger than 24 h (a concurrent runner), never another name, a plain
    # file, or a symlink. RED on a reaper with no age floor (young dir gone) or none (old stays).
    import time

    now = time.time()

    def mk(name, age_h, kind="dir"):
        p = tmp_path / name
        if kind == "dir":
            p.mkdir()
            (p / "estop.latch").write_text("x")
        else:
            p.write_text("x")
        os.utime(p, (now - age_h * 3600, now - age_h * 3600))
        return p

    old = mk("snn_selfchecks_home_old", 25)
    young = mk("snn_selfchecks_home_young", 23)
    other = mk("other_dir_old", 48)
    afile = mk("snn_selfchecks_home_file", 48, kind="file")
    reaped = rs._reap_stale_homes(str(tmp_path), now=now)
    assert reaped == [str(old)]
    assert not old.exists() and young.exists() and other.exists() and afile.exists()
    assert rs._reap_stale_homes(str(tmp_path / "missing"), now=now) == []


def test_timing_deferral_partitions_the_battery_exactly():
    # The interface lane b3-battery codes against: `--battery --defer-timing` + `--timing-only` ==
    # `--battery`. RED if a module is dropped from either run or run in both.
    assert rs.TIMING_DEFERRED == ["gates.amdahl_gate"]
    full_in, full_rest = rs.battery_partition()
    d_in, d_rest = rs.battery_partition(defer_timing=True)
    t_in, t_rest = rs.battery_partition(timing_only=True)
    assert t_in == rs.TIMING_DEFERRED and t_rest == []
    assert set(d_in).isdisjoint(t_in) and d_rest == full_rest
    assert sorted(d_in + d_rest + t_in + t_rest) == sorted(full_in + full_rest)
    assert sorted(full_in + full_rest) == sorted(rs.MODULES)
    assert not set(rs.TIMING_DEFERRED) & set(d_in + d_rest)


def test_timing_deferral_partition_refuses_a_module_it_would_drop(monkeypatch):
    # a TIMING_DEFERRED entry that is not in-process would silently vanish from --timing-only
    monkeypatch.setattr(rs, "TIMING_DEFERRED", ["research.nothing"])
    with pytest.raises(ValueError, match="not in the in-process set"):
        rs.battery_partition(timing_only=True)


def test_cli_timing_flags_run_exactly_the_battery(monkeypatch):
    # main() wiring: the two flagged runs together execute --battery's modules once each, and
    # --timing-only runs in-process only (no subprocess half, no tools gates).
    monkeypatch.setenv("NORD_HOME", rs._HOME)
    calls = []
    monkeypatch.setattr(
        rs,
        "run_all_in_process",
        lambda emit=print, modules=None: calls.append(("in", list(modules))) or True,
    )
    monkeypatch.setattr(
        rs,
        "run_all",
        lambda carry=False, modules=None: calls.append(("sub", list(modules))) or True,
    )

    def run(*argv):
        calls.clear()
        assert rs.main(["run_selfchecks.py", *argv]) == 0
        return [(k, m) for k, mods in calls for m in mods]

    battery = run("--battery")
    deferred = run("--battery", "--defer-timing")
    timing = run("--timing-only")
    assert sorted(deferred + timing) == sorted(battery)
    assert all(k == "in" for k, _ in timing) and [m for _, m in timing] == rs.TIMING_DEFERRED
    assert ("in", "gates.amdahl_gate") in battery and ("in", "gates.amdahl_gate") not in deferred
    assert rs.main(["run_selfchecks.py", "--defer-timing"]) == 2
    assert rs.main(["run_selfchecks.py", "--timing-only", "--defer-timing"]) == 2

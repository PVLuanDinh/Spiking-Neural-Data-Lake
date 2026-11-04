"""P4b -- the signal-calibration state goes IN the one lake, and a restart is not amnesia.

This suite exists because the defect it guards against is SILENT by construction. All three consumer
sites in `engine/brain.py` sit inside a bare `except Exception`, and the load handler additionally
NULLS both calibrators. So a persistence path that stops working does not raise, does not log, and
does not move a dashboard: `n` simply resets to 0 on every boot, the live `r["n"] >= 50` guard in
`Brain._tick_ensemble` then never fires, and `self._lc.salience_th` is pinned to the LocusCoeruleus
module default forever while every gate stays green. There was ZERO test coverage of
`SignalCalibrator`, `signal_calib` or `salience_th` in this tree before 2026-08-05.

Every assertion here reads the state back through a path the writer does not own, and the two that
matter most read it back in a process the writer is not running in.

Run: pytest tests/test_signal_calibrator_lake.py
"""

import importlib
import json
import os
import subprocess
import sys

import pytest

SRC_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _fresh_home(tmp_path, monkeypatch):
    """Point NORD_HOME at `tmp_path` and hand back a home module with an empty BinLog cache."""
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    from snn_data_lake.substrate import home as H

    importlib.reload(H)
    return H


def _sidecar_blob(name, p, values):
    """The PRE-P4b on-disk format: `json.dump({name: P2 state}, indent=2, sort_keys=True)`."""
    from snn_data_lake.substrate.signal_calibrator import SignalCalibrator

    c = SignalCalibrator(name, p=p)
    for v in values:
        c.observe(v)
    return json.dumps({c.name: c.snapshot()}, indent=2, sort_keys=True), c.result()


def _spread(n):
    """A stream with real spread in [0.30, 0.70] -- P75 lands near 0.60, comfortably > 0."""
    return [0.30 + 0.40 * ((i % 5) / 4.0) for i in range(n)]


# --------------------------------------------------------------------------- the carrier ----------
def test_state_lands_in_the_lake_and_no_sidecar_is_created(tmp_path, monkeypatch):
    """P4b: what the brain learns goes IN the lake under a record tag, not in a JSON blob beside it.

    THIS is the assertion that discriminates the CARRIER, and it has to be the absence of the file
    rather than the presence of a value, for a reason found by running the red rather than by
    reasoning about it: `home._migrate_sidecar` imports `<home>/signal_calib.json` on first read, so
    a `save_all` that regressed to writing the sidecar is still READ BACK correctly the first time
    and every value-shaped assertion downstream stays green. What that world actually looks like is
    worse than a clean break -- the migration copies the sidecar into the lake once, the lake then
    wins forever, and `close()` goes on writing a file nobody reads again.

    RED, run 2026-08-05: revert `save_all` to `json.dump` at `state_path(_STATE_NAME)`. Fires on the
    second assert -- "a sidecar file was written beside brain_lake.bin".
    """
    H = _fresh_home(tmp_path, monkeypatch)
    from snn_data_lake.substrate.signal_calibrator import SignalCalibrator

    c = SignalCalibrator("salience_th", p=0.75)
    for v in _spread(200):
        c.observe(v)
    assert SignalCalibrator.save_all([c]) is True, "the first save of a non-empty state must land"

    assert H.load_named_state("signal_calib.json") is not None, (
        "the calibration state is not in the one lake -- this is the P4b defect itself"
    )
    assert not os.path.exists(os.path.join(str(tmp_path), "signal_calib.json")), (
        "a sidecar file was written beside brain_lake.bin"
    )


def test_an_unchanged_state_is_not_re_appended(tmp_path, monkeypatch):
    """The lake is append-only and consolidate() keeps every non-FACT record forever, so saving a
    byte-identical blob on every shutdown must add nothing.

    RED: drop the unchanged-blob check in `home.save_named_state` and the second save returns True.
    """
    _fresh_home(tmp_path, monkeypatch)
    from snn_data_lake.substrate.signal_calibrator import SignalCalibrator

    c = SignalCalibrator("salience_th", p=0.75)
    for v in _spread(200):
        c.observe(v)
    assert SignalCalibrator.save_all([c]) is True
    assert SignalCalibrator.save_all([c]) is False, "an unchanged state grew the one lake"


# --------------------------------------------------------------------- the cross-process proof ----
@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
def test_written_in_one_process_read_in_a_separate_one(tmp_path, monkeypatch):
    """P4b's honesty guard: "it persists" is a claim about a NEW process.

    Neither the writer nor the reader is this pytest process. A child writes and EXITS; a second,
    independent child reads the value back out of the lake and asserts it. An in-process round-trip
    would pass against a purely in-memory store and prove nothing.

    The writer deliberately does NOT assert that its save landed -- it only prints the value it
    recorded. Putting the assertion in the writer would let it, not the reader, be the thing that
    fails, and the claim under test is about the READER's process.

    RED, run 2026-08-05: make `save_all` a no-op (`return False`, nothing written) and the reader
    child exits 1 on "the lake held no calibration state in a NEW process".
    """
    monkeypatch.setenv("NORD_HOME", str(tmp_path))
    env = dict(os.environ, NORD_HOME=str(tmp_path), PYTHONPATH=SRC_ROOT)

    writer = (
        "from snn_data_lake.substrate.signal_calibrator import SignalCalibrator\n"
        "c = SignalCalibrator('salience_th', p=0.75)\n"
        "for i in range(200):\n"
        "    c.observe(0.30 + 0.40 * ((i % 5) / 4.0))\n"
        "SignalCalibrator.save_all([c])\n"
        "print(c.result()['value'])\n"
    )
    w = subprocess.run([sys.executable, "-c", writer], env=env, capture_output=True, text=True)
    assert w.returncode == 0, f"writer process failed: {w.stderr}"
    written_value = float(w.stdout.strip().splitlines()[-1])

    reader = (
        "from snn_data_lake.substrate.signal_calibrator import SignalCalibrator\n"
        "r = SignalCalibrator.load_result('salience_th')\n"
        "assert r is not None, 'the lake held no calibration state in a NEW process'\n"
        "assert r['n'] == 200, r\n"
        f"assert abs(r['value'] - {written_value!r}) < 1e-9, r\n"
        "print('READBACK', r['value'], r['n'])\n"
    )
    rd = subprocess.run([sys.executable, "-c", reader], env=env, capture_output=True, text=True)
    assert rd.returncode == 0, f"reader process failed: {rd.stderr}"
    assert "READBACK" in rd.stdout, rd.stdout


# ------------------------------------------------------------------------ the legacy migration ----
@pytest.mark.pool  # D-176: spawns a brain child python (>= 1.5 GB); never beside xdist
def test_a_pre_p4b_sidecar_migrates_and_is_kept_renamed(tmp_path, monkeypatch):
    """An operator's already-recorded operating point must survive the policy fix, not be reset by
    it.

    RED: delete the `_migrate_sidecar(name) if path is None else None` branch in
    `home.load_named_state` and the first assert fires with None -- a cold start beside a full
    sidecar.
    """
    H = _fresh_home(tmp_path, monkeypatch)
    from snn_data_lake.substrate.signal_calibrator import SignalCalibrator

    legacy = os.path.join(str(tmp_path), "signal_calib.json")
    text, expected = _sidecar_blob("salience_th", 0.75, _spread(200))
    with open(legacy, "w", encoding="utf-8") as f:
        f.write(text)

    got = SignalCalibrator.load_result("salience_th")
    assert got is not None, "an existing sidecar was reported as a cold start"
    assert got["n"] == expected["n"] and abs(got["value"] - expected["value"]) < 1e-9, got

    assert H.load_named_state("signal_calib.json") is not None, (
        "the migration returned the blob but never put it IN the lake"
    )
    # kept, never deleted -- renamed so a lake reset cannot import it again (arc-2 panel A-m1)
    assert os.path.exists(legacy + ".imported") and not os.path.exists(legacy)

    # ...and the imported state is durable in its own right: a SEPARATE process reads it back from
    # the lake with the sidecar renamed away, so nothing can be re-reading the file.
    env = dict(os.environ, NORD_HOME=str(tmp_path), PYTHONPATH=SRC_ROOT)
    child = (
        "from snn_data_lake.substrate.signal_calibrator import SignalCalibrator\n"
        "r = SignalCalibrator.load_result('salience_th')\n"
        "assert r is not None and r['n'] == 200, r\n"
    )
    rc = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True)
    assert rc.returncode == 0, f"a migrated state was not readable in a NEW process: {rc.stderr}"


# ------------------------------------------------------- the behaviour, which is what matters ----
def test_a_restarted_brain_recovers_its_salience_threshold(tmp_path, monkeypatch):
    """THE assertion this whole change exists for: a second boot recovers `salience_th` instead of
    cold-starting at the LocusCoeruleus module default.

    The samples are fed to the brain's OWN calibrator directly rather than produced by ticking, and
    that is a deliberate honesty limit rather than a shortcut. MEASURED 2026-08-05 on a seed=0
    persistent brain: 300 ticks record n=146 salience samples (only wake ticks reach
    `_tick_ensemble`) and their P75 is exactly 0.0 -- `salience = max(0.0, threat)` is all-zero in
    today's dormant-arm regime, so `salience_th` stays 0.5. A tick-driven version of this test would
    therefore be measuring the arm, not the persistence path, and would be permanently vacuous.
    What IS under test here is the round trip and the live consumer -- both real, both on the real
    Brain, neither simulated.

    RED, run 2026-08-05, and it is the exact half-applied state this change was warned about:
    leave `Brain.close()` calling `SignalCalibrator.save_all(state_path("signal_calib.json"), [..])`
    -- the OLD argument order against the NEW signature. `calibrators` is then a string, `c.name`
    raises AttributeError, `close()`'s bare `except Exception: pass` swallows it, and NOTHING is
    persisted. Fires with restored n=0 against a lake holding n=200. Note that no error is printed
    and no other test moves: that silence is the whole reason this file exists.
    """
    _fresh_home(tmp_path, monkeypatch)
    from snn_data_lake.engine.brain import make_brain

    first = make_brain(seed=0, persist=True)
    if first._salience_calib is None or first._lc is None:
        first.close()
        pytest.skip("this brain has no salience calibrator / LC to restore")
    default_th = float(first._lc.salience_th)
    for v in _spread(200):
        first._salience_calib.observe(v)
    recorded = first._salience_calib.result()
    first.close()

    assert recorded["n"] == 200 and recorded["value"] > 0.0, recorded

    second = make_brain(seed=0, persist=True)
    restored = second._salience_calib.result()
    assert restored["n"] == recorded["n"], (
        f"a restarted brain booted with n={restored['n']} beside a lake holding "
        f"n={recorded['n']} -- P4b failure, and the silent one"
    )
    assert abs(restored["value"] - recorded["value"]) < 1e-9, restored

    # ...and the LIVE consumer takes it: a wake tick moves the LC gate off its frozen default.
    for _ in range(8):
        second.tick()
    live_th = float(second._lc.salience_th)
    second.close()
    assert live_th != default_th, (
        f"the restored quantile never reached the LC gate: salience_th still {live_th}"
    )
    assert abs(live_th - recorded["value"]) < 0.05, (
        f"the LC gate took {live_th}, not the restored quantile {recorded['value']}"
    )


def test_a_brain_with_no_stored_state_holds_the_default(tmp_path, monkeypatch):
    """The CONTROL for the test above, and the shape of the world when the save silently stops.

    With nothing in the lake the calibrator starts at n=0, the `n >= 50` data-sufficiency guard
    never fires, and `salience_th` stays exactly where the module put it. This test passing while
    the one above fails is the whole failure mode, made visible.
    """
    _fresh_home(tmp_path, monkeypatch)
    from snn_data_lake.engine.brain import make_brain

    b = make_brain(seed=0, persist=True)
    if b._salience_calib is None or b._lc is None:
        b.close()
        pytest.skip("this brain has no salience calibrator / LC")
    default_th = float(b._lc.salience_th)
    assert b._salience_calib.result()["n"] == 0, "a fresh home must start with no recorded samples"
    for _ in range(8):
        b.tick()
    held = float(b._lc.salience_th)
    b.close()
    assert held == default_th, (
        f"with no recorded distribution the LC gate must hold its default, got {held}"
    )

"""
SNDL BRAIN DAEMON -- run the brain ALWAYS-ON: learning continuously + persisting across restarts.

Ticks the full circadian brain forever -- wake (perception / cognition / three-factor + e-prop learning)
and sleep (neurogenesis growth + replay consolidation) -- with persistence ON, so learned state
accumulates in data/ and a restart RESUMES where it left off (it does NOT start over). A bounded inter-
tick pause keeps it a polite background process rather than a 100%-CPU spinner.

SAFETY: the arm is the internal SIMULATOR (ArmSim); NO real hardware is actuated unless the brain is
separately configured for a serial arm, and the reflex / E-STOP / actuator gate are always in force
inside every tick. Autonomous self-build is OPT-IN and OFF here by default: set env NORD_SELFMOD=1 to let
the brain gate-verify + apply its own improvements every 8 nights (sandbox-verified, arm-safety
firewalled, rolled back byte-exact on any gate failure); unset = learn only, never self-modify.

  Learn only:            ./.venv-lake/Scripts/python.exe brain_daemon.py
  Learn + self-build:    NORD_SELFMOD=1 ./.venv-lake/Scripts/python.exe brain_daemon.py
  Lighter (stdlib tier): NORD_TORCH=0 ./.venv-lake/Scripts/python.exe brain_daemon.py

Auto-start at Windows logon: register the schtasks command documented in the repo (or run brain_daemon.bat
via Task Scheduler). Stop it: end the task / kill the python process. ASCII-only output (Windows cp1252).
"""

# ruff: noqa: E402, E501  -- root-level runner: intentional pre-import path setup + long descriptive lines

import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # repo root -> relative data/ + model paths resolve
sys.path.insert(0, "src")

from snn_data_lake.cortex.brain import Brain

TICK_SLEEP = float(os.environ.get("SNDL_TICK_SLEEP", "0.05"))  # inter-tick pause, seconds (polite bg)
LOG_EVERY = int(os.environ.get("SNDL_LOG_EVERY", "2000"))  # heartbeat log cadence, in ticks


def main():
    selfbuild = "ON" if os.environ.get("NORD_SELFMOD") == "1" else "off"
    b = Brain(seed=0)  # persistence spine default ON -> learning accumulates in data/ across restarts
    print(
        f"[brain-daemon] started: persist ON, self-build={selfbuild}, tick_sleep={TICK_SLEEP}s. "
        "Learning continuously; reflex/E-STOP in force; sim arm (no real hardware unless configured).",
        flush=True,
    )
    n = 0
    while True:
        try:
            b.tick()  # one circadian iteration; safety lives inside; sleep-gated growth + self-build
        except Exception as e:  # noqa: BLE001 -- a bad tick must never kill the always-on daemon
            print(f"[brain-daemon] tick {n} error (continuing): {type(e).__name__}: {e}", flush=True)
        n += 1
        if n % LOG_EVERY == 0:
            try:
                st = b.state()
                print(
                    f"[brain-daemon] tick {n}: nights={st.get('nights_elapsed')} "
                    f"grown_units={st.get('grown_units')} reasoning_acc={st.get('reasoning_accuracy')} "
                    f"selfbuild_last={getattr(b, '_selfbuild_last', None)}",
                    flush=True,
                )
            except Exception:  # noqa: BLE001 -- logging must never crash the loop
                pass
        time.sleep(TICK_SLEEP)


if __name__ == "__main__":
    main()

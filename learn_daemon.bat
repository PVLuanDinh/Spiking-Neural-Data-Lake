@echo off
REM ============================================================================================
REM SNDL LEARNING daemon launcher -- always-on FULL-brain learning + chat->durable-lake + AUTO-UPDATE.
REM Register via Task Scheduler at logon so the brain runs + learns continuously without manual launch.
REM See learn_daemon.py. This is the "only the brain learning daemon" target (the arm is the internal
REM simulator substrate, dormant -- never externally actuated).
REM
REM   FULL brain + chat ingest to the durable spike lake + autonomous self-build (default, below).
REM   Self-build OFF: comment out the NORD_SELFMOD line.
REM   AUTO-UPDATE: the daemon exits when a NEW COMMIT lands; this loop relaunches it with the new build.
REM ============================================================================================
cd /d D:\AI-project\spiking-neural-data-lake

set NORD_SELFMOD=1
REM ^ AUTONOMOUS self-build ON: every 8 nights the brain gate-verifies + applies its own improvements
REM   (sandbox-verified, arm-safety firewalled, rolled back byte-exact on any gate failure). Comment out
REM   this line to make the daemon learn-only. The self-build pauses the tick for minutes when it fires.

set NORD_TIME_CELLS_RESTORE=1
REM ^ NEURAL GROWTH SURVIVES RESTART. Growing the time-cell population is on by default everywhere
REM   (specs/003); RESTORING the grown width is opt-in, and this is the one place it is opted into.
REM   The split is not decoration: an unconditional restore makes brain construction depend on what
REM   the previous process left on disk instead of on its seed, which measured as non-deterministic
REM   construction in ~42% of trials. That is fatal for the gates (run_selfchecks/pytest must be
REM   reproducible) and is exactly what the always-on daemon WANTS -- a brain that relaunches on every
REM   new commit and forgets every neuron it grew is not growing. Gates stay deterministic because
REM   they never set this. Restore refuses and records rather than half-restoring on a fmt, length or
REM   lineage (seed/drive/noise_sd) mismatch; see time_cells.restored().

:loop
REM GPU-prioritized, always-on: run the live brain under the CUDA venv (.venv-cu128 = torch+cu128, RTX).
REM The torch/perception/neurogenesis tier then runs on the GPU; NORD_GPU=1 is set inside learn_daemon.py.
REM (The CPU gate env .venv-lake stays for run_selfchecks/pytest -- deterministic; not this always-on path.)
".venv-cu128\Scripts\python.exe" learn_daemon.py >> chat_learn.log 2>&1
echo [%date% %time%] learn-daemon exited (code %errorlevel%); relaunching in 8s >> chat_learn.log
timeout /t 8 /nobreak >nul
goto loop

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

set NORD_ESTOP_SELF_CLEAR=1
REM ^ THE BRAIN MAY RE-ARM ITSELF after an emergency stop, but ONLY while the breach condition
REM   reads clear, and every clear is counted into `estop_self_clears`. Owner-authorised 2026-08-03.
REM   Default OFF everywhere else: ten "the E-STOP freeze must hold" assertions across
REM   autonomous_loop / brain / live_brain encode "a latched arm stays latched across later ticks",
REM   and enabling this globally would have meant rewriting all ten. The DAEMON is where it matters,
REM   because a cockpit actuation route can trip an E-STOP and nothing on the live path could
REM   previously clear one -- a tripped stop deadlocked the arm until restart.

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
REM GPU-prioritized, always-on: run the live brain under .venv-lake (cp314t free-threaded,
REM torch 2.11.0+cu128, RTX 5070 verified). REPOINTED 2026-08-19: the previous target
REM .venv-cu128 was DELETED from disk at some point, and because this loop swallows the
REM launch failure into chat_learn.log, the daemon silently crash-looped (cmd code 3,
REM "cannot find the path") with NOBODY noticing -- the brain was down while the log grew.
REM Sharing the gate env does not break gate determinism: the NORD_* env vars are set only
REM in this launcher, never by run_selfchecks/pytest.
".venv-lake\Scripts\python.exe" learn_daemon.py >> chat_learn.log 2>&1
echo [%date% %time%] learn-daemon exited (code %errorlevel%); relaunching in 8s >> chat_learn.log
timeout /t 8 /nobreak >nul
goto loop

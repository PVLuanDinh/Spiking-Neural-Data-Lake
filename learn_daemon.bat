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
REM RESILIENCE, added 2026-08-29 after the SECOND silent crash-loop.
REM
REM The 2026-08-19 note below already named the defect exactly -- "this loop swallows the launch
REM failure into chat_learn.log ... with NOBODY noticing" -- and then only the TARGET was repointed.
REM Two days later the same class of failure recurred: the uv base interpreter behind
REM .venv-lake stopped resolving for the Task Scheduler context, and the loop retried every 8s for
REM 8.2 DAYS -- 21,438 identical failures, 29 MB of log, the brain down the whole time, nothing
REM surfaced. Repointing the path again would repeat the 2026-08-19 mistake.
REM
REM Four changes, each aimed at one way the old loop hid a dead brain:
REM   1. PREFLIGHT   -- prove the interpreter RUNS (not merely that the stub file exists; the stub
REM                     existed the whole time and it was its base that had gone).
REM   2. BACKOFF     -- 8s doubling to 300s, so a hard failure costs a handful of lines a minute
REM                     instead of 10,000 a day.
REM   3. ESCALATION  -- after 5 consecutive failures write DAEMON_DOWN.txt with the exit code and
REM                     reason. A file that appears is discoverable; line 21,438 of a 29 MB log is
REM                     not. Cleared on the next successful start.
REM   4. LOG BOUND   -- rotate chat_learn.log past ~64 MB so the evidence stays readable.
REM
REM A clean exit (code 0 = auto-update relaunch) RESETS the backoff and the failure count; only
REM non-zero exits escalate. That keeps the auto-update path exactly as fast as it was.
REM ============================================================================================
cd /d D:\AI-project\spiking-neural-data-lake
setlocal enabledelayedexpansion

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
REM ^ Since specs/022 T038 this switch restores GROWN STATE generally: the time-cell width AND the
REM   arm's workspace-coverage ledger (both lake-persisted; both refuse a lineage mismatch).
REM ^ NEURAL GROWTH SURVIVES RESTART. Growing the time-cell population is on by default everywhere
REM   (specs/003); RESTORING the grown width is opt-in, and this is the one place it is opted into.
REM   The split is not decoration: an unconditional restore makes brain construction depend on what
REM   the previous process left on disk instead of on its seed, which measured as non-deterministic
REM   construction in ~42% of trials. That is fatal for the gates (run_selfchecks/pytest must be
REM   reproducible) and is exactly what the always-on daemon WANTS -- a brain that relaunches on every
REM   new commit and forgets every neuron it grew is not growing. Gates stay deterministic because
REM   they never set this. Restore refuses and records rather than half-restoring on a fmt, length or
REM   lineage (seed/drive/noise_sd) mismatch; see time_cells.restored().

set PY=.venv-lake\Scripts\python.exe
set DELAY=8
set FAILS=0
set MARKER=DAEMON_DOWN.txt

:loop

REM -- 4. LOG BOUND: keep the evidence readable ------------------------------------------------
REM `if exist` FIRST: without it, %%~zF expands to EMPTY on a missing log and cmd aborts on the
REM malformed `if  GTR ...`. Found by testing the launcher on a fresh directory -- in this repo the
REM log always exists, so the crash would only ever have fired on a clean checkout.
if exist chat_learn.log (
  for %%F in (chat_learn.log) do if %%~zF GTR 67108864 (
    if exist chat_learn.log.1 del chat_learn.log.1
    move /y chat_learn.log chat_learn.log.1 >nul
    echo [%date% %time%] rotated chat_learn.log at 64MB >> chat_learn.log
  )
)

REM -- 1. PREFLIGHT: prove the interpreter RUNS, not that its stub exists -----------------------
REM    The stub existed throughout the 8.2-day outage; its uv base interpreter was what vanished.
REM    `-c pass` is the cheapest thing that fails exactly when the base cannot be resolved.
"%PY%" -c "pass" >nul 2>&1
if errorlevel 1 (
  set /a FAILS+=1
  REM RC is set here too: the marker prints it, and on this path the daemon never ran, so
  REM without this the escalation file reported a BLANK exit code -- found in test case A.
  set RC=preflight
  echo [%date% %time%] PREFLIGHT FAILED: "%PY%" cannot start ^(base interpreter unresolved^); attempt !FAILS!, retry in !DELAY!s >> chat_learn.log
  goto backoff
)

REM -- run the brain ---------------------------------------------------------------------------
REM GPU-prioritized, always-on: run the live brain under .venv-lake (cp314t free-threaded,
REM torch 2.11.0+cu128, RTX 5070 verified). REPOINTED 2026-08-19: the previous target
REM .venv-cu128 was DELETED from disk at some point, and because this loop swallows the
REM launch failure into chat_learn.log, the daemon silently crash-looped (cmd code 3,
REM "cannot find the path") with NOBODY noticing -- the brain was down while the log grew.
REM Sharing the gate env does not break gate determinism: the NORD_* env vars are set only
REM in this launcher, never by run_selfchecks/pytest.
"%PY%" learn_daemon.py >> chat_learn.log 2>&1
set RC=!errorlevel!

if !RC! EQU 0 (
  REM clean exit == auto-update relaunch. Reset everything; this path stays as fast as before.
  echo [%date% %time%] learn-daemon exited cleanly ^(code 0, auto-update^); relaunching in 8s >> chat_learn.log
  set DELAY=8
  set FAILS=0
  if exist %MARKER% del %MARKER%
  timeout /t 8 /nobreak >nul
  goto loop
)

set /a FAILS+=1
echo [%date% %time%] learn-daemon exited (code !RC!); consecutive failures !FAILS!; retry in !DELAY!s >> chat_learn.log

:backoff
REM -- 3. ESCALATION: make a dead brain DISCOVERABLE ---------------------------------------------
REM    A marker file that appears in the repo root is findable. Line 21,438 of a 29 MB log is not.
if !FAILS! GEQ 5 (
  echo SNDL DAEMON DOWN > %MARKER%
  echo first noticed: %date% %time% >> %MARKER%
  echo consecutive failures: !FAILS! >> %MARKER%
  echo last exit code: !RC! >> %MARKER%
  echo interpreter: %PY% >> %MARKER%
  echo. >> %MARKER%
  echo The brain is NOT running. Per P4c the daemon is the brain's drive, so this is not >> %MARKER%
  echo a stopped service -- it is a stopped brain. Check the tail of chat_learn.log. >> %MARKER%
  echo Most likely cause: the venv's BASE interpreter no longer resolves in this context. >> %MARKER%
  echo Verify with:  %PY% -c "pass" >> %MARKER%
  echo This file is deleted automatically on the next successful start. >> %MARKER%
)

REM -- 2. BACKOFF: 8 -> 16 -> 32 ... capped at 300s ----------------------------------------------
timeout /t !DELAY! /nobreak >nul
set /a DELAY=!DELAY!*2
if !DELAY! GTR 300 set DELAY=300
goto loop

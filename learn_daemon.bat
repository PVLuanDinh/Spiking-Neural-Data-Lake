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

:loop
".venv-lake\Scripts\python.exe" learn_daemon.py >> chat_learn.log 2>&1
echo [%date% %time%] learn-daemon exited (code %errorlevel%); relaunching in 8s >> chat_learn.log
timeout /t 8 /nobreak >nul
goto loop

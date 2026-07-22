@echo off
REM ============================================================================================
REM SNDL brain daemon launcher -- always-on, auto-restart on crash. Register via Task Scheduler
REM at logon so the brain runs + learns continuously without manual launch. See brain_daemon.py.
REM
REM   Learn only (default):  this file as-is.
REM   Learn + AUTONOMOUS self-build: uncomment the NORD_SELFMOD line below (the brain then gate-
REM   verifies + applies its own improvements every 8 nights; sandbox-verified, arm-safety
REM   firewalled, rolled back byte-exact on any gate failure).
REM ============================================================================================
cd /d D:\AI-project\spiking-neural-data-lake

REM set NORD_SELFMOD=1
REM set NORD_TORCH=0    REM  (lighter stdlib perception tier, if the always-on torch tier is too heavy)

:loop
".venv-lake\Scripts\python.exe" brain_daemon.py >> brain_daemon.log 2>&1
echo [%date% %time%] brain-daemon exited (code %errorlevel%); restarting in 10s >> brain_daemon.log
timeout /t 10 /nobreak >nul
goto loop

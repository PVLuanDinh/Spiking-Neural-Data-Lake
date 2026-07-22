@echo off
REM ============================================================================================
REM SNDL CHAT-LEARNING daemon launcher -- the brain learns from the project chat transcripts with
REM NO arm sim / no motor loop (free conversational text is display-only). Auto-restart on crash.
REM Register via Task Scheduler at logon so it runs continuously. Learn-only: no NORD_SELFMOD.
REM This is the "only the brain learning daemon" -- use INSTEAD of brain_daemon.bat (the arm one).
REM ============================================================================================
cd /d D:\AI-project\spiking-neural-data-lake

:loop
".venv-lake\Scripts\python.exe" chat_learn_daemon.py >> chat_learn.log 2>&1
echo [%date% %time%] chat-learn daemon exited (code %errorlevel%); restarting in 10s >> chat_learn.log
timeout /t 10 /nobreak >nul
goto loop

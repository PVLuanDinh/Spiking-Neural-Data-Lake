@echo off
setlocal enabledelayedexpansion
REM ===================================================================
REM  Spiking Neural Data Lake -- Windows installer launcher.
REM  Finds Python 3.11+ and runs packaging\install.py, which builds a
REM  virtual environment (.venv) and installs the package + every extra
REM  it can on this machine (gpu/torch, genn are best-effort).
REM
REM  Just double-click this file, or:  install.bat --skip gpu,genn
REM ===================================================================
set "HERE=%~dp0"
set "PYEXE="

REM Prefer the py launcher (picks the newest Python), else python on PATH.
where py >nul 2>nul && set "PYEXE=py -3"
if not defined PYEXE (
  where python >nul 2>nul && set "PYEXE=python"
)

if not defined PYEXE (
  echo.
  echo Python 3.11+ was not found on PATH.
  echo Install it from https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^),
  echo open a new terminal, and re-run install.bat.
  echo.
  pause
  exit /b 1
)

echo Using Python launcher: %PYEXE%
echo.
%PYEXE% "%HERE%packaging\install.py" %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (
  echo Install exited with code %RC%.
)
pause
exit /b %RC%

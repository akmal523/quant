@echo off
REM start.bat - launch the app using the repo's own virtual environment (v10.8.0).
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo No virtual environment found. Run install.bat first.
  exit /b 1
)

.venv\Scripts\python.exe -m quant.cli dash %*

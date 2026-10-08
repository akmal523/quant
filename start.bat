@echo off
REM start.bat — start the app and open the browser (v10.8.2).
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo No environment found. Run install.bat first.
  exit /b 1
)
.venv\Scripts\python -m quant.cli dash %*

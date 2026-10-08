@echo off
REM install.bat — one-command install for Windows (v10.8.2).
setlocal
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[dashboard]"
.venv\Scripts\python -m quant.cli dash

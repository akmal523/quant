@echo off
REM install.bat - one-command install for Windows (v10.8.0).
REM Creates a virtual environment inside the repo and installs the package.
cd /d "%~dp0"

python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -e ".[dashboard]"

echo Installed. Start the app with: start.bat

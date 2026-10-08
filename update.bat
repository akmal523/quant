@echo off
REM update.bat — pull, reinstall, migrate the database, print old and new version.
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo No environment found. Run install.bat first.
  exit /b 1
)
for /f %%v in ('.venv\Scripts\python -c "import quant; print(quant.__version__)"') do set OLD=%%v
.venv\Scripts\python -m quant.cli backup
git pull --ff-only
.venv\Scripts\python -m pip install -e ".[dashboard]"
.venv\Scripts\python -c "from quant.data.database import init_db; init_db()"
for /f %%v in ('.venv\Scripts\python -c "import quant; print(quant.__version__)"') do set NEW=%%v
echo Updated %OLD% to %NEW%.

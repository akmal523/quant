#!/usr/bin/env bash
# update.sh — pull, reinstall, migrate the database, print old and new version.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "No environment found. Run ./install.sh first."
  exit 1
fi

OLD=$(.venv/bin/python -c "import quant; print(quant.__version__)")
.venv/bin/python -m quant.cli backup || true
git pull --ff-only
.venv/bin/python -m pip install -e ".[dashboard]"
.venv/bin/python -c "from quant.data.database import init_db; init_db()"
NEW=$(.venv/bin/python -c "import quant; print(quant.__version__)")
echo "Updated $OLD to $NEW."

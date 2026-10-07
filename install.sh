#!/usr/bin/env bash
# install.sh — one-command install for macOS/Linux (v10.8.0).
#
# Creates a virtual environment inside the repo, installs the package, and
# prints the next step. Requires Python 3.11 or newer.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dashboard]"

echo "Installed. Start the app with: ./start.sh"

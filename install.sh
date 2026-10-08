#!/usr/bin/env bash
# install.sh — one-command install for macOS/Linux (v10.8.2).
# Creates the environment, installs, then starts the app. No `quant` command and
# no manual activation are needed for normal use.
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dashboard]"

exec .venv/bin/python -m quant.cli dash

#!/usr/bin/env bash
# start.sh — launch the app using the repo's own virtual environment (v10.8.0).
#
# No activation step and no `quant` on PATH are required.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "No virtual environment found. Run ./install.sh first."
  exit 1
fi

exec .venv/bin/python -m quant.cli dash "$@"

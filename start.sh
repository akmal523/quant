#!/usr/bin/env bash
# start.sh — start the app and open the browser (v10.8.2).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "No environment found. Run ./install.sh first."
  exit 1
fi

exec .venv/bin/python -m quant.cli dash "$@"

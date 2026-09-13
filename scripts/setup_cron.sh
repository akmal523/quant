#!/usr/bin/env bash
# setup_cron.sh — Configure local cron job for the daily workflow (Phase 4, 4.2).
#
# Runs data_updater.py + main.py at 18:00 CET (after European markets close and
# TR settles). discovery.py runs weekly (Sundays). Output logged to cron.log.
#
# Usage:  bash setup_cron.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CRON_LOG="$PROJECT_DIR/cron.log"

# 18:00 CET daily = 17:00 UTC in winter, 16:00 UTC in summer (DST).
# Use TZ=CET to let cron resolve local time correctly.
# Entry points are the thin root wrappers; discovery runs as a module.
DAILY_CMD="cd $PROJECT_DIR && TZ=Europe/Berlin $PYTHON data_updater.py >> $CRON_LOG 2>&1 && TZ=Europe/Berlin $PYTHON main.py >> $CRON_LOG 2>&1"
WEEKLY_CMD="cd $PROJECT_DIR && TZ=Europe/Berlin $PYTHON -m quant.execution.discovery >> $CRON_LOG 2>&1"

# Remove any previously installed lines for this project to avoid duplicates.
crontab -l 2>/dev/null | grep -v "data_updater.py" | grep -v "discovery.py" | crontab - 2>/dev/null || true

# Install new entries.
(
    crontab -l 2>/dev/null || true
    echo "0 18 * * * $DAILY_CMD"
    echo "0 18 * * 0 $WEEKLY_CMD"
) | crontab -

echo "Cron installed:"
crontab -l | grep -E "data_updater|discovery"
echo "Logs: $CRON_LOG"
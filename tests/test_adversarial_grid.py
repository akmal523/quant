"""test_adversarial_grid.py — edge cases and adversarial scenarios (v10.7.4, Part 5).

Intent: the system must survive empty, degenerate, and malformed inputs without
crashing, lying, or leaking. Every failure is a plain message, never a traceback.

Invariants:
  - Zero holdings -> empty advice, no crash.
  - Negative values raise DataError; duplicates are deduplicated.
  - Orphan tiers and unpriceable symbols are reported, not fatal.
  - The runner lock prevents concurrent runs.
  - Notification payloads carry no totals and never the token.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from quant import paths
from quant.engine import lock
from quant.errors import DataError
from quant.portfolio.portfolio import load_portfolio, portfolio_issues
from quant.portfolio.tier_manager import validate_tiers_csv
from quant.ui.copy import DB_BUSY_RETRY

# ── Part 5.2: adversarial data ────────────────────────────────────────────────

def test_negative_value_raises_data_error(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "BAD,10,-100,0\n", encoding="utf-8")
    with pytest.raises(DataError):
        load_portfolio(str(path))


def test_duplicate_rows_deduplicated(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "AMZN,100,1000,50\nAMZN,100,1000,50\n", encoding="utf-8")
    df = load_portfolio(str(path))
    assert len(df) == 1


def test_portfolio_issues_reports_duplicates(tmp_path):
    path = tmp_path / "portfolio.csv"
    path.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "AMZN,100,1000,50\nAMZN,100,1000,50\n", encoding="utf-8")
    issues = portfolio_issues(str(path))
    assert any("duplicate" in issue for issue in issues)


def test_orphan_tier_warning():
    tiers = pd.DataFrame([{"symbol": "NOT_HELD", "tier": "FORTRESS",
                           "last_updated": "2026-10-01", "notes": ""}])
    portfolio = pd.DataFrame([{"Symbol": "AMZN", "Current_Value_EUR": 1000.0}])
    _valid, errors = validate_tiers_csv(tiers, portfolio)
    assert any("not in portfolio" in error for error in errors)


# ── Part 5.3: race and timing ─────────────────────────────────────────────────

def _write_lock(tmp_path, pid, age_min, command="other"):
    started = (datetime.now() - timedelta(minutes=age_min)).isoformat(timespec="seconds")
    (tmp_path / ".runner.lock").write_text(
        json.dumps({"pid": pid, "started_at": started, "command": command}),
        encoding="utf-8")


def test_lock_prevents_concurrent_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    _write_lock(tmp_path, 1, 0)  # pid 1 is always alive
    res = lock.acquire("run", attempts=2, total_seconds=0.1, sleep_fn=lambda _s: None)
    assert res.acquired is False


def test_db_busy_retry_message():
    assert "Database is busy" in DB_BUSY_RETRY


def test_tiers_reload_on_change(tmp_path):
    from quant.portfolio.tier_manager import load_tiers, tier_map

    path = tmp_path / "tiers.csv"
    path.write_text("symbol,tier,last_updated,notes\nAMZN,ALPHA,2026-10-01,\n",
                    encoding="utf-8")
    assert tier_map(load_tiers(str(path)))["AMZN"] == "ALPHA"
    path.write_text("symbol,tier,last_updated,notes\nAMZN,FORTRESS,2026-10-01,\n",
                    encoding="utf-8")
    assert tier_map(load_tiers(str(path)))["AMZN"] == "FORTRESS"

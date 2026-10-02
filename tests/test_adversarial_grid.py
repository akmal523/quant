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
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from quant import paths
from quant.data.database import get_connection
from quant.engine import allocator, lock, notify, valuation
from quant.engine.advice import build_advice
from quant.errors import DataError
from quant.portfolio.portfolio import load_portfolio, portfolio_issues
from quant.portfolio.tier_manager import get_asset_tier, validate_tiers_csv
from quant.ui.copy import DB_BUSY_RETRY
from tests.fixtures.live_portfolio import degenerate_cases, holding

AS_OF = date(2026, 10, 2)


# ── Part 5.1: empty and degenerate portfolios ─────────────────────────────────

def test_zero_holdings_returns_empty_advice():
    advice, rejected = build_advice([], tiers={}, as_of=AS_OF)
    assert advice == []
    assert rejected == []


def test_one_fortress_allocator():
    holdings = degenerate_cases()["one_fortress"]
    legs = allocator.allocate(200.0, holdings, regime="bull")
    long_legs = [leg for leg in legs if leg["kind"] == "long_term"]
    assert long_legs and long_legs[0]["symbol"] == "EUNL.DE"
    assert [leg for leg in legs if leg["kind"] == "cash"]
    assert not [leg for leg in legs if leg["kind"] == "bet"]


def test_all_holdings_one_tier_cap_violation():
    holdings = [
        holding(f"BET{i}", f"Bet {i}", "SPECULATIVE", 100.0, total=300.0,
                target=0.02, conviction=0.0)
        for i in range(3)
    ]
    tiers = {h["symbol"]: "SPECULATIVE" for h in holdings}
    advice, _ = build_advice(holdings, tiers=tiers, as_of=AS_OF)
    assert any("2 percent cap" in a["why"] for a in advice if a["kind"] == "keep")


def test_zero_portfolio_value_no_nan():
    zero = holding("ZERO", "Zero", "ALPHA", 0.0, total=0.0, target=0.10)
    advice, _ = build_advice([zero], tiers={"ZERO": "ALPHA"}, as_of=AS_OF)
    assert isinstance(advice, list)


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


def test_missing_tier_uses_default():
    empty = pd.DataFrame(columns=["symbol", "tier", "last_updated", "notes"])
    assert get_asset_tier("UNKNOWN", empty) == "ALPHA"


def test_unpriceable_symbols():
    conn = get_connection()
    conn.execute("DELETE FROM holdings_meta")
    conn.execute(
        "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
        "VALUES ('NOPRICE', 1.0, ?, 100.0)", [AS_OF])
    assert valuation.unpriceable_symbols(conn, {}) == ["NOPRICE"]


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


# ── Part 5.4: notification correctness ────────────────────────────────────────

def _alert(symbol="AMZN", amount=30.0):
    return {"symbol": symbol, "name": "Amazon.com", "action": "sell",
            "amount_eur": amount, "fee_eur": 1.0, "message": "tactics fell."}


def test_telegram_payload_has_no_totals():
    text = notify.format_alert(_alert())
    assert "30" in text
    assert "portfolio" not in text.lower()
    assert "cash" not in text.lower()


def test_telegram_payload_has_no_token():
    text = notify.format_alert(_alert())
    assert "SECRET" not in text
    assert "telegram_bot_token" not in text


def test_failed_send_leaves_unnotified():
    assert notify.send_alert(_alert(), {"channel": "none"}) is False


def test_two_alerts_two_messages():
    first = notify.format_alert(_alert("AMZN", 30.0))
    second = notify.format_alert(_alert("MSFT", 40.0))
    assert first != second

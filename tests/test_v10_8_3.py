"""test_v10_8_3.py — live view, class invest plan, timed list, editable history.

Covers the v10.8.3 fixes: the pages read the live holdings (not a review
artifact), the invest block balances classes toward the strategy, the ONE list is
timed and directional, the History ledger is editable, and the instrument type can
be overridden.
"""
from __future__ import annotations

from datetime import date

import pytest

# ── A. live holdings view ─────────────────────────────────────────────────────

def _seed_portfolio(tmp_path, monkeypatch, rows):
    from quant import paths

    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        + "".join(f"{s},{e},{v},{p}\n" for s, e, v, p in rows))
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    tiers = tmp_path / "tiers.csv"
    tiers.write_text("symbol,tier,last_updated,notes\n"
                     "EUNL.DE,FORTRESS,2026-01-01,\n"
                     "AMZN,ALPHA,2026-01-01,\n")
    monkeypatch.setattr(paths, "DATA_TIERS", str(tiers))


def test_holdings_view_from_live_portfolio(tmp_path, monkeypatch):
    _seed_portfolio(tmp_path, monkeypatch, [
        ("EUNL.DE", 95.5, 287.58, 9.58),
        ("AMZN", 150.0, 150.2, 0.2),
    ])
    from quant.engine.holdings_view import holdings_view

    view = {h["symbol"]: h for h in holdings_view()}
    assert view["EUNL.DE"]["value_eur"] == pytest.approx(287.58)
    assert view["EUNL.DE"]["tier"] == "FORTRESS"
    assert view["AMZN"]["tier"] == "ALPHA"
    # current_weight is a fraction of the invested pool.
    total = 287.58 + 150.2
    assert view["EUNL.DE"]["current_weight"] == pytest.approx(287.58 / total)


def test_holdings_view_empty_without_csv(tmp_path, monkeypatch):
    from quant import paths
    from quant.engine.holdings_view import holdings_view

    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(tmp_path / "missing.csv"))
    assert holdings_view() == []


# ── D. class-based invest plan ────────────────────────────────────────────────

def _holdings():
    return [
        {"symbol": "EUNL.DE", "name": "World", "tier": "FORTRESS",
         "value_eur": 300.0, "current_weight": 0.30, "target_weight": 0.50},
        {"symbol": "AMZN", "name": "Amazon", "tier": "ALPHA",
         "value_eur": 400.0, "current_weight": 0.40, "target_weight": 0.20},
    ]


def test_invest_plan_balances_classes_and_never_sells():
    from quant.engine.invest_plan import propose_by_class

    plan = propose_by_class(1000.0, _holdings(), "balanced")
    assert plan["orders"], "expected at least one buy"
    assert all(o["amount_eur"] > 0 for o in plan["orders"])
    assert plan["total"] <= 1000.0
    # The long-term class is under target, so it receives a savings-plan buy.
    assert any(o["tier"] == "FORTRESS" and o["how"] == "Savings plan"
               for o in plan["orders"])


def test_invest_plan_class_rows_have_now_and_target():
    from quant.engine.invest_plan import propose_by_class

    plan = propose_by_class(500.0, _holdings(), "conservative")
    tiers = {r["tier"] for r in plan["classes"]}
    assert tiers == {"FORTRESS", "ALPHA", "SPECULATIVE"}
    for r in plan["classes"]:
        assert 0.0 <= r["now_pct"] <= 100.0
        assert 0.0 <= r["target_pct"] <= 100.0


def test_invest_plan_zero_amount_is_empty():
    from quant.engine.invest_plan import propose_by_class

    assert propose_by_class(0.0, _holdings())["orders"] == []


# ── C. timed ONE list ─────────────────────────────────────────────────────────

def test_timed_items_tags_horizon_class_direction():
    from quant.engine.decisions import timed_items
    from quant.ui import copy as C

    holdings = [{"symbol": "AMZN", "tier": "ALPHA"}]
    advice = [
        {"kind": "sell_part", "symbol": "AMZN", "company_name": "Amazon",
         "eur": 50.0, "why": "above target", "source": "drift"},
        {"kind": "buy", "symbol": "AMZN", "company_name": "Amazon",
         "eur": 100.0, "why": "below target", "source": "drift"},
        {"kind": "keep", "symbol": "AMZN", "company_name": "Amazon",
         "eur": None, "why": "fine", "source": "monitor"},
    ]
    items = timed_items(holdings, advice)
    assert len(items) == 2  # keep is not actionable
    sell = next(i for i in items if i["direction"] == C.DIRECTION_SELL)
    buy = next(i for i in items if i["direction"] == C.DIRECTION_BUY)
    assert sell["horizon"] == C.HORIZON_TODAY
    assert buy["horizon"] == C.HORIZON_MONTH
    assert sell["class"] == "ALPHA"


# ── G. editable history ledger ────────────────────────────────────────────────

def test_replace_trades_updates_inserts_and_deletes():
    from quant.data.database import get_connection
    from quant.engine.ledger import replace_trades

    conn = get_connection()
    conn.execute("DELETE FROM trades")
    replace_trades([
        {"id": None, "date": date(2026, 1, 2), "symbol": "AMZN",
         "action": "buy", "amount_eur": 100.0, "realized_pnl_eur": None},
        {"id": None, "date": date(2026, 2, 2), "symbol": "AMZN",
         "action": "sell", "amount_eur": 60.0, "realized_pnl_eur": 5.0},
    ])
    rows = conn.execute("SELECT id, amount_eur FROM trades ORDER BY date").fetchall()
    assert len(rows) == 2
    first_id = rows[0][0]
    # Correct the first row and drop the second.
    replace_trades([
        {"id": first_id, "date": date(2026, 1, 2), "symbol": "AMZN",
         "action": "buy", "amount_eur": 120.0, "realized_pnl_eur": None},
    ])
    rows = conn.execute("SELECT id, amount_eur FROM trades").fetchall()
    assert len(rows) == 1
    assert rows[0][1] == pytest.approx(120.0)
    conn.execute("DELETE FROM trades")


# ── I. instrument-type override ───────────────────────────────────────────────

def test_set_instrument_class_writes_registry():
    from quant.data.database import get_connection
    from quant.data.registry_repair import set_instrument_class

    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO asset_registry (symbol, instrument_class) "
        "VALUES ('AMZN', 'EQUITY')")
    assert set_instrument_class("AMZN", "ETF") is True
    row = conn.execute(
        "SELECT instrument_class FROM asset_registry WHERE symbol = 'AMZN'").fetchone()
    assert row[0] == "ETF"
    conn.execute("DELETE FROM asset_registry WHERE symbol = 'AMZN'")


# ── H. tax keys the History page reads ────────────────────────────────────────

def test_tax_summary_exposes_the_keys_the_page_reads():
    from quant.portfolio.tax_accounting import calculate_yearly_tax_summary

    summary = calculate_yearly_tax_summary(2026)
    for key in ("realized_gains_eur", "sparerpauschbetrag_remaining_eur",
                "estimated_tax_eur"):
        assert key in summary

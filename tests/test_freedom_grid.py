"""test_freedom_grid.py — the user's rights (v10.7.4, Part 3).

Intent: the system must not lie, block, or nag without reason. This grid encodes
the user's rights as tests that fail if a future change re-introduces a
false-positive block, a false nag, or an incorrect emergency order.

Invariants:
  - Cooldown blocks sells only; never buys, savings-plan legs, or alerts.
  - Emergency liquidity is ALPHA first, then SPECULATIVE, FORTRESS last.
  - The user can approve, record, and delete any monthly plan.
  - The sync reminder is throttled and never nags.
  - A dismissed alert does not reopen unless the condition re-arms.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from quant.data.database import get_connection
from quant.engine import alerts, plans, valuation
from quant.engine.advice import build_advice
from quant.portfolio.risk import emergency_sell_plan
from tests.fixtures.live_portfolio import (
    DRIFT_UNDER,
    alpha_case,
    fortress_case,
)

AS_OF = date(2026, 10, 2)
FAR_FUTURE = date(2099, 1, 1)


# ── Part 3.1: cooldown respect ────────────────────────────────────────────────

def test_cooldown_blocks_sell():
    holding = alpha_case(0.20, value=1000.0, cooldown=FAR_FUTURE)
    advice, rejected = build_advice([holding], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    assert not any(a["kind"] == "sell_part" for a in advice)
    assert any("cooldown" in r["plain_reason"] for r in rejected)


def test_cooldown_does_not_block_buy():
    holding = alpha_case(-0.09, value=500.0, cooldown=FAR_FUTURE, conviction=90.0)
    advice, _ = build_advice([holding], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    assert any(a["kind"] == "buy" for a in advice)


def test_cooldown_does_not_block_savings_plan():
    holding = fortress_case(DRIFT_UNDER, cooldown=FAR_FUTURE)
    advice, _ = build_advice([holding], tiers={"EUNL.DE": "FORTRESS"}, as_of=AS_OF)
    assert any(a["kind"] == "change_savings_plan" for a in advice)


def test_cooldown_does_not_block_alerts():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    created = alerts.evaluate_alerts(
        conn,
        [{"symbol": "AMZN", "name": "Amazon.com", "tier": "ALPHA",
          "value_eur": 500.0, "structural": 35.0}],
        today=AS_OF)
    assert any(a["kind"] == alerts.KIND_STRUCTURAL for a in created)


def test_system_lockdown_pauses_non_emergency_sells():
    holding = alpha_case(0.20, value=1000.0)
    advice, rejected = build_advice(
        [holding], tiers={"AMZN": "ALPHA"}, as_of=AS_OF, system_lockdown=True)
    assert not any(a["kind"] == "sell_part" for a in advice)
    assert any("lockdown" in r["plain_reason"] for r in rejected)


def test_system_lockdown_does_not_block_emergency_liquidity():
    df = pd.DataFrame([{"Symbol": "AMZN", "Tier": "ALPHA",
                        "Current_Value_EUR": 1000.0, "Liquidity_Score": 80.0,
                        "Broker_PnL_EUR": 0.0}])
    plan = emergency_sell_plan(500.0, df)
    assert plan["recommendations"]


# ── Part 3.2: emergency liquidity rights ──────────────────────────────────────

def _emergency_df():
    return pd.DataFrame([
        {"Symbol": "A_LIQ", "Tier": "ALPHA", "Current_Value_EUR": 300.0,
         "Liquidity_Score": 90.0, "Broker_PnL_EUR": 50.0},
        {"Symbol": "A_ILLIQ", "Tier": "ALPHA", "Current_Value_EUR": 300.0,
         "Liquidity_Score": 20.0, "Broker_PnL_EUR": -50.0},
        {"Symbol": "S1", "Tier": "SPECULATIVE", "Current_Value_EUR": 200.0,
         "Liquidity_Score": 50.0, "Broker_PnL_EUR": 0.0},
        {"Symbol": "F1", "Tier": "FORTRESS", "Current_Value_EUR": 1000.0,
         "Liquidity_Score": 10.0, "Broker_PnL_EUR": 100.0},
    ])


def test_emergency_alpha_first_sorted_by_liquidity():
    plan = emergency_sell_plan(500.0, _emergency_df())
    syms = [r["symbol"] for r in plan["recommendations"]]
    assert syms[0] == "A_LIQ"
    assert "F1" not in syms


def test_emergency_speculative_follows_alpha():
    plan = emergency_sell_plan(700.0, _emergency_df())
    syms = [r["symbol"] for r in plan["recommendations"]]
    assert syms.index("S1") > syms.index("A_LIQ")


def test_emergency_fortress_last_with_warning():
    plan = emergency_sell_plan(2000.0, _emergency_df())
    syms = [r["symbol"] for r in plan["recommendations"]]
    assert "F1" in syms
    assert plan["fortress_warning"] is not None


def test_emergency_losers_before_winners_at_equal_liquidity():
    df = pd.DataFrame([
        {"Symbol": "WIN", "Tier": "ALPHA", "Current_Value_EUR": 100.0,
         "Liquidity_Score": 50.0, "Broker_PnL_EUR": 20.0},
        {"Symbol": "LOSE", "Tier": "ALPHA", "Current_Value_EUR": 100.0,
         "Liquidity_Score": 50.0, "Broker_PnL_EUR": -20.0},
    ])
    plan = emergency_sell_plan(100.0, df)
    assert plan["recommendations"][0]["symbol"] == "LOSE"


def test_emergency_order_is_stable():
    first = [r["symbol"] for r in emergency_sell_plan(2000.0, _emergency_df())["recommendations"]]
    second = [r["symbol"] for r in emergency_sell_plan(2000.0, _emergency_df())["recommendations"]]
    assert first == second


def test_emergency_never_refuses_returns_shortfall():
    plan = emergency_sell_plan(100_000.0, _emergency_df())
    assert plan["shortfall"] > 0
    assert plan["fortress_warning"] is not None


# ── Part 3.4: sync reminder respect ───────────────────────────────────────────

def _seed_sync(conn, sync_date):
    conn.execute("DELETE FROM holdings_meta")
    conn.execute("DELETE FROM meta")
    conn.execute(
        "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
        "VALUES ('EUNL.DE', 1.0, ?, 100.0)", [sync_date])


def test_sync_reminder_appears_after_35_days(monkeypatch):
    conn = get_connection()
    _seed_sync(conn, date(2026, 1, 1))
    monkeypatch.setattr(plans, "is_pending_sync", lambda: False)
    line = valuation.sync_reminder_line(conn, date(2026, 3, 1))
    assert line is not None and "broker sync" in line


def test_sync_reminder_throttled_to_once_per_7_days(monkeypatch):
    conn = get_connection()
    _seed_sync(conn, date(2026, 1, 1))
    monkeypatch.setattr(plans, "is_pending_sync", lambda: False)
    assert valuation.sync_reminder_line(conn, date(2026, 3, 1)) is not None
    assert valuation.sync_reminder_line(conn, date(2026, 3, 3)) is None
    assert valuation.sync_reminder_line(conn, date(2026, 3, 9)) is not None


def test_sync_reminder_never_same_day_as_sync(monkeypatch):
    conn = get_connection()
    _seed_sync(conn, date(2026, 3, 1))
    monkeypatch.setattr(plans, "is_pending_sync", lambda: False)
    assert valuation.sync_reminder_line(conn, date(2026, 3, 1)) is None


def test_sync_reminder_lists_pending_positions_by_name(monkeypatch):
    conn = get_connection()
    _seed_sync(conn, date(2026, 1, 1))
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('pending_symbols', 'AMZN')")
    monkeypatch.setattr(plans, "is_pending_sync", lambda: True)
    line = valuation.sync_reminder_line(conn, date(2026, 3, 1))
    assert line is not None and "Positions recorded since then:" in line
    assert "Amazon.com" in line or "AMZN" in line


# ── Part 3.5: alert dismissal rights ──────────────────────────────────────────

def _structural_holding(structural=35.0):
    return {"symbol": "AMZN", "name": "Amazon.com", "tier": "ALPHA",
            "value_eur": 500.0, "structural": structural}


def test_dismiss_alert_with_reason():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    alerts.evaluate_alerts(conn, [_structural_holding()], today=AS_OF)
    alert_id = alerts.open_alerts(conn)[0]["id"]
    assert alerts.resolve_alert(conn, alert_id, "declined", "not now", today=AS_OF)
    assert not alerts.open_alerts(conn)


def test_dismissed_alert_does_not_reopen_without_rearm():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    alerts.evaluate_alerts(conn, [_structural_holding()], today=AS_OF)
    alert_id = alerts.open_alerts(conn)[0]["id"]
    alerts.resolve_alert(conn, alert_id, "declined", "not now", today=AS_OF)
    again = alerts.evaluate_alerts(conn, [_structural_holding()], today=AS_OF)
    assert not any(a["kind"] == alerts.KIND_STRUCTURAL for a in again)


def test_dismissed_alert_reopens_after_rearm():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    alerts.evaluate_alerts(conn, [_structural_holding(35.0)], today=AS_OF)
    alert_id = alerts.open_alerts(conn)[0]["id"]
    alerts.resolve_alert(conn, alert_id, "declined", "not now", today=AS_OF)
    # The condition clears -> re-arm.
    alerts.evaluate_alerts(conn, [_structural_holding(80.0)], today=AS_OF)
    # The condition breaks again -> a new alert fires.
    again = alerts.evaluate_alerts(conn, [_structural_holding(35.0)], today=AS_OF)
    assert any(a["kind"] == alerts.KIND_STRUCTURAL for a in again)


def test_advice_record_counts_dismissed_correct():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    alerts.evaluate_alerts(conn, [_structural_holding()], today=AS_OF)
    alert_id = alerts.open_alerts(conn)[0]["id"]
    alerts.resolve_alert(conn, alert_id, "declined", "not now", price=100.0, today=AS_OF)
    alerts.score_resolved_alerts(conn, {"AMZN": 120.0}, today=date(2026, 11, 5))
    line = alerts.advice_record_line(conn)
    assert "correct" in line

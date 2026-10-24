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

from quant.data.database import get_connection
from quant.engine import alerts
from quant.engine.advice import build_advice
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

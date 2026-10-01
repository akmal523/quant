"""
test_v10_7_0_phase4.py — Monthly allocator, plan storage, actuals (v10.7.0).

Intent: lock the monthly ritual. The long base is 70 percent; the active pool
goes to cash in a bear regime or without HIGH conviction; bets are gated. The
plan round-trips, and actuals replace the auto-flows and update holdings_meta.

Invariants: allocator is pure; I/O tests use the isolated test DB.
"""
from __future__ import annotations

from datetime import date

from quant.data.database import get_connection
from quant.engine import allocator, flows, plans

# ── Allocator ─────────────────────────────────────────────────────────────────

def test_long_base_is_seventy_percent():
    legs = allocator.allocate(200.0, holdings=[], regime="bull")
    long_legs = [leg for leg in legs if leg["kind"] == "long_term"]
    assert sum(leg["amount_eur"] for leg in long_legs) == 140.0


def test_no_high_conviction_sends_active_to_cash():
    legs = allocator.allocate(200.0, holdings=[], regime="bull")
    cash = [leg for leg in legs if leg["kind"] == "cash"]
    assert cash and cash[0]["amount_eur"] == 60.0


def test_bear_regime_sends_active_to_cash():
    legs = allocator.allocate(200.0, holdings=[], regime="bear")
    cash = [leg for leg in legs if leg["kind"] == "cash"]
    assert cash and "falling" in cash[0]["reason"]


def test_high_conviction_active_gets_the_pool():
    holdings = [{"symbol": "AMZN", "name": "Amazon.com", "tier": "ALPHA",
                 "value_eur": 500.0, "conviction": 80.0}]
    legs = allocator.allocate(200.0, holdings=holdings, regime="bull")
    active = [leg for leg in legs if leg["kind"] == "active"]
    assert active and active[0]["symbol"] == "AMZN"
    assert active[0]["amount_eur"] == 60.0


def test_bets_gated_without_speculative_or_flag():
    legs = allocator.allocate(200.0, holdings=[], regime="bull", bets_enabled=False)
    assert not [leg for leg in legs if leg["kind"] == "bet"]


def test_fortress_gap_distribution():
    holdings = [
        {"symbol": "EUNL.DE", "name": "MSCI World", "tier": "FORTRESS",
         "value_eur": 300.0, "current_weight": 0.30, "target_weight": 0.50},
        {"symbol": "IWDA.AS", "name": "World", "tier": "FORTRESS",
         "value_eur": 100.0, "current_weight": 0.10, "target_weight": 0.20},
    ]
    legs = allocator.allocate(200.0, holdings=holdings, regime="bull")
    long_legs = {leg["symbol"]: leg["amount_eur"] for leg in legs
                 if leg["kind"] == "long_term"}
    # EUNL has the larger gap (0.20 vs 0.10) -> gets more.
    assert long_legs["EUNL.DE"] > long_legs["IWDA.AS"]
    # Rounding to 5 EUR steps may leave a small shortfall below the long base.
    assert 130.0 <= sum(long_legs.values()) <= 140.0


# ── Plan storage and actuals ──────────────────────────────────────────────────

def test_plan_roundtrip():
    conn = get_connection()
    conn.execute("DELETE FROM monthly_plans")
    legs = [{"symbol": "EUNL.DE", "name": "MSCI World", "amount_eur": 140.0,
             "kind": "long_term", "reason": "r", "fee_eur": 0.0}]
    plans.save_plan(conn, "2026-11", 200.0, legs,
                    approved_date=date(2026, 11, 1), execution_date=date(2026, 11, 1))
    loaded = plans.load_plan(conn, "2026-11")
    assert loaded["budget_eur"] == 200.0
    assert loaded["legs"][0]["symbol"] == "EUNL.DE"
    assert plans.is_approved(conn, "2026-11")


def test_enter_actuals_replaces_flows_and_updates_meta():
    conn = get_connection()
    conn.execute("DELETE FROM flows")
    conn.execute("DELETE FROM holdings_meta")
    flows.write_planned_sparplan_flows(
        conn, "2026-11", [{"symbol": "EUNL.DE", "amount_eur": 140.0}],
        date(2026, 11, 1))
    recon = plans.enter_actuals(
        conn, "2026-11",
        [{"symbol": "EUNL.DE", "amount_eur": 140.0, "date": date(2026, 11, 1)}],
        price_lookup={"EUNL.DE": 100.0})
    assert recon[0]["planned"] == 140.0
    assert recon[0]["actual"] == 140.0
    # Auto-flow replaced by the actual.
    remaining = flows.load_flows(conn)
    assert len(remaining) == 1
    assert remaining[0]["note"] == "actual sparplan 2026-11"
    # holdings_meta updated: 140 / 100 = 1.4 shares.
    row = conn.execute(
        "SELECT shares FROM holdings_meta WHERE symbol = 'EUNL.DE'").fetchone()
    assert abs(row[0] - 1.4) < 1e-9


def test_reconciliation_line_plain():
    line = plans.reconciliation_line(
        {"symbol": "EUNL.DE", "planned": 140.0, "actual": 140.0, "deviation": 0.0})
    assert "Planned 140 EUNL.DE" in line
    assert "bought 140 EUNL.DE" in line
    assert "Deviation +0" in line

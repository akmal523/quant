"""
test_v10_7_0_phase4.py — Monthly allocator, plan storage, actuals (v10.7.0).

Intent: lock the monthly ritual. The long base is 70 percent; the active pool
goes to cash in a bear regime or without HIGH conviction; bets are gated. The
plan round-trips, and actuals replace the auto-flows and update holdings_meta.

Invariants: allocator is pure; I/O tests use the isolated test DB.
"""
from __future__ import annotations

from quant.engine import allocator

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
    # R10: the long base goes to the SINGLE largest-gap holding (EUNL, 0.20).
    assert long_legs == {"EUNL.DE": 140.0}



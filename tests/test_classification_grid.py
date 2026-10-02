"""test_classification_grid.py — the tier contract (v10.7.4, Part 1).

Intent: a ticker's tier is a binding contract. This grid proves the contract
holds under every realistic combination of drift, regime, and cooldown, and that
a considered-but-suppressed action is never silent.

Invariants:
  - FORTRESS never yields sell_part, under any input.
  - ALPHA obeys the six sizing laws (R1 corrected numbers).
  - SPECULATIVE yields notes and alerts, never forced sells.
  - A tier change in tiers.csv flips the advice set within one run.
"""
from __future__ import annotations

from datetime import date

import pytest

from quant.data.database import get_connection
from quant.engine import alerts
from quant.engine.advice import build_advice
from quant.ui.copy import ADVICE_BUY, ADVICE_SELL_PART
from tests.fixtures.live_portfolio import (
    DRIFT_FAR,
    DRIFT_NEAR,
    DRIFT_UNDER,
    alpha_case,
    fortress_case,
    spec_portfolio,
)

AS_OF = date(2026, 10, 2)
FAR_FUTURE = date(2099, 1, 1)


# ── Part 1.1: FORTRESS invariants ─────────────────────────────────────────────

FORTRESS_MATRIX = [
    # (drift, regime, cooldown, allowed advice kinds)
    (DRIFT_NEAR, "rising", None, {"keep", "change_savings_plan"}),
    (DRIFT_FAR, "rising", None, {"change_savings_plan"}),
    (DRIFT_FAR, "bear", None, {"change_savings_plan"}),
    (DRIFT_FAR, "bear", FAR_FUTURE, {"change_savings_plan"}),
    (DRIFT_UNDER, "rising", None, {"change_savings_plan"}),
    (DRIFT_UNDER, "bear", FAR_FUTURE, {"change_savings_plan"}),
]


@pytest.mark.parametrize("drift,regime,cooldown,allowed", FORTRESS_MATRIX)
def test_fortress_advice_set(drift, regime, cooldown, allowed):
    """Every FORTRESS row yields exactly one advice, inside the allowed set."""
    holding = fortress_case(drift, cooldown=cooldown)
    advice, rejected = build_advice(
        [holding], tiers={"EUNL.DE": "FORTRESS"}, regime=regime, as_of=AS_OF)
    kinds = {a["kind"] for a in advice if a.get("symbol") == "EUNL.DE"}
    assert kinds, "a FORTRESS holding always gets exactly one advice"
    assert kinds <= allowed
    assert "sell_part" not in kinds
    assert not [r for r in rejected if r["considered_action"] == ADVICE_SELL_PART]


def test_fortress_over_target_plan_change_wording():
    """R2: far over target -> a plan-change note that never implies a sale."""
    advice, _ = build_advice(
        [fortress_case(DRIFT_FAR)], tiers={"EUNL.DE": "FORTRESS"}, as_of=AS_OF)
    plan = [a for a in advice if a["kind"] == "change_savings_plan"]
    assert plan
    assert "lowering or pausing" in plan[0]["why"]
    assert "sell" not in plan[0]["why"].lower()


def test_fortress_under_target_top_up_wording():
    """Under target -> the shared top-up sentence."""
    advice, _ = build_advice(
        [fortress_case(DRIFT_UNDER)], tiers={"EUNL.DE": "FORTRESS"}, as_of=AS_OF)
    plan = [a for a in advice if a["kind"] == "change_savings_plan"]
    assert plan
    assert "Top up" in plan[0]["why"]


def test_fortress_structural_break_is_signal_not_sell():
    """A structural break on a FORTRESS holding is a signal, never a sale."""
    advice, _ = build_advice(
        [fortress_case(0.0)], tiers={"EUNL.DE": "FORTRESS"},
        open_alerts=[{"symbol": "EUNL.DE", "action": "sell", "amount_eur": 100.0,
                      "message": "structure fell to 30."}],
        as_of=AS_OF)
    kinds = {a["kind"] for a in advice if a.get("symbol") == "EUNL.DE"}
    assert "sell_part" not in kinds
    assert kinds <= {"keep", "change_savings_plan"}


# ── Part 1.2: ALPHA invariants (the six sizing laws) ──────────────────────────

ALPHA_MATRIX = [
    # value, drift, cooldown, regime, conviction, sell, buy, rejected
    (143.0, 0.069, None, "rising", 0.0, False, False, True),
    (500.0, 0.069, None, "rising", 0.0, True, False, False),
    (500.0, 0.069, FAR_FUTURE, "rising", 0.0, False, False, True),
    (500.0, 0.069, None, "bear", 0.0, True, False, False),
    (500.0, -0.09, None, "rising", 90.0, False, True, False),
    (500.0, -0.09, FAR_FUTURE, "rising", 90.0, False, True, False),
    (500.0, -0.09, None, "rising", 60.0, False, False, True),
    (90.0, -0.09, None, "rising", 90.0, False, False, True),
    # R9: the bear regime suppresses buys.
    (500.0, -0.09, None, "bear", 90.0, False, False, True),
    (500.0, -0.09, FAR_FUTURE, "bear", 90.0, False, False, True),
]


@pytest.mark.parametrize(
    "value,drift,cooldown,regime,conviction,sell,buy,rejected", ALPHA_MATRIX)
def test_alpha_sizing_laws(value, drift, cooldown, regime, conviction, sell, buy, rejected):
    holding = alpha_case(drift, value=value, cooldown=cooldown, conviction=conviction)
    advice, rej = build_advice(
        [holding], tiers={"AMZN": "ALPHA"}, regime=regime, as_of=AS_OF)
    kinds = {a["kind"] for a in advice if a.get("symbol") == "AMZN"}
    assert ("sell_part" in kinds) == sell
    assert ("buy" in kinds) == buy
    if rejected:
        assert rej, "a considered action must never be silent"


def test_alpha_sell_amount_is_thirty():
    """R1: 500 EUR at +6.9 percent -> min(34.5, 499) floored to 5 = 30 EUR."""
    advice, _ = build_advice(
        [alpha_case(0.069, value=500.0)], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    sell = [a for a in advice if a["kind"] == "sell_part"]
    assert sell and sell[0]["eur"] == 30.0


def test_alpha_buy_amount_is_forty():
    """R1: a 45 EUR buy rounds down to 40 EUR."""
    advice, _ = build_advice(
        [alpha_case(-0.09, value=500.0, conviction=90.0)],
        tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    buy = [a for a in advice if a["kind"] == "buy"]
    assert buy and buy[0]["eur"] == 40.0


def test_alpha_medium_conviction_is_keep_plus_rejected():
    """R3: a considered buy with MEDIUM conviction is never silent."""
    advice, rejected = build_advice(
        [alpha_case(-0.09, value=500.0, conviction=60.0)],
        tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    assert any(a["kind"] == "keep" for a in advice)
    assert any(r["considered_action"] == ADVICE_BUY for r in rejected)


def test_alpha_buy_below_100_eur_is_rejected():
    """R4: the untouchable law applies to ACTIVE buys."""
    advice, rejected = build_advice(
        [alpha_case(-0.09, value=90.0, conviction=90.0)],
        tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    assert not any(a["kind"] == "buy" for a in advice)
    assert any("below 100 EUR" in r["plain_reason"] for r in rejected)


def test_bear_regime_suppresses_buy_and_emits_to_cash():
    """R9: regime override -> no buy, one to_cash, one rejected note."""
    advice, rejected = build_advice(
        [alpha_case(-0.09, value=500.0, conviction=90.0)],
        tiers={"AMZN": "ALPHA"}, regime="bear", as_of=AS_OF)
    assert not any(a["kind"] == "buy" for a in advice)
    assert len([a for a in advice if a["kind"] == "to_cash"]) == 1
    notes = [r for r in rejected if r["considered_action"] == ADVICE_BUY]
    assert notes and "suppressed by bear regime" in notes[0]["plain_reason"]


# ── Part 1.3: SPECULATIVE invariants ──────────────────────────────────────────

def test_speculative_stop_loss_alert():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    created = alerts.evaluate_alerts(
        conn,
        [{"symbol": "BET", "name": "Small Bet", "tier": "SPECULATIVE",
          "value_eur": 40.0, "entry_price": 100.0, "current_price": 45.0}],
        today=AS_OF)
    assert any(a["kind"] == alerts.KIND_SPECULATIVE for a in created)


def test_speculative_no_alert_under_stop():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    created = alerts.evaluate_alerts(
        conn,
        [{"symbol": "BET", "name": "Small Bet", "tier": "SPECULATIVE",
          "value_eur": 40.0, "entry_price": 100.0, "current_price": 60.0}],
        today=AS_OF)
    assert not any(a["kind"] == alerts.KIND_SPECULATIVE for a in created)


def test_speculative_take_profit_advisory_high_conviction():
    holdings = spec_portfolio(spec_value=40.0, base_value=10_000.0,
                              pnl_pct=1.0, conviction=90.0)
    advice, _ = build_advice(
        holdings, tiers={"EUNL.DE": "FORTRESS", "BET": "SPECULATIVE"}, as_of=AS_OF)
    spec = [a for a in advice if a.get("symbol") == "BET"]
    assert spec and spec[0]["kind"] == "keep"
    assert "taking profit" in spec[0]["why"]


def test_speculative_take_profit_low_conviction_is_keep():
    holdings = spec_portfolio(spec_value=40.0, base_value=10_000.0,
                              pnl_pct=1.0, conviction=0.0)
    advice, _ = build_advice(
        holdings, tiers={"EUNL.DE": "FORTRESS", "BET": "SPECULATIVE"}, as_of=AS_OF)
    spec = [a for a in advice if a.get("symbol") == "BET"]
    assert spec and spec[0]["kind"] == "keep"
    assert "taking profit" not in spec[0]["why"]


def test_speculative_cap_violation_note():
    holdings = spec_portfolio(spec_value=200.0, base_value=1000.0)
    advice, _ = build_advice(
        holdings, tiers={"EUNL.DE": "FORTRESS", "BET": "SPECULATIVE"}, as_of=AS_OF)
    spec = [a for a in advice if a.get("symbol") == "BET"]
    assert spec and spec[0]["kind"] == "keep"
    assert "2 percent cap" in spec[0]["why"]


# ── Part 1.4: tier transitions ────────────────────────────────────────────────

def test_transition_alpha_to_fortress_removes_sell():
    holding = alpha_case(0.20, value=1000.0)
    advice_alpha, _ = build_advice([holding], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    assert any(a["kind"] == "sell_part" for a in advice_alpha)
    advice_fortress, _ = build_advice([holding], tiers={"AMZN": "FORTRESS"}, as_of=AS_OF)
    assert not any(a["kind"] == "sell_part" for a in advice_fortress)


def test_transition_fortress_to_alpha_allows_drift_sell():
    holding = fortress_case(0.20, value=1000.0)
    advice_fortress, _ = build_advice(
        [holding], tiers={"EUNL.DE": "FORTRESS"}, as_of=AS_OF)
    assert not any(a["kind"] == "sell_part" for a in advice_fortress)
    advice_alpha, _ = build_advice([holding], tiers={"EUNL.DE": "ALPHA"}, as_of=AS_OF)
    assert any(a["kind"] == "sell_part" for a in advice_alpha)


def test_transition_alpha_to_speculative_removes_drift_advice():
    holding = alpha_case(0.20, value=1000.0)
    advice_alpha, _ = build_advice([holding], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    assert any(a["kind"] == "sell_part" for a in advice_alpha)
    advice_spec, _ = build_advice([holding], tiers={"AMZN": "SPECULATIVE"}, as_of=AS_OF)
    kinds = {a["kind"] for a in advice_spec if a.get("symbol") == "AMZN"}
    assert kinds == {"keep"}

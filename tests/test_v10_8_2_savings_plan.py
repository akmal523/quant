"""test_v10_8_2_savings_plan.py — the invest block (v10.8.2, B2 / B6 1-3)."""
from __future__ import annotations

from quant.engine.savings_plan import propose_once, propose_plans


def _holdings():
    # EUNL is clearly underweight (Fortress); SXRV is overweight (Alpha).
    return [
        {"symbol": "EUNL.DE", "name": "iShares Core MSCI World", "tier": "FORTRESS",
         "value_eur": 300.0, "current_weight": 0.30, "target_weight": 0.50,
         "conviction": 0.0},
        {"symbol": "SXRV.DE", "name": "iShares NASDAQ 100", "tier": "ALPHA",
         "value_eur": 400.0, "current_weight": 0.40, "target_weight": 0.20,
         "conviction": 0.0},
    ]


def test_monthly_plan_rows_sum_to_amount_minus_unallocated():
    out = propose_plans(200.0, _holdings())
    assert out["total"] + out["not_allocated"] == 200.0
    rows = {r["symbol"]: r for r in out["rows"]}
    # Overweight Alpha gets no new plan (the rules give it zero).
    assert rows["SXRV.DE"]["suggested"] == 0.0
    # Change equals suggested minus current.
    for r in out["rows"]:
        assert r["change"] == round(r["suggested"] - r["current"], 2)


def test_overweight_holding_gets_zero_and_change_is_negative():
    out = propose_plans(200.0, _holdings(),
                        current_plans={"SXRV.DE": 40.0, "EUNL.DE": 100.0})
    rows = {r["symbol"]: r for r in out["rows"]}
    assert rows["SXRV.DE"]["suggested"] == 0.0
    assert rows["SXRV.DE"]["change"] == -40.0


def test_already_fit_when_within_five_eur_per_row():
    first = propose_plans(200.0, _holdings())
    plans = {r["symbol"]: r["suggested"] for r in first["rows"]}
    out = propose_plans(200.0, _holdings(), current_plans=plans)
    assert out["already_fit"] is True
    assert all(abs(r["change"]) <= 5.0 for r in out["rows"])


def test_once_returns_whole_orders_never_a_sell():
    orders = propose_once(200.0, _holdings(), min_order_eur=25.0)
    assert all(o["amount_eur"] > 0 for o in orders)
    # Only the underweight holding is bought; nothing sells.
    assert {o["symbol"] for o in orders} == {"EUNL.DE"}

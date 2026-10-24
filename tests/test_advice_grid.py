"""test_advice_grid.py — advice correctness (v10.7.4, Part 2).

Intent: the user trusts the system not to recommend buys that are mathematically
irrational, and to split the monthly budget honestly. This grid asserts the math
under realistic portfolio states.

Invariants:
  - A buy clears the fee hurdle (R1: expected alpha >= fee).
  - Conviction maps to action deterministically (R9: bear overrides).
  - The allocator conserves the budget and states truthful gaps (R10).
  - Modified Dietz excludes deposits and is honest about negative returns.
"""
from __future__ import annotations

from datetime import date

import pytest

from quant.engine import flows, sizing
from quant.engine.advice import build_advice
from tests.fixtures.live_portfolio import alpha_case

AS_OF = date(2026, 10, 2)
FAR_FUTURE = date(2099, 1, 1)


# ── Part 2.1: fee hurdle invariants ───────────────────────────────────────────


def test_fee_hurdle_blocks_small_buy():
    holding_ = alpha_case(-0.10, value=1000.0, conviction=90.0, expected_alpha_bps=50)
    advice, rejected = build_advice([holding_], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    assert not any(a["kind"] == "buy" for a in advice)
    assert any("does not clear" in r["plain_reason"] for r in rejected)


def test_fee_hurdle_allows_large_buy():
    holding_ = alpha_case(-0.20, value=1000.0, conviction=90.0, expected_alpha_bps=50)
    advice, _ = build_advice([holding_], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    buy = [a for a in advice if a["kind"] == "buy"]
    assert buy and buy[0]["eur"] == 200.0


# ── Part 2.2: conviction vs action ────────────────────────────────────────────

CONVICTION_MATRIX = [
    # conviction, cooldown, regime, buy, keep, rejected
    (90.0, None, "rising", True, False, False),
    (90.0, FAR_FUTURE, "rising", True, False, False),
    (90.0, None, "bear", False, False, True),
    (60.0, None, "rising", False, True, True),
    (40.0, None, "rising", False, False, False),
]


@pytest.mark.parametrize(
    "conviction,cooldown,regime,buy,keep,rejected", CONVICTION_MATRIX)
def test_conviction_vs_action(conviction, cooldown, regime, buy, keep, rejected):
    holding_ = alpha_case(-0.09, value=500.0, cooldown=cooldown, conviction=conviction)
    advice, rej = build_advice(
        [holding_], tiers={"AMZN": "ALPHA"}, regime=regime, as_of=AS_OF)
    kinds = {a["kind"] for a in advice if a.get("symbol") == "AMZN"}
    assert ("buy" in kinds) == buy
    # v10.8.2: the cash concept is gone; no advice kind is "to_cash".
    assert not any(a["kind"] == "to_cash" for a in advice)
    assert ("keep" in kinds) == keep
    assert bool(rej) == rejected


# ── Part 2.4: Modified Dietz correctness ──────────────────────────────────────

def test_modified_dietz_zero_flows():
    ret = flows.modified_dietz(1000.0, 1100.0, [], date(2026, 1, 1), date(2026, 1, 31))
    assert abs(ret - 0.10) < 1e-9


def test_modified_dietz_single_mid_period_buy_excluded():
    f = [{"date": date(2026, 1, 16), "type": "buy", "amount_eur": 200.0}]
    ret = flows.modified_dietz(1000.0, 1200.0, f, date(2026, 1, 1), date(2026, 1, 31))
    expected = (1200.0 - 1000.0 - 200.0) / (1000.0 + 0.5 * 200.0)
    assert abs(ret - expected) < 1e-9
    assert abs(ret) < 1e-9


def test_modified_dietz_dividend_is_outflow():
    f = [{"date": date(2026, 1, 16), "type": "dividend", "amount_eur": 20.0}]
    ret = flows.modified_dietz(1000.0, 1000.0, f, date(2026, 1, 1), date(2026, 1, 31))
    expected = (1000.0 - 1000.0 + 20.0) / (1000.0 - 10.0)
    assert abs(ret - expected) < 1e-9


def test_modified_dietz_multiple_flows():
    f = [
        {"date": date(2026, 1, 10), "type": "buy", "amount_eur": 200.0},
        {"date": date(2026, 1, 20), "type": "sell", "amount_eur": 50.0},
        {"date": date(2026, 1, 25), "type": "dividend", "amount_eur": 20.0},
    ]
    ret = flows.modified_dietz(1000.0, 1200.0, f, date(2026, 1, 1), date(2026, 1, 31))
    net = 200.0 - 50.0 - 20.0
    weighted = (21 / 30) * 200.0 + (11 / 30) * (-50.0) + (6 / 30) * (-20.0)
    expected = (1200.0 - 1000.0 - net) / (1000.0 + weighted)
    assert abs(ret - expected) < 1e-9


def test_modified_dietz_negative_return_with_deposit():
    f = [{"date": date(2026, 1, 16), "type": "buy", "amount_eur": 500.0}]
    ret = flows.modified_dietz(1000.0, 1400.0, f, date(2026, 1, 1), date(2026, 1, 31))
    assert ret is not None and ret < 0


FEE_HURDLE_MATRIX = [
    # fee, expected alpha (bps), buy amount, allowed
    (1.0, 50, 200.0, True),
    (1.0, 50, 100.0, False),
    (1.0, 200, 50.0, True),
    (0.0, 10, 5.0, True),
]


@pytest.mark.parametrize("fee,alpha,amount,allowed", FEE_HURDLE_MATRIX)
def test_fee_hurdle_law(fee, alpha, amount, allowed):
    assert sizing.passes_fee_hurdle(alpha, amount, fee) is allowed

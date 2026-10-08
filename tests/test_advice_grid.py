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

from quant.engine import allocator, flows, sizing
from quant.engine.advice import build_advice
from tests.fixtures.live_portfolio import alpha_case, holding

AS_OF = date(2026, 10, 2)
FAR_FUTURE = date(2099, 1, 1)


# ── Part 2.1: fee hurdle invariants ───────────────────────────────────────────

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


# ── Part 2.3: savings-plan allocator math ─────────────────────────────────────

def _allocator_fixtures():
    """The full cross product: FORTRESS x HIGH x regime x bets."""
    return [
        (n_fortress, n_high, regime, bets)
        for n_fortress in (0, 1, 3)
        for n_high in (0, 1, 3)
        for regime in ("bull", "bear")
        for bets in (False, True)
    ]


def _build_alloc_fixture(n_fortress, n_high, bets):
    holdings = []
    for i in range(n_fortress):
        holdings.append(holding(
            f"F{i}", f"Fortress {i}", "FORTRESS", 1000.0, total=10_000.0,
            target=0.30, current_weight=0.10 + 0.05 * i))
    for i in range(n_high):
        holdings.append(holding(
            f"A{i}", f"Alpha {i}", "ALPHA", 1000.0, total=10_000.0,
            target=0.20, current_weight=0.10, conviction=90.0))
    candidates = []
    if bets:
        holdings.append(holding(
            "BET", "Small Bet", "SPECULATIVE", 50.0, total=10_000.0,
            target=0.02, conviction=90.0))
        candidates.append({"symbol": "BET", "name": "Small Bet",
                           "tier": "SPECULATIVE", "conviction": 90.0})
    return holdings, candidates


@pytest.mark.parametrize("n_fortress,n_high,regime,bets", _allocator_fixtures())
def test_allocator_math(n_fortress, n_high, regime, bets):
    holdings, candidates = _build_alloc_fixture(n_fortress, n_high, bets)
    budget = 200.0
    legs = allocator.allocate(
        budget, holdings, regime=regime, candidates=candidates, bets_enabled=bets)

    # 1. Budget conservation.
    assert abs(sum(leg["amount_eur"] for leg in legs) - budget) < 1e-6

    # 2. Every leg obeys its tier's rules.
    tier_of = {h["symbol"]: h["tier"] for h in holdings}
    for leg in legs:
        if leg["kind"] == "long_term":
            if n_fortress > 0:
                assert tier_of.get(leg["symbol"]) == "FORTRESS"
            else:
                assert leg["symbol"] == allocator.DEFAULT_BROAD_ETF
        elif leg["kind"] == "active":
            assert tier_of.get(leg["symbol"]) == "ALPHA"
        elif leg["kind"] == "bet":
            assert tier_of.get(leg["symbol"]) == "SPECULATIVE"
        elif leg["kind"] == "cash":
            assert leg["symbol"] is None

    # 3. The long base goes to the SINGLE largest-gap FORTRESS holding.
    if n_fortress > 0:
        long_legs = [leg for leg in legs if leg["kind"] == "long_term"]
        assert len(long_legs) == 1
        assert long_legs[0]["symbol"] == "F0"

    # 4. Reasons are truthful: the gap number matches the actual drift.
    for leg in legs:
        if leg["kind"] in ("long_term", "active") and leg["symbol"] in tier_of:
            h = next(x for x in holdings if x["symbol"] == leg["symbol"])
            gap = abs(h["target_weight"] - h["current_weight"]) * 100
            assert f"a gap of {gap:.0f} points" in leg["reason"]


def test_allocator_no_fortress_uses_broad_etf():
    legs = allocator.allocate(200.0, holdings=[], regime="bull")
    long_legs = [leg for leg in legs if leg["kind"] == "long_term"]
    assert long_legs and long_legs[0]["symbol"] == allocator.DEFAULT_BROAD_ETF
    assert long_legs[0]["amount_eur"] == 140.0


def test_allocator_bear_sends_active_to_cash():
    holdings, _ = _build_alloc_fixture(1, 1, False)
    legs = allocator.allocate(200.0, holdings, regime="bear")
    assert not [leg for leg in legs if leg["kind"] == "active"]
    assert [leg for leg in legs if leg["kind"] == "cash"]


def test_allocator_bets_zero_without_speculative():
    holdings, _ = _build_alloc_fixture(1, 1, False)
    legs = allocator.allocate(200.0, holdings, regime="bull", bets_enabled=True)
    assert not [leg for leg in legs if leg["kind"] == "bet"]


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

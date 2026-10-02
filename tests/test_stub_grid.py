"""test_stub_grid.py — dynamic sensitivity grid (v10.7.5, Part 2).

Intent: a stub that returns constants fails these tests. For every pure math and
scoring function, assert the output RESPONDS to its inputs in the documented
direction, and (where the domain defines it) monotonicity.

Allowlist for legitimately insensitive functions (documented constants, not
stubs): ``regime_confidence`` band edges, ``tier_word``/``class_word`` enum maps,
``fmt_*`` formatters. They are pure lookups, not computations, and are covered by
their own contract tests.

Invariants: pure unit tests; no network, no I/O.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quant.analytics.scoring import (
    calculate_conviction,
    deflated_sharpe_ratio,
    etf_tactical_grade,
    evaluate_structural_grade,
    evaluate_tactical_grade,
    fit_market_regime,
    stewardship_score_v2,
)
from quant.engine import allocator, flows, sizing
from quant.features.indicators import atr, garch_volatility, rsi
from quant.portfolio.risk import (
    calculate_liquidity_score,
    calculate_risk_penalty,
    emergency_sell_plan,
)


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D"))


def _calm(n: int = 300, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _series(100.0 + np.cumsum(rng.normal(0.0, 0.1, n)))


def _violent(n: int = 300, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _series(100.0 + np.cumsum(rng.normal(0.0, 2.0, n)))


def _price_hist(volume: float, close: float = 100.0, n: int = 30) -> pd.DataFrame:
    return pd.DataFrame({
        "Volume": [volume] * n, "Close": [close] * n,
        "High": [close * 1.01] * n, "Low": [close * 0.99] * n,
    })


# ── Structural grade ──────────────────────────────────────────────────────────

def test_structural_grade_sensitive_and_monotone():
    base = evaluate_structural_grade(10.0, 1.0, 0.30, 20.0)
    assert base != evaluate_structural_grade(30.0, 1.0, 0.30, 20.0)  # sensitive to PE
    assert evaluate_structural_grade(10.0, 1.0, 0.30, 20.0) >= \
        evaluate_structural_grade(30.0, 1.0, 0.30, 20.0)  # non-increasing in PE
    assert evaluate_structural_grade(10.0, 1.0, 0.30, 20.0) >= \
        evaluate_structural_grade(10.0, 5.0, 0.30, 20.0)  # non-increasing in PEG
    assert evaluate_structural_grade(10.0, 1.0, 0.30, 20.0) >= \
        evaluate_structural_grade(10.0, 1.0, 0.05, 20.0)  # non-decreasing in ROE
    assert evaluate_structural_grade(10.0, 1.0, 0.30, 20.0) >= \
        evaluate_structural_grade(10.0, 1.0, 0.30, 5.0)  # non-decreasing in stewardship


# ── Stewardship ───────────────────────────────────────────────────────────────

def test_stewardship_sensitive_and_monotone():
    low_de = stewardship_score_v2({"DebtToEquity": 0.3, "ROE": 0.3, "ICR": 10.0})
    high_de = stewardship_score_v2({"DebtToEquity": 1.5, "ROE": 0.3, "ICR": 10.0})
    assert low_de != high_de
    assert low_de >= high_de  # non-increasing in DE
    assert stewardship_score_v2({"ROE": 0.3}) >= stewardship_score_v2({"ROE": 0.05})
    assert stewardship_score_v2({"ICR": 10.0}) >= stewardship_score_v2({"ICR": 1.0})


# ── Tactical grade ────────────────────────────────────────────────────────────

def test_tactical_grade_sensitive_and_monotone():
    assert evaluate_tactical_grade(0.8, 0.0, 0.0) >= evaluate_tactical_grade(0.2, 0.0, 0.0)
    assert evaluate_tactical_grade(0.5, 50.0, 0.0) >= evaluate_tactical_grade(0.5, -50.0, 0.0)
    assert evaluate_tactical_grade(0.5, 0.0, 0.0) >= evaluate_tactical_grade(0.5, 0.0, 20.0)
    assert evaluate_tactical_grade(0.5, 0.0, 0.0, 1.0) >= \
        evaluate_tactical_grade(0.5, 0.0, 0.0, 0.0)


def test_etf_tactical_grade_bounded_and_monotone():
    assert etf_tactical_grade(0.8, 1.0) >= etf_tactical_grade(0.2, 1.0)
    assert etf_tactical_grade(0.5, 2.0) >= etf_tactical_grade(0.5, -2.0)
    for bull in (0.0, 0.5, 1.0):
        for z in (-5.0, 0.0, 5.0):
            assert 0.0 <= etf_tactical_grade(bull, z) <= 100.0


# ── Risk penalty ──────────────────────────────────────────────────────────────

def test_risk_penalty_sensitive_and_monotone():
    calm = pd.Series(np.random.default_rng(2).normal(0.0, 0.001, 200))
    violent = pd.Series(np.random.default_rng(2).normal(-0.05, 0.05, 200))
    assert calculate_risk_penalty(violent) >= calculate_risk_penalty(calm)
    assert calculate_risk_penalty(pd.Series([0.01] * 200)) == 0.0


# ── Modified Dietz ────────────────────────────────────────────────────────────

def test_modified_dietz_linear_in_end_value():
    start, end = date(2026, 1, 1), date(2026, 1, 31)
    r1 = flows.modified_dietz(1000.0, 1050.0, [], start, end)
    r2 = flows.modified_dietz(1000.0, 1100.0, [], start, end)
    assert abs((r2 - r1) - r1) < 1e-9  # equal increments -> linear


def test_modified_dietz_deposit_is_not_profit():
    start, end = date(2026, 1, 1), date(2026, 1, 31)
    f = [{"date": date(2026, 1, 16), "type": "buy", "amount_eur": 200.0}]
    ret = flows.modified_dietz(1000.0, 1200.0, f, start, end)
    assert abs(ret) < 1e-9


# ── Liquidity score ───────────────────────────────────────────────────────────

def test_liquidity_score_sensitive_and_bounded():
    low = calculate_liquidity_score("X", _price_hist(10_000.0))
    high = calculate_liquidity_score("X", _price_hist(1_000_000.0))
    assert high >= low  # non-decreasing in volume
    for vol in (0.0, 10_000.0, 1_000_000.0):
        assert 0.0 <= calculate_liquidity_score("X", _price_hist(vol)) <= 100.0


# ── Emergency sell plan ───────────────────────────────────────────────────────

def _emergency_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Symbol": "A1", "Tier": "ALPHA", "Current_Value_EUR": 300.0,
         "Liquidity_Score": 90.0, "Broker_PnL_EUR": 0.0},
        {"Symbol": "A2", "Tier": "ALPHA", "Current_Value_EUR": 300.0,
         "Liquidity_Score": 20.0, "Broker_PnL_EUR": 0.0},
    ])


def test_emergency_plan_monotone_in_amount():
    small = emergency_sell_plan(100.0, _emergency_df())
    large = emergency_sell_plan(500.0, _emergency_df())
    assert large["total_available"] >= small["total_available"]
    assert [r["symbol"] for r in small["recommendations"]] == \
        [r["symbol"] for r in large["recommendations"]][:len(small["recommendations"])]


# ── Allocator ─────────────────────────────────────────────────────────────────

def test_allocator_budget_conservation_and_long_base():
    for budget in (5.0, 50.0, 200.0, 1000.0, 10_000.0):
        legs = allocator.allocate(budget, [], regime="bull")
        assert abs(sum(leg["amount_eur"] for leg in legs) - budget) < 1e-6
        long_base = sum(leg["amount_eur"] for leg in legs if leg["kind"] == "long_term")
        assert abs(long_base - budget * 0.70) < 5.0  # 70 percent, within rounding


# ── Conviction ────────────────────────────────────────────────────────────────

def test_conviction_sensitive_and_thresholds():
    assert calculate_conviction(100.0, 100.0, 100.0) == "HIGH"
    assert calculate_conviction(50.0, 50.0, 50.0) == "LOW"
    assert calculate_conviction(70.0, 70.0, 70.0) == "MEDIUM"
    assert calculate_conviction(90.0, 90.0, 90.0) != calculate_conviction(10.0, 10.0, 10.0)


# ── Deflated Sharpe ───────────────────────────────────────────────────────────

def test_deflated_sharpe_monotone_in_trials():
    assert deflated_sharpe_ratio(1.0, num_trials=1) == 1.0  # raw Sharpe at one trial
    assert deflated_sharpe_ratio(1.0, num_trials=100) < 1.0
    assert deflated_sharpe_ratio(1.0, num_trials=100) <= \
        deflated_sharpe_ratio(1.0, num_trials=10)


# ── Sizing ────────────────────────────────────────────────────────────────────

def test_sell_amount_monotone_and_capped():
    assert sizing.sell_amount(50.0, 1000.0) >= sizing.sell_amount(30.0, 1000.0)
    assert sizing.sell_amount(2000.0, 1000.0) <= 1000.0 - 1.0
    assert sizing.can_sell("ALPHA", 50.0, 200.0) is None  # below 100 EUR


# ── Indicators ────────────────────────────────────────────────────────────────

def test_indicators_differ_calm_vs_violent():
    calm, violent = _calm(), _violent()
    assert rsi(calm) != rsi(violent)
    assert atr(calm, calm, calm) != atr(violent, violent, violent)
    calm_vol = garch_volatility(calm)
    violent_vol = garch_volatility(violent)
    assert calm_vol is not None and violent_vol is not None
    assert violent_vol.mean() > calm_vol.mean()


# ── Market regime ─────────────────────────────────────────────────────────────

def test_fit_market_regime_differs_bull_vs_bear():
    n = 300
    rng = np.random.default_rng(7)
    bull = _series(100.0 + np.cumsum(rng.normal(0.4, 0.3, n)))
    bear = _series(100.0 + np.cumsum(rng.normal(-0.4, 0.3, n)))
    vol = _series(np.abs(rng.normal(0.01, 0.003, n)))
    bull_score = fit_market_regime(bull, vol)
    bear_score = fit_market_regime(bear, vol)
    assert bull_score != bear_score
    assert bull_score > bear_score

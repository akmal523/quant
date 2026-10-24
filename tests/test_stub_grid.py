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
    etf_tactical_grade,
    evaluate_structural_grade,
    evaluate_tactical_grade,
    fit_market_regime,
    stewardship_score_v2,
)
from quant.engine import flows, sizing
from quant.portfolio.risk import (
    calculate_risk_penalty,
)


def _series(values) -> pd.Series:
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D"))


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


# ── Sizing ────────────────────────────────────────────────────────────────────

def test_sell_amount_monotone_and_capped():
    assert sizing.sell_amount(50.0, 1000.0) >= sizing.sell_amount(30.0, 1000.0)
    assert sizing.sell_amount(2000.0, 1000.0) <= 1000.0 - 1.0
    assert sizing.can_sell("ALPHA", 50.0, 200.0) is None  # below 100 EUR


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

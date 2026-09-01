"""
optimizer.py — Portfolio optimization (cvxpy).

Intent: replace independent BUY/SELL labels with risk-aware portfolio weights.
Uses expected scores as return proxy, Ledoit-Wolf shrunk covariance for risk,
and enforces max weight / sector / turnover constraints.

Phase 4 (2.1, 2.2):
  - Cash is a first-class asset with the broker cash APY as its expected return.
  - The 2 EUR round-trip fee is a hurdle: an active trade must beat the cash
    yield after fees, else capital is allocated to Cash.
  - Smart Balance risk buckets (Safety/Core/Alpha) are hard inequality
    constraints, preventing 100% allocation into volatile tech stocks.

Invariants: weights sum to 1, all >= 0, each <= max_weight. Pure computation.
Dependencies: cvxpy, numpy, sklearn (LedoitWolf), config, risk.
"""
from __future__ import annotations

import numpy as np

from config import (
    MAX_POSITION_PCT, TARGET_VOLATILITY,
    ROUND_TRIP_FEE_EUR,
    SAFETY_BUCKET_MIN, CORE_BUCKET_MIN, ALPHA_BUCKET_MAX,
)
from risk import daily_risk_free_rate


def shrunk_covariance(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance for stable weights (Pillar 6.1)."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf()
    lw.fit(returns)
    return lw.covariance_


def optimize_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    current_weights: np.ndarray | None = None,
    max_weight: float = MAX_POSITION_PCT,
    sector_cap: float = 0.30,
    target_vol: float = TARGET_VOLATILITY,
    risk_aversion: float = 1.0,
    turnover_cost: float = 0.02,
    sector_map: np.ndarray | None = None,
    bucket_map: np.ndarray | None = None,
    cash_index: int | None = None,
) -> np.ndarray:
    """Solve mean-variance optimization with constraints.

    Args:
        expected_returns: (n,) return proxy (e.g. composite_score).
        cov_matrix: (n,n) shrunk covariance.
        current_weights: (n,) current holdings for turnover penalty (optional).
        max_weight: per-asset cap.
        sector_cap: max total weight per sector (requires sector_map).
        target_vol: target portfolio volatility.
        risk_aversion: risk penalty multiplier.
        turnover_cost: penalty on |w - current|.
        sector_map: (n,) integer sector ids (optional).
        bucket_map: (n,) bucket ids: 0=SAFETY, 1=CORE, 2=ALPHA (optional).
        cash_index: index of the cash asset in the weight vector (optional).
            When set, the cash asset's expected return is the broker cash APY
            daily yield, and the 2 EUR round-trip fee hurdle is applied to
            active (alpha) trades.

    Returns:
        (n,) optimal weights. Falls back to equal-weight on solver failure.
    """
    import cvxpy as cp

    n = len(expected_returns)
    w = cp.Variable(n)
    ret = expected_returns @ w
    risk = cp.quad_form(w, cov_matrix)

    objective = cp.Maximize(ret - risk_aversion * risk)
    constraints = [cp.sum(w) == 1, w >= 0, w <= max_weight]

    # Phase 4 (2.1): cash is a first-class asset with the broker cash yield.
    if cash_index is not None:
        expected_returns = expected_returns.copy()
        expected_returns[cash_index] = daily_risk_free_rate()

    # Phase 4 (2.2): Smart Balance risk buckets as hard inequality constraints.
    if bucket_map is not None:
        safety_mask = (bucket_map == 0).astype(float)
        core_mask   = (bucket_map == 1).astype(float)
        alpha_mask  = (bucket_map == 2).astype(float)
        constraints.append(safety_mask @ w >= SAFETY_BUCKET_MIN)
        constraints.append(core_mask   @ w >= CORE_BUCKET_MIN)
        constraints.append(alpha_mask  @ w <= ALPHA_BUCKET_MAX)

    if current_weights is not None:
        turnover = cp.norm1(w - current_weights)
        objective = cp.Maximize(
            ret - risk_aversion * risk - turnover_cost * turnover
        )

    if sector_map is not None:
        for sid in np.unique(sector_map):
            mask = (sector_map == sid).astype(float)
            constraints.append(mask @ w <= sector_cap)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        if w.value is None:
            raise RuntimeError("solver returned None")
        return np.clip(w.value, 0.0, max_weight)
    except Exception:
        # Fallback: equal weight within caps.
        return np.full(n, min(1.0 / n, max_weight))


def minimum_trade_size(expected_alpha_bps: float, round_trip_fee_eur: float = ROUND_TRIP_FEE_EUR) -> float:
    """Compute the minimum capital required for an active trade to clear fees.

    Intent: reject signals where expected alpha does not exceed the 2 EUR
    round-trip hurdle. Formula (Phase 4, 1.1):
        Minimum Capital = (Round_Trip_Fee / Expected_Alpha_BPS) * 10000
    Example: alpha 200 bps (2%) -> (2 / 200) * 10000 = 100 EUR.
    Invariants: returns >= 0; pure function.
    """
    if expected_alpha_bps <= 0:
        return float("inf")
    return float((round_trip_fee_eur / expected_alpha_bps) * 10000.0)


def passes_fee_hurdle(
    expected_alpha_bps: float,
    capital_eur: float,
    round_trip_fee_eur: float = ROUND_TRIP_FEE_EUR,
) -> bool:
    """True if the trade's expected alpha clears the 2 EUR round-trip fee.

    Intent: route capital to Cash when an active trade cannot mathematically
    beat the fee + risk-free yield. Pure function.
    """
    min_size = minimum_trade_size(expected_alpha_bps, round_trip_fee_eur)
    return capital_eur >= min_size
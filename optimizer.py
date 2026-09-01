"""
optimizer.py — Portfolio optimization (cvxpy).

Intent: replace independent BUY/SELL labels with risk-aware portfolio weights.
Uses expected scores as return proxy, Ledoit-Wolf shrunk covariance for risk,
and enforces max weight / sector / turnover constraints.
Invariants: weights sum to 1, all >= 0, each <= max_weight. Pure computation.
Dependencies: cvxpy, numpy, sklearn (LedoitWolf).
"""
from __future__ import annotations

import numpy as np

from config import MAX_POSITION_PCT, TARGET_VOLATILITY


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
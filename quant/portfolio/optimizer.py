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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from quant.config import (
    MAX_POSITION_PCT, TARGET_VOLATILITY,
    ROUND_TRIP_FEE_EUR,
    SAFETY_BUCKET_MIN, CORE_BUCKET_MIN, ALPHA_BUCKET_MAX,
    CVAR_ALPHA, MIN_TRADE_VOL_MULT,
)
from quant.portfolio.risk import daily_risk_free_rate


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
    regime: str | None = None,
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

    # v10.4.0 (Phase 3): regime-conditional cap. Bear/Chop hard-caps single-asset
    # weight (e.g. 2%) so the optimizer cannot concentrate in a hostile regime.
    if regime is not None:
        from quant.portfolio.regime_constraints import apply_regime_constraints
        max_weight = apply_regime_constraints(max_weight, regime)

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


def cvar_of_weights(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    alpha: float = CVAR_ALPHA,
) -> float:
    """Empirical Conditional Value at Risk (Expected Shortfall) of a weight vector.

    Intent: verify/measure the average loss in the worst `alpha` tail. Used to
    validate the Mean-CVaR optimizer and to report tail risk.
    Invariants: returns a non-negative loss fraction; pure function (no I/O).
    """
    if returns_matrix is None or len(returns_matrix) == 0:
        return 0.0
    port_returns = np.asarray(returns_matrix) @ np.asarray(weights)
    losses = -port_returns
    var = np.quantile(losses, 1.0 - alpha)
    tail = losses[losses >= var]
    return float(tail.mean()) if len(tail) else float(var)


def optimize_portfolio_cvar(
    expected_returns: np.ndarray,
    returns_matrix: np.ndarray,
    max_weight: float = MAX_POSITION_PCT,
    sector_cap: float = 0.30,
    risk_aversion: float = 1.0,
    alpha: float = CVAR_ALPHA,
    sector_map: np.ndarray | None = None,
    bucket_map: np.ndarray | None = None,
    regime: str | None = None,
) -> np.ndarray:
    """Mean-CVaR optimization via the Rockafellar-Uryasev linear formulation.

    Intent (v10.4.0, Phase 3): Mean-Variance assumes normal returns and ignores
    tail risk. CVaR optimizes the average loss in the worst `alpha` scenarios,
    protecting against black swans. The objective maximizes
    `expected_returns @ w - risk_aversion * CVaR(w)`.

    Formulation (Rockafellar-Uryasev):
        CVaR = VaR + 1/(alpha*N) * sum(u)
        s.t.  u_i >= -r_i @ w - VaR,  u_i >= 0
    where r_i is the i-th historical return row.

    Invariants: weights sum to 1, all >= 0, each <= max_weight. Falls back to
    equal weight on solver failure. Pure computation (no I/O).
    """
    import cvxpy as cp

    # v10.4.0 (Phase 3): regime-conditional cap (see optimize_portfolio).
    if regime is not None:
        from quant.portfolio.regime_constraints import apply_regime_constraints
        max_weight = apply_regime_constraints(max_weight, regime)

    R = np.asarray(returns_matrix, dtype=float)
    n = len(expected_returns)
    if R.ndim != 2 or R.shape[1] != n or R.shape[0] == 0:
        return np.full(n, min(1.0 / n, max_weight))

    N = R.shape[0]
    w = cp.Variable(n)
    var = cp.Variable()
    u = cp.Variable(N)

    port_ret = R @ w
    cvar = var + (1.0 / (alpha * N)) * cp.sum(u)

    objective = cp.Maximize(expected_returns @ w - risk_aversion * cvar)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight,
        u >= -port_ret - var,
        u >= 0,
    ]

    if bucket_map is not None:
        safety_mask = (bucket_map == 0).astype(float)
        core_mask = (bucket_map == 1).astype(float)
        alpha_mask = (bucket_map == 2).astype(float)
        constraints.append(safety_mask @ w >= SAFETY_BUCKET_MIN)
        constraints.append(core_mask @ w >= CORE_BUCKET_MIN)
        constraints.append(alpha_mask @ w <= ALPHA_BUCKET_MAX)

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


def minimum_trade_size_vol_aware(
    expected_alpha_bps: float,
    asset_annual_vol: float | None,
    round_trip_fee_eur: float = ROUND_TRIP_FEE_EUR,
    vol_mult: float = MIN_TRADE_VOL_MULT,
) -> float:
    """Volatility-aware minimum trade size (v10.4.0, Phase 2).

    Intent: the 1 EUR fee is asymmetric. A 1 EUR fee on a 50 EUR trade is a 2%
    instant loss. Higher-volatility assets have noisier edges, so the fixed fee
    must be amortized over a larger notional. Scales the base fee-hurdle capital
    by (1 + vol_mult * annual_vol).
    Invariants: returns >= base minimum_trade_size; inf propagates; pure function.
    """
    base = minimum_trade_size(expected_alpha_bps, round_trip_fee_eur)
    if base == float("inf"):
        return base
    if asset_annual_vol is None or asset_annual_vol <= 0:
        return base
    return float(base * (1.0 + vol_mult * float(asset_annual_vol)))


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


def calculate_min_trade_size(
    target_weight: float,
    current_weight: float,
    portfolio_value_eur: float,
    round_trip_fee_eur: float = ROUND_TRIP_FEE_EUR,
    min_alpha_bps: float = 100.0,
) -> float:
    """Calculate minimum trade size to clear the fee hurdle.

    Intent: a rebalance trade must be large enough to justify the 2 EUR
    round-trip fee. Returns the max of (drift value, fee-hurdle capital,
    MIN_TRADE_SIZE_EUR). Pure function (no I/O).
    Invariants: returns >= MIN_TRADE_SIZE_EUR.
    """
    from quant.config import MIN_TRADE_SIZE_EUR
    drift = abs(target_weight - current_weight)
    drift_value_eur = drift * portfolio_value_eur
    min_size_for_fees = (round_trip_fee_eur / min_alpha_bps) * 10000.0
    return max(drift_value_eur, min_size_for_fees, MIN_TRADE_SIZE_EUR)


def check_volume_liquidity(
    symbol: str,
    trade_size_eur: float,
    df_market_data: pd.DataFrame,
) -> tuple[bool, str]:
    """Check if a trade size is reasonable given daily volume.

    Intent: avoid trades that exceed 1% of the 20-day average daily volume
    (ADV), which would move the market / fail to fill. Uses the market_history
    Volume column. Pure function (no I/O).
    Invariants: returns (is_valid, reason).
    """
    if df_market_data is None or df_market_data.empty:
        return False, "No market data"
    if "Volume" not in df_market_data.columns or "Close" not in df_market_data.columns:
        return False, "Missing Volume/Close columns"

    vol = df_market_data["Volume"].dropna()
    close = df_market_data["Close"].dropna()
    if len(vol) < 20 or len(close) == 0:
        return False, "Insufficient history"

    avg_daily_volume = vol.tail(20).mean()
    current_price = close.iloc[-1]
    if current_price <= 0 or avg_daily_volume <= 0:
        return False, "Zero price/volume"

    shares_to_trade = trade_size_eur / current_price
    max_shares = avg_daily_volume * 0.01
    if shares_to_trade > max_shares:
        return False, (
            f"Trade {shares_to_trade:.0f} sh exceeds 1% ADV {max_shares:.0f}"
        )
    return True, "OK"
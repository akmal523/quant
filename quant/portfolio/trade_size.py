"""
trade_size.py — trade-size gates for the review audit (fee hurdle, liquidity).

Intent: a rebalance suggestion must be large enough to justify the round-trip
fee and small enough to fill: ``calculate_min_trade_size`` returns the larger of
the drift value, the fee-hurdle capital and ``MIN_TRADE_SIZE_EUR``;
``check_volume_liquidity`` rejects a trade above 1% of the 20-day average daily
volume. Used by :func:`quant.portfolio.portfolio.enhanced_portfolio_audit`.

(Until v10.8.10 this module was ``optimizer.py`` and also held an unused
mean-CVaR optimizer.)

Invariants: pure functions, no I/O.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from quant.config import (
    ROUND_TRIP_FEE_EUR,
)


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

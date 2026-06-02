"""
indicators.py — Stochastic volatility modeling + classic technical indicators.
Replaces deterministic linear oscillators with GARCH(1,1) conditional variance,
with EWMA fallback for short histories. Provides RSI and ATR for backtesting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def calculate_log_returns(close_prices: pd.Series) -> pd.Series:
    """Compute log returns for stationarity. Scaled x100 for GARCH optimizer stability."""
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    return log_returns * 100


def garch_volatility(close_prices: pd.Series, horizon: int = 1) -> pd.Series | None:
    """
    GARCH(1,1) conditional volatility (annualized).
    Auto-scales returns to unit variance before fitting to suppress DataScaleWarning
    for low-price assets (e.g., ETFs priced < 1 EUR).
    Returns None if data < 252 obs or model fails to converge.
    """
    if len(close_prices) < 252:
        return None

    returns = calculate_log_returns(close_prices)

    if returns.empty or returns.std() == 0:
        return None

    try:
        # Auto-scale weak returns to unit variance for GARCH numerical stability
        std_est = returns.std()
        if std_est < 1.0:
            scale = max(std_est, 1e-8)
            returns_scaled = returns / scale
        else:
            scale = 1.0
            returns_scaled = returns

        am = arch_model(returns_scaled, vol='Garch', p=1, q=1, mean='Zero', dist='t', rescale=True)
        res = am.fit(update_freq=0, disp='off')
        conditional_volatility = res.conditional_volatility * scale
        annualized_vol = conditional_volatility * np.sqrt(252) / 100
        vol_series = pd.Series(index=close_prices.index, dtype=float)
        vol_series.update(annualized_vol)
        return vol_series.bfill()
    except Exception:
        return None


def ewma_volatility(close_prices: pd.Series, span: int = 20) -> pd.Series:
    """
    Exponentially weighted moving average volatility (annualized).
    Used as fallback when GARCH cannot converge or data is too short.
    """
    returns = np.log(close_prices / close_prices.shift(1)).dropna()
    ewma_std = returns.ewm(span=span).std()
    annualized = ewma_std * np.sqrt(252)
    vol_series = pd.Series(index=close_prices.index, dtype=float)
    vol_series.update(annualized)
    return vol_series.bfill().fillna(0.0)


def rsi(close: pd.Series, period: int = 14) -> float | None:
    """Relative Strength Index. Returns None if insufficient data."""
    if len(close) < period + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, float('nan'))
    return float(100.0 - (100.0 / (1.0 + rs.iloc[-1]))) if not np.isnan(rs.iloc[-1]) else None


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float | None:
    """Average True Range. Returns None if insufficient data."""
    if len(close) < period + 1:
        return None
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def add_all_indicators(h: pd.DataFrame) -> pd.DataFrame:
    """
    Integrate stochastic + classic volatility metrics into DataFrame.
    GARCH(1,1) is primary; falls back to EWMA for short histories or non-convergence.
    """
    h = h.copy()

    garch_vol = garch_volatility(h["Close"])
    if garch_vol is not None:
        h["GARCH_Vol"] = garch_vol
    else:
        # EWMA fallback for short histories or failed GARCH convergence
        h["GARCH_Vol"] = ewma_volatility(h["Close"])

    return h

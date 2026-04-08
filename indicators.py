"""
indicators.py — pure technical indicator calculations (no I/O, no side effects).
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> float | None:
    """Relative Strength Index (Wilder smoothing via rolling mean approximation)."""
    if len(series) < period + 5:
        return None
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi_s = 100 - (100 / (1 + rs))
    v = rsi_s.iloc[-1]
    return float(v) if pd.notna(v) else None


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> float | None:
    """Average True Range."""
    try:
        h_l  = high - low
        h_pc = (high - close.shift(1)).abs()
        l_pc = (low  - close.shift(1)).abs()
        tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        val  = tr.rolling(period).mean().iloc[-1]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def safe_float(v) -> float | None:
    """Return float or None; filters NaN and Inf."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None

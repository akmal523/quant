"""
indicators.py — Pure technical indicator calculations (no I/O, no side effects).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> float | None:
    """Relative Strength Index (Wilder smoothing)."""
    if len(series) < period + 5:
        return None
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi_s = 100 - (100 / (1 + rs))
    rsi_s = rsi_s.dropna()
    if rsi_s.empty:
        return None
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


def add_all_indicators(h: pd.DataFrame) -> pd.DataFrame:
    """
    Attach EMA20, SMA50, SMA200, ATR(14), RSI(14) columns to a price DataFrame.
    Operates on currency-normalised Close/High/Low columns.
    Returns a new DataFrame (does not mutate input).
    """
    h = h.copy()

    h["EMA20"]  = ema(h["Close"], 20)
    h["SMA50"]  = sma(h["Close"], 50)
    h["SMA200"] = sma(h["Close"], 200)

    # ATR — vectorised
    h_l  = h["High"] - h["Low"]
    h_pc = (h["High"] - h["Close"].shift(1)).abs()
    l_pc = (h["Low"]  - h["Close"].shift(1)).abs()
    tr   = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    h["ATR"] = tr.rolling(14).mean()

    # RSI — vectorised
    delta    = h["Close"].diff()
    gain     = delta.clip(lower=0).rolling(14).mean()
    loss     = (-delta.clip(upper=0)).rolling(14).mean()
    rs       = gain / loss.replace(0, np.nan)
    h["RSI"] = 100 - (100 / (1 + rs))

    return h

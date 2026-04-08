"""
indicators.py — pure technical indicator calculations (no I/O, no side effects).

Changes from v4.5:
- rsi(): replaced rolling-mean approximation with proper Wilder EWM smoothing
- macd(): new — MACD line, signal line, histogram, and crossover flag
- bollinger(): new — %B position and bandwidth
- volume_trend(): new — detects price-up / volume-down divergence (weak signal flag)
- ema200(): convenience wrapper
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# MOVING AVERAGES
# ─────────────────────────────────────────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    """Standard EMA (adjust=False for recursive definition)."""
    return series.ewm(span=span, adjust=False).mean()


def ema200(series: pd.Series) -> float | None:
    """200-period EMA — long-term trend filter. Returns latest value or None."""
    if len(series) < 200:
        return None
    v = ema(series, 200).iloc[-1]
    return float(v) if pd.notna(v) else None


# ─────────────────────────────────────────────────────────────────────────────
# RSI — Wilder smoothing
# ─────────────────────────────────────────────────────────────────────────────

def rsi(series: pd.Series, period: int = 14) -> float | None:
    """
    Relative Strength Index using proper Wilder EWM smoothing
    (alpha = 1/period, adjust=False).  This matches Bloomberg / TradingView.

    The previous implementation used rolling().mean() as an approximation —
    that converges to Wilder only after a very large number of bars.
    """
    if len(series) < period + 5:
        return None
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    # Wilder smoothing: com = period - 1  ↔  alpha = 1/period
    avg_gain = gain.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_s = 100 - (100 / (1 + rs))
    v = rsi_s.iloc[-1]
    return float(v) if pd.notna(v) else None


# ─────────────────────────────────────────────────────────────────────────────
# ATR
# ─────────────────────────────────────────────────────────────────────────────

def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> float | None:
    """Average True Range (Wilder smoothing, consistent with rsi above)."""
    try:
        h_l = high - low
        h_pc = (high - close.shift(1)).abs()
        l_pc = (low - close.shift(1)).abs()
        tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
        val = tr.ewm(com=period - 1, min_periods=period, adjust=False).mean().iloc[-1]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────────────────────────────────────

def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9
         ) -> dict:
    """
    MACD indicator.

    Returns a dict with:
      macd_line   — EMA(fast) − EMA(slow)
      signal_line — EMA(signal) of macd_line
      histogram   — macd_line − signal_line
      bullish_cross — True if histogram just turned positive (signal crossover)
      bearish_cross — True if histogram just turned negative
    All values are floats or None on insufficient data.
    """
    out = dict(macd_line=None, signal_line=None, histogram=None,
               bullish_cross=False, bearish_cross=False)
    if len(series) < slow + signal + 5:
        return out
    ml = ema(series, fast) - ema(series, slow)
    sl = ml.ewm(span=signal, adjust=False).mean()
    hist = ml - sl
    out["macd_line"] = float(ml.iloc[-1]) if pd.notna(ml.iloc[-1]) else None
    out["signal_line"] = float(sl.iloc[-1]) if pd.notna(sl.iloc[-1]) else None
    if len(hist) >= 2 and pd.notna(hist.iloc[-1]) and pd.notna(hist.iloc[-2]):
        h_now = float(hist.iloc[-1])
        h_prev = float(hist.iloc[-2])
        out["histogram"] = round(h_now, 6)
        out["bullish_cross"] = h_now > 0 and h_prev <= 0
        out["bearish_cross"] = h_now < 0 and h_prev >= 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BOLLINGER BANDS
# ─────────────────────────────────────────────────────────────────────────────

def bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> dict:
    """
    Bollinger Bands.

    Returns a dict with:
      bb_upper, bb_mid, bb_lower  — band levels (float | None)
      bb_pct_b  — %B: where price sits within the bands (0=lower, 1=upper)
                  values outside [0,1] mean price is outside the bands
      bb_bandwidth — (upper − lower) / mid  — measures band width / volatility
    """
    out = dict(bb_upper=None, bb_mid=None, bb_lower=None,
               bb_pct_b=None, bb_bandwidth=None)
    if len(series) < period:
        return out
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    m = float(mid.iloc[-1])
    u = float(upper.iloc[-1])
    l = float(lower.iloc[-1])
    p = float(series.iloc[-1])
    if not all(pd.notna(x) for x in [m, u, l]):
        return out
    out["bb_upper"] = round(u, 4)
    out["bb_mid"] = round(m, 4)
    out["bb_lower"] = round(l, 4)
    band_range = u - l
    if band_range > 0:
        out["bb_pct_b"] = round((p - l) / band_range, 3)
        out["bb_bandwidth"] = round(band_range / m * 100, 2)  # as % of mid
    return out


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME TREND
# ─────────────────────────────────────────────────────────────────────────────

def volume_trend(close: pd.Series, volume: pd.Series,
                 lookback: int = 20) -> dict:
    """
    Detect price-direction / volume-direction divergence over the last `lookback` bars.

    Returns:
      price_up    — True if price end > price start over the window
      volume_up   — True if average volume (last 5 bars) > average (prior 15 bars)
      weak_signal — True when price is rising but volume is falling
                    (classic distribution / exhaustion warning)
      vol_ratio   — recent_vol / prior_vol (float | None)
    """
    out = dict(price_up=None, volume_up=None, weak_signal=False, vol_ratio=None)
    if len(close) < lookback or len(volume) < lookback:
        return out
    c = close.tail(lookback)
    v = volume.tail(lookback)
    price_up = float(c.iloc[-1]) > float(c.iloc[0])
    recent_vol = float(v.tail(5).mean())
    prior_vol = float(v.head(lookback - 5).mean())
    if prior_vol > 0:
        ratio = recent_vol / prior_vol
        volume_up = ratio > 1.0
        out["vol_ratio"] = round(ratio, 3)
        out["volume_up"] = volume_up
        out["price_up"] = price_up
        out["weak_signal"] = price_up and not volume_up
    return out


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def safe_float(v) -> float | None:
    """Return float or None; filters NaN and Inf."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None

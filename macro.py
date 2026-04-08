"""
macro.py — macro series download, rolling correlations, and ROC signals.

Changes from v4.5:
- corr_with_macro() now returns both full-history Pearson AND a 6-month
  rolling window correlation — rolling captures regime shifts that a static
  full-history number masks.
- macro_roc() added: computes 1-month and 3-month rate-of-change on each
  macro series so callers can detect directional momentum (e.g. rates rising,
  oil falling) rather than just the level.
"""

import pandas as pd
import yfinance as yf
from scipy import stats
from config import START_DATE
from tickers import MACRO_TICKERS

# Rolling window used for regime-sensitive correlation (trading days)
ROLLING_WINDOW_DAYS = 126   # ~6 calendar months


def load_macro(start: str = START_DATE) -> dict:
    """
    Download macro reference series from Yahoo Finance.
    Returns a dict of {name: pd.Series}.
    Silently skips any ticker that fails to download.
    """
    macro: dict = {}
    for name, sym in MACRO_TICKERS.items():
        try:
            df = yf.download(sym, start=start, auto_adjust=True, progress=False)
            if not df.empty:
                macro[name] = df["Close"].squeeze()
        except Exception:
            pass
    return macro


def corr_with_macro(price: pd.Series, macro: dict) -> dict:
    """
    Compute two sets of correlations between *price* and each macro series:

    1. full_<name>  — Pearson r over the entire shared history (monthly returns)
    2. roll_<name>  — Pearson r over the last 6 months of daily returns
                      (captures current regime vs long-run average)

    Requires ≥ 12 overlapping monthly observations for full-history;
    ≥ 60 overlapping daily observations for the rolling window.
    Returns a flat dict.
    """
    out: dict = {}
    if price is None or len(price) < 24:
        return out

    # Monthly returns for full-history correlation
    m_px = price.resample("ME").last().pct_change().dropna()
    # Daily returns for rolling correlation
    d_px = price.pct_change().dropna()

    for name, m in macro.items():
        try:
            # ── Full-history (monthly) ───────────────────────────────────
            m_m = m.resample("ME").last().pct_change().dropna()
            al_m = pd.concat([m_px, m_m], axis=1).dropna()
            if len(al_m) >= 12:
                r_full, _ = stats.pearsonr(al_m.iloc[:, 0], al_m.iloc[:, 1])
                out[f"full_{name}"] = round(r_full, 3)

            # ── Rolling 6-month (daily) ──────────────────────────────────
            d_m = m.pct_change().dropna()
            al_d = pd.concat([d_px, d_m], axis=1).dropna()
            if len(al_d) >= 60:
                window = al_d.tail(ROLLING_WINDOW_DAYS)
                if len(window) >= 40:
                    r_roll, _ = stats.pearsonr(window.iloc[:, 0], window.iloc[:, 1])
                    out[f"roll_{name}"] = round(r_roll, 3)
        except Exception:
            pass

    return out


def macro_roc(macro: dict) -> dict:
    """
    Rate-of-change signals on each macro series.

    Returns a dict with two keys per series:
      roc1m_<name>  — 1-month (≈21 trading-day) percentage change in levels
      roc3m_<name>  — 3-month (≈63 trading-day) percentage change in levels

    Use these as directional signals: e.g. roc1m_OIL < 0 means oil is
    falling in the near term, which may be a positive tailwind for
    energy-importing sectors.
    """
    out: dict = {}
    for name, series in macro.items():
        s = series.dropna()
        for label, days in (("roc1m", 21), ("roc3m", 63)):
            if len(s) > days:
                current = float(s.iloc[-1])
                past = float(s.iloc[-days - 1])
                if past != 0:
                    out[f"{label}_{name}"] = round((current - past) / abs(past) * 100, 2)
    return out

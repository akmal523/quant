"""
macro.py — macro series download and correlation helpers.
"""
import pandas as pd
import yfinance as yf
from scipy import stats

from config  import START_DATE
from tickers import MACRO_TICKERS


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
    Compute monthly Pearson correlations between *price* and each macro series.
    Returns a dict of {macro_name: correlation_coefficient}.
    Requires at least 12 overlapping monthly observations; skips otherwise.
    """
    out: dict = {}
    if price is None or len(price) < 24:
        return out
    m_px = price.resample("ME").last().pct_change().dropna()
    for name, m in macro.items():
        try:
            m_m = m.resample("ME").last().pct_change().dropna()
            al  = pd.concat([m_px, m_m], axis=1).dropna()
            if len(al) < 12:
                continue
            r, _ = stats.pearsonr(al.iloc[:, 0], al.iloc[:, 1])
            out[name] = round(r, 3)
        except Exception:
            pass
    return out

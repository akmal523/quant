"""
currency.py — FX helpers: rate fetching, OHLCV normalisation to EUR.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

from config   import CONVERT_TO_EUR
from universe import CURRENCY_SYMBOLS

# Module-level EUR/USD rate cache (fetched once per process).
_eur_rate_cache: float | None = None


def get_eur_rate() -> float:
    """
    Fetch live EUR/USD rate. Cached for lifetime of the process.
    Falls back to 1.08 on failure.
    """
    global _eur_rate_cache
    if _eur_rate_cache is not None:
        return _eur_rate_cache
    try:
        df = yf.download("EURUSD=X", period="5d", auto_adjust=True, progress=False)
        if not df.empty:
            rate = float(df["Close"].iloc[-1])
            _eur_rate_cache = rate
            print(f"  [FX] EUR/USD = {rate:.4f}")
            return rate
    except Exception:
        pass
    _eur_rate_cache = 1.08
    print(f"  [FX] EUR/USD fallback = {_eur_rate_cache}")
    return _eur_rate_cache


def currency_symbol(code: str) -> str:
    return CURRENCY_SYMBOLS.get((code or "").upper(), (code or "") + " ")


def apply_fx_conversion(
    hist: pd.DataFrame,
    from_currency: str,
    to_currency: str = "EUR",
) -> pd.DataFrame:
    """
    Normalise OHLCV Close/High/Low/Open columns to target currency.

    Conversion chain:
      GBX → GBP  (divide by 100, pence normalisation)
      USD → EUR  (via EURUSD=X rate — 1/rate)
      GBP → EUR  (via GBPEUR=X direct rate)
      other → EUR  (via <CCY>EUR=X, fallback to USD intermediary)
      EUR → EUR  (pass-through)
    """
    src = (from_currency or "").upper().strip()
    tgt = (to_currency  or "EUR").upper().strip()

    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in hist.columns]
    h = hist.copy()

    # ── GBX (pence) → GBP ────────────────────────────────────────────────────
    if src == "GBX":
        for c in price_cols:
            h[c] = h[c] / 100.0
        src = "GBP"

    if src == tgt:
        return h

    # ── Fetch FX multiplier ───────────────────────────────────────────────────
    multiplier: float | None = None

    if tgt == "EUR":
        if src == "USD":
            # EURUSD=X = USD per 1 EUR → invert for USD→EUR
            multiplier = 1.0 / get_eur_rate()
        else:
            ticker = f"{src}EUR=X"
            try:
                fx_df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
                if not fx_df.empty:
                    multiplier = float(fx_df["Close"].iloc[-1])
            except Exception:
                pass
            if multiplier is None:
                # Fallback: convert through USD
                try:
                    usd_ticker = f"{src}USD=X"
                    fx_usd = yf.download(usd_ticker, period="5d", auto_adjust=True, progress=False)
                    if not fx_usd.empty:
                        usd_rate = float(fx_usd["Close"].iloc[-1])
                        multiplier = usd_rate / get_eur_rate()
                except Exception:
                    multiplier = 1.0 / get_eur_rate()  # last-resort: treat as USD

    if multiplier is None:
        return h  # cannot determine rate; return as-is

    for c in price_cols:
        h[c] = h[c] * multiplier

    return h


def format_price(value, currency_code: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    v   = float(value)
    sym = currency_symbol(currency_code)
    return f"{sym}{v:,.4f}"

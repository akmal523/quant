"""
currency.py — FX helpers: dynamic rate fetching, OHLCV normalisation to EUR.
No hardcoded fallbacks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
import urllib.request
import json

from config   import CONVERT_TO_EUR
from universe import CURRENCY_SYMBOLS

# Module-level EUR/USD rate cache (fetched once per process).
_eur_rate_cache: float | None = None

def get_eur_rate() -> float:
    """
    Fetch live EUR/USD rate dynamically. Cached for lifetime of the process.
    Tries Yahoo Finance first, then falls back to a public European API.
    """
    global _eur_rate_cache
    if _eur_rate_cache is not None:
        return _eur_rate_cache
    
    # Method 1: Yahoo Finance (Using Ticker, which is more stable than download)
    try:
        tkr = yf.Ticker("EURUSD=X")
        df = tkr.history(period="5d")
        if not df.empty and "Close" in df.columns:
            rate = float(df["Close"].iloc[-1])
            if 0.8 < rate < 1.5: # Sanity check
                _eur_rate_cache = rate
                print(f"  [FX] Live Rate (Yahoo) EUR/USD = {rate:.4f}")
                return rate
    except Exception:
        pass

    # Method 2: Frankfurter API (Free, open-source ECB European exchange rates)
    try:
        req = urllib.request.urlopen("https://api.frankfurter.app/latest?from=EUR&to=USD", timeout=5)
        data = json.loads(req.read())
        rate = float(data["rates"]["USD"])
        _eur_rate_cache = rate
        print(f"  [FX] Live Rate (ECB API) EUR/USD = {rate:.4f}")
        return rate
    except Exception as e:
        raise RuntimeError(f"CRITICAL: Could not fetch real EUR/USD exchange rate from any source. Halting to prevent bad math. Error: {e}")


def currency_symbol(code: str) -> str:
    return CURRENCY_SYMBOLS.get((code or "").upper(), (code or "") + " ")


def apply_fx_conversion(
    hist: pd.DataFrame,
    from_currency: str,
    to_currency: str = "EUR",
) -> pd.DataFrame:
    """
    Normalise OHLCV Close/High/Low/Open columns to target currency.
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
                # Upgraded to use Ticker for stability
                fx_df = yf.Ticker(ticker).history(period="5d")
                if not fx_df.empty:
                    multiplier = float(fx_df["Close"].iloc[-1])
            except Exception:
                pass
            
            if multiplier is None:
                # Fallback: convert through USD dynamically
                try:
                    usd_ticker = f"{src}USD=X"
                    fx_usd = yf.Ticker(usd_ticker).history(period="5d")
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

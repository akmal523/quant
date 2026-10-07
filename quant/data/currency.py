"""
currency.py — FX helpers: dynamic rate fetching, OHLCV normalisation to EUR.
Graceful degradation: falls back to cached or 1.0 rate rather than crashing.
"""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd
import yfinance as yf

from quant.cli.output import reporter
from quant.data.universe import CURRENCY_SYMBOLS

# Module-level EUR/USD rate cache (fetched once per process).
_eur_rate_cache: float | None = None

def get_eur_rate() -> float:
    """
    Fetch live EUR/USD rate dynamically. Cached for lifetime of the process.
    Tries Yahoo Finance first, then falls back to a public European API.
    If both fail, returns last-known rate or 1.0 with a warning.
    """
    global _eur_rate_cache
    if _eur_rate_cache is not None:
        return _eur_rate_cache

    # Method 1: Yahoo Finance
    try:
        tkr = yf.Ticker("EURUSD=X")
        df = tkr.history(period="5d")
        if not df.empty and "Close" in df.columns:
            rate = float(df["Close"].iloc[-1])
            if 0.8 < rate < 1.5:
                _eur_rate_cache = rate
                reporter.detail(f"  [FX] Live Rate (Yahoo) EUR/USD = {rate:.4f}")
                return rate
    except Exception:
        pass

    # Method 2: Frankfurter API (ECB)
    try:
        req = urllib.request.urlopen(
            "https://api.frankfurter.app/latest?from=EUR&to=USD", timeout=5)
        data = json.loads(req.read())
        rate = float(data["rates"]["USD"])
        _eur_rate_cache = rate
        reporter.detail(f"  [FX] Live Rate (ECB API) EUR/USD = {rate:.4f}")
        return rate
    except Exception:
        pass

    # Graceful fallback: warn but don't crash
    reporter.detail("  [FX] WARNING: could not fetch live EUR/USD rate; using fallback 1.0.")
    _eur_rate_cache = 1.0
    return _eur_rate_cache


def currency_symbol(code: str) -> str:
    return CURRENCY_SYMBOLS.get((code or "").upper(), (code or "") + " ")


def deduce_currency(symbol: str) -> str:
    """Deduce native currency from a yahoo ticker suffix. Pure function.

    Intent: portfolio audit needs per-symbol native currency to convert prices
    to EUR. Symbols without a suffix are US-listed (USD). Exchange suffixes map
    to their listing currency. Invariants: returns an ISO-ish code; unknown
    suffixes default to USD.
    """
    if "." not in symbol:
        return "USD"
    suffix = symbol.split(".")[-1].upper()
    eur_zones = {"DE", "PA", "AS", "MI", "MC", "BR", "VI", "HE"}
    if suffix in eur_zones:
        return "EUR"
    if suffix == "L":
        return "GBX"
    if suffix == "SW":
        return "CHF"
    if suffix == "CO":
        return "DKK"
    if suffix == "OL":
        return "NOK"
    if suffix == "ST":
        return "SEK"
    if suffix == "TO":
        return "CAD"
    if suffix == "AX":
        return "AUD"
    if suffix == "KS":
        return "KRW"
    return "USD"


def get_fx_to_eur(symbol: str) -> float:
    """Return multiplier converting a symbol's native price to EUR.

    Intent: portfolio audit converts native-currency prices to EUR for real PnL.
    EUR -> 1.0; USD -> 1/EURUSD (live). GBX -> GBP -> EUR via USD proxy. Other
    currencies fall back to apply_fx_conversion on a single-row frame; unknown
    rates degrade to 1.0 (no crash). Dependencies: get_eur_rate, apply_fx_conversion.
    """
    ccy = deduce_currency(symbol)
    if ccy == "EUR":
        return 1.0
    if ccy == "USD":
        return 1.0 / get_eur_rate()
    if ccy == "GBX":
        return (1.0 / 100.0) / get_eur_rate()
    dummy = pd.DataFrame({"Close": [1.0]})
    conv = apply_fx_conversion(dummy, ccy, "EUR")
    return float(conv["Close"].iloc[0])


def _native_close(symbol: str, as_of=None, conn=None) -> float | None:
    """The native-currency close on (or before) a date; None when absent."""
    sql = "SELECT Close FROM market_history WHERE Symbol = ?"
    params: list = [symbol]
    if as_of is not None:
        sql += " AND Date <= ?"
        params.append(str(as_of)[:10])
    sql += " ORDER BY Date DESC LIMIT 1"
    try:
        if conn is not None:
            row = conn.execute(sql, params).fetchone()
        else:
            from quant.data.database import read_only_connection

            with read_only_connection() as c:
                row = c.execute(sql, params).fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not row or row[0] is None:
        return None
    try:
        return float(row[0])
    except (TypeError, ValueError):
        return None


def price_in_eur(symbol: str, as_of=None, conn=None) -> float | None:
    """The EUR price of a symbol on (or before) a date (v10.8.0, 2.2).

    Intent: valuation, the daily job, the Overview estimate, buy and sell
    recording, the chart series and the audit all need a price in EUR. The
    native close comes from ``market_history``; the FX multiplier comes from
    :func:`get_fx_to_eur` (the documented nearest-available rate). Returns None
    when no close exists.

    ``as_of`` may be a date, an ISO string, or None (latest). ``conn`` is an
    optional open connection; when omitted a short-lived read-only connection is
    used.
    """
    close = _native_close(symbol, as_of, conn)
    if close is None:
        return None
    try:
        return close * get_fx_to_eur(symbol)
    except Exception:  # noqa: BLE001
        return close


def apply_fx_conversion(
    hist: pd.DataFrame,
    from_currency: str,
    to_currency: str = "EUR",
) -> pd.DataFrame:
    """
    Normalise OHLCV Close/High/Low/Open columns to target currency.
    Returns the original DataFrame unchanged if conversion is not needed or fails.
    """
    src = (from_currency or "").upper().strip()
    tgt = (to_currency  or "EUR").upper().strip()

    price_cols = [c for c in ("Open", "High", "Low", "Close") if c in hist.columns]
    h = hist.copy()

    # GBX (pence) -> GBP
    if src == "GBX":
        for c in price_cols:
            h[c] = h[c] / 100.0
        src = "GBP"

    if src == tgt:
        return h

    # Fetch FX multiplier
    multiplier: float | None = None

    if tgt == "EUR":
        if src == "USD":
            multiplier = 1.0 / get_eur_rate()
        else:
            ticker = f"{src}EUR=X"
            try:
                fx_df = yf.Ticker(ticker).history(period="5d")
                if not fx_df.empty:
                    multiplier = float(fx_df["Close"].iloc[-1])
            except Exception:
                pass

            if multiplier is None:
                # Fallback: convert through USD
                try:
                    usd_ticker = f"{src}USD=X"
                    fx_usd = yf.Ticker(usd_ticker).history(period="5d")
                    if not fx_usd.empty:
                        usd_rate = float(fx_usd["Close"].iloc[-1])
                        multiplier = usd_rate / get_eur_rate()
                except Exception:
                    multiplier = 1.0 / get_eur_rate()  # treat as USD

    if multiplier is None:
        return h  # cannot determine rate; return as-is

    for c in price_cols:
        h[c] = h[c] * multiplier

    return h


def format_price(value, currency_code: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "\u2014"
    v   = float(value)
    sym = currency_symbol(currency_code)
    return f"{sym}{v:,.4f}"

"""
currency.py — multi-currency FX fetching and EUR-equivalent price formatting.

Changes from v4.5:
- Expanded from EUR/USD only to cover all currencies present in the watchlist:
  USD, GBP, NOK, DKK, KRW, AUD, CAD.
- _FX_CACHE stores {ISO_CODE: eur_per_unit} so any price can be converted
  to EUR with a single lookup.
- get_eur_rate() retained for backward-compatibility; delegates to get_fx_rates().
- eur_equivalent(value, currency) — new helper for clean EUR conversion.
- All rates are fetched in one batch at startup; fallback values used if any
  ticker fails.
"""

import numpy as np
import yfinance as yf
from config import CONVERT_TO_EUR
from tickers import CURRENCY_SYMBOLS

# ─────────────────────────────────────────────────────────────────────────────
# FX TABLE
# Maps ISO currency code → Yahoo Finance EUR cross ticker
# Convention: all rates stored as EUR per 1 unit of foreign currency
# i.e. _FX_CACHE["USD"] = 0.92  means 1 USD = €0.92
# ─────────────────────────────────────────────────────────────────────────────

_FX_PAIRS: dict[str, str] = {
    "USD": "USDEUR=X",
    "GBP": "GBPEUR=X",
    "NOK": "NOKEUR=X",
    "DKK": "DKKEUR=X",
    "KRW": "KRWEUR=X",
    "AUD": "AUDEUR=X",
    "CAD": "CADEUR=X",
    "CHF": "CHFEUR=X",
    "SEK": "SEKEUR=X",
    "JPY": "JPYEUR=X",
}

# Fallback rates (EUR per unit) — updated periodically, used only if live fetch fails
_FX_FALLBACK: dict[str, float] = {
    "USD": 0.925,
    "GBP": 1.165,
    "NOK": 0.086,
    "DKK": 0.134,
    "KRW": 0.000068,
    "AUD": 0.605,
    "CAD": 0.685,
    "CHF": 1.040,
    "SEK": 0.088,
    "JPY": 0.0063,
    "EUR": 1.0,
}

_FX_CACHE: dict[str, float] = {}


def get_fx_rates() -> dict[str, float]:
    """
    Fetch live EUR-per-unit rates for all watchlist currencies.
    Results are cached for the lifetime of the process.
    Returns {ISO_CODE: eur_per_unit}.
    """
    global _FX_CACHE
    if _FX_CACHE:
        return _FX_CACHE

    _FX_CACHE["EUR"] = 1.0
    tickers_to_fetch = list(_FX_PAIRS.values())

    try:
        raw = yf.download(
            tickers_to_fetch, period="5d",
            auto_adjust=True, progress=False, group_by="ticker"
        )
        for iso, yfx in _FX_PAIRS.items():
            try:
                if yfx in raw.columns.get_level_values(0):
                    series = raw[yfx]["Close"].dropna()
                else:
                    series = raw["Close"][yfx].dropna() if "Close" in raw else None
                if series is not None and not series.empty:
                    rate = float(series.iloc[-1])
                    _FX_CACHE[iso] = rate
                    continue
            except Exception:
                pass
            # Fallback for this currency
            _FX_CACHE[iso] = _FX_FALLBACK.get(iso, 1.0)
    except Exception:
        # Full fallback
        for iso, fb in _FX_FALLBACK.items():
            _FX_CACHE.setdefault(iso, fb)

    # Print summary
    fetched = {k: v for k, v in _FX_CACHE.items() if k != "EUR"}
    print(f"  [FX] Rates (EUR per unit): " +
          ", ".join(f"{k}={v:.4g}" for k, v in sorted(fetched.items())))
    return _FX_CACHE


def get_eur_rate() -> float:
    """
    Return EUR/USD rate (USD per 1 EUR) for backward-compatibility.
    Equivalent to 1 / get_fx_rates()["USD"].
    """
    rates = get_fx_rates()
    usd_per_eur = rates.get("USD", _FX_FALLBACK["USD"])
    return round(1.0 / usd_per_eur, 6) if usd_per_eur > 0 else 1.08


def eur_equivalent(value: float, currency_code: str) -> float | None:
    """
    Convert a price in `currency_code` to its EUR equivalent.
    Returns None if the currency is unknown or value is invalid.
    """
    if value is None:
        return None
    rates = get_fx_rates()
    iso = currency_code.upper()
    rate = rates.get(iso)
    if rate is None:
        return None
    return value * rate


def currency_symbol(currency_code: str) -> str:
    """Return the display symbol for a 3-letter currency code."""
    code = (currency_code or "").upper()
    return CURRENCY_SYMBOLS.get(code, code + " ")


def format_price(value, currency_code: str) -> str:
    """
    Format a price with the appropriate currency symbol.
    When CONVERT_TO_EUR is True, appends EUR equivalent in parentheses
    for all non-EUR currencies (not just USD as before).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    v = float(value)
    iso = (currency_code or "USD").upper()
    sym = currency_symbol(iso)
    base = f"{sym}{v:,.2f}"

    if CONVERT_TO_EUR and iso != "EUR":
        eur_val = eur_equivalent(v, iso)
        if eur_val is not None:
            return f"{base} (€{eur_val:,.2f})"

    return base

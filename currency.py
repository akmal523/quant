"""
currency.py — currency helpers: EUR/USD rate fetching and price formatting.
"""
import numpy as np
import yfinance as yf

from config  import CONVERT_TO_EUR
from tickers import CURRENCY_SYMBOLS

# Module-level cache so we fetch the rate only once per run.
_eur_rate_cache: float | None = None


def get_eur_rate() -> float:
    """
    Fetch the current EUR/USD rate (USD per 1 EUR).
    Result is cached for the lifetime of the process.
    Falls back to 1.08 if the feed is unavailable.
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


def currency_symbol(currency_code: str) -> str:
    """Return the display symbol for a 3-letter currency code."""
    code = (currency_code or "").upper()
    return CURRENCY_SYMBOLS.get(code, code + " ")


def format_price(value, currency_code: str) -> str:
    """
    Format a price with the appropriate currency symbol.

    USD  → shows the USD price and, when CONVERT_TO_EUR is True,
           appends the EUR equivalent in parentheses.
    EUR  → shown as-is (covers .DE / .PA / .AS / .MI tickers).
    GBP  → shown as-is (pence-to-pounds normalisation is done upstream).
    All other currencies → native symbol, no conversion.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    v   = float(value)
    sym = currency_symbol(currency_code)
    base = f"{sym}{v:,.2f}"
    if CONVERT_TO_EUR and currency_code.upper() == "USD":
        eur_val = v / get_eur_rate()
        return f"{base} (€{eur_val:,.2f})"
    return base

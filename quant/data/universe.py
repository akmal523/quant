"""
universe.py — Shared universe helpers (Plan 3, Phase 1).

Intent: the hardcoded ~300 SECTOR_UNIVERSE is DELETED. The broad 1000+ ticker
pool now lives in the universe_master table, built by universe_builder.py and
filtered by funnel.py. This module keeps only the shared, universe-agnostic
helpers: ETF detection, geo-risk tables, and currency display symbols.

Invariants:
  - is_etf() never raises and never recurses (reads broker_registry directly).
  - CURRENCY_SYMBOLS / GEO tables are pure data (no I/O).

Dependencies: pandas (lazy), os (lazy) for broker_registry read.
"""
from __future__ import annotations

from quant import paths

# ─── ETF Detection ─────────────────────────────────────────────────────────────

# Known ETF/Index tickers that name-based heuristics would miss. Kept small;
# the authoritative source is broker_registry.csv instrument_class.
KNOWN_ETFS: set[str] = {
    "ICLN", "GLD", "SLV", "COPX", "GDX", "GDXJ", "SH", "SDS",
    "URA", "URNM", "BATT", "LIT", "QTUM", "SOXX", "SMH", "BOTZ",
    "ROBO", "AIQ", "MOO", "PHO", "IQQQ.DE", "GLUG.L", "KBE", "IHF",
    "VNQI", "IPRP.L", "MTUM", "USMV", "QUAL", "PDBC", "TLT",
    "ITPS.L", "IEAG.L", "XEON.DE", "EUNL.DE", "IWDA.AS", "URTH",
    "VOO", "CSPX.L", "EIMI.L", "SWRD.L", "EXW1.DE", "EXIA.DE",
    "ISF.L", "SXRV.DE", "5J50.DE", "DFEN", "ITA", "CIBR", "BUG",
    "XLE", "TAN", "QQQ", "TQQQ",
}


def is_etf(symbol: str) -> bool:
    """Return True if the symbol is an ETF/Index.

    Intent: drive bifurcated scoring (ETFs bypass Fundamentals/NLP). Checks the
    known-ETF set first, then broker_registry.csv instrument_class. Reads the
    registry directly (NOT taxonomy.classify_instrument) to avoid a circular
    call: classify_instrument -> is_etf -> classify_instrument.
    Invariants: returns bool; never raises; no recursion.
    """
    if symbol in KNOWN_ETFS:
        return True
    try:
        import os

        import pandas as pd
        if os.path.exists(paths.DATA_BROKER_REGISTRY):
            reg = pd.read_csv(paths.DATA_BROKER_REGISTRY)
            row = reg[reg["yahoo_ticker"] == symbol]
            if not row.empty and str(row.iloc[0].get("instrument_class", "")) == "ETF":
                return True
    except Exception:
        pass
    return False


# ─── Currency Display Symbols ─────────────────────────────────────────────────

CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",    "EUR": "€",    "GBP": "£",    "GBX": "p",
    "NOK": "kr",   "DKK": "kr",   "SEK": "kr",   "AUD": "A$",
    "CAD": "C$",   "JPY": "¥",    "KRW": "₩",    "CHF": "Fr",
    "HKD": "HK$",  "SGD": "S$",   "NZD": "NZ$",  "BRL": "R$",
    "ZAR": "R",    "MXN": "$",    "PLN": "zl",   "CZK": "Kc",
}


if __name__ == "__main__":
    print("universe.py: broad universe lives in universe_master (universe_builder.py).")
    print(f"  KNOWN_ETFS: {len(KNOWN_ETFS)} | CURRENCY_SYMBOLS: {len(CURRENCY_SYMBOLS)}")

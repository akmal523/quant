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


# ─── Geo Risk Tables ──────────────────────────────────────────────────────────

GEO_BASE: dict[str, int] = {
    "United States":              2,
    "Germany":                    2,
    "France":                     2,
    "United Kingdom":             2,
    "Netherlands":                2,
    "Switzerland":                2,
    "Norway":                     2,
    "Denmark":                     2,
    "Sweden":                     2,
    "Finland":                    2,
    "Canada":                     2,
    "Australia":                  3,
    "New Zealand":                2,
    "Japan":                      3,
    "South Korea":                5,
    "Taiwan":                     6,
    "Hong Kong":                  5,
    "Singapore":                  3,
    "China":                      7,
    "India":                      4,
    "Brazil":                     5,
    "Mexico":                     5,
    "South Africa":               5,
    "Russia":                     10,
    "Ukraine":                    10,
    "Israel":                     7,
    "Iran":                       10,
    "Turkey":                     6,
    "Austria":                    3,
    "Belgium":                    2,
    "Spain":                      3,
    "Italy":                      3,
    "Portugal":                   3,
    "Greece":                     4,
    "Poland":                     3,
    "Hungary":                    4,
    "Czech Republic":             3,
    "Romania":                    4,
    "Chile":                      4,
    "Colombia":                   5,
    "Peru":                       5,
    "Kazakhstan":                 6,
    "Zambia":                     6,
    "Democratic Republic Congo":  7,
}

GEO_KEYWORDS: list[str] = [
    "sanction",    "war",          "conflict",    "invasion",    "default",
    "recession",   "fraud",        "investigation","ban",         "tariff",
    "embargo",     "collapse",     "crisis",       "downgrade",   "lawsuit",
    "bankruptcy",  "seizure",      "nationalise",  "nationalize", "fine",
    "probe",       "scandal",      "corruption",   "coup",        "protest",
    "strike",      "cyberattack",  "hack",         "data breach", "penalty",
]

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

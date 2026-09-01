"""
taxonomy.py — Asset Taxonomy & Broker Registry (Phase 4, Module 3).

Intent: bifurcate the analysis engine by instrument_class and resolve Trade
Republic routing keys (ISIN / tr_ticker / exchange) from the broker_registry.csv.
ETFs bypass Fundamentals/NLP; Commodities use macro; Equities run full pipeline.

Invariants:
  - instrument_class ∈ {EQUITY, ETF, COMMODITY, CASH}
  - universe_status ∈ {ACTIVE, WATCHLIST, CORE}
  - broker_registry.csv is the single source of truth for ISIN/tr_ticker.

Dependencies: pandas, database.get_connection, universe.is_etf.
"""
from __future__ import annotations

import os
import time
import pandas as pd

from database import get_connection

BROKER_REGISTRY_PATH = "broker_registry.csv"

# Instrument classes that bypass the heavy Fundamentals/NLP pipeline.
ETF_CLASSES = {"ETF"}
COMMODITY_CLASSES = {"COMMODITY"}
CASH_CLASSES = {"CASH"}

# ETF/commodity display-name keywords for heuristic classification fallback.
_FUND_KEYWORDS = ["etf", "ishares", "vaneck", "spdr", "select", "trust", "fund", "ucits"]
_COMMODITY_KEYWORDS = ["gold", "silver", "commodit", "oil", "copper", "uranium etf"]


def load_broker_registry(path: str = BROKER_REGISTRY_PATH) -> pd.DataFrame:
    """Load broker_registry.csv. Returns empty DataFrame if missing."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "yahoo_ticker", "isin", "tr_ticker", "exchange", "currency", "instrument_class",
        ])
    return pd.read_csv(path)


def resolve_broker(symbol: str, path: str = BROKER_REGISTRY_PATH) -> dict:
    """Resolve a yahoo_ticker to its Trade Republic routing metadata.

    Intent: map yahoo ticker -> ISIN / tr_ticker / exchange / currency for
    execution on LS Exchange / Tradegate. Falls back to symbol itself.
    Invariants: always returns a dict with all keys present.
    """
    reg = load_broker_registry(path)
    row = reg[reg["yahoo_ticker"] == symbol]
    if row.empty:
        return {
            "yahoo_ticker": symbol,
            "isin": "",
            "tr_ticker": symbol,
            "exchange": "LS Exchange",
            "currency": "",
            "instrument_class": classify_instrument(symbol),
        }
    r = row.iloc[0]
    return {
        "yahoo_ticker": symbol,
        "isin": str(r.get("isin", "")),
        "tr_ticker": str(r.get("tr_ticker", symbol)),
        "exchange": str(r.get("exchange", "LS Exchange")),
        "currency": str(r.get("currency", "")),
        "instrument_class": str(r.get("instrument_class", "EQUITY")),
    }


def classify_instrument(symbol: str, name: str = "") -> str:
    """Classify an asset into EQUITY | ETF | COMMODITY | CASH.

    Priority: broker_registry override > name keywords > universe.is_etf heuristic.
    Intent: drive bifurcated scoring pipelines (Module 3.1).
    """
    # 1. Broker registry explicit override.
    reg = load_broker_registry()
    row = reg[reg["yahoo_ticker"] == symbol]
    if not row.empty and str(row.iloc[0].get("instrument_class", "")):
        return str(row.iloc[0]["instrument_class"])

    # 2. Name-based heuristic.
    name_l = name.lower()
    if any(kw in name_l for kw in _COMMODITY_KEYWORDS):
        return "COMMODITY"
    if any(kw in name_l for kw in _FUND_KEYWORDS):
        return "ETF"

    # 3. Fallback to universe.is_etf heuristic.
    try:
        from universe import is_etf
        if is_etf(symbol):
            return "ETF"
    except Exception:
        pass

    return "EQUITY"


def get_instrument_class(symbol: str, name: str = "") -> str:
    """Public accessor for instrument_class with DB persistence."""
    cls = classify_instrument(symbol, name)
    _upsert_registry(symbol, name, cls)
    return cls


def _upsert_registry(symbol: str, name: str, instrument_class: str) -> None:
    """Persist classification into asset_registry (idempotent).

    Intent (Phase 4, 3.2): new symbols default to WATCHLIST, NOT ACTIVE.
    Only discovery.graduate() or manual pin promotes to ACTIVE. This keeps the
    heavy-analysis universe lean instead of auto-activating every scanned symbol.
    Invariants: existing rows keep their universe_status (ON CONFLICT only
    updates name/instrument_class/updated_at).
    """
    try:
        conn = get_connection()
        conn.execute(
            """INSERT INTO asset_registry (symbol, name, instrument_class, universe_status, updated_at)
               VALUES (?, ?, ?, 'WATCHLIST', ?)
               ON CONFLICT (symbol) DO UPDATE SET
                 name = excluded.name,
                 instrument_class = excluded.instrument_class,
                 updated_at = excluded.updated_at""",
            [symbol, name, instrument_class, time.time()],
        )
    except Exception:
        # Registry table may not exist yet (init_db not called). Non-fatal.
        pass


def is_etf_class(symbol: str, name: str = "") -> bool:
    """True if instrument is an ETF (bypasses Fundamentals/NLP)."""
    return classify_instrument(symbol, name) in ETF_CLASSES


def is_commodity_class(symbol: str, name: str = "") -> bool:
    """True if instrument is a commodity/ETC (macro-scored)."""
    return classify_instrument(symbol, name) in COMMODITY_CLASSES


def is_cash_class(symbol: str, name: str = "") -> bool:
    """True if instrument is a cash-equivalent (e.g. XEON money-market ETF)."""
    return classify_instrument(symbol, name) in CASH_CLASSES
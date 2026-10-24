"""
taxonomy.py — Asset Taxonomy & Broker Registry (Phase 4, Module 3).

Intent: bifurcate the analysis engine by instrument_class and resolve Trade
Republic routing keys (ISIN / tr_ticker / exchange) from the broker_registry.csv.
ETFs bypass Fundamentals/NLP; Commodities use macro; Equities run full pipeline.

Invariants:
  - instrument_class ∈ {EQUITY, ETF, COMMODITY, CASH}
  - universe_status ∈ {CORE, ACTIVE, WATCHLIST, DELISTED}
  - structure ∈ {PLAIN, INVERSE, LEVERAGED}
  - broker_registry.csv is the single source of truth for ISIN/tr_ticker.

Dependencies: pandas, database.get_connection, universe.is_etf.
"""
from __future__ import annotations

import os
import threading
import time

import pandas as pd

from quant import paths
from quant.data.database import get_connection

BROKER_REGISTRY_PATH = paths.DATA_BROKER_REGISTRY

# DuckDB allows only one writer at a time. data_updater.py runs fetch_single in a
# ThreadPoolExecutor, and each worker calls get_instrument_class() -> here. Without
# this lock those concurrent INSERTs block each other and freeze the whole run.
# Reentrant so a locked writer can safely call another writer (e.g. set_core ->
# log_universe_event) without deadlocking.
_DB_WRITE_LOCK = threading.RLock()


# ── Universe State Machine (Phase 5 / v10.2) ──────────────────────────────────
# CORE is immutable: never graduated, never demoted. Only ACTIVE (graduated
# satellites) age out. DELISTED stops all fetching.
CORE_STATUSES = {"CORE"}

# Structure flags: INVERSE/LEVERAGED products decay over time and must never be
# accumulation vehicles (never SPARPLAN, never CORE bucket).
PLAIN_STRUCTURE = "PLAIN"
INVERSE_STRUCTURE = "INVERSE"
LEVERAGED_STRUCTURE = "LEVERAGED"
ALL_STRUCTURES = {PLAIN_STRUCTURE, INVERSE_STRUCTURE, LEVERAGED_STRUCTURE}

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


def sync_broker_registry(path: str = BROKER_REGISTRY_PATH) -> int:
    """Sync broker_registry.csv ISIN/tr_ticker/exchange/currency into asset_registry.

    Intent (Phase 5 / v10.2): the registry is the single source of truth for
    routing keys. Populate asset_registry.isin so the dashboard Data Health
    section and routing instructions have real ISINs.
    Invariants: returns count of rows updated; best-effort (never raises).
    """
    reg = load_broker_registry(path)
    if reg.empty:
        return 0
    conn = get_connection()
    updated = 0
    with _DB_WRITE_LOCK:
        for _, r in reg.iterrows():
            sym = str(r.get("yahoo_ticker", ""))
            if not sym:
                continue
            isin = str(r.get("isin", "") or "")
            if isin in ("", "nan"):
                isin = None            # H3.5 (F2): missing cells are NULL, never ''
            tr_ticker = str(r.get("tr_ticker", "") or sym)
            if tr_ticker == "nan":
                tr_ticker = sym
            exchange = str(r.get("exchange", "") or "LS Exchange")
            if exchange == "nan":
                exchange = "LS Exchange"
            currency = str(r.get("currency", "") or "")
            if currency in ("", "nan"):
                currency = None        # H3.5 (F2): missing cells are NULL, never ''
            instrument_class = str(r.get("instrument_class", "") or "EQUITY")
            if instrument_class == "nan":
                instrument_class = "EQUITY"
            conn.execute(
                """INSERT INTO asset_registry (symbol, instrument_class, isin, tr_ticker,
                                              exchange, currency, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (symbol) DO UPDATE SET
                     instrument_class = excluded.instrument_class,
                     isin = excluded.isin,
                     tr_ticker = excluded.tr_ticker,
                     exchange = excluded.exchange,
                     currency = excluded.currency,
                     updated_at = excluded.updated_at""",
                [sym, instrument_class, isin, tr_ticker, exchange, currency, time.time()],
            )
            updated += 1
    return updated


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
            "isin_source": "",
            "tr_ticker": symbol,
            "exchange": "LS Exchange",
            "currency": "",
            "instrument_class": classify_instrument(symbol),
        }
    r = row.iloc[0]
    isin = str(r.get("isin", "") or "")
    if isin == "nan":
        isin = ""
    # v10.5.2 (A6): provenance drives the confirm-in-broker caveat in the UI.
    isin_source = str(r.get("isin_source", "") or "")
    if isin_source == "nan":
        isin_source = ""
    return {
        "yahoo_ticker": symbol,
        "isin": isin,
        "isin_source": isin_source,
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
        from quant.data.universe import is_etf
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
        with _DB_WRITE_LOCK:
            conn = get_connection()
            conn.execute(
                """INSERT INTO asset_registry
                   (symbol, name, instrument_class, universe_status, updated_at)
                   VALUES (?, ?, ?, 'WATCHLIST', ?)
                   ON CONFLICT (symbol) DO UPDATE SET
                     name = COALESCE(excluded.name, name),
                     instrument_class = excluded.instrument_class,
                     updated_at = excluded.updated_at""",
                [symbol, (name or None), instrument_class, time.time()],
            )
    except Exception:
        # Registry table may not exist yet (init_db not called). Non-fatal.
        pass


def log_universe_event(symbol: str, event: str, reason: str) -> None:
    """Append a state-change row to universe_events (audit trail).

    Intent (Phase 5 / v10.2): every GRADUATE / DEMOTE / DELIST / PIN / ADD is
    recorded so the dashboard can render an auditable event log.
    Invariants: best-effort; never raises (table may not exist yet).
    """
    try:
        with _DB_WRITE_LOCK:
            conn = get_connection()
            conn.execute(
                "INSERT INTO universe_events (symbol, event, reason) VALUES (?, ?, ?)",
                [symbol, event, reason],
            )
    except Exception:
        pass


def set_core(symbol: str) -> None:
    """Set a symbol to CORE status (immutable sleeve).

    Intent (Phase 5 / v10.2): CORE assets are never graduated, never demoted.
    Invariants: status becomes CORE; event logged.
    """
    with _DB_WRITE_LOCK:
        conn = get_connection()
        conn.execute(
            """INSERT INTO asset_registry (symbol, universe_status, updated_at)
               VALUES (?, 'CORE', ?)
               ON CONFLICT (symbol) DO UPDATE SET
                 universe_status = 'CORE',
                 updated_at = excluded.updated_at""",
            [symbol, time.time()],
        )
        log_universe_event(symbol, "PIN", "set to CORE (immutable sleeve)")


def mark_delisted(symbol: str, reason: str = "consecutive fetch failures") -> None:
    """Mark a symbol DELISTED and stop all fetching.

    Intent (Phase 5 / v10.2): after MAX_FETCH_FAILURES consecutive failures, a
    symbol is archived. ZNWD.L is the first case. Delisted symbols are excluded
    from build_fetch_list() and main.py active_symbols.
    Invariants: status becomes DELISTED; event logged.
    """
    conn = get_connection()
    conn.execute(
        """INSERT INTO asset_registry (symbol, universe_status, updated_at)
           VALUES (?, 'DELISTED', ?)
           ON CONFLICT (symbol) DO UPDATE SET
             universe_status = 'DELISTED',
             updated_at = excluded.updated_at""",
        [symbol, time.time()],
    )
    log_universe_event(symbol, "DELIST", reason)


def set_structure(symbol: str, structure: str) -> None:
    """Set the structure flag for a symbol.

    Intent (Phase 5 / v10.2): repair script and dashboard mark inverse/leveraged
    products so routing excludes them from accumulation.
    Invariants: structure must be in ALL_STRUCTURES; event logged.
    """
    if structure not in ALL_STRUCTURES:
        raise ValueError(f"invalid structure {structure!r}; must be one of {ALL_STRUCTURES}")
    conn = get_connection()
    conn.execute(
        """INSERT INTO asset_registry (symbol, structure, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT (symbol) DO UPDATE SET
             structure = excluded.structure,
             updated_at = excluded.updated_at""",
        [symbol, structure, time.time()],
    )
    log_universe_event(symbol, "ADD", f"structure set to {structure}")

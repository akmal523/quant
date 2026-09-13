"""
registry_repair.py — Validated ISIN backfill (v10.5.2, A6/F1).

Intent: fill MISSING ISIN cells in data/broker_registry.csv from a source
hierarchy — data/isin_curated.csv (user-verified) > existing non-empty cells
(never overwritten) > live Yahoo instrument metadata. Every written value passes
is_valid_isin; invalid or absent metadata yields "not found", never a guess.
Provenance is recorded in the isin_source column (user/curated/yahoo).

Invariants:
  - Idempotent: a second run changes nothing.
  - Missing cells only; existing ISINs are never overwritten.
  - No synthesized identifiers (F1): a value is written only if checksum-valid.

Dependencies: pandas, quant.paths, quant.data.identifiers.
"""
from __future__ import annotations

import os

import pandas as pd

from quant import paths
from quant.data.identifiers import is_valid_isin


def load_curated(path: str = paths.DATA_ISIN_CURATED) -> dict[str, str]:
    """Load user-verified ISINs. Aborts with a plain error on an invalid row.

    Intent (F1): a curated row is the highest-trust source, so a malformed one is
    a hard error naming the row, never a silent skip.
    """
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, comment="#")
    if df.empty or "symbol" not in df.columns or "isin" not in df.columns:
        return {}
    curated: dict[str, str] = {}
    for _, row in df.iterrows():
        sym = str(row.get("symbol", "") or "").strip()
        isin = str(row.get("isin", "") or "").strip()
        if not sym:
            continue
        if not is_valid_isin(isin):
            raise ValueError(
                f"curated row {sym}: invalid ISIN {isin!r} "
                f"(fails ISO 6166 checksum)"
            )
        curated[sym] = isin
    return curated


def _yahoo_isin(symbol: str) -> str | None:
    """Best-effort live ISIN from Yahoo instrument metadata. Never raises."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        value = getattr(ticker, "isin", None)
        if not value:
            info = ticker.get_info()
            value = info.get("isin")
        value = str(value).strip() if value else ""
        return value or None
    except Exception:  # noqa: BLE001
        return None


def repair_isins(
    registry_path: str = paths.DATA_BROKER_REGISTRY,
    curated_path: str = paths.DATA_ISIN_CURATED,
    metadata_source=None,
) -> dict:
    """Fill missing ISIN cells only. Returns a plain summary dict.

    Source hierarchy: curated > existing non-empty cells > live metadata.
    ``metadata_source`` is injectable for hermetic tests.
    """
    metadata_source = metadata_source or _yahoo_isin
    curated = load_curated(curated_path)
    if not os.path.exists(registry_path):
        return {"filled": 0, "not_found": 0, "curated": 0}

    df = pd.read_csv(registry_path)
    if "isin" not in df.columns:
        df["isin"] = ""
    added_source_col = "isin_source" not in df.columns
    if added_source_col:
        df["isin_source"] = ""

    changed = added_source_col
    filled = 0
    not_found = 0
    curated_used = 0

    for i, row in df.iterrows():
        isin = str(row.get("isin", "") or "").strip()
        if isin and isin.lower() != "nan":
            # Pre-existing cell: never overwritten. Default provenance = user.
            if not str(row.get("isin_source", "") or "").strip():
                df.at[i, "isin_source"] = "user"
                changed = True
            continue

        sym = str(row.get("yahoo_ticker", "") or "").strip()
        if not sym:
            continue

        candidate = curated.get(sym)
        source = "curated"
        if not candidate:
            candidate = metadata_source(sym)
            source = "yahoo"

        if candidate and is_valid_isin(candidate):
            df.at[i, "isin"] = str(candidate).strip().upper()
            df.at[i, "isin_source"] = source
            filled += 1
            changed = True
            if source == "curated":
                curated_used += 1
        else:
            not_found += 1

    if changed:
        df.to_csv(registry_path, index=False)
    return {"filled": filled, "not_found": not_found, "curated": curated_used}

def _currency(symbol: str) -> str:
    """Currency from the exchange suffix (pure heuristic; not an identifier)."""
    suffix = symbol.split(".")[-1].upper() if "." in symbol else ""
    if suffix in {"DE", "PA", "AS", "MI", "MC", "BR", "VI", "HE"}:
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
    return "USD"


def ensure_registry_rows(registry_path: str = paths.DATA_BROKER_REGISTRY,
                         symbols: list[str] | None = None) -> dict:
    """Create missing broker_registry rows for symbols the product already knows.

    Intent (H3.1): a HELD symbol with no registry row is unroutable — worse than a
    missing ISIN. Before the missing-cells repair runs, add a row (empty ISIN,
    provenance tracked) for every HELD or CORE/ACTIVE symbol (bounded).
    Invariants: idempotent; existing rows are never modified; missing rows only.
    """
    from quant.execution.taxonomy import classify_instrument

    columns = ["yahoo_ticker", "isin", "tr_ticker", "exchange", "currency",
               "instrument_class"]
    if os.path.exists(registry_path):
        df = pd.read_csv(registry_path)
    else:
        df = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    known = {str(v).strip().upper() for v in df["yahoo_ticker"].dropna()}

    if symbols is None:
        symbols = []
        try:
            from quant.portfolio.portfolio import load_portfolio

            pf = load_portfolio(str(paths.DATA_PORTFOLIO))
            if not pf.empty and "Symbol" in pf.columns:
                symbols.extend(str(x) for x in pf["Symbol"].tolist())
        except Exception:  # noqa: BLE001
            pass
        try:
            # BOUNDED: only the routable set (CORE/ACTIVE registry rows), never the
            # whole 1000+ universe_master, which would trigger a bulk ISIN fetch.
            from quant.data.database import read_only_connection

            with read_only_connection() as conn:
                rows = conn.execute(
                    "SELECT symbol FROM asset_registry "
                    "WHERE universe_status IN ('CORE', 'ACTIVE')"
                ).fetchall()
            symbols.extend(str(r[0]) for r in rows)
        except Exception:  # noqa: BLE001
            pass

    added = 0
    for sym in symbols:
        sym = str(sym or "").strip()
        if not sym or sym.upper() in known:
            continue
        df.loc[len(df)] = {
            "yahoo_ticker": sym, "isin": "", "tr_ticker": sym,
            "exchange": "LS Exchange", "currency": _currency(sym),
            "instrument_class": classify_instrument(sym),
        }
        known.add(sym.upper())
        added += 1
    if added:
        df.to_csv(registry_path, index=False)
    return {"added": added}


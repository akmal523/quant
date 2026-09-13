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

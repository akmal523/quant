"""
names.py — Display-name cleaning + backfill (v10.5.3, R5).

Intent: one home for the friendly-name pipeline. Fill order: curated
(data/names_curated.csv) > cleaned Yahoo longName > symbol. Clean rules drop
boilerplate tokens; an empty result falls back to the symbol. Backfill fills
MISSING cells only and never overwrites curated/existing values.

Invariants:
  - clean_display_name never returns empty (falls back to the symbol).
  - backfill_display_names is idempotent and missing-cells-only.
  - Cleaning is pure (no I/O); backfill touches only the rows it reads.

Dependencies: pandas, re, quant.paths.
"""
from __future__ import annotations

import os
import re

from quant import paths

# Boilerplate tokens removed by the clean rules (spec 5.1).
_BOILERPLATE = ["UCITS", "ETF", "USD", "EUR", "PLC", "Inc", "Corp",
                "NV", "SA", "SE", "AG", "Ltd"]
_PAREN = re.compile(r"\((?:acc|dist)\)", re.IGNORECASE)


def clean_display_name(long_name: str | None, symbol: str) -> str:
    """Clean a Yahoo longName into a short display name. Never empty."""
    if long_name is None or str(long_name).strip() == "":
        return symbol
    s = _PAREN.sub(" ", str(long_name))
    for tok in _BOILERPLATE:
        s = re.sub(rf"(?i)\b{re.escape(tok)}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -,.:")
    return s or symbol


def load_curated_names(path: str | None = None) -> dict[str, str]:
    """Load data/names_curated.csv (symbol -> display_name). Curated wins forever."""
    import pandas as pd

    path = path or paths.DATA_NAMES_CURATED
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, comment="#")
    except Exception:  # noqa: BLE001
        return {}
    if df.empty or "symbol" not in df.columns or "display_name" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        sym = str(r.get("symbol", "") or "").strip()
        name = str(r.get("display_name", "") or "").strip()
        if sym and name:
            out[sym] = name
    return out


def _yahoo_identity(symbol: str) -> tuple[str, str]:
    """Best-effort (long_name, currency) from Yahoo metadata. Never raises."""
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).get_info()
        name = str(info.get("longName") or info.get("shortName") or "")
        return name, str(info.get("currency") or "")
    except Exception:  # noqa: BLE001
        return "", ""


def backfill_display_names(conn, metadata_source=None, curated_path: str | None = None) -> dict:
    """Fill MISSING display_name / name / currency cells. Returns a summary.

    ``metadata_source`` is injectable (returns (long_name, currency)) for hermetic
    tests. Curated values are never overwritten; existing cells are left alone.
    """
    metadata_source = metadata_source or _yahoo_identity
    curated = load_curated_names(curated_path)
    rows = conn.execute(
        "SELECT symbol, display_name, name, currency FROM asset_registry"
    ).fetchall()
    d_fill = n_fill = c_fill = 0
    for sym, dn, nm, cur in rows:
        sym = str(sym or "")
        if not sym:
            continue
        need_dn = not (dn and str(dn).strip())
        need_nm = not (nm and str(nm).strip())
        need_cur = not (cur and str(cur).strip())
        if not (need_dn or need_nm or need_cur):
            continue
        long_name, meta_cur = "", ""
        if need_dn or need_nm:
            if sym in curated:
                long_name = curated[sym]
            else:
                long_name, meta_cur = metadata_source(sym)
        if need_dn:
            conn.execute("UPDATE asset_registry SET display_name = ? WHERE symbol = ?",
                         [clean_display_name(long_name, sym), sym])
            d_fill += 1
        if need_nm and long_name:
            conn.execute("UPDATE asset_registry SET name = ? WHERE symbol = ?",
                         [long_name, sym])
            n_fill += 1
        if need_cur and meta_cur:
            conn.execute("UPDATE asset_registry SET currency = ? WHERE symbol = ?",
                         [meta_cur, sym])
            c_fill += 1
    return {"display_filled": d_fill, "name_filled": n_fill, "currency_filled": c_fill}

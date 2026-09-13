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
import time

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


def _state_path() -> str:
    return os.path.join(str(paths.OUTPUTS_DIR), "names_state.json")


def read_names_state() -> dict:
    """Read outputs/names_state.json (the last backfill outcome). {} when absent."""
    import json

    try:
        with open(_state_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _write_names_state(state: dict) -> None:
    import json
    import tempfile

    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(paths.OUTPUTS_DIR), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, _state_path())
    except Exception:  # noqa: BLE001
        pass


def probe_metadata(symbol: str) -> dict:
    """Read-only metadata probe for quant doctor (raw longName or the exact error)."""
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).get_info()
        return {"symbol": symbol,
                "long_name": str(info.get("longName") or info.get("shortName") or ""),
                "currency": str(info.get("currency") or ""),
                "error": None}
    except Exception as e:  # noqa: BLE001
        return {"symbol": symbol, "long_name": "", "currency": "",
                "error": f"{type(e).__name__}: {e}"}


def _blank(v) -> bool:
    return v is None or str(v).strip() == "" or str(v).strip().lower() == "nan"


def _symbolish(value, symbol: str) -> bool:
    """True when a cell is empty OR merely repeats the ticker (a poisoned fill).

    Intent (H2): a display_name/name equal to the symbol carries no information,
    so it must be treated as MISSING and repaired on the next backfill.
    """
    return _blank(value) or str(value).strip().upper() == str(symbol).strip().upper()


def ensure_display_names() -> dict:
    """Startup backfill: migrate + fill missing names. Never raises.

    The ONLY write-enabled connection outside quant update/run/publish (app
    startup, spec B1/B4). Opens a short-lived connection and closes immediately;
    later UI reads stay read_only. Idempotent: missing cells only.
    """
    try:
        from quant.data.database import connect_with_retry, get_connection, migrate_registry_display_name
    except Exception:  # noqa: BLE001
        return {}
    own = False
    conn = None
    try:
        conn = connect_with_retry(attempts=2, delay=0.3)
        own = True
    except Exception:  # noqa: BLE001
        try:
            conn = get_connection()
        except Exception:  # noqa: BLE001
            # Lock held (another app instance / update running): skip backfill for
            # this session; the app still starts. Retried on the next start/update.
            import logging

            logging.getLogger("quant.ui").warning(
                "Display names backfill skipped: database busy.")
            print("Display names backfill skipped: database busy.")
            _write_names_state({"ts": time.time(), "rows_total": 0, "filled": 0,
                                "still_missing": 0, "skipped_reason": "lock",
                                "unreachable_symbols": 0})
            return {"skipped_reason": "lock"}
    try:
        migrate_registry_display_name(conn)
        summary = backfill_display_names(conn)
        try:
            rows_total = conn.execute("SELECT COUNT(*) FROM asset_registry").fetchone()[0]
            still_missing = conn.execute(
                "SELECT COUNT(*) FROM asset_registry "
                "WHERE display_name IS NULL OR trim(display_name) = ''"
            ).fetchone()[0]
        except Exception:  # noqa: BLE001
            rows_total = still_missing = 0
        _write_names_state({
            "ts": time.time(), "rows_total": int(rows_total),
            "filled": summary.get("display_filled", 0),
            "still_missing": int(still_missing),
            "skipped_reason": "metadata_unreachable" if summary.get("unreachable") else None,
            "unreachable_symbols": summary.get("unreachable", 0),
        })
        return summary
    except Exception:  # noqa: BLE001
        return {}
    finally:
        if own and conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


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
    unreachable = 0
    for sym, dn, nm, cur in rows:
        sym = str(sym or "")
        if not sym:
            continue
        need_dn = _symbolish(dn, sym)
        need_nm = _symbolish(nm, sym)
        need_cur = _blank(cur)
        if not (need_dn or need_nm or need_cur):
            continue
        long_name, meta_cur = "", ""
        need_meta = (sym not in curated) and _symbolish(nm, sym)
        if need_dn or need_nm or need_cur:
            if sym in curated:
                long_name = curated[sym]
            elif not _symbolish(nm, sym):
                # Reuse whatever real name we already have before the network.
                long_name = str(nm)
        if need_meta:
            long_name, meta_cur = metadata_source(sym)
            if not long_name:
                unreachable += 1
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
    return {"display_filled": d_fill, "name_filled": n_fill,
            "currency_filled": c_fill, "unreachable": unreachable}

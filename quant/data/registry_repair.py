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

import logging
import os
import time

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


def _registry_bound() -> int:
    """CSV ceiling: portfolio + CORE_ETFS + a small buffer (H3-fix).

    Intent: the CSV is user-owned routing input, not the working universe. It is
    fed only by portfolio + CORE/ACTIVE + curated rows, never the 1000+ pool.
    """
    try:
        from quant.config import CORE_ETFS
        from quant.portfolio.portfolio import load_portfolio

        pf = load_portfolio(str(paths.DATA_PORTFOLIO))
        n_port = 0 if pf.empty else len(pf)
        return n_port + len(CORE_ETFS) + 20
    except Exception:  # noqa: BLE001
        return 50


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

    # Bounded invariant: refuse to grow the registry past the routable ceiling.
    _bound = _registry_bound()
    if len(df) > _bound:
        logging.getLogger("quant.data").warning(
            "broker_registry has %d rows, above the bounded ceiling %d; "
            "refusing to add more", len(df), _bound)
        return {"added": 0, "warning": "registry_above_bound"}

    # H3-fix part 2: never accept a bulk universe_master source. A caller passing
    # the 1000+ discovery pool gets a logged no-op, not a bulk insert.
    if symbols is not None and len(symbols) > _bound:
        logging.getLogger("quant.data").warning(
            "ensure_registry_rows refused %d symbols (bulk universe_master source); "
            "the registry is fed only by portfolio + CORE/ACTIVE + curated.",
            len(symbols))
        return {"added": 0, "warning": "bulk_source_refused"}

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


# ── H3-fix part 2: the working universe (canonical definition) ────────────────
# The SINGLE source of the term "working universe" (W). asset_registry holds
# exactly W, nothing else. universe_master is EXCLUDED everywhere — the defect
# class being killed is bulk-inserting the 1000+ discovery pool.
#
#   W = funnel survivors (cached)
#     ∪ CORE ∪ ACTIVE (asset_registry.universe_status)
#     ∪ portfolio.csv symbols
#     ∪ broker_registry.csv symbols
#     ∪ curated ISIN/name symbols
#     ∪ BROAD_ETFS (the named always-tracked constant)
#
# Writers of asset_registry and their source sets are documented in CONTEXT.md
# ("Registry write paths").

def working_universe_sources(
    conn=None,
    *,
    portfolio_path: str = paths.DATA_PORTFOLIO,
    registry_path: str = paths.DATA_BROKER_REGISTRY,
    isin_curated_path: str = paths.DATA_ISIN_CURATED,
    names_curated_path: str = paths.DATA_NAMES_CURATED,
) -> dict[str, set[str]]:
    """Per-source membership of W. Pure reads; never raises."""
    sources: dict[str, set[str]] = {
        "broad_etfs": set(), "portfolio": set(), "broker_registry": set(),
        "curated": set(), "core_active": set(), "survivors": set(),
    }
    try:
        from quant.data.universe_builder import BROAD_ETFS

        sources["broad_etfs"] = {str(s).strip().upper()
                                 for s in BROAD_ETFS.values() if str(s).strip()}
    except Exception:  # noqa: BLE001
        pass
    try:
        from quant.portfolio.portfolio import load_portfolio

        pf = load_portfolio(str(portfolio_path))
        if not pf.empty and "Symbol" in pf.columns:
            sources["portfolio"] = {str(x).strip().upper()
                                    for x in pf["Symbol"].tolist() if str(x).strip()}
    except Exception:  # noqa: BLE001
        pass
    try:
        if os.path.exists(registry_path):
            reg = pd.read_csv(registry_path)
            if "yahoo_ticker" in reg.columns:
                sources["broker_registry"] = {str(x).strip().upper()
                                              for x in reg["yahoo_ticker"].dropna()
                                              if str(x).strip()}
    except Exception:  # noqa: BLE001
        pass
    try:
        sources["curated"] = {str(s).strip().upper()
                              for s in load_curated(isin_curated_path) if str(s).strip()}
    except Exception:  # noqa: BLE001
        pass
    try:
        from quant.data.names import load_curated_names

        sources["curated"] |= {str(s).strip().upper()
                               for s in load_curated_names(names_curated_path)
                               if str(s).strip()}
    except Exception:  # noqa: BLE001
        pass
    if conn is not None:
        try:
            sources["core_active"] = {str(r[0]).strip().upper() for r in conn.execute(
                "SELECT symbol FROM asset_registry WHERE universe_status IN ('CORE','ACTIVE')"
            ).fetchall() if r[0]}
        except Exception:  # noqa: BLE001
            pass
        try:
            sources["survivors"] = {str(r[0]).strip().upper() for r in conn.execute(
                "SELECT symbol FROM funnel_survivors"
            ).fetchall() if r[0]}
        except Exception:  # noqa: BLE001
            pass
    for key in sources:
        sources[key].discard("")
    return sources


def working_universe(conn=None, **kwargs) -> set[str]:
    """The working universe W (see module note). Union of all sources."""
    union: set[str] = set()
    for members in working_universe_sources(conn, **kwargs).values():
        union |= members
    return union


def sync_registry_to_working_universe(conn, **kwargs) -> dict:
    """Two-way sync of asset_registry to W (H3-fix part 2).

    Intent: asset_registry holds exactly W. Insert missing W members (status
    WATCHLIST; class from taxonomy; currency by suffix heuristic); prune rows
    outside W. Membership only: surviving rows keep every cell. Idempotent.
    A fresh (empty) registry is a no-op — nothing stale to heal.
    """
    n_before = int(conn.execute("SELECT COUNT(*) FROM asset_registry").fetchone()[0])
    if n_before == 0:
        return {"added": 0, "pruned": 0, "total": 0, "skipped": "empty"}

    wu = working_universe(conn, **kwargs)
    rows = conn.execute("SELECT symbol FROM asset_registry").fetchall()
    existing = {str(r[0]).strip().upper(): str(r[0]) for r in rows if r[0]}
    to_prune = [existing[u] for u in existing if u not in wu]
    to_add = sorted(wu - set(existing))

    if to_prune:
        placeholders = ",".join(["?"] * len(to_prune))
        conn.execute(
            f"DELETE FROM asset_registry WHERE symbol IN ({placeholders})", to_prune)

    added = 0
    if to_add:
        from quant.data.universe_builder import BROAD_ETFS
        from quant.execution.taxonomy import classify_instrument

        # H3.5 (F3): classify WITH a known name so keyword rules fire (e.g.
        # SGLN.L "Silver ETF" -> COMMODITY). classify_instrument prefers the CSV
        # class when a broker_registry row exists, else the taxonomy heuristics.
        broad = {str(s).strip().upper(): n for n, s in BROAD_ETFS.items()}
        for sym in to_add:
            nm = broad.get(sym, "") or None
            conn.execute(
                """INSERT INTO asset_registry
                       (symbol, name, instrument_class, currency, universe_status, updated_at)
                   VALUES (?, ?, ?, ?, 'WATCHLIST', ?)
                   ON CONFLICT (symbol) DO NOTHING""",
                [sym, nm, classify_instrument(sym, nm or ""), _currency(sym), time.time()],
            )
            added += 1

    # Fill blank currency cells with the pure suffix heuristic (H3-fix part 2;
    # a venue suffix, not an identifier). Non-blank cells are never touched.
    for (sym,) in conn.execute(
        "SELECT symbol FROM asset_registry "
        "WHERE currency IS NULL OR trim(currency) = ''"
    ).fetchall():
        if sym:
            conn.execute(
                "UPDATE asset_registry SET currency = ? WHERE symbol = ?",
                [_currency(str(sym)), sym],
            )

    total = int(conn.execute("SELECT COUNT(*) FROM asset_registry").fetchone()[0])
    if added or to_prune:
        logging.getLogger("quant.data").info(
            "registry sync: +%d -%d", added, len(to_prune))
    return {"added": added, "pruned": len(to_prune), "total": total}


def _isin_cache_path() -> str:
    return os.path.join(str(paths.OUTPUTS_DIR), "isin_cache.json")


def _read_isin_cache() -> dict:
    import json

    try:
        with open(_isin_cache_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _write_isin_cache(cache: dict) -> None:
    import json
    import tempfile

    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(paths.OUTPUTS_DIR), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        os.replace(tmp, _isin_cache_path())
    except Exception:  # noqa: BLE001
        pass


def heal_registry_isins(conn, symbols=None, *,
                        curated_path: str = paths.DATA_ISIN_CURATED,
                        metadata_source=None) -> dict:
    """Fill missing asset_registry.isin for W rows (H3-fix part 2, spec 3).

    Source hierarchy: curated (data/isin_curated.csv) > cache
    (outputs/isin_cache.json) > live metadata. Every written value passes
    is_valid_isin (F1). Missing cells only; per-symbol failures stay silent.
    Idempotent: a filled cell is never refetched.
    """
    metadata_source = metadata_source or _yahoo_isin
    cache = _read_isin_cache()
    curated = {str(k).strip().upper(): v for k, v in load_curated(curated_path).items()}
    want = {str(s).strip().upper() for s in symbols} if symbols else None

    rows = conn.execute("SELECT symbol, isin FROM asset_registry").fetchall()
    filled = 0
    not_found = 0
    cache_dirty = False
    for sym, isin in rows:
        sym_u = str(sym or "").strip().upper()
        if not sym_u or (want is not None and sym_u not in want):
            continue
        cur = str(isin or "").strip()
        if cur and cur.lower() != "nan":
            continue
        candidate = curated.get(sym_u)
        if not candidate:
            entry = cache.get(sym_u)
            candidate = entry.get("isin") if isinstance(entry, dict) else entry
        if not candidate:
            candidate = metadata_source(sym_u)
            if candidate:
                cache[sym_u] = {"isin": str(candidate).strip().upper(),
                                "retrieved_at": time.time()}
                cache_dirty = True
        if candidate and is_valid_isin(candidate):
            conn.execute("UPDATE asset_registry SET isin = ? WHERE symbol = ?",
                         [str(candidate).strip().upper(), sym])
            filled += 1
        else:
            not_found += 1
    if cache_dirty:
        _write_isin_cache(cache)
    return {"filled": filled, "not_found": not_found, "curated": len(curated)}


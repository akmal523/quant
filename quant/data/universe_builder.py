"""
universe_builder.py — Broad Universe Loader (Plan 3, Phase 1).

Intent: replace the hardcoded ~300 SECTOR_UNIVERSE with a dynamic 1000+ ticker
universe built from major index constituents (S&P 500, Nasdaq 100, Russell 1000)
plus a curated broad-ETF list. Writes to the universe_master table.

State Transition: index constituents + ETFs -> dedupe -> upsert universe_master
-> funnel.py filters down to top survivors.

Invariants:
  - universe_master.symbol is unique (deduplicated across sources).
  - Symbols normalized to uppercase, stripped of whitespace.
  - Best-effort: a failed source does not abort the others.
  - Idempotent: re-running upserts, never duplicates.

Dependencies: pandas, database.get_connection, taxonomy.classify_instrument.
"""
from __future__ import annotations

import io
import time
import urllib.request

import pandas as pd

from quant.data.database import get_connection, init_db
from quant.execution.taxonomy import classify_instrument

# Curated broad-ETF list (migrated from the deleted SECTOR_UNIVERSE "Broad ETFs"
# plus the CORE_ETFS sleeve). These are always tracked accumulation vehicles.
BROAD_ETFS: dict[str, str] = {
    "MSCI World":                "URTH",
    "MSCI World (IWDA)":         "IWDA.AS",
    "MSCI World Xetra (EUNL)":   "EUNL.DE",
    "MSCI World (SWRD)":         "SWRD.L",
    "MSCI EM IMI (EIMI)":        "EIMI.L",
    "S&P 500 (CSPX)":            "CSPX.L",
    "S&P 500 (VOO)":             "VOO",
    "EURO STOXX 50 (EXW1)":      "EXW1.DE",
    "DAX (EXIA)":                "EXIA.DE",
    "FTSE 100 (ISF)":            "ISF.L",
    "Short S&P 500 (SH)":        "SH",
    "2x Short S&P (SDS)":        "SDS",
    "Commodities (PDBC)":        "PDBC",
    "Treasury Bonds 20yr (TLT)": "TLT",
    "EUR Govt Bonds (IEAG)":     "IEAG.L",
    "TIPS Inflation (ITPS)":     "ITPS.L",
    "MSCI Momentum (MTUM)":      "MTUM",
    "MSCI Min Vol (USMV)":       "USMV",
    "MSCI Quality (QUAL)":       "QUAL",
    "Euro Cash (XEON)":          "XEON.DE",
    "Nasdaq 100 UCITS (SXRV)":   "SXRV.DE",
    "Global Aero & Def (5J50)":  "5J50.DE",
    "Uranium ETF (URA)":         "URA",
    "Uranium ETF (URNM)":        "URNM",
    "Clean Energy (ICLN)":       "ICLN",
    "Energy Select (XLE)":       "XLE",
    "Copper ETF (COPX)":         "COPX",
    "Lithium ETF (LIT)":         "LIT",
    "Battery ETF (BATT)":        "BATT",
    "Quantum ETF (QTUM)":        "QTUM",
    "Semiconductor ETF (SOXX)":  "SOXX",
    "AI & Robotics (BOTZ)":      "BOTZ",
    "AI ETF (AIQ)":              "AIQ",
    "Water ETF (PHO)":           "PHO",
    "Global Water (IQQQ.DE)":    "IQQQ.DE",
    "Clean Water (GLUG.L)":      "GLUG.L",
    "Healthcare ETF (IHF)":      "IHF",
    "Defense ETF (DFEN)":        "DFEN",
    "Cybersecurity ETF (CIBR)":  "CIBR",
    "Cybersecurity ETF (BUG)":   "BUG",
    "Gold Miners (GDX)":         "GDX",
    "Gold Shares (GLD)":         "GLD",
    "Silver Shares (SLV)":       "SLV",
    "Gold Miners Jr (GDXJ)":     "GDXJ",
    "Silver ETF (SGLN.L)":       "SGLN.L",
    "Global ex-US REIT (VNQI)":  "VNQI",
    "EU REIT ETF (IPRP.L)":      "IPRP.L",
    "Bank ETF (KBE)":            "KBE",
    "Agriculture ETF (MOO)":     "MOO",
}

# Symbols the Wikipedia lists contain but Yahoo does not carry (no price data).
# Removed from universe_master so the funnel never probes them again.
NONEXISTENT_SYMBOLS: set[str] = {"LBRDK", "WBS"}

# Index constituent sources. Each maps to a Wikipedia "List of ..." page whose
# table contains a ticker/symbol column. Best-effort: a missing/failed page is
# skipped. The dedicated list pages (not the index overview pages) carry the
# full constituent tables.
INDEX_SOURCES: dict[str, str] = {
    "SP500":       "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "NASDAQ100":   "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
    "RUSSELL1000": "https://en.wikipedia.org/wiki/List_of_Russell_1000_companies",
}

# Column names (case-insensitive) that identify a ticker column in a table.
_SYMBOL_COL_HINTS = ("ticker symbol", "ticker", "symbol", "ticker_symbol")
# Column names (case-insensitive) that identify a company/security name column.
_NAME_COL_HINTS = ("company", "security", "company name", "name")


def _normalize(symbol) -> str:
    """Normalize a ticker: uppercase, strip whitespace, drop empty/NaN."""
    if symbol is None:
        return ""
    try:
        if pd.isna(symbol):
            return ""
    except (TypeError, ValueError):
        pass
    return str(symbol).strip().upper()


def _find_col(df: pd.DataFrame, hints: tuple[str, ...]):
    """Return the first column whose lowercased name contains a hint."""
    cols = {str(c).strip().lower(): c for c in df.columns}
    for hint in hints:
        if hint in cols:
            return cols[hint]
    return None


def _extract_rows_from_table(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (symbol, name) pairs from a constituent table.

    Intent: constituent tables carry both a ticker and a company/security name.
    Find both columns; fall back to the symbol when no name column exists.
    Share-class dots (BRK.B) become dashes (BRK-B) for Yahoo.
    Invariants: symbols normalized/non-empty; name defaults to the symbol.
    """
    sym_col = _find_col(df, _SYMBOL_COL_HINTS)
    if sym_col is None:
        return []
    name_col = _find_col(df, _NAME_COL_HINTS)
    out: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        s = _normalize(row[sym_col])
        if not s:
            continue
        sym = s.replace(".", "-")
        name = ""
        if name_col is not None:
            name = str(row[name_col]).strip()
        out.append((sym, name or sym))
    return out


def _extract_symbols_from_table(df: pd.DataFrame) -> list[str]:
    """Pull ticker symbols from a single parsed HTML table.

    Thin wrapper over ``_extract_rows_from_table`` (kept for callers/tests that
    only need the symbols).
    Invariants: returns a list of normalized, non-empty Yahoo-style symbols.
    """
    return [sym for sym, _ in _extract_rows_from_table(df)]


def _fetch_html(url: str) -> str:
    """Fetch a URL's HTML with a browser User-Agent.

    Intent: Wikipedia returns HTTP 403 to pandas' default read_html UA. A real
    browser UA avoids the block. Invariants: returns HTML string or raises.
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def load_index_constituents() -> dict[str, tuple[str, str]]:
    """Fetch index constituents across all sources, deduplicated.

    Intent: build {symbol: (name, source)} from Wikipedia tables. A failed
    source is logged and skipped; the rest still load.
    Invariants: returns a dict keyed by normalized symbol; first source wins.
    """
    result: dict[str, tuple[str, str]] = {}
    for source, url in INDEX_SOURCES.items():
        try:
            html = _fetch_html(url)
            tables = pd.read_html(io.StringIO(html))
        except Exception as e:  # noqa: BLE001
            print(f"  [UNIVERSE] {source}: failed to load ({e})")
            continue
        rows: list[tuple[str, str]] = []
        for t in tables:
            rows.extend(_extract_rows_from_table(t))
        # Dedupe within source, keep first occurrence.
        seen: set[str] = set()
        for sym, name in rows:
            if sym not in seen:
                seen.add(sym)
                result.setdefault(sym, (name, source))
        print(f"  [UNIVERSE] {source}: {len(seen)} constituents")
    return result


def build_universe_master() -> int:
    """Fetch index constituents + broad ETFs and upsert into universe_master.

    Intent: single entry point for the cron job. Idempotent — re-running
    refreshes names/classes without duplicating rows. New symbols default to
    WATCHLIST so discovery can graduate them to ACTIVE.
    Invariants: returns count of symbols upserted; never raises.
    """
    init_db()
    conn = get_connection()

    # Prune stale US class-share rows written with a dot (BRK.B, HEI.A, ...):
    # the extraction below now writes Yahoo's dashed form (BRK-B). The pattern
    # only matches a single trailing A/B/C class suffix, so European exchange
    # suffixes (.L / .DE / .AS) are untouched. Also drop known non-existent
    # tickers (Yahoo returns no data) so the funnel stops probing them.
    conn.execute("DELETE FROM universe_master WHERE regexp_matches(symbol, '^[A-Z]+\\.[ABC]$')")
    if NONEXISTENT_SYMBOLS:
        placeholders = ",".join(["?"] * len(NONEXISTENT_SYMBOLS))
        conn.execute(
            f"DELETE FROM universe_master WHERE symbol IN ({placeholders})",
            sorted(NONEXISTENT_SYMBOLS),
        )

    constituents = load_index_constituents()
    # Merge broad ETFs (source "ETF") — ETFs take precedence over index dupes.
    # The BROAD_ETFS key IS the friendly name ("MSCI World", "Gold Shares (GLD)").
    for name, sym in BROAD_ETFS.items():
        constituents.setdefault(sym, (name, "ETF"))

    count = 0
    for sym, (name, source) in constituents.items():
        if sym in NONEXISTENT_SYMBOLS:
            continue
        cls = classify_instrument(sym, "")
        conn.execute(
            """INSERT INTO universe_master (symbol, name, source, instrument_class,
                                            universe_status, updated_at)
               VALUES (?, ?, ?, ?, 'WATCHLIST', ?)
               ON CONFLICT (symbol) DO UPDATE SET
                 name = excluded.name,
                 source = excluded.source,
                 instrument_class = excluded.instrument_class,
                 updated_at = excluded.updated_at""",
            [sym, name or sym, source, cls, time.time()],
        )
        count += 1

    # v10.5.3 (R5): backfill display_name/name/currency (missing cells only).
    try:
        from quant.data.names import backfill_display_names

        summary = backfill_display_names(conn)
        print(f"  [NAMES] display {summary['display_filled']}, "
              f"name {summary['name_filled']}, currency {summary['currency_filled']}")
    except Exception as e:  # noqa: BLE001
        print(f"  [NAMES] backfill skipped: {e}")

    print(f"  [UNIVERSE] universe_master: {count} symbols upserted")
    return count


def load_universe_master() -> list[str]:
    """Return all symbols currently in universe_master (for the funnel)."""
    try:
        conn = get_connection()
        rows = conn.execute("SELECT symbol FROM universe_master").fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


if __name__ == "__main__":
    build_universe_master()
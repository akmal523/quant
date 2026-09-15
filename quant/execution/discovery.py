"""
discovery.py — Universe Graduation Engine (Phase 4, Module 3.2).

Intent: replace the hardcoded 277-stock universe with a dynamic Core & Satellite
model. A lightweight weekly scan of the WATCHLIST (1,000 potential stocks)
downloads only the last 5 days of price/volume data. If a watchlist stock
crosses a 52-week high OR volume exceeds 3x its 20-day average, it "graduates"
to the ACTIVE universe (heavy FinBERT/GARCH analysis). Active stocks with no
signals for 6 months are demoted back to the watchlist.

Phase 5 (v10.2) state machine fixes:
  - CORE status is immutable: never graduated, never demoted.
  - New graduates get a GRADUATION_GRACE_MONTHS grace period before the demotion
    check can age them out (fixes graduate-then-demote-in-same-run).
  - Demotion anchor is the most recent of graduated_at / last_signal_date.
  - Consecutive fetch failures (MAX_FETCH_FAILURES) mark a symbol DELISTED.
  - Every state change is logged to universe_events (audit trail).

Invariants:
  - universe_status ∈ {CORE, ACTIVE, WATCHLIST, DELISTED}
  - graduation is idempotent (INSERT OR REPLACE on asset_registry).
  - Only last WATCHLIST_LOOKBACK_DAYS of data fetched for watchlist symbols.

Dependencies: config, database, taxonomy, yfinance, pandas.
"""
from __future__ import annotations

from quant import paths
import time
import datetime as dt
import pandas as pd

from quant.config import (
    WATCHLIST_LOOKBACK_DAYS, WATCHLIST_VOLUME_MULT,
    WATCHLIST_52W_HIGH_DAYS, ACTIVE_DEMOTE_MONTHS,
    GRADUATION_GRACE_MONTHS, MAX_FETCH_FAILURES,
)
from quant.data.database import get_connection, init_db
from quant.execution.taxonomy import (
    get_instrument_class, log_universe_event, mark_delisted,
    CORE_STATUSES,
)

# Legacy watchlist CSV (fallback only). Plan 3 (Phase 1): the graduation engine
# now scans the broad universe_master pool (1000+ index constituents) instead.
WATCHLIST_CSV = paths.DATA_WATCHLIST


def load_watchlist(path: str = WATCHLIST_CSV) -> list[str]:
    """Load watchlist symbols from CSV (one symbol per line or 'Symbol' column)."""
    import os
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path)
        if "Symbol" in df.columns:
            return df["Symbol"].astype(str).str.strip().dropna().tolist()
        # Single-column CSV.
        return df.iloc[:, 0].astype(str).str.strip().dropna().tolist()
    except Exception:
        return []


def fetch_recent(symbol: str, days: int = WATCHLIST_LOOKBACK_DAYS) -> pd.DataFrame | None:
    """Fetch only the last `days` of price/volume data for a watchlist symbol.

    Intent: keep the weekly scan lightweight (5 days, not 5 years).
    Invariants: returns df with Close/Volume columns or None on failure.
    Dependencies: yfinance (imported lazily to keep pure logic testable).
    """
    try:
        import yfinance as yf
        start = (dt.date.today() - dt.timedelta(days=days + 30)).isoformat()
        df = yf.Ticker(symbol).history(start=start, auto_adjust=True)
        if df.empty or "Close" not in df.columns:
            return None
        return df.dropna(subset=["Close"])
    except Exception:
        return None


def detect_graduation(symbol: str, df: pd.DataFrame) -> tuple[bool, str]:
    """Detect whether a watchlist symbol should graduate to ACTIVE.

    Criteria (Phase 4, 3.2):
      1. Crosses a 52-week high (within the fetched window).
      2. Volume exceeds 3x its 20-day average.
    Invariants: returns (bool, reason).
    """
    if df is None or len(df) < 20:
        return False, "insufficient data"

    close = df["Close"]
    volume = df["Volume"] if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    # 52-week high breakout: last close STRICTLY exceeds the prior high
    # (excluding today). A flat series must NOT count as a breakout.
    window = min(WATCHLIST_52W_HIGH_DAYS, len(close))
    prior_high = close.iloc[-window:-1].max()
    if close.iloc[-1] > prior_high and close.iloc[-1] > 0:
        return True, "52-week high"

    # Volume anomaly: last volume > 3x 20-day average.
    avg_vol = volume.rolling(20).mean().iloc[-1]
    if avg_vol and avg_vol > 0 and volume.iloc[-1] > WATCHLIST_VOLUME_MULT * avg_vol:
        return True, f"volume {WATCHLIST_VOLUME_MULT:.0f}x 20d avg"

    return False, "no anomaly"


def months_between(start, end) -> float:
    """Months between two dates (ISO strings or datetime.date). Negative if start is after end."""
    if isinstance(start, str):
        s = dt.date.fromisoformat(start)
    else:
        s = start
    if isinstance(end, str):
        e = dt.date.fromisoformat(end)
    else:
        e = end
    return (e.year - s.year) * 12 + (e.month - s.month) + (e.day - s.day) / 30.0


def graduate(symbol: str, reason: str) -> None:
    """Promote a watchlist symbol to ACTIVE universe status.

    Intent (Phase 5 / v10.2): CORE symbols are never graduated. Sets
    graduated_at to today so the grace period starts now.
    Invariants: status becomes ACTIVE; event logged.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT universe_status FROM asset_registry WHERE symbol = ?", [symbol]
    ).fetchone()
    if row and row[0] in CORE_STATUSES:
        return  # CORE is immutable; never graduate.
    conn.execute(
        """INSERT INTO asset_registry (symbol, instrument_class, universe_status, graduated_at, updated_at)
           VALUES (?, ?, 'ACTIVE', ?, ?)
           ON CONFLICT (symbol) DO UPDATE SET
             universe_status = 'ACTIVE',
             graduated_at = excluded.graduated_at,
             updated_at = excluded.updated_at""",
        [symbol, get_instrument_class(symbol), dt.date.today().isoformat(), time.time()],
    )
    log_universe_event(symbol, "GRADUATE", reason)
    print(f" [GRADUATE] {symbol} -> ACTIVE ({reason})")


def demote_stale_active(months: int = ACTIVE_DEMOTE_MONTHS,
                        grace_months: int = GRADUATION_GRACE_MONTHS,
                        graduated_this_run: set[str] | None = None) -> int:
    """Demote ACTIVE assets with no signals for `months` back to WATCHLIST.

    Intent (Phase 5 / v10.2): keep the heavy-analysis universe lean, but never
    demote CORE assets, never demote a symbol graduated in the current run, and
    give new graduates a grace period. The demotion anchor is the most recent of
    graduated_at / last_signal_date.
    Invariants: returns count of demoted symbols.
    """
    conn = get_connection()
    now = dt.date.today().isoformat()
    graduated_this_run = graduated_this_run or set()

    rows = conn.execute(
        """SELECT symbol, graduated_at, last_signal_date FROM asset_registry
           WHERE universe_status = 'ACTIVE'""",
    ).fetchall()

    demoted = 0
    for sym, graduated_at, last_signal_date in rows:
        if sym in graduated_this_run:
            continue  # never demote a symbol graduated in the current run.
        # Demotion anchor: most recent activity. Skip if no activity recorded.
        dates = [d for d in [graduated_at, last_signal_date] if d]
        if not dates:
            continue
        anchor = max(dates)
        if months_between(anchor, now) < grace_months:
            continue  # grace period, do not demote.
        if months_between(anchor, now) < months:
            continue  # not stale yet.
        conn.execute(
            "UPDATE asset_registry SET universe_status = 'WATCHLIST', updated_at = ? WHERE symbol = ?",
            [time.time(), sym],
        )
        log_universe_event(sym, "DEMOTE", f"no signals for {months} months")
        print(f" [DEMOTE] {sym} -> WATCHLIST (no signals for {months} months)")
        demoted += 1
    return demoted


def run_discovery() -> dict:
    """Run the weekly watchlist scan and return a summary.

    Intent: entry point for the cron job. Scans watchlist, graduates anomalies,
    demotes stale active assets, tracks fetch failures -> DELISTED.
    Invariants: returns dict with graduated/demoted/scanned/delisted counts.
    """
    init_db()
    conn = get_connection()
    # Plan 3 (Phase 1): scan the broad universe_master pool for graduation.
    # Fall back to the legacy watchlist.csv only if universe_master is empty.
    from quant.data.universe_builder import load_universe_master
    watchlist = load_universe_master()
    if not watchlist:
        watchlist = load_watchlist()
    if not watchlist:
        print(" [!] No universe_master / watchlist found — skipping discovery scan.")
        return {"scanned": 0, "graduated": 0, "demoted": 0, "delisted": 0}

    graduated = 0
    graduated_this_run: set[str] = set()
    delisted = 0

    for sym in watchlist:
        # Skip DELISTED symbols entirely (stop retrying forever).
        row = conn.execute(
            "SELECT universe_status FROM asset_registry WHERE symbol = ?", [sym]
        ).fetchone()
        if row and row[0] == "DELISTED":
            continue

        df = fetch_recent(sym)
        if df is None:
            # H3-fix part 2: NEVER materialize a universe_master symbol into
            # asset_registry. Fetch-failure tracking updates tracked rows only;
            # an unknown (broad-pool) symbol is skipped, not inserted. This is
            # the defect class that bloated the registry to 1090 rows.
            failures = conn.execute(
                "SELECT fetch_failures FROM asset_registry WHERE symbol = ?", [sym]
            ).fetchone()
            if failures is None:
                continue
            count = (failures[0] or 0) + 1
            conn.execute(
                "UPDATE asset_registry SET fetch_failures = ?, updated_at = ? "
                "WHERE symbol = ?",
                [count, time.time(), sym],
            )
            if count >= MAX_FETCH_FAILURES:
                mark_delisted(sym, "possibly delisted (consecutive fetch failures)")
                delisted += 1
            continue

        # Fetch succeeded: reset the failure counter.
        conn.execute(
            "UPDATE asset_registry SET fetch_failures = 0 WHERE symbol = ?", [sym]
        )

        ok, reason = detect_graduation(sym, df)
        if ok:
            graduate(sym, reason)
            graduated += 1
            graduated_this_run.add(sym)

    demoted = demote_stale_active(graduated_this_run=graduated_this_run)
    print(f"\nDiscovery complete: scanned {len(watchlist)}, "
          f"graduated {graduated}, demoted {demoted}, delisted {delisted}.")
    return {"scanned": len(watchlist), "graduated": graduated,
            "demoted": demoted, "delisted": delisted}


if __name__ == "__main__":
    run_discovery()

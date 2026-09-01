"""
discovery.py — Universe Graduation Engine (Phase 4, Module 3.2).

Intent: replace the hardcoded 277-stock universe with a dynamic Core & Satellite
model. A lightweight weekly scan of the WATCHLIST (1,000 potential stocks)
downloads only the last 5 days of price/volume data. If a watchlist stock
crosses a 52-week high OR volume exceeds 3x its 20-day average, it "graduates"
to the ACTIVE universe (heavy FinBERT/GARCH analysis). Active stocks with no
signals for 6 months are demoted back to the watchlist.

Invariants:
  - universe_status ∈ {ACTIVE, WATCHLIST, CORE}
  - graduation is idempotent (INSERT OR REPLACE on asset_registry).
  - Only last WATCHLIST_LOOKBACK_DAYS of data fetched for watchlist symbols.

Dependencies: config, database, taxonomy, yfinance, pandas.
"""
from __future__ import annotations

import time
import datetime as dt
import pandas as pd

from config import (
    WATCHLIST_LOOKBACK_DAYS, WATCHLIST_VOLUME_MULT,
    WATCHLIST_52W_HIGH_DAYS, ACTIVE_DEMOTE_MONTHS,
)
from database import get_connection, init_db
from taxonomy import get_instrument_class

# Default watchlist: symbols not in the active SECTOR_UNIVERSE but tracked for
# potential graduation. Extend this CSV with up to ~1,000 tickers.
WATCHLIST_CSV = "watchlist.csv"


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


def graduate(symbol: str, reason: str) -> None:
    """Promote a watchlist symbol to ACTIVE universe status."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO asset_registry (symbol, instrument_class, universe_status, graduated_at, updated_at)
           VALUES (?, ?, 'ACTIVE', ?, ?)
           ON CONFLICT (symbol) DO UPDATE SET
             universe_status = 'ACTIVE',
             graduated_at = excluded.graduated_at,
             updated_at = excluded.updated_at""",
        [symbol, get_instrument_class(symbol), dt.date.today().isoformat(), time.time()],
    )
    print(f" [GRADUATE] {symbol} -> ACTIVE ({reason})")


def demote_stale_active(months: int = ACTIVE_DEMOTE_MONTHS) -> int:
    """Demote ACTIVE assets with no signals for `months` back to WATCHLIST.

    Intent: keep the heavy-analysis universe lean. Assets that generate no
    signals for 6 months are demoted to the lightweight watchlist.
    Invariants: returns count of demoted symbols.
    """
    conn = get_connection()
    cutoff = (dt.date.today() - dt.timedelta(days=months * 30)).isoformat()
    rows = conn.execute(
        """SELECT symbol FROM asset_registry
           WHERE universe_status = 'ACTIVE'
             AND (last_signal_date IS NULL OR last_signal_date < ?)""",
        [cutoff],
    ).fetchall()
    for (sym,) in rows:
        conn.execute(
            "UPDATE asset_registry SET universe_status = 'WATCHLIST', updated_at = ? WHERE symbol = ?",
            [time.time(), sym],
        )
        print(f" [DEMOTE] {sym} -> WATCHLIST (no signals for {months} months)")
    return len(rows)


def run_discovery() -> dict:
    """Run the weekly watchlist scan and return a summary.

    Intent: entry point for the cron job. Scans watchlist, graduates anomalies,
    demotes stale active assets.
    Invariants: returns dict with graduated/demoted/scanned counts.
    """
    init_db()
    watchlist = load_watchlist()
    if not watchlist:
        print(" [!] No watchlist.csv found — skipping discovery scan.")
        return {"scanned": 0, "graduated": 0, "demoted": 0}

    graduated = 0
    for sym in watchlist:
        df = fetch_recent(sym)
        ok, reason = detect_graduation(sym, df)
        if ok:
            graduate(sym, reason)
            graduated += 1

    demoted = demote_stale_active()
    print(f"\nDiscovery complete: scanned {len(watchlist)}, "
          f"graduated {graduated}, demoted {demoted}.")
    return {"scanned": len(watchlist), "graduated": graduated, "demoted": demoted}


if __name__ == "__main__":
    run_discovery()
"""
valuation.py — Holdings metadata and estimated revaluation (v10.7.0, Section 9).

Intent: broker is truth. On every CSV sync, shares = Current_Value_EUR / price
at sync time (currency conversion via the existing currency module). Between
syncs, the daily revalue uses shares * latest close and labels the value
"estimated, as of <date>". Broker PnL stays the synced truth until the next
sync. A symbol with no price (cash-like) is skipped and keeps its last value.

Invariants:
  - Pure helpers (compute_shares, estimated_label) do no I/O.
  - I/O helpers take an explicit connection; they never open their own.
  - sync_holdings_meta never raises on a malformed row; it skips it.
"""
from __future__ import annotations

from datetime import date
from typing import Any


def _lookup(price_lookup: Any, symbol: str) -> float | None:
    """Resolve a price from a mapping or a callable; None on any failure."""
    if price_lookup is None:
        return None
    try:
        if callable(price_lookup):
            return price_lookup(symbol)
        return price_lookup.get(symbol)
    except Exception:  # noqa: BLE001
        return None


def revalue_holdings(conn: Any, price_lookup: Any) -> dict[str, float]:
    """Estimated value = shares * latest close. Returns {symbol: value}.

    A symbol with no resolvable price is skipped (keeps its last value).
    """
    out: dict[str, float] = {}
    try:
        rows = conn.execute("SELECT symbol, shares FROM holdings_meta").fetchall()
    except Exception:  # noqa: BLE001
        return out
    for symbol, shares in rows:
        price = _lookup(price_lookup, symbol)
        if price is None:
            continue
        try:
            out[str(symbol)] = float(shares) * float(price)
        except (TypeError, ValueError):
            continue
    return out


def unpriceable_symbols(conn: Any, price_lookup: Any) -> list[str]:
    """Symbols in holdings_meta with no resolvable price (Part 5.2)."""
    out: list[str] = []
    try:
        rows = conn.execute("SELECT symbol FROM holdings_meta").fetchall()
    except Exception:  # noqa: BLE001
        return out
    for row in rows:
        symbol = str(row[0])
        if _lookup(price_lookup, symbol) is None:
            out.append(symbol)
    return out


def _last_sync_date(conn) -> date | None:
    """The most recent holdings_meta sync date, or None."""
    try:
        row = conn.execute("SELECT MAX(sync_date) FROM holdings_meta").fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not row or row[0] is None:
        return None
    last = row[0]
    if isinstance(last, str):
        last = date.fromisoformat(last[:10])
    return last


def days_since_last_sync(conn: Any, today: date | None = None) -> int | None:
    """Days since the most recent holdings_meta sync; None when never synced."""
    today = today or date.today()
    last = _last_sync_date(conn)
    if last is None:
        return None
    return (today - last).days


def _meta_get(conn, key: str) -> str | None:
    """Read a meta value; None on any failure."""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
        return row[0] if row else None
    except Exception:  # noqa: BLE001
        return None


def _meta_set(conn, key: str, value: str) -> None:
    """Write a meta value; never raises."""
    try:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                     [key, str(value)])
    except Exception:  # noqa: BLE001
        pass

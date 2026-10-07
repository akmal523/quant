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

from quant.config import SYNC_REMINDER_DAYS


def compute_shares(value_eur: Any, price_eur: Any) -> float | None:
    """shares = value / price. None when price is missing or non-positive."""
    try:
        value = float(value_eur)
        price = float(price_eur)
    except (TypeError, ValueError):
        return None
    if price <= 0:
        return None
    return value / price


def estimated_label(as_of: date | str | None) -> str:
    """The visible label for an estimated value: 'estimated, as of <date>'."""
    from quant.ui import copy as ui_copy

    return ui_copy.ESTIMATED_LABEL.format(date=ui_copy.fmt_date(as_of))


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


def sync_holdings_meta(
    conn: Any,
    portfolio_df: Any,
    price_lookup: Any,
    sync_date: date | None = None,
    first_time_only: bool = False,
) -> int:
    """Write holdings_meta from a broker CSV sync. Returns rows written.

    Intent: shares are derived from the broker value and the price at sync time.
    A row with no resolvable price is skipped (keeps any prior shares).

    ``first_time_only`` (v10.7.2, Part 5.2): the review path seeds holdings_meta
    for symbols it has never seen, but never overwrites an existing row, so the
    recorded ``sync_date`` keeps reflecting the user's last CSV export (and the
    35-day reminder still fires).
    """
    sync_date = sync_date or date.today()
    written = 0
    if portfolio_df is None or getattr(portfolio_df, "empty", True):
        return 0
    existing: set[str] = set()
    if first_time_only:
        try:
            existing = {str(r[0]) for r in
                        conn.execute("SELECT symbol FROM holdings_meta").fetchall()}
        except Exception:  # noqa: BLE001
            existing = set()
    for _, row in portfolio_df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol:
            continue
        if first_time_only and symbol in existing:
            continue
        value = row.get("Current_Value_EUR", 0)
        invested = row.get("Invested_EUR", value)
        price = _lookup(price_lookup, symbol)
        shares = compute_shares(value, price)
        if shares is None:
            continue
        try:
            invested_f = float(invested)
        except (TypeError, ValueError):
            invested_f = float(value or 0)
        conn.execute(
            "INSERT OR REPLACE INTO holdings_meta "
            "(symbol, shares, sync_date, invested_at_sync) VALUES (?, ?, ?, ?)",
            [symbol, shares, sync_date, invested_f],
        )
        written += 1
    if written:
        record_snapshot(conn, sync_date)
    return written


def record_snapshot(conn: Any, when: date | None = None) -> int:
    """Append the current holdings_meta shares as a dated snapshot (v10.8.0).

    Intent: the Overview value chart needs an append-only history of positions
    that is independent of reviews. Every sync appends one row per symbol for
    the sync date. A re-sync on the same day replaces that day's rows
    (idempotent); a later sync appends a new date and never rewrites the past.

    Returns rows written. Never raises.
    """
    when = when or date.today()
    try:
        rows = conn.execute("SELECT symbol, shares FROM holdings_meta").fetchall()
    except Exception:  # noqa: BLE001
        return 0
    written = 0
    for symbol, shares in rows:
        try:
            conn.execute(
                "INSERT OR REPLACE INTO position_snapshots "
                "(snapshot_date, symbol, shares) VALUES (?, ?, ?)",
                [when, str(symbol), float(shares or 0.0)],
            )
            written += 1
        except Exception:  # noqa: BLE001
            continue
    return written


def sync_from_portfolio_csv(
    conn: Any,
    price_lookup: Any,
    sync_date: date | None = None,
    filepath: str | None = None,
    first_time_only: bool = False,
) -> int:
    """Load the broker CSV and sync holdings_meta. Returns rows written.

    Convenience wrapper for the daily job and the review path. Never raises.
    """
    try:
        from quant.portfolio.portfolio import load_portfolio

        df = load_portfolio(filepath) if filepath else load_portfolio()
    except Exception:  # noqa: BLE001
        return 0
    return sync_holdings_meta(conn, df, price_lookup, sync_date,
                              first_time_only=first_time_only)


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


def write_value_history(conn: Any, as_of: date, invested_eur: float) -> None:
    """Upsert one row into portfolio_value_history (invested pool only)."""
    conn.execute(
        "INSERT OR REPLACE INTO portfolio_value_history (date, invested_eur) "
        "VALUES (?, ?)",
        [as_of, float(invested_eur)],
    )


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


def pending_position_names(conn: Any) -> list[str]:
    """Display names of positions recorded since the last CSV sync (R7)."""
    raw = _meta_get(conn, "pending_symbols") or ""
    out: list[str] = []
    for symbol in [s for s in str(raw).split(",") if s]:
        try:
            from quant.data.names import display_name

            out.append(display_name(symbol))
        except Exception:  # noqa: BLE001
            out.append(symbol)
    return out


def sync_reminder_line(conn: Any, today: date | None = None) -> str | None:
    """One gentle line when the last broker sync is older than the threshold.

    v10.7.3 (Part 3.4): when positions were recorded since the last CSV sync
    (the pending-sync marker), the reminder mentions them explicitly.
    v10.7.4 (R7): the reminder appears at most once per 7 days, never on the
    same day as a successful sync, and lists pending positions by name.
    """
    from quant.ui import copy as ui_copy

    today = today or date.today()
    days = days_since_last_sync(conn, today)
    if days is None or days <= SYNC_REMINDER_DAYS:
        return None
    # Never on the same day as a successful sync.
    if _last_sync_date(conn) == today:
        return None
    # At most once per 7 days.
    last_reminder = _meta_get(conn, "sync_reminder_last")
    if last_reminder:
        try:
            delta = (today - date.fromisoformat(str(last_reminder)[:10])).days
            # Only a reminder within the last 7 days suppresses; a future or
            # stale stamp never does.
            if 0 <= delta < 7:
                return None
        except ValueError:
            pass
    _meta_set(conn, "sync_reminder_last", today.isoformat())
    try:
        from quant.engine import plans

        if plans.is_pending_sync():
            names = pending_position_names(conn)
            if names:
                return ui_copy.SYNC_REMINDER_PENDING_NAMES.format(
                    n=days, names=", ".join(names))
            return ui_copy.SYNC_REMINDER_PENDING.format(n=days)
    except Exception:  # noqa: BLE001
        pass
    return ui_copy.SYNC_REMINDER.format(n=days)

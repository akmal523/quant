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
    conn,
    portfolio_df,
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
    return written


def sync_from_portfolio_csv(
    conn,
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


def revalue_holdings(conn, price_lookup: Any) -> dict[str, float]:
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


def write_value_history(conn, as_of: date, invested_eur: float) -> None:
    """Upsert one row into portfolio_value_history (invested pool only)."""
    conn.execute(
        "INSERT OR REPLACE INTO portfolio_value_history (date, invested_eur) "
        "VALUES (?, ?)",
        [as_of, float(invested_eur)],
    )


def days_since_last_sync(conn, today: date | None = None) -> int | None:
    """Days since the most recent holdings_meta sync; None when never synced."""
    today = today or date.today()
    try:
        row = conn.execute("SELECT MAX(sync_date) FROM holdings_meta").fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not row or row[0] is None:
        return None
    last = row[0]
    if isinstance(last, str):
        last = date.fromisoformat(last[:10])
    return (today - last).days


def sync_reminder_line(conn, today: date | None = None) -> str | None:
    """One gentle line when the last broker sync is older than the threshold."""
    from quant.ui import copy as ui_copy

    days = days_since_last_sync(conn, today)
    if days is None or days <= SYNC_REMINDER_DAYS:
        return None
    return ui_copy.SYNC_REMINDER.format(n=days)

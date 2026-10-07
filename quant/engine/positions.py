"""positions.py — the ONE source of current positions (v10.8.0, 2.1).

Intent: every page, the allocator, advice, alerts, Telegram and reports read the
same positions. Before v10.8.0 the same holding's value came from three places
(the holdings_meta estimate, the review artifact, the broker CSV), so the pages
disagreed. This module returns one list of position dicts.

Invariants:
  - Pure read: no writes.
  - Never raises; returns [] on failure.
  - value_eur is in EUR (via price_in_eur).

Dependencies: quant.paths, quant.data.currency, quant.portfolio.portfolio.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any

from quant import paths


def _as_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _meta_rows(conn) -> dict:
    """{symbol: (shares, sync_date, invested)} from holdings_meta."""
    if conn is None:
        return {}
    try:
        rows = conn.execute(
            "SELECT symbol, shares, sync_date, invested_at_sync FROM holdings_meta"
        ).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    out: dict = {}
    for sym, shares, sync_date, invested in rows:
        out[str(sym)] = (float(shares or 0), sync_date, float(invested or 0))
    return out


def _csv_mtime_date() -> date | None:
    try:
        return date.fromtimestamp(os.path.getmtime(paths.DATA_PORTFOLIO))
    except Exception:  # noqa: BLE001
        return None


def _estimate_is_fresher(meta: dict) -> bool:
    """True when holdings_meta is newer than the broker CSV.

    A pending-sync marker (actuals or a quick event recorded since the last CSV
    export) always means the estimate is fresher. Otherwise the newest
    holdings_meta sync_date is compared to the CSV file's modification date.
    """
    try:
        from quant.engine import plans

        if plans.is_pending_sync():
            return True
    except Exception:  # noqa: BLE001
        pass
    csv_day = _csv_mtime_date()
    if csv_day is None:
        return False
    newest: date | None = None
    for _sym, (_shares, sync_date, _inv) in meta.items():
        d = _as_date(sync_date)
        if d is not None and (newest is None or d > newest):
            newest = d
    return newest is not None and newest > csv_day


def _build(portfolio, conn, price_in_eur) -> list[dict]:
    meta = _meta_rows(conn)
    fresher = _estimate_is_fresher(meta)
    out: list[dict] = []
    for _, r in portfolio.iterrows():
        sym = str(r["Symbol"])
        csv_value = float(r.get("Current_Value_EUR", 0) or 0)
        csv_entry = float(r.get("Avg_Entry_Price", 0) or 0)
        csv_pnl = float(r.get("Broker_PnL_EUR", 0) or 0)
        if fresher and sym in meta:
            shares, sync_date, invested = meta[sym]
            price = price_in_eur(sym, conn=conn)
            value = shares * price if price is not None else csv_value
            entry = invested / shares if shares else csv_entry
            out.append({
                "symbol": sym, "shares": shares, "value_eur": value,
                "entry_eur": entry, "profit_eur": value - invested,
                "as_of": sync_date, "estimated": True,
            })
        else:
            invested = csv_value - csv_pnl
            shares = csv_value / csv_entry if csv_entry else 0.0
            out.append({
                "symbol": sym, "shares": shares, "value_eur": csv_value,
                "entry_eur": csv_entry, "profit_eur": csv_pnl,
                "as_of": None, "estimated": False,
            })
    return out


def positions_now(conn: Any =None) -> list[dict]:
    """The current positions: symbol, shares, value_eur, entry_eur, profit_eur,
    as_of, estimated.

    Source precedence: the broker CSV is the base truth. When holdings_meta is
    fresher than the CSV, the value is shares * price_in_eur (estimated=True).
    """
    from quant.data.currency import price_in_eur
    from quant.portfolio.portfolio import load_portfolio

    try:
        portfolio = load_portfolio(paths.DATA_PORTFOLIO)
    except Exception:  # noqa: BLE001
        return []
    if portfolio is None or portfolio.empty:
        return []

    if conn is not None:
        return _build(portfolio, conn, price_in_eur)
    try:
        from quant.data.database import read_only_connection

        with read_only_connection() as c:
            return _build(portfolio, c, price_in_eur)
    except Exception:  # noqa: BLE001
        return _build(portfolio, None, price_in_eur)

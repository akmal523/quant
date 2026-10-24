"""positions.py — the ONE source of current positions (v10.8.0; v10.8.10).

Every page, the ONE list and Telegram read the same positions. Per row of the
holdings table:

  invested = typed value - typed profit       (what was paid, EUR)
  shares   = invested / average cost per share
  value    = the valuation store's latest value for the symbol (the same number
             the Portfolio header sums), else the typed value when the symbol
             has no price yet
  profit   = value - invested

v10.8.10: values were the typed numbers (possibly weeks old), so weights,
drift and stop-losses ignored the market; shares were computed as value / cost
(too many shares for any position in profit); a holdings_meta "estimate" branch
depended on a writer that no longer exists.

Invariants: pure read; never raises; returns [] on failure.
"""
from __future__ import annotations

from typing import Any

from quant import paths


def _store_values(conn: Any) -> tuple[dict, Any]:
    """``({symbol: value_eur}, as_of)`` from the latest valuation date."""
    try:
        rows = conn.execute(
            "SELECT symbol, value_eur, date FROM valuation "
            "WHERE date = (SELECT MAX(date) FROM valuation)").fetchall()
    except Exception:  # noqa: BLE001
        return {}, None
    return {str(s): float(v or 0) for s, v, _d in rows}, (rows[0][2] if rows else None)


def _build(portfolio, conn) -> list[dict]:
    live, as_of = _store_values(conn) if conn is not None else ({}, None)
    out: list[dict] = []
    for _, r in portfolio.iterrows():
        sym = str(r["Symbol"])
        typed_value = float(r.get("Current_Value_EUR", 0) or 0)
        entry = float(r.get("Avg_Entry_Price", 0) or 0)
        invested = typed_value - float(r.get("Broker_PnL_EUR", 0) or 0)
        value = live.get(sym, typed_value)
        out.append({
            "symbol": sym,
            "shares": invested / entry if entry > 0 else 0.0,
            "value_eur": value,
            "invested_eur": invested,
            "entry_eur": entry,
            "profit_eur": value - invested,
            "pnl_pct": (value / invested - 1.0) if invested > 0 else None,
            "as_of": as_of if sym in live else None,
            "estimated": sym in live,
        })
    return out


def positions_now(conn: Any = None) -> list[dict]:
    """The current positions: symbol, shares, value_eur, invested_eur, entry_eur,
    profit_eur, pnl_pct, as_of, estimated (True = marked to the latest close)."""
    from quant.portfolio.portfolio import load_portfolio

    portfolio = load_portfolio(paths.DATA_PORTFOLIO)
    if portfolio is None or portfolio.empty:
        return []
    if conn is not None:
        return _build(portfolio, conn)
    try:
        from quant.data.database import read_only_connection

        with read_only_connection() as c:
            return _build(portfolio, c)
    except Exception:  # noqa: BLE001
        return _build(portfolio, None)

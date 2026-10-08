"""
recovery.py — rebuild the holdings table from the newest saved snapshot (v10.8.1, A3).

Intent: if the table file was lost (a botched update, a re-clone), the database
still holds ``holdings_meta`` (shares + invested per symbol). This rebuilds the
five-column table so the user can review and confirm it through the normal flow;
nothing is written to disk here.

Invariants:
  - Pure read: no writes.
  - Never raises; returns [] on failure.
  - Value falls back to the invested amount when no EUR price resolves.
"""
from __future__ import annotations

from typing import Any


def restore_last_saved_holdings(conn: Any = None) -> list[dict]:
    """Rebuild table rows from ``holdings_meta`` + the latest EUR price.

    Returns ``[{Symbol, Avg_Entry_Price, Current_Value_EUR, Broker_PnL_EUR}]``.
    Einstandskurs = invested / shares; Profit = Value - invested.
    """
    from quant.data.currency import price_in_eur

    rows = _meta_rows(conn)
    if not rows:
        return []
    out: list[dict] = []
    for symbol, shares, invested in rows:
        price = price_in_eur(symbol, conn=conn)
        value = shares * price if (price is not None and shares) else invested
        entry = (invested / shares) if shares else 0.0
        out.append({
            "Symbol": symbol,
            "Avg_Entry_Price": round(entry, 4),
            "Current_Value_EUR": round(value, 2),
            "Broker_PnL_EUR": round(value - invested, 2),
        })
    return out


def _meta_rows(conn: Any) -> list[tuple[str, float, float]]:
    """(symbol, shares, invested_at_sync) from holdings_meta. Never raises."""
    try:
        if conn is None:
            from quant.data.database import read_only_connection

            with read_only_connection() as c:
                return _query(c)
        return _query(conn)
    except Exception:  # noqa: BLE001
        return []


def _query(conn: Any) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    rows = conn.execute(
        "SELECT symbol, shares, invested_at_sync FROM holdings_meta"
    ).fetchall()
    for symbol, shares, invested in rows:
        s = float(shares or 0)
        if s <= 0:
            continue
        out.append((str(symbol), s, float(invested or 0)))
    return out

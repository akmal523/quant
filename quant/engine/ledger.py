"""ledger.py — the ONE transactional write for a recorded transaction (v10.8.0, 2.4).

Intent: before v10.8.0 a buy, sell, or dividend was written to ``flows`` and
``trades`` in two separate connections, so a failure between them left one
ledger updated and the UI said "Could not record". This module writes both in
ONE transaction, computes the FIFO realized gain for a sell, and returns the fee
and gain so every caller (UI, tax, performance) reads the same facts.

Invariants:
  - One connection, one transaction: a failure leaves nothing half-written.
  - FIFO realized gain is seeded by the broker's entry price for opening lots.
  - The fee default is zero for a savings-plan execution, else the broker fee.
"""
from __future__ import annotations

from datetime import date

from quant.config import ACTIVE_TRADE_FEE_EUR


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def default_fee(action: str, savings_plan: bool = False) -> float:
    """The fee for a transaction: zero for a savings plan, else the broker fee."""
    if savings_plan:
        return 0.0
    return float(ACTIVE_TRADE_FEE_EUR) if str(action).lower() in ("buy", "sell") else 0.0


def replace_trades(rows: list[dict]) -> int:
    """Replace the trades ledger with ``rows`` in one transaction (v10.8.3).

    Each row: ``{id?, date, symbol, action, amount_eur, realized_pnl_eur}``. A row
    with an existing ``id`` is updated; a row without one is inserted; any id not
    present in ``rows`` is deleted. Returns the number of rows written. Used by the
    History page so the user can correct the derived numbers.
    """
    from quant.data.database import write_connection

    with write_connection() as conn:
        existing = {int(r[0]) for r in conn.execute("SELECT id FROM trades").fetchall()}
        keep: set[int] = set()
        for r in rows:
            rid = r.get("id")
            date = r.get("date")
            symbol = str(r.get("symbol") or "")
            action = str(r.get("action") or "").lower()
            amount = float(r.get("amount_eur") or 0)
            pnl = r.get("realized_pnl_eur")
            pnl = None if pnl is None or pnl == "" else float(pnl)
            if rid is not None and int(rid) in existing:
                conn.execute(
                    "UPDATE trades SET date = ?, symbol = ?, action = ?, "
                    "amount_eur = ?, realized_pnl_eur = ? WHERE id = ?",
                    [date, symbol, action, amount, pnl, int(rid)])
                keep.add(int(rid))
            else:
                conn.execute(
                    "INSERT INTO trades (date, symbol, action, amount_eur, "
                    "realized_pnl_eur) VALUES (?, ?, ?, ?, ?)",
                    [date, symbol, action, amount, pnl])
        for rid in existing - keep:
            conn.execute("DELETE FROM trades WHERE id = ?", [rid])
    return len(rows)

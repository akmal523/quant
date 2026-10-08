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


def fifo_realized_gain(lots: list[dict], sell_shares: float, sell_price: float,
                       entry_price: float = 0.0) -> float:
    """FIFO realized gain for a sell, seeded by the broker entry price.

    ``lots`` is ``[{"shares", "price_eur"}]`` oldest first. Lots are consumed
    FIFO; any shortfall is priced at the broker's ``entry_price`` (the opening
    position), as German rules require FIFO on the ledger.
    """
    remaining = float(sell_shares)
    gain = 0.0
    for lot in lots:
        if remaining <= 0:
            break
        lot_shares = float(lot.get("shares") or 0.0)
        lot_price = float(lot.get("price_eur") or 0.0)
        if lot_shares <= 0:
            continue
        take = min(remaining, lot_shares)
        gain += (float(sell_price) - lot_price) * take
        remaining -= take
    if remaining > 0 and entry_price > 0:
        gain += (float(sell_price) - float(entry_price)) * remaining
    return round(gain, 2)


def record_transaction(
    *,
    when,
    action: str,
    symbol: str,
    amount_eur: float,
    shares: float | None = None,
    price_eur: float | None = None,
    fee_eur: float | None = None,
    realized_pnl_eur: float | None = None,
    savings_plan: bool = False,
    entry_price_eur: float = 0.0,
) -> dict:
    """Write one transaction to both ledgers in a single transaction.

    Returns ``{"fee_eur", "realized_pnl_eur", "shares"}``.
    """
    from quant.data.database import write_connection

    when = _as_date(when)
    action = str(action).lower()
    fee = default_fee(action, savings_plan) if fee_eur is None else float(fee_eur)
    pnl = realized_pnl_eur

    with write_connection() as conn:
        if action == "sell" and pnl is None:
            rows = conn.execute(
                "SELECT shares, price_eur FROM trades "
                "WHERE symbol = ? AND action = 'buy' ORDER BY date",
                [symbol],
            ).fetchall()
            lots = [{"shares": r[0], "price_eur": r[1]} for r in rows]
            pnl = fifo_realized_gain(
                lots, float(shares or 0.0), float(price_eur or 0.0), entry_price_eur)
        conn.execute(
            "INSERT INTO flows (date, type, amount_eur, symbol, note) "
            "VALUES (?, ?, ?, ?, ?)",
            [when, action, float(amount_eur), symbol, "recorded"],
        )
        conn.execute(
            "INSERT INTO trades (date, symbol, action, shares, price_eur, "
            "amount_eur, fee_eur, realized_pnl_eur) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [when, symbol, action, shares, price_eur, float(amount_eur), fee, pnl],
        )
    return {"fee_eur": fee, "realized_pnl_eur": pnl, "shares": shares}


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

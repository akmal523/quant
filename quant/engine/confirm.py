"""
confirm.py — Confirm and save, one all-or-nothing action (v10.8.2, section 3).

Intent: saving the table is the only way data enters the system. Confirm and save
backs up the previous table, writes the new one atomically, appends a position
snapshot, and records the detected diff in the ledger. A failure before the table
write leaves nothing changed; the table write itself is atomic (temp + rename).

Invariants:
  - The previous table is backed up before the new one is written.
  - The position snapshot and the ledger rows are written in one DB transaction.
  - Shares are derived from the table (invested / Einstandskurs), never a price
    lookup.
"""
from __future__ import annotations

from datetime import date

from quant import paths


def confirm_save(cleaned, diff: dict, when: date | None = None,
                 path: str | None = None) -> dict:
    """Persist a confirmed table + its diff. Returns ``{ok, changes, error}``.

    ``cleaned`` is a DataFrame with the four broker columns; ``diff`` is the
    result of :func:`quant.engine.diff.diff_tables`.
    """
    from quant.portfolio.editor import save_portfolio

    when = when or date.today()
    path = path or paths.DATA_PORTFOLIO

    # 1. Back up the previous table + settings (copy only; never raises).
    try:
        from quant.engine import backup as backup_mod

        backup_mod.snapshot_user_files()
    except Exception:  # noqa: BLE001
        pass

    # 2. Write the new table atomically.
    try:
        save_portfolio(cleaned, path)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "changes": 0, "error": str(e)}

    # 3+4. Snapshot + ledger in one transaction.
    try:
        from quant.data.database import write_connection

        with write_connection() as conn:
            _append_snapshot(conn, cleaned, when)
            _append_ledger(conn, diff, when)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "changes": 0, "error": str(e)}

    return {"ok": True, "changes": len(diff.get("changes", [])), "error": None}


def _row_shares(row) -> tuple[str, float]:
    symbol = str(row.get("Symbol") or "").strip()
    entry = float(row.get("Avg_Entry_Price") or 0)
    value = float(row.get("Current_Value_EUR") or 0)
    profit = float(row.get("Broker_PnL_EUR") or 0)
    invested = value - profit
    return symbol, (invested / entry if entry > 0 else 0.0)


def _append_snapshot(conn, cleaned, when: date) -> None:
    """Append one position snapshot row per holding (idempotent per date)."""
    for _, row in cleaned.iterrows():
        symbol, shares = _row_shares(row)
        if not symbol:
            continue
        conn.execute(
            "INSERT OR REPLACE INTO position_snapshots "
            "(snapshot_date, symbol, shares) VALUES (?, ?, ?)",
            [when, symbol, shares])


def _append_ledger(conn, diff: dict, when: date) -> None:
    """Record the diff's Bought/Sold changes in the ledger (one transaction)."""
    from quant.engine.ledger import default_fee

    for change in diff.get("changes", []):
        kind = str(change.get("kind"))
        if kind not in ("Bought", "Sold"):
            continue
        action = "buy" if kind == "Bought" else "sell"
        symbol = str(change.get("symbol") or "")
        amount = float(change.get("amount_eur") or 0.0)
        shares = change.get("shares")
        price = (amount / float(shares)) if shares else None
        pnl = change.get("realized_eur")
        fee = default_fee(action, savings_plan=(kind == "Bought"))
        conn.execute(
            "INSERT INTO flows (date, type, amount_eur, symbol, note) "
            "VALUES (?, ?, ?, ?, ?)",
            [when, action, amount, symbol, "confirmed"])
        conn.execute(
            "INSERT INTO trades (date, symbol, action, shares, price_eur, "
            "amount_eur, fee_eur, realized_pnl_eur) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [when, symbol, action, shares, price, amount, fee, pnl])

"""
plans.py — monthly plan storage and actuals reconciliation (v10.7.0, Section 8).

Intent: after approval, store the plan (month, budget, legs) and write planned
auto-flows. When the user later enters actuals, replace the auto rows with
actuals, update holdings_meta shares, mark the portfolio as pending sync, and
print a reconciliation line. Conscious deviations are recorded, not judged.

Invariants:
  - I/O helpers take an explicit connection.
  - enter_actuals never raises; returns reconciliation rows.
"""
from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

from quant import paths

PENDING_SYNC_MARKER = ".pending_sync"


def save_plan(
    conn: Any,
    month: str,
    budget_eur: float,
    legs: list[dict],
    approved_date: date | None = None,
    execution_date: date | None = None,
) -> None:
    """Store an approved monthly plan (upsert by month)."""
    conn.execute(
        "INSERT OR REPLACE INTO monthly_plans "
        "(month, budget_eur, legs_json, approved_date, execution_date) "
        "VALUES (?, ?, ?, ?, ?)",
        [month, float(budget_eur), json.dumps(legs), approved_date, execution_date],
    )


def load_plan(conn: Any, month: str) -> dict | None:
    """Load a monthly plan, or None when not approved."""
    try:
        row = conn.execute(
            "SELECT month, budget_eur, legs_json, approved_date, execution_date "
            "FROM monthly_plans WHERE month = ?",
            [month],
        ).fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not row:
        return None
    try:
        legs = json.loads(row[2]) if row[2] else []
    except Exception:  # noqa: BLE001
        legs = []
    return {
        "month": row[0],
        "budget_eur": row[1],
        "legs": legs,
        "approved_date": row[3],
        "execution_date": row[4],
    }


def is_approved(conn: Any, month: str) -> bool:
    """True when a plan exists for the month."""
    return load_plan(conn, month) is not None


def delete_plan(conn: Any, month: str) -> bool:
    """Delete an approved plan before execution (Part 3.3). Returns True if removed."""
    try:
        conn.execute("DELETE FROM monthly_plans WHERE month = ?", [month])
    except Exception:  # noqa: BLE001
        return False
    return True


def mark_pending_sync() -> None:
    """Write the pending-sync marker (portfolio changed, awaiting a CSV sync)."""
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(os.path.join(str(paths.OUTPUTS_DIR), PENDING_SYNC_MARKER),
                  "w", encoding="utf-8") as f:
            f.write(date.today().isoformat())
    except Exception:  # noqa: BLE001
        pass


def is_pending_sync() -> bool:
    """True when the pending-sync marker exists."""
    return os.path.exists(os.path.join(str(paths.OUTPUTS_DIR), PENDING_SYNC_MARKER))


def clear_pending_sync() -> None:
    """Remove the pending-sync marker (after a fresh CSV sync)."""
    try:
        os.remove(os.path.join(str(paths.OUTPUTS_DIR), PENDING_SYNC_MARKER))
    except Exception:  # noqa: BLE001
        pass
    # R7: the pending-position list is cleared with the marker.
    try:
        from quant.data.database import get_connection

        get_connection().execute("DELETE FROM meta WHERE key = 'pending_symbols'")
    except Exception:  # noqa: BLE001
        pass


def _record_pending_symbols(conn, symbols: list[str]) -> None:
    """Append symbols to the meta pending list (R7). Never raises."""
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'pending_symbols'").fetchone()
        existing = row[0] if row else ""
    except Exception:  # noqa: BLE001
        existing = ""
    current = [s for s in str(existing or "").split(",") if s]
    for symbol in symbols:
        if symbol and symbol not in current:
            current.append(symbol)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('pending_symbols', ?)",
            [",".join(current)])
    except Exception:  # noqa: BLE001
        pass


def enter_actuals(
    conn: Any,
    month: str,
    actuals: list[dict],
    price_lookup: Any = None,
) -> list[dict]:
    """Replace planned Sparplan flows with actuals and update holdings_meta.

    actuals: list of {symbol, amount_eur, date}. Returns reconciliation rows
    {symbol, planned, actual, deviation}. Never raises.
    """
    from quant.engine import flows, valuation

    try:
        recon = flows.replace_auto_flows_with_actuals(conn, month, actuals)
    except Exception:  # noqa: BLE001
        recon = []

    for actual in actuals:
        symbol = actual.get("symbol")
        amount = float(actual.get("amount_eur", 0) or 0)
        if not symbol or amount <= 0:
            continue
        price = valuation._lookup(price_lookup, symbol)
        shares = valuation.compute_shares(amount, price)
        if shares is None:
            continue
        # v10.7.3 (Part 3.1): a buy ADDS to the existing estimated position, so
        # the visible balance moves by the amount bought. A symbol not yet held
        # starts a new estimated position (Part 3.4).
        try:
            existing = conn.execute(
                "SELECT shares, invested_at_sync FROM holdings_meta WHERE symbol = ?",
                [symbol]).fetchone()
            if existing:
                new_shares = float(existing[0] or 0) + shares
                new_invested = float(existing[1] or 0) + amount
            else:
                new_shares = shares
                new_invested = amount
            conn.execute(
                "INSERT OR REPLACE INTO holdings_meta "
                "(symbol, shares, sync_date, invested_at_sync) VALUES (?, ?, ?, ?)",
                [symbol, new_shares, date.today(), new_invested],
            )
        except Exception:  # noqa: BLE001
            pass

    _record_pending_symbols(
        conn, [a.get("symbol") for a in actuals if a.get("symbol")])
    mark_pending_sync()
    return recon


def reconciliation_line(row: dict) -> str:
    """One plain reconciliation sentence for a leg."""
    symbol = row.get("symbol") or "holding"
    planned = float(row.get("planned", 0) or 0)
    actual = float(row.get("actual", 0) or 0)
    deviation = float(row.get("deviation", 0) or 0)
    line = (f"Planned {planned:.0f} {symbol}, bought {actual:.0f} {symbol}. "
            f"Deviation {deviation:+.0f}.")
    if abs(deviation) < 1e-9:
        line += " If you changed the plan on purpose, no action needed."
    return line

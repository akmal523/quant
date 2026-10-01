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
    conn,
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


def load_plan(conn, month: str) -> dict | None:
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


def is_approved(conn, month: str) -> bool:
    """True when a plan exists for the month."""
    return load_plan(conn, month) is not None


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


def enter_actuals(
    conn,
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
        try:
            conn.execute(
                "INSERT OR REPLACE INTO holdings_meta "
                "(symbol, shares, sync_date, invested_at_sync) VALUES (?, ?, ?, ?)",
                [symbol, shares, date.today(), amount],
            )
        except Exception:  # noqa: BLE001
            pass

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

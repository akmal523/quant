"""
reminders.py — the savings-plan reminder lifecycle (v10.8.4, T4).

Intent: the old Portfolio reminder was a floating banner that never cleared.
A reminder is now a STATEFUL object:

  - it triggers when the plan execution date has passed this month AND the
    expected flow is not yet reflected in the ledger/holdings;
  - it clears automatically once a confirm reflects the flow;
  - it can be dismissed explicitly ("Done / Nothing changed") and the dismissal
    persists across reruns and days (stored in the ``meta`` key-value table).

It renders as a task item in the action area, never a banner between the chart
and the decisions.

Invariants:
  - Pure date math; the ``meta`` read/write never raises.
  - Returns one dict ``{kind, amount_eur, date, message, status, key}`` or None.
"""
from __future__ import annotations

from datetime import date
from typing import Any

KIND_SAVINGS = "savings_plan"


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    return 30 if month in (4, 6, 9, 11) else 31


def plan_date(today: date, day: int) -> date:
    """This month's plan execution date (clamped to the month's last day)."""
    return date(today.year, today.month, min(int(day), _days_in_month(
        today.year, today.month)))


def _meta_get(conn: Any, key: str) -> str | None:
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
        return row[0] if row else None
    except Exception:  # noqa: BLE001
        return None


def _meta_set(conn: Any, key: str, value: str) -> None:
    try:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                     [key, str(value)])
    except Exception:  # noqa: BLE001
        pass


def _expected_amount(conn: Any, amount_eur: float | None) -> float:
    """The expected flow: the passed amount, else the sum of Plan_EUR_month."""
    if amount_eur is not None:
        return float(amount_eur)
    try:
        from quant.portfolio.portfolio import load_portfolio

        df = load_portfolio()
        if df is not None and not df.empty and "Plan_EUR_month" in df.columns:
            return float(df["Plan_EUR_month"].fillna(0).sum())
    except Exception:  # noqa: BLE001
        pass
    return 0.0


def is_reflected(conn: Any, when: date, today: date) -> bool:
    """True when a buy flow dated in [when, today] exists (the plan landed)."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM flows WHERE date >= ? AND date <= ? "
            "AND type = 'buy'", [when, today]).fetchone()
        return bool(row and row[0])
    except Exception:  # noqa: BLE001
        return False


def _meta_key(when: date) -> str:
    return f"savings_reminder_dismissed_{when.isoformat()}"


def savings_plan_reminder(conn: Any, today: date | None = None,
                          day: int | None = None,
                          amount_eur: float | None = None) -> dict | None:
    """The reminder object, or None when not triggered / cleared / dismissed."""
    today = today or date.today()
    if day is None:
        try:
            from quant.portfolio.account import load_account

            day = load_account().savings_plan_day
        except Exception:  # noqa: BLE001
            day = None
    if not day:
        return None
    when = plan_date(today, int(day))
    if today < when:
        return None
    if _meta_get(conn, _meta_key(when)) is not None:
        return None
    if is_reflected(conn, when, today):
        return None
    amount = _expected_amount(conn, amount_eur)
    if amount <= 0:
        return None   # no savings plan in the table: nothing to remind (v10.8.10)
    from quant.ui import copy as C

    message = C.SAVINGS_PLAN_REMINDER.format(
        amount=f"{amount:.0f}", date=C.fmt_date(when))
    return {"kind": KIND_SAVINGS, "amount_eur": amount, "date": when,
            "message": message, "status": "open", "key": _meta_key(when)}


def dismiss(conn: Any, today: date | None = None, day: int | None = None) -> bool:
    """Mark the reminder done for this month (persists across reruns/days)."""
    today = today or date.today()
    if day is None:
        try:
            from quant.portfolio.account import load_account

            day = load_account().savings_plan_day
        except Exception:  # noqa: BLE001
            return False
    if not day:
        return False
    when = plan_date(today, int(day))
    _meta_set(conn, _meta_key(when), today.isoformat())
    return True

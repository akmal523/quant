"""
steps.py — the "Your steps this week" generator (v10.7.0, Section 10.5).

Intent: the Overview shows concrete steps, each with what, EUR, from where, why,
and the fee. Everything else is noise and must not appear. The "Not this week"
block explains considered-but-rejected actions in one line each.

Sources, in order:
  1. open alerts (each becomes a step with its action);
  2. unapproved monthly decision after the 25th of the month;
  3. approved plan not yet executed;
  4. actuals missing 7+ days after the execution day;
  5. a savings-plan leg change suggestion only if a FORTRESS target gap exceeds
     10 points.

Invariants: pure functions; no I/O.
"""
from __future__ import annotations

from datetime import date
from typing import Any

FORTRESS_GAP_SUGGESTION = 0.10


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def build_steps(
    today: date,
    open_alerts: list[dict] | None = None,
    plan: dict | None = None,
    holdings: list[dict] | None = None,
    actuals_entered: bool = False,
) -> list[dict]:
    """Return the ordered list of concrete steps for this week."""
    from quant.ui import copy as ui_copy

    steps: list[dict] = []

    for alert in open_alerts or []:
        steps.append({
            "what": alert.get("message", ""),
            "amount_eur": alert.get("amount_eur"),
            "source": "alert",
            "why": alert.get("message", ""),
            "fee_eur": alert.get("fee_eur"),
        })

    if plan is None and today.day >= 25:
        steps.append({
            "what": ui_copy.MONTHLY_MAKE_DECISION_STEP,
            "amount_eur": None,
            "source": "monthly",
            "why": "the budget is re-entered every month.",
            "fee_eur": None,
        })

    if plan is not None and not actuals_entered:
        steps.append({
            "what": ui_copy.MONTHLY_EXECUTE_STEP,
            "amount_eur": plan.get("budget_eur"),
            "source": "plan",
            "why": "the plan is approved but not yet executed.",
            "fee_eur": None,
        })
        execution = _as_date(plan.get("execution_date"))
        if execution is not None and (today - execution).days >= 7:
            steps.append({
                "what": ui_copy.MONTHLY_ENTER_ACTUALS_STEP,
                "amount_eur": None,
                "source": "actuals",
                "why": "the math stays honest only with actuals.",
                "fee_eur": None,
            })

    for holding in holdings or []:
        if str(holding.get("tier", "")).upper() != "FORTRESS":
            continue
        gap = _num(holding.get("target_weight")) - _num(holding.get("current_weight"))
        if gap > FORTRESS_GAP_SUGGESTION:
            name = holding.get("name") or holding.get("symbol")
            steps.append({
                "what": f"Consider raising the {name} leg of the savings plan.",
                "amount_eur": None,
                "source": "fortress_gap",
                "why": (f"{_num(holding.get('current_weight')) * 100:.0f} percent of "
                        f"invested vs {_num(holding.get('target_weight')) * 100:.0f} "
                        f"percent target."),
                "fee_eur": 0.0,
            })

    return steps


def rejected_actions(holdings: list[dict] | None = None) -> list[str]:
    """The "Not this week" block: considered-but-rejected actions, one line each."""
    out: list[str] = []
    for holding in holdings or []:
        value = _num(holding.get("value_eur"))
        tier = str(holding.get("tier", "")).upper()
        name = holding.get("name") or holding.get("symbol")
        if tier == "FORTRESS":
            continue
        # The 2 EUR round-trip fee is pointless above 1 percent of the position.
        if value > 0 and (2.0 / value) > 0.01:
            out.append(
                f"Do not sell {name}: position {value:.0f} EUR; the 2 EUR fee makes "
                f"any sale pointless.")
    return out

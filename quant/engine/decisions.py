"""decisions.py — the ONE decision list (v10.8.0, Phase 1, redesign 3.2).

Intent: every surface that shows the user what to do renders from one list,
grouped into three buckets:

  - "Needs your input": a holding that blocks advice (missing ISIN).
  - "Recommended": an actionable advice (buy, top up, sell part, plan change).
  - "Optional": a keep, or a considered-but-suppressed note.

Invariants:
  - Pure: no I/O.
  - Every advice record lands in exactly one group.
  - A blocked holding is never also "Recommended".
"""
from __future__ import annotations

from quant.ui import copy as ui_copy

# The advice kinds that ask the user to act now.
_ACTIONABLE = {"buy", "top_up", "sell_part", "change_savings_plan", "to_cash"}


def _label(item: dict) -> str:
    """The "Human name (TICKER)" label for an advice or holding dict."""
    name = str(item.get("company_name") or item.get("name") or "").strip()
    symbol = str(item.get("symbol") or "").strip()
    if name and symbol and name != symbol:
        return f"{name} ({symbol})"
    return name or symbol


def build_decision_list(advice: list[dict] | None,
                        holdings: list[dict] | None = None,
                        rejected: list[dict] | None = None) -> list[dict]:
    """Group every decision into the three buckets. Pure; no I/O.

    Args:
        advice: the Advice records from :func:`quant.engine.advice.build_advice`.
        holdings: the position dicts (for the ``blocked`` flag).
        rejected: the RejectedNote records (the "Not this week" notes).

    Returns ``[{group, verb, label, amount_eur, reason, symbol}]`` in group
    order: Needs your input, then Recommended, then Optional.
    """
    out: list[dict] = []
    blocked_syms: set[str] = set()
    for h in holdings or []:
        if not h.get("blocked"):
            continue
        sym = str(h.get("symbol") or "")
        blocked_syms.add(sym)
        out.append({
            "group": ui_copy.DECISION_GROUP_INPUT,
            "verb": "blocked",
            "label": _label(h),
            "amount_eur": None,
            "reason": ui_copy.NEEDS_ATTENTION_ISIN.format(symbol=sym),
            "symbol": sym,
        })
    for a in advice or []:
        sym = str(a.get("symbol") or "")
        if sym and sym in blocked_syms:
            continue
        kind = str(a.get("kind") or "")
        group = (ui_copy.DECISION_GROUP_RECOMMENDED if kind in _ACTIONABLE
                 else ui_copy.DECISION_GROUP_OPTIONAL)
        out.append({
            "group": group,
            "verb": kind,
            "label": _label(a),
            "amount_eur": a.get("eur"),
            "reason": str(a.get("why") or ""),
            "symbol": sym,
        })
    for n in rejected or []:
        out.append({
            "group": ui_copy.DECISION_GROUP_OPTIONAL,
            "verb": "rejected",
            "label": str(n.get("symbol") or ""),
            "amount_eur": None,
            "reason": str(n.get("plain_reason") or ""),
            "symbol": str(n.get("symbol") or ""),
        })
    return out


def group_order() -> list[str]:
    """The three group headers, in display order."""
    return [ui_copy.DECISION_GROUP_INPUT, ui_copy.DECISION_GROUP_RECOMMENDED,
            ui_copy.DECISION_GROUP_OPTIONAL]

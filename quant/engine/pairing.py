"""
pairing.py — pair a sell with where its proceeds go (v10.8.4, T5; v10.8.7, R-C1).

Intent: the ONE list must never tell the user to sell without saying what to do
with the money. Every sell item proposes its proceeds routing inside the SAME
item, by this fixed priority:

  1. top up the largest UNDERWEIGHT holding vs its target (the gap in points);
  2. else the highest-score idea that passes the sizing laws;
  3. else hold the proceeds at the broker, untracked (v10.8.7, R-C1: there is no
     tracked-cash target; the only permitted trace of cash is this reasoned line).

The result renders as one line, for example:
  "Sell ~65 EUR of Nasdaq 100 (SXRV.DE) -> buy MSCI World (EUNL.DE), 4pp under target".

Invariants:
  - Pure: no I/O.
  - Never proposes selling to buy the same symbol.
  - Deterministic: ties break by the larger gap, then alphabetically.
"""
from __future__ import annotations

from typing import Any

ROUTE_UNDERWEIGHT = "underweight"
ROUTE_IDEA = "idea"
ROUTE_HOLD = "hold"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label(item: dict) -> str:
    from quant.ui.search import label_for

    name = str(item.get("name") or item.get("company_name") or item.get("symbol") or "")
    symbol = str(item.get("symbol") or "")
    return label_for(name, symbol) if symbol else name


def largest_underweight(holdings: list[dict], exclude: str = "") -> dict | None:
    """The held symbol furthest below its target (by value gap). None if none."""
    best: dict | None = None
    best_gap = 0.0
    for h in holdings or []:
        sym = str(h.get("symbol") or "")
        if not sym or sym == exclude:
            continue
        current = _num(h.get("current_weight"))
        target = _num(h.get("target_weight"))
        gap = target - current
        if gap <= 1e-9:
            continue
        # Tie-break: larger gap wins; equal gaps break alphabetically.
        if (gap > best_gap + 1e-12
                or (abs(gap - best_gap) <= 1e-12 and best is not None
                    and sym < str(best.get("symbol")))):
            best_gap = gap
            best = {**h, "gap": gap}
    return best


def best_idea(candidates: list[dict] | None, held: set, exclude: str = "") -> dict | None:
    """The highest-score candidate idea not already a holding. None if none."""
    best: dict | None = None
    best_score = float("-inf")
    for c in candidates or []:
        sym = str(c.get("symbol") or "")
        if not sym or sym == exclude or sym in held:
            continue
        score = _num(c.get("score"), float("-inf"))
        if score > best_score:
            best_score = score
            best = c
    return best


def route_sell(sell: dict, holdings: list[dict] | None = None,
               candidates: list[dict] | None = None) -> dict:
    """Where a sell's proceeds should go. Returns ``{kind, symbol, text}``.

    Priority: underweight holding -> best idea -> hold proceeds at the broker
    (untracked). v10.8.7 (R-C1): there is no tracked-cash target.
    """
    holdings = holdings or []
    exclude = str(sell.get("symbol") or "")
    held = {str(h.get("symbol") or "") for h in holdings}

    under = largest_underweight(holdings, exclude=exclude)
    if under is not None:
        pp = under["gap"] * 100.0
        text = f"buy {_label(under)}, {pp:.0f}pp under target"
        return {"kind": ROUTE_UNDERWEIGHT, "symbol": under.get("symbol"),
                "name": under.get("name"), "text": text}

    idea = best_idea(candidates, held, exclude=exclude)
    if idea is not None:
        text = f"buy {_label(idea)}, top idea"
        return {"kind": ROUTE_IDEA, "symbol": idea.get("symbol"),
                "name": idea.get("name"), "text": text}

    return {"kind": ROUTE_HOLD, "symbol": None,
            "text": "no eligible buy passes the hurdle; hold proceeds at the "
                    "broker, untracked"}


def paired_line(sell: dict, holdings: list[dict] | None = None,
                candidates: list[dict] | None = None) -> str:
    """The full one-line pairing for a sell item (T5).

    ``sell`` carries ``label`` (the "Name (TICKER)" form) and ``amount_eur``.
    """
    amount = _num(sell.get("amount_eur"))
    label = str(sell.get("label") or _label(sell))
    route = route_sell(sell, holdings, candidates)
    return f"Sell ~{amount:.0f} EUR of {label} -> {route['text']}"

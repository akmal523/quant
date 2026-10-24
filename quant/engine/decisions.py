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


def _label(item: dict) -> str:
    """The "Human name (TICKER)" label for an advice or holding dict.

    v10.8.5 (D8): routed through :func:`quant.ui.search.label_for`, which never
    doubles a ticker already present in the name ("Nasdaq 100 (SXRV)" stays
    "Nasdaq 100 (SXRV)", not "... (SXRV) (SXRV.DE)").
    """
    name = str(item.get("company_name") or item.get("name") or "").strip()
    symbol = str(item.get("symbol") or "").strip()
    if name and symbol and name != symbol:
        try:
            from quant.ui.search import label_for

            return label_for(name, symbol)
        except Exception:  # noqa: BLE001
            return f"{name} ({symbol})"
    return name or symbol


# v10.8.2 (section 6): the first line is always "Verb Name (TICKER) amount".
# v10.8.4 (T9): "Sell about X EUR of Name (TICKER) (TICKER.DE)" became
# "Sell ~X EUR of Name (TICKER.DE)": one verb, one ticker.
_VERB_TEMPLATES = {
    "buy": "Buy {amount} EUR of {label}",
    "top_up": "Buy {amount} EUR of {label}",
    "sell_part": "Sell ~{amount} EUR of {label}",
    "change_savings_plan": "Change the savings plan for {label}",
    # v10.8.5 (T3): the widened verb classes (one item each in the ONE list).
    "rebalance": "Rebalance {label}",
    "threat": "Review {label}",
    "plan_change": "Raise the savings plan for {label}",
    "swap": "Swap {label}",
}


def open_since_line(item: dict) -> str:
    """The 'open since <date>' state line for a repeated advice (T5).

    A repeated item is never byte-identical without showing how long it has been
    open; an item with no open_since returns an empty string.
    """
    value = item.get("open_since")
    if value is None:
        return ""
    return ui_copy.ADVICE_OPEN_SINCE.format(date=ui_copy.fmt_date(value))


def _amount_text(amount) -> str:
    """Whole EUR with a thousands separator (v10.8.2 money rule)."""
    try:
        return f"{float(amount):,.0f}"
    except (TypeError, ValueError):
        return ""


def format_item_line(item: dict) -> str:
    """The first line of a decision item: "Verb Name (TICKER) amount".

    v10.8.2 (section 6): never a bare reason. Falls back to the label when the
    kind has no verb template, so a line is always produced.
    """
    kind = str(item.get("verb") or item.get("kind") or "").strip().lower()
    label = str(item.get("label") or _label(item))
    amount = _amount_text(item.get("amount_eur"))
    tpl = _VERB_TEMPLATES.get(kind)
    if tpl is None:
        return label
    if amount:
        return tpl.format(amount=amount, label=label)
    return tpl.format(amount="", label=label).replace("  ", " ").strip()

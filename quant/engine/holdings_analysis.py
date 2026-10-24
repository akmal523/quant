"""
holdings_analysis.py — derived Holdings Analysis cells (v10.8.8, T2/R-P2).

Intent: the Holdings Analysis status columns were filled with placeholders
("held", a bare 0, "coverage -") instead of being derived from advice state,
weights and the run manifest. This module is the ONE pure source for the two
derived cells:

  - ``reason_cell``: the Reason column. If the symbol has open advice it reads
    "<Verb> ~<amount> EUR, open since <date>"; otherwise
    "no open advice; drift <x.x> pp vs <target>% target". Never "held".
  - ``threat_alerts`` / ``alert_count``: the Alerts column. The count of open
    Threat-verb items for the symbol (stop / drawdown / score floor, plus the
    market regime trigger for the Active class); 0 only when truly zero.

Invariants:
  - Pure: no I/O, no streamlit.
  - Never raises; a missing input yields a reasoned empty state, never a dash.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Verb -> plain word for the Reason cell (the ONE list's verbs).
_VERB_WORDS = {
    "sell_part": "Sell part", "buy": "Buy", "threat": "Review", "swap": "Swap",
    "plan_up": "Raise savings plan", "plan_down": "Lower savings plan",
    # verbs written to the advice log by older versions (History shows them)
    "sell": "Sell", "rebalance": "Rebalance", "plan_change": "Plan change",
    "change_savings_plan": "Change savings plan", "trim": "Sell part",
    "top_up": "Buy", "hold": "Hold",
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_date(value: Any) -> str:
    """A date as '1 Oct 2026' (empty when unknown)."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return f"{value.day} {_MONTHS[value.month - 1]} {value.year}"
    try:
        d = date.fromisoformat(str(value)[:10])
        return f"{d.day} {_MONTHS[d.month - 1]} {d.year}"
    except (ValueError, TypeError):
        return str(value)


def verb_word(verb: Any) -> str:
    """The plain word for an advice verb (P2: never a raw key)."""
    v = str(verb or "").strip()
    if v in _VERB_WORDS:
        return _VERB_WORDS[v]
    return v.replace("_", " ").capitalize() or "Advice"


def _open_for(symbol: str, open_advice: list[dict] | None) -> dict | None:
    """The first open advice item for ``symbol`` (None when none)."""
    sym = str(symbol)
    for item in open_advice or []:
        if str(item.get("symbol") or "") == sym:
            return item
    return None


def reason_cell(symbol: str, open_advice: list[dict] | None,
                drift: float | None = 0.0, target: float | None = 0.0) -> str:
    """The derived Reason cell (R-P2). Never "held".

    With open advice: "<Verb> ~<amount> EUR, open since <date>". Otherwise:
    "no open advice; drift <x.x> pp vs <target>% target".
    """
    item = _open_for(symbol, open_advice)
    if item is not None:
        verb = verb_word(item.get("verb"))
        amount = _num(item.get("amount_eur"), 0.0)
        since = _fmt_date(item.get("open_since"))
        tail = f", open since {since}" if since else ""
        return f"{verb} ~{amount:.0f} EUR{tail}"
    if target is None:
        return "no open advice; no target set"
    return (f"no open advice; drift {_num(drift) * 100:+.1f} pp vs "
            f"{_num(target) * 100:.0f}% target")


def _tier_of(symbol: str, holdings: list[dict] | None) -> str:
    for h in holdings or []:
        if str(h.get("symbol") or "") == str(symbol):
            return str(h.get("tier", "")).upper()
    return ""


def threat_alerts(symbol: str, holdings: list[dict] | None,
                  scores: dict | None = None, regime: Any = None) -> list[dict]:
    """The open Threat-verb triggers for ``symbol`` (R-P2).

    Reuses :func:`quant.engine.action_taxonomy.threat_items`; the market-level
    regime trigger is included for the Active class. Each item carries its
    ``reason`` (the trigger: stop / drawdown / score floor / regime).
    """
    from quant.engine import action_taxonomy as T

    sym = str(symbol)
    items = T.threat_items(holdings, scores, regime)
    out = [it for it in items if str(it.get("symbol") or "") == sym]
    if _tier_of(sym, holdings) == "ALPHA":
        out += [it for it in items if it.get("symbol") is None]
    return out


def alert_count(symbol: str, holdings: list[dict] | None,
                scores: dict | None = None, regime: Any = None) -> int:
    """The Alerts cell: the number of open Threat-verb triggers (0 when none)."""
    return len(threat_alerts(symbol, holdings, scores, regime))

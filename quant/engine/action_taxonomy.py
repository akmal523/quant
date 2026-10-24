"""
action_taxonomy.py — the decision verb classes (v10.8.5, T3; v10.8.7, T4/R-C1).

Intent: the ONE list must carry more than rebalancing. Each item is still a single
action with one reason and an evidence pointer; nothing new is added to the page.
The classes:

  - rebalance    drift over threshold from the target weight.
  - threat       a stop-loss/take-profit hit, a drawdown from entry, a score below
                 the floor, or a bear regime forcing class de-risking.
  - swap         a held Active/Speculative asset whose composite trails the best
                 Research idea by more than a hurdle (fees + tax + confidence).
                 Rendered as ONE paired item (sell X -> buy Y).
  - plan_change  a Long-term class under target for K months of plan runs.

v10.8.7 (R-C1): the ``cash_deploy`` verb is removed. Uninvested cash is not an
entity; a sell without a paired buy is reasoned "hold proceeds at the broker,
untracked" (see :mod:`quant.engine.pairing`).

Invariants:
  - Pure: no I/O.
  - A FORTRESS holding never yields a sell.
  - Each returned item carries ``verb``, ``symbol``, ``label``, ``amount_eur``,
    ``reason``, ``horizon``, ``class`` and an ``evidence`` pointer.
"""
from __future__ import annotations

from typing import Any

from quant.config import (
    PLAN_CHANGE_GAP,
    PLAN_CHANGE_MONTHS,
    REBALANCE_DRIFT_THRESHOLD,
    SCORE_FLOOR,
    SPEC_STOP,
    SPEC_TAKE,
    SWAP_HURDLE,
    THREAT_DRAWDOWN,
)

VERB_REBALANCE = "rebalance"
VERB_THREAT = "threat"
VERB_SWAP = "swap"
VERB_PLAN = "plan_change"

HORIZON_TODAY = "Today"
HORIZON_MONTH = "This month"

# Only these classes are swap candidates; Long-term is never sold for an idea.
_SWAPPABLE_TIERS = ("ALPHA", "SPECULATIVE")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label(item: dict) -> str:
    try:
        from quant.ui.search import label_for

        name = str(item.get("name") or item.get("company_name") or item.get("symbol") or "")
        symbol = str(item.get("symbol") or "")
        return label_for(name, symbol) if symbol else name
    except Exception:  # noqa: BLE001
        return str(item.get("symbol") or "")


def _item(verb, symbol, label, amount, reason, horizon, cls, evidence) -> dict:
    return {"verb": verb, "symbol": symbol, "label": label, "amount_eur": amount,
            "reason": reason, "horizon": horizon, "class": cls,
            "evidence": evidence}


def _tier(h: dict) -> str:
    return str(h.get("tier", "")).upper()


# ── Rebalance ─────────────────────────────────────────────────────────────────

def rebalance_items(holdings: list[dict] | None,
                    threshold: float = REBALANCE_DRIFT_THRESHOLD) -> list[dict]:
    """Drift over threshold from the target weight (FORTRESS excluded)."""
    out: list[dict] = []
    for h in holdings or []:
        tier = _tier(h)
        if tier == "FORTRESS":
            continue
        cur = _num(h.get("current_weight"))
        tgt = _num(h.get("target_weight"))
        drift = cur - tgt
        if abs(drift) <= threshold:
            continue
        out.append(_item(
            VERB_REBALANCE, str(h.get("symbol")), _label(h), None,
            f"{abs(drift) * 100:.0f}pp off its {tgt * 100:.0f}% target.",
            HORIZON_TODAY if drift > 0 else HORIZON_MONTH, tier,
            "Research"))
    return out


# ── Threat ────────────────────────────────────────────────────────────────────

def threat_items(holdings: list[dict] | None, scores: dict | None = None,
                 regime: Any = None) -> list[dict]:
    """A stop/take hit, a drawdown, a score floor, or a bear regime."""
    scores = scores or {}
    out: list[dict] = []
    for h in holdings or []:
        sym = str(h.get("symbol"))
        tier = _tier(h)
        entry = _num(h.get("entry_price") or h.get("cost_per_share"))
        price = _num(h.get("current_price"))
        pnl_pct = h.get("pnl_pct")
        if pnl_pct is None and entry > 0 and price > 0:
            pnl_pct = price / entry - 1.0
        # Speculative stop / take.
        if tier == "SPECULATIVE" and pnl_pct is not None:
            if pnl_pct <= SPEC_STOP:
                out.append(_item(VERB_THREAT, sym, _label(h), None,
                                 f"stop-loss hit ({pnl_pct * 100:.0f}%).",
                                 HORIZON_TODAY, tier, "Holdings analysis"))
                continue
            if pnl_pct >= SPEC_TAKE:
                out.append(_item(VERB_THREAT, sym, _label(h), None,
                                 f"take-profit hit (+{pnl_pct * 100:.0f}%).",
                                 HORIZON_TODAY, tier, "Holdings analysis"))
                continue
        # Drawdown from entry.
        if pnl_pct is not None and pnl_pct <= -THREAT_DRAWDOWN:
            out.append(_item(VERB_THREAT, sym, _label(h), None,
                             f"down {abs(pnl_pct) * 100:.0f}% from entry.",
                             HORIZON_TODAY, tier, "Holdings analysis"))
            continue
        # Score floor: the held composite, only when the score is trustworthy
        # (medium/high confidence) and never for a Long-term holding (v10.8.10).
        s = scores.get(sym) or {}
        comp = _score_of(s)
        trusted = str(s.get("confidence") or "medium") in ("medium", "high")
        if (comp is not None and trusted and tier != "FORTRESS"
                and _num(comp) < SCORE_FLOOR):
            out.append(_item(VERB_THREAT, sym, _label(h), None,
                             f"score {_num(comp):.0f} below the {SCORE_FLOOR:.0f} floor.",
                             HORIZON_TODAY, tier, "Research"))
    # Bear regime (market-level, once).
    from quant.portfolio.regime_constraints import is_bear

    if is_bear(regime):
        out.append(_item(VERB_THREAT, None, "the Active class", None,
                         "falling regime; de-risk the Active part.",
                         HORIZON_TODAY, "ALPHA", "Market"))
    return out


# ── Swap ──────────────────────────────────────────────────────────────────────

_SCORE_KEYS = ("composite", "active_score", "score", "structural_grade")


def _score_of(item: dict) -> Any:
    """The first known score field (composite first), or None when unknown."""
    for key in _SCORE_KEYS:
        if item.get(key) is not None:
            return item.get(key)
    return None


def _composite(item: dict) -> float:
    return _num(_score_of(item))


def swap_items(holdings: list[dict] | None, ideas: list[dict] | None,
               scores: dict | None = None, hurdle: float = SWAP_HURDLE) -> list[dict]:
    """A held Active/Speculative asset that trails the best idea after the hurdle."""
    scores = scores or {}
    out: list[dict] = []
    ideas = ideas or []
    for h in holdings or []:
        tier = _tier(h)
        if tier not in _SWAPPABLE_TIERS:
            continue
        sym = str(h.get("symbol"))
        s = scores.get(sym) or {}
        if _score_of({**h, **s}) is None:
            continue  # an unscored holding is never swapped on a fake 0
        held_score = _composite({**h, **s})
        best = None
        best_score = float("-inf")
        for idea in ideas:
            if str(idea.get("symbol")) == sym:
                continue
            sc = _composite(idea)
            if sc > best_score:
                best_score = sc
                best = idea
        if best is None:
            continue
        # The hurdle stands for the fee + tax + confidence gap (points).
        if best_score - held_score <= hurdle * 100.0:
            continue
        out.append(_item(
            VERB_SWAP, sym, _label(h), _num(h.get("value_eur")),
            f"{_label(best)} {best_score / 100:.2f} vs {_label(h)} "
            f"{held_score / 100:.2f} after hurdle.",
            HORIZON_TODAY, tier, "Research",
        ) | {"target_symbol": str(best.get("symbol")), "target_name": best.get("name")})
    return out


# ── Plan change ───────────────────────────────────────────────────────────────

def plan_change_items(holdings: list[dict] | None,
                      months_under: int | dict | None = None,
                      gap: float = PLAN_CHANGE_GAP,
                      months_required: int = PLAN_CHANGE_MONTHS) -> list[dict]:
    """A Long-term holding under target for K consecutive months of plan runs.

    ``months_under`` is either one count for every holding or a per-symbol map
    (see :func:`quant.engine.today_list.months_under_target`).
    """
    out: list[dict] = []
    for h in holdings or []:
        if _tier(h) != "FORTRESS":
            continue
        sym = str(h.get("symbol"))
        months = (int((months_under or {}).get(sym, 0))
                  if isinstance(months_under, dict) else int(months_under or 0))
        if months < months_required:
            continue
        cur = _num(h.get("current_weight"))
        tgt = _num(h.get("target_weight"))
        if (tgt - cur) > gap:
            out.append(_item(
                VERB_PLAN, sym, _label(h), None,
                f"under target for {months} months; raise the savings plan.",
                HORIZON_MONTH, "FORTRESS", "Savings plan"))
    return out


# ── Aggregator ────────────────────────────────────────────────────────────────

def build(holdings: list[dict] | None, scores: dict | None = None,
          ideas: list[dict] | None = None, regime: Any = None,
          months_under: int | dict | None = None) -> list[dict]:
    """All verb classes as ONE list. Today items sort before This-month.

    v10.8.7 (T4): a swappable position never gets a standalone sell, so a swap
    item and a rebalance item for the same symbol never coexist.
    """
    holdings = holdings or []
    items: list[dict] = []
    items += rebalance_items(holdings)
    items += threat_items(holdings, scores, regime)
    swaps = swap_items(holdings, ideas, scores)
    items += swaps
    items += plan_change_items(holdings, months_under=months_under)
    swap_syms = {str(s.get("symbol")) for s in swaps}
    items = [i for i in items
             if not (i.get("verb") == VERB_REBALANCE
                     and str(i.get("symbol")) in swap_syms)]
    order = {HORIZON_TODAY: 0, HORIZON_MONTH: 1}
    items.sort(key=lambda i: order.get(i.get("horizon"), 2))
    return items

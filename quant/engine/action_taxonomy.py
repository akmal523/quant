"""
action_taxonomy.py — the five decision verb classes (v10.8.5, T3).

Intent: the ONE list must carry more than rebalancing. Each item is still a single
action with one reason and an evidence pointer; nothing new is added to the page.
The five classes:

  - rebalance    drift over threshold from the target weight.
  - threat       a stop-loss/take-profit hit, a drawdown from entry, a score below
                 the floor, or a bear regime forcing class de-risking.
  - swap         a held Active/Speculative asset whose composite trails the best
                 Research idea by more than a hurdle (fees + tax + confidence).
                 Rendered as ONE paired item (sell X -> buy Y).
  - plan_change  a Long-term class under target for K months of plan runs.
  - cash_deploy  cash over its target weight, routed into the top eligible idea.

Invariants:
  - Pure: no I/O.
  - A FORTRESS holding never yields a sell.
  - Each returned item carries ``verb``, ``symbol``, ``label``, ``amount_eur``,
    ``reason``, ``horizon``, ``class`` and an ``evidence`` pointer.
"""
from __future__ import annotations

from typing import Any

from quant.config import REBALANCE_DRIFT_THRESHOLD

VERB_REBALANCE = "rebalance"
VERB_THREAT = "threat"
VERB_SWAP = "swap"
VERB_PLAN = "plan_change"
VERB_CASH = "cash_deploy"

HORIZON_TODAY = "Today"
HORIZON_MONTH = "This month"

# Trigger thresholds (documented, conservative).
THREAT_DRAWDOWN = 0.20       # 20% below entry
SCORE_FLOOR = 40.0           # composite/structural below this is a threat
SPEC_STOP = -0.50
SPEC_TAKE = 1.00
SWAP_HURDLE = 0.05           # 5 points after fees/tax/confidence
DEFAULT_CASH_TARGET = 0.10


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
        # Score floor.
        s = scores.get(sym) or {}
        comp = s.get("active_score")
        if comp is None:
            comp = s.get("structural_grade")
        if comp is not None and _num(comp) < SCORE_FLOOR:
            out.append(_item(VERB_THREAT, sym, _label(h), None,
                             f"score {_num(comp):.0f} below the {SCORE_FLOOR:.0f} floor.",
                             HORIZON_TODAY, tier, "Research"))
    # Bear regime (market-level, once).
    label = regime.get("label") if isinstance(regime, dict) else regime
    if str(label).lower() == "bear":
        out.append(_item(VERB_THREAT, None, "the Active class", None,
                         "falling regime; de-risk the Active part.",
                         HORIZON_TODAY, "ALPHA", "Market"))
    return out


# ── Swap ──────────────────────────────────────────────────────────────────────

def _composite(item: dict) -> float:
    for key in ("composite", "active_score", "score", "structural_grade"):
        if item.get(key) is not None:
            return _num(item.get(key))
    return 0.0


def swap_items(holdings: list[dict] | None, ideas: list[dict] | None,
               scores: dict | None = None, hurdle: float = SWAP_HURDLE) -> list[dict]:
    """A held Active/Speculative asset that trails the best idea after the hurdle."""
    scores = scores or {}
    out: list[dict] = []
    ideas = ideas or []
    for h in holdings or []:
        tier = _tier(h)
        if tier == "FORTRESS":
            continue
        sym = str(h.get("symbol"))
        s = scores.get(sym) or {}
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

def plan_change_items(holdings: list[dict] | None, months_under: int = 0,
                      gap: float = 0.10, months_required: int = 3) -> list[dict]:
    """A Long-term class under target for K months of plan runs."""
    if months_under < months_required:
        return []
    out: list[dict] = []
    for h in holdings or []:
        if _tier(h) != "FORTRESS":
            continue
        cur = _num(h.get("current_weight"))
        tgt = _num(h.get("target_weight"))
        if (tgt - cur) > gap:
            out.append(_item(
                VERB_PLAN, str(h.get("symbol")), _label(h), None,
                f"under target for {months_under} months; raise the savings plan.",
                HORIZON_MONTH, "FORTRESS", "Savings plan"))
    return out


# ── Cash deploy ───────────────────────────────────────────────────────────────

def cash_deploy_items(cash: float, invested: float, ideas: list[dict] | None,
                      cash_target: float = DEFAULT_CASH_TARGET) -> list[dict]:
    """Cash over its target weight, routed into the top eligible idea."""
    invested = _num(invested)
    pool = invested + _num(cash)
    if pool <= 0:
        return []
    cash_weight = _num(cash) / pool
    if cash_weight <= cash_target:
        return []
    best = None
    best_score = float("-inf")
    for idea in ideas or []:
        sc = _composite(idea)
        if sc > best_score:
            best_score = sc
            best = idea
    target_sym = str(best.get("symbol")) if best else None
    target_name = best.get("name") if best else None
    excess = round((cash_weight - cash_target) * pool, 2)
    return [_item(
        VERB_CASH, target_sym, target_name or "the top idea", excess,
        f"cash is {cash_weight * 100:.0f}% vs the {cash_target * 100:.0f}% target.",
        HORIZON_MONTH, "ALPHA", "Research")]


# ── Aggregator ────────────────────────────────────────────────────────────────

def build(holdings: list[dict] | None, scores: dict | None = None,
          ideas: list[dict] | None = None, regime: Any = None,
          cash: float = 0.0, invested: float | None = None,
          cash_target: float = DEFAULT_CASH_TARGET,
          months_under: int = 0) -> list[dict]:
    """All five verb classes as ONE list. Today items sort before This-month."""
    holdings = holdings or []
    if invested is None:
        invested = sum(_num(h.get("value_eur")) for h in holdings)
    items: list[dict] = []
    items += rebalance_items(holdings)
    items += threat_items(holdings, scores, regime)
    items += swap_items(holdings, ideas, scores)
    items += plan_change_items(holdings, months_under=months_under)
    items += cash_deploy_items(cash, invested, ideas, cash_target)
    order = {HORIZON_TODAY: 0, HORIZON_MONTH: 1}
    items.sort(key=lambda i: order.get(i.get("horizon"), 2))
    return items

"""
invest_plan.py — the class-based invest suggestion (v10.8.3; v10.8.7, T3).

Intent: the user enters ONE amount. The app compares the current mix of the three
classes (FORTRESS / ALPHA / SPECULATIVE tiers) against the mix the chosen strategy
(risk profile) wants, and suggests buys that move the portfolio toward it. It is a
suggestion only: the app never controls the broker's plans and never sells here.

v10.8.7 (T3/R-C3):
  - The class Now comes from the ONE weight math (:mod:`quant.engine.weights`).
  - An instrument with an open trim/sell gets NO buy suggestion; if its class is
    under target the money routes to the next eligible instrument or the exclusion
    is noted in one line.
  - One-off EUR and savings-plan EUR/m are separate dimensions: the one-off row
    sums to at most the entered amount; the Long-term savings-plan increase is
    shown as "+Z EUR/m" and never added into the one-off sum.
  - A non-held idea can fill an active class (How = "New position").

Invariants:
  - Pure: no I/O.
  - Never suggests a sell.
  - The one-off buys never exceed the entered amount.
"""
from __future__ import annotations

from quant.config import BETS_MAX, DEFAULT_RISK_PROFILE, RISK_PROFILES
from quant.engine.allocator import DEFAULT_BROAD_ETF, DEFAULT_BROAD_ETF_NAME, ROUND_STEP
from quant.engine.sizing import round_down
from quant.engine.weights import CLASS_ORDER, class_weights

MIN_ORDER_EUR = 25.0

HOW_SAVINGS_PLAN = "Savings plan"
HOW_ONEOFF = "One-off"
HOW_NEW_POSITION = "New position"

# The classes that receive one-off money; FORTRESS is a savings plan (EUR/m).
_ACTIVE_CLASSES = ("ALPHA", "SPECULATIVE")
# Advice verbs that block a buy into the same instrument (T3).
_BLOCKING_VERBS = {"sell_part", "trim", "swap", "threat", "rebalance"}


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _class_targets(strategy: str) -> dict[str, float]:
    """The target fraction of invested per class, from the strategy's bounds."""
    long_min, _active_max, _max_pos = RISK_PROFILES.get(
        strategy, RISK_PROFILES[DEFAULT_RISK_PROFILE])
    spec = BETS_MAX
    fortress = long_min
    alpha = max(0.0, 1.0 - fortress - spec)
    return {"FORTRESS": fortress, "ALPHA": alpha, "SPECULATIVE": spec}


def _distribute(class_buy: float, members: list[dict], new_pool: float) -> list[tuple]:
    """Split class_buy across members by their underweight gap, else equally."""
    from quant.engine.weights import target_weight

    if class_buy <= 0 or not members:
        return []
    gaps = []
    for h in members:
        tw = target_weight(str(h.get("symbol")))
        gap = max(0.0, tw * new_pool - _num(h.get("value_eur")))
        gaps.append((gap, h))
    total_gap = sum(g for g, _ in gaps)
    if total_gap > 0:
        return [(h, class_buy * gap / total_gap) for gap, h in gaps if gap > 0]
    share = class_buy / len(members)
    return [(h, share) for h in members]


def _blocked_symbols(open_advice: list[dict] | None) -> set[str]:
    """Symbols with an open trim/sell (no buy suggestion until it clears)."""
    out: set[str] = set()
    for a in open_advice or []:
        verb = str(a.get("verb") or a.get("kind") or "").lower()
        if verb in _BLOCKING_VERBS:
            sym = str(a.get("symbol") or "")
            if sym:
                out.add(sym)
    return out


def _best_candidate(candidates: list[dict] | None, blocked: set[str],
                    held: set[str]) -> dict | None:
    """The highest-score non-held, non-blocked candidate idea."""
    best = None
    best_score = float("-inf")
    for c in candidates or []:
        sym = str(c.get("symbol") or "")
        if not sym or sym in blocked or sym in held:
            continue
        score = _num(c.get("score"), float("-inf"))
        if score > best_score:
            best_score = score
            best = c
    return best


def propose_by_class(
    amount_eur: float,
    holdings: list[dict] | None = None,
    strategy: str = DEFAULT_RISK_PROFILE,
    candidates: list[dict] | None = None,
    open_advice: list[dict] | None = None,
) -> dict:
    """Suggest buys per class toward the strategy's target mix.

    Returns ``{classes, orders, savings_plan, not_allocated, total,
    savings_plan_eur_month, excluded}`` where ``classes`` is one row per tier
    (now fraction, target fraction, suggested EUR), ``orders`` are the one-off
    buys, ``savings_plan`` the EUR/m suggestions, ``total`` the one-off sum and
    ``savings_plan_eur_month`` the separate monthly sum.
    """
    amount = _num(amount_eur)
    holdings = holdings or []
    if amount <= 0:
        return {"classes": [], "orders": [], "savings_plan": [],
                "not_allocated": 0.0, "total": 0.0,
                "savings_plan_eur_month": 0.0, "excluded": []}

    invested = sum(_num(h.get("value_eur")) for h in holdings)
    new_pool = invested + amount
    targets = _class_targets(strategy)
    now = class_weights(holdings)  # fractions of the invested pool (R-C3)
    now_value = {c: now.get(c, 0.0) * invested for c in CLASS_ORDER}

    blocked = _blocked_symbols(open_advice)
    held = {str(h.get("symbol")) for h in holdings}

    # One-off budget goes to the active classes; the Long-term class is a
    # savings-plan (EUR/m) suggestion, a separate dimension (T3).
    buys = {c: max(0.0, targets[c] * new_pool - now_value[c])
            for c in _ACTIVE_CLASSES}
    total_buy = sum(buys.values())
    if total_buy > amount > 0:
        scale = amount / total_buy
        buys = {c: v * scale for c, v in buys.items()}
    fortress_gap = max(0.0, targets["FORTRESS"] * new_pool - now_value["FORTRESS"])

    members = {c: [] for c in CLASS_ORDER}
    for h in holdings:
        tier = str(h.get("tier", "ALPHA")).upper()
        if tier in members:
            members[tier].append(h)

    orders: list[dict] = []
    savings_plan: list[dict] = []
    excluded: list[dict] = []
    not_allocated = 0.0
    used = {c: 0.0 for c in CLASS_ORDER}

    for tier in _ACTIVE_CLASSES:
        rows = _distribute(buys[tier], members[tier], new_pool)
        if not rows and buys[tier] > 0:
            idea = _best_candidate(candidates, blocked, held)
            amt = round_down(buys[tier], ROUND_STEP)
            if idea is not None and amt >= MIN_ORDER_EUR:
                orders.append({"symbol": str(idea.get("symbol")),
                               "name": idea.get("name") or idea.get("symbol"),
                               "amount_eur": amt, "how": HOW_NEW_POSITION,
                               "tier": tier})
                used[tier] = amt
                continue
            not_allocated += buys[tier]
            continue
        for h, raw in rows:
            sym = str(h.get("symbol"))
            if sym in blocked:
                excluded.append({"symbol": sym, "name": h.get("name") or sym,
                                 "reason": "open sell/trim; no buy until it clears."})
                not_allocated += raw
                continue
            amt = round_down(raw, ROUND_STEP)
            if amt < MIN_ORDER_EUR:
                not_allocated += amt
                continue
            orders.append({"symbol": sym, "name": h.get("name") or sym,
                           "amount_eur": amt, "how": HOW_ONEOFF, "tier": tier})
            used[tier] += amt

    # The Long-term savings-plan suggestion (EUR/m), separate from the one-off.
    if fortress_gap > 0:
        sp_amt = round_down(fortress_gap, ROUND_STEP)
        if sp_amt >= MIN_ORDER_EUR:
            savings_plan.append({"symbol": DEFAULT_BROAD_ETF,
                                 "name": DEFAULT_BROAD_ETF_NAME,
                                 "amount_eur": sp_amt, "how": HOW_SAVINGS_PLAN,
                                 "tier": "FORTRESS"})
            used["FORTRESS"] = sp_amt

    classes = [{
        "tier": c,
        "now_pct": now.get(c, 0.0) * 100.0,
        "target_pct": targets[c] * 100.0,
        "suggested_eur": round(used[c], 2),
    } for c in CLASS_ORDER]

    one_off_total = sum(o["amount_eur"] for o in orders)
    sp_total = sum(o["amount_eur"] for o in savings_plan)
    return {"classes": classes, "orders": orders, "savings_plan": savings_plan,
            "not_allocated": round(amount - one_off_total, 2),
            "total": round(one_off_total, 2),
            "savings_plan_eur_month": round(sp_total, 2),
            "excluded": excluded}

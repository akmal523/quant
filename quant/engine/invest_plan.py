"""
invest_plan.py — the class-based invest suggestion (v10.8.3).

Intent: the user enters ONE amount. The app compares the current mix of the three
classes (FORTRESS / ALPHA / SPECULATIVE tiers) against the mix the chosen strategy
(risk profile) wants, and suggests buys that move the portfolio toward it. It is a
suggestion only: the app never controls the broker's plans and never sells here.

Long-term (FORTRESS) buys are the savings plan (free on the buy side); active
(ALPHA) and small-bet (SPECULATIVE) buys are one-off orders (1 EUR fee).

Invariants:
  - Pure: no I/O.
  - Never suggests a sell.
  - The suggested buys never exceed the entered amount.
"""
from __future__ import annotations

from quant.config import BETS_MAX, DEFAULT_RISK_PROFILE, RISK_PROFILES
from quant.engine.allocator import DEFAULT_BROAD_ETF, DEFAULT_BROAD_ETF_NAME, ROUND_STEP
from quant.engine.sizing import round_down

MIN_ORDER_EUR = 25.0

HOW_SAVINGS_PLAN = "Savings plan"
HOW_ONEOFF = "One-off"
CLASS_ORDER = ("FORTRESS", "ALPHA", "SPECULATIVE")


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


def _class_now(holdings: list[dict]) -> dict[str, float]:
    out = {c: 0.0 for c in CLASS_ORDER}
    for h in holdings:
        tier = str(h.get("tier", "ALPHA")).upper()
        if tier in out:
            out[tier] += _num(h.get("value_eur"))
    return out


def _distribute(class_buy: float, members: list[dict], new_pool: float) -> list[tuple]:
    """Split class_buy across members by their underweight gap, else equally."""
    from quant.config import TARGET_WEIGHTS_INVESTED

    if class_buy <= 0 or not members:
        return []
    gaps = []
    for h in members:
        tw = _num(TARGET_WEIGHTS_INVESTED.get(str(h.get("symbol")), 0.0))
        gap = max(0.0, tw * new_pool - _num(h.get("value_eur")))
        gaps.append((gap, h))
    total_gap = sum(g for g, _ in gaps)
    if total_gap > 0:
        return [(h, class_buy * gap / total_gap) for gap, h in gaps if gap > 0]
    share = class_buy / len(members)
    return [(h, share) for h in members]


def propose_by_class(
    amount_eur: float,
    holdings: list[dict] | None = None,
    strategy: str = DEFAULT_RISK_PROFILE,
    candidates: list[dict] | None = None,
) -> dict:
    """Suggest buys per class toward the strategy's target mix.

    Returns ``{classes, orders, not_allocated, total}`` where ``classes`` is one
    row per tier (now fraction, target fraction, suggested EUR) and ``orders`` is
    ``{symbol, name, amount_eur, how, tier}``.
    """
    amount = _num(amount_eur)
    holdings = holdings or []
    if amount <= 0:
        return {"classes": [], "orders": [], "not_allocated": 0.0, "total": 0.0}

    invested = sum(_num(h.get("value_eur")) for h in holdings)
    new_pool = invested + amount
    targets = _class_targets(strategy)
    now_value = _class_now(holdings)
    now_total = invested or 1.0

    # Buy per class = the gap to target on the post-investment pool.
    buys = {c: max(0.0, targets[c] * new_pool - now_value[c]) for c in CLASS_ORDER}
    total_buy = sum(buys.values())
    if total_buy > amount > 0:
        scale = amount / total_buy
        buys = {c: v * scale for c, v in buys.items()}

    members = {c: [] for c in CLASS_ORDER}
    for h in holdings:
        tier = str(h.get("tier", "ALPHA")).upper()
        if tier in members:
            members[tier].append(h)

    orders: list[dict] = []
    not_allocated = 0.0
    used = {c: 0.0 for c in CLASS_ORDER}
    for tier in CLASS_ORDER:
        how = HOW_SAVINGS_PLAN if tier == "FORTRESS" else HOW_ONEOFF
        rows = _distribute(buys[tier], members[tier], new_pool)
        if not rows and buys[tier] > 0:
            if tier == "FORTRESS":
                orders.append({"symbol": DEFAULT_BROAD_ETF,
                               "name": DEFAULT_BROAD_ETF_NAME,
                               "amount_eur": round_down(buys[tier], ROUND_STEP),
                               "how": how, "tier": tier})
                used[tier] = round_down(buys[tier], ROUND_STEP)
                continue
            not_allocated += buys[tier]
            continue
        for h, raw in rows:
            amt = round_down(raw, ROUND_STEP)
            if amt < MIN_ORDER_EUR:
                not_allocated += amt
                continue
            orders.append({"symbol": str(h.get("symbol")),
                           "name": h.get("name") or h.get("symbol"),
                           "amount_eur": amt, "how": how, "tier": tier})
            used[tier] += amt

    classes = [{
        "tier": c,
        "now_pct": (now_value[c] / now_total * 100.0) if now_total else 0.0,
        "target_pct": targets[c] * 100.0,
        "suggested_eur": round(used[c], 2),
    } for c in CLASS_ORDER]

    total = sum(used.values())
    return {"classes": classes, "orders": orders,
            "not_allocated": round(amount - total, 2), "total": round(total, 2)}

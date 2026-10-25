"""
action_taxonomy.py — the rules behind the ONE list (v10.8.10: the only ones).

One rule set, four kinds of item. Each item is one action with one reason, an
evidence pointer, and a horizon (Today / This month):

  rebalance    an Active or Speculative holding more than
               ``REBALANCE_DRIFT_THRESHOLD`` above its target -> "Sell ~X EUR"
               (Today; paired with where the money goes), or as far below ->
               "Buy X EUR" (This month; not in a bear market). Holdings without
               a target get no drift advice.
  plan change  a Long-term holding more than ``PLAN_CHANGE_GAP`` below / above
               its target -> raise / lower its savings plan (This month).
               Long-term holdings are never sold.
  threat       Speculative stop-loss / take-profit; a drawdown from cost beyond
               ``THREAT_DRAWDOWN``; a fall of ``THREAT_CRASH_7D`` within 7 days;
               a trusted score below ``SCORE_FLOOR`` (not Long-term); the
               Speculative part above ``BETS_MAX``; a bear market (once).
  swap         a scored Active/Speculative holding that trails the best Idea by
               more than ``SWAP_HURDLE``; one paired item, and that symbol gets
               no separate buy/sell.

Sizing laws (:mod:`quant.engine.sizing`): sells round down to 5 EUR, never
below 25 EUR, never from a position under 100 EUR; buys round to 10 EUR.

v10.8.10 replaces ``advice.build_advice`` (whose Active buys needed a conviction
field that never existed, and whose Long-term items duplicated the plan change
here) and the separate ``alerts`` engine (a second threat system).

Invariants: pure, no I/O; every item carries verb, symbol, label, amount_eur,
reason, horizon, class and evidence.
"""
from __future__ import annotations

from typing import Any

from quant.config import (
    BETS_MAX,
    PLAN_CHANGE_GAP,
    REBALANCE_DRIFT_THRESHOLD,
    SCORE_FLOOR,
    SPEC_STOP,
    SPEC_TAKE,
    SWAP_HURDLE,
    THREAT_CRASH_7D,
    THREAT_DRAWDOWN,
)
from quant.engine import sizing
from quant.ui.search import label_for

VERB_SELL = "sell_part"
VERB_BUY = "buy"
VERB_PLAN_UP = "plan_up"
VERB_PLAN_DOWN = "plan_down"
VERB_THREAT = "threat"
VERB_SWAP = "swap"

HORIZON_TODAY = "Today"
HORIZON_MONTH = "This month"

# Only these classes are traded on drift / swapped; Long-term uses the plan.
_ACTIVE_TIERS = ("ALPHA", "SPECULATIVE")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _label(item: dict) -> str:
    symbol = str(item.get("symbol") or "")
    name = str(item.get("name") or symbol)
    return label_for(name, symbol) if symbol else name


def _item(verb, symbol, label, amount, reason, horizon, cls, evidence) -> dict:
    return {"verb": verb, "symbol": symbol, "label": label, "amount_eur": amount,
            "reason": reason, "horizon": horizon, "class": cls,
            "evidence": evidence}


def _tier(h: dict) -> str:
    return str(h.get("tier", "")).upper()


def _is_bear(regime: Any) -> bool:
    from quant.portfolio.regime_constraints import is_bear

    return is_bear(regime)


# ── Rebalance (Active / Speculative) ──────────────────────────────────────────

def rebalance_items(holdings: list[dict] | None, regime: Any = None,
                    threshold: float = REBALANCE_DRIFT_THRESHOLD) -> list[dict]:
    """Drift beyond the threshold -> a sized sell (Today) or buy (This month)."""
    holdings = holdings or []
    total = sum(_num(h.get("value_eur")) for h in holdings)
    out: list[dict] = []
    for h in holdings:
        tier, tgt = _tier(h), h.get("target_weight")
        if tier not in _ACTIVE_TIERS or tgt is None:
            continue
        cur, tgt = _num(h.get("current_weight")), _num(tgt)
        drift = cur - tgt
        sym, value = str(h.get("symbol")), _num(h.get("value_eur"))
        if drift > threshold:
            # Never a Long-term holding here (_ACTIVE_TIERS), so only the
            # untouchable and size laws apply.
            amount = (None if sizing.is_untouchable(value)
                      else sizing.sell_amount(drift * total, value))
            if amount is not None:
                out.append(_item(
                    VERB_SELL, sym, _label(h), amount,
                    f"{drift * 100:.0f}pp above its {tgt * 100:.0f}% target.",
                    HORIZON_TODAY, tier, "Holdings analysis"))
        elif -drift > threshold and not (tier == "ALPHA" and _is_bear(regime)):
            amount = sizing.round_buy(-drift * total)
            # Buys are whole 10 EUR steps, so >= 25 and > 25 are the same.
            if amount >= sizing.SELL_MIN_EUR:  # pragma: no mutate
                out.append(_item(
                    VERB_BUY, sym, _label(h), amount,
                    f"{-drift * 100:.0f}pp below its {tgt * 100:.0f}% target.",
                    HORIZON_MONTH, tier, "Holdings analysis"))
    return out


# ── Plan change (Long-term) ───────────────────────────────────────────────────

def plan_change_items(holdings: list[dict] | None,
                      gap: float = PLAN_CHANGE_GAP) -> list[dict]:
    """A Long-term holding far off its target: raise or lower its savings plan."""
    out: list[dict] = []
    for h in holdings or []:
        if _tier(h) != "FORTRESS" or h.get("target_weight") is None:
            continue
        cur, tgt = _num(h.get("current_weight")), _num(h.get("target_weight"))
        sym = str(h.get("symbol"))
        if tgt - cur > gap:
            out.append(_item(
                VERB_PLAN_UP, sym, _label(h), None,
                f"{(tgt - cur) * 100:.0f}pp below its {tgt * 100:.0f}% target.",
                HORIZON_MONTH, "FORTRESS", "Savings plan"))
        elif cur - tgt > gap:
            out.append(_item(
                VERB_PLAN_DOWN, sym, _label(h), None,
                f"{(cur - tgt) * 100:.0f}pp above its {tgt * 100:.0f}% target; "
                "Long-term holdings are never sold.",
                HORIZON_MONTH, "FORTRESS", "Savings plan"))
    return out


# ── Threat ────────────────────────────────────────────────────────────────────

def _score_of(item: dict) -> Any:
    """The first known score field (composite first), or None when unknown."""
    for key in ("composite", "score", "structural_grade"):
        if item.get(key) is not None:
            return item.get(key)
    return None


def threat_items(holdings: list[dict] | None, scores: dict | None = None,
                 regime: Any = None) -> list[dict]:
    """Price, score and market threats (one item per holding at most)."""
    scores = scores or {}
    holdings = holdings or []
    out: list[dict] = []
    for h in holdings:
        sym, tier = str(h.get("symbol")), _tier(h)
        pnl = h.get("pnl_pct")
        change_7d = h.get("change_7d")
        reason = None
        if tier == "SPECULATIVE" and pnl is not None and pnl <= SPEC_STOP:
            reason = f"stop-loss hit ({pnl * 100:.0f}%)."
        elif tier == "SPECULATIVE" and pnl is not None and pnl >= SPEC_TAKE:
            reason = f"take-profit hit (+{pnl * 100:.0f}%)."
        elif tier != "FORTRESS" and pnl is not None and pnl <= -THREAT_DRAWDOWN:
            reason = f"down {abs(pnl) * 100:.0f}% from what you paid."
        elif change_7d is not None and change_7d <= -THREAT_CRASH_7D:
            reason = f"fell {abs(change_7d) * 100:.0f}% in the last 7 days."
        else:
            s = scores.get(sym) or {}
            comp = _score_of(s)
            trusted = str(s.get("confidence") or "medium") in ("medium", "high")
            if (comp is not None and trusted and tier != "FORTRESS"
                    and _num(comp) < SCORE_FLOOR):
                reason = f"score {_num(comp):.0f} below the {SCORE_FLOOR:.0f} floor."
        if reason is not None:
            out.append(_item(VERB_THREAT, sym, _label(h), None, reason,
                             HORIZON_TODAY, tier, "Holdings analysis"))
    total = sum(_num(h.get("value_eur")) for h in holdings)
    spec = sum(_num(h.get("value_eur")) for h in holdings if _tier(h) == "SPECULATIVE")
    if total > 0 and spec / total > BETS_MAX:
        out.append(_item(VERB_THREAT, None, "the Speculative part", None,
                         f"{spec / total * 100:.0f}% of your money, above the "
                         f"{BETS_MAX:.0%} cap.", HORIZON_TODAY, "SPECULATIVE",
                         "Holdings analysis"))
    if _is_bear(regime):
        out.append(_item(VERB_THREAT, None, "the Active part", None,
                         "the market is falling; no new Active buys, review risk.",
                         HORIZON_TODAY, "ALPHA", "Market"))
    return out


# ── Swap ──────────────────────────────────────────────────────────────────────

def swap_items(holdings: list[dict] | None, ideas: list[dict] | None,
               scores: dict | None = None, hurdle: float = SWAP_HURDLE) -> list[dict]:
    """A scored Active/Speculative holding that trails the best idea after the hurdle."""
    scores = scores or {}
    out: list[dict] = []
    ideas = ideas or []
    for h in holdings or []:
        tier = _tier(h)
        if tier not in _ACTIVE_TIERS:
            continue
        sym = str(h.get("symbol"))
        s = scores.get(sym) or {}
        held = _score_of({**h, **s})
        if held is None:
            continue  # an unscored holding is never swapped on a fake 0
        best = max((i for i in ideas if str(i.get("symbol")) != sym),
                   key=lambda i: _num(_score_of(i)), default=None)
        if best is None:
            continue
        best_score = _num(_score_of(best))
        if best_score - _num(held) <= hurdle * 100.0:
            continue
        out.append(_item(
            VERB_SWAP, sym, _label(h), _num(h.get("value_eur")),
            f"{_label(best)} scores {best_score:.0f} vs {_num(held):.0f} for "
            f"{_label(h)}, more than the fee/tax hurdle.",
            HORIZON_TODAY, tier, "Research",
        ) | {"target_symbol": str(best.get("symbol")), "target_name": best.get("name")})
    return out


# ── The ONE list ──────────────────────────────────────────────────────────────

def build(holdings: list[dict] | None, scores: dict | None = None,
          ideas: list[dict] | None = None, regime: Any = None) -> list[dict]:
    """All rules as ONE list, Today before This month."""
    holdings = holdings or []
    swaps = swap_items(holdings, ideas, scores)
    swapped = {str(s["symbol"]) for s in swaps}
    items = [i for i in rebalance_items(holdings, regime)
             if str(i["symbol"]) not in swapped]
    items += plan_change_items(holdings)
    items += threat_items(holdings, scores, regime)
    items += swaps
    items.sort(key=lambda i: i["horizon"] != HORIZON_TODAY)   # stable: Today first
    return items

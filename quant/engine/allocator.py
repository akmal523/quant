"""
allocator.py — the monthly savings-plan split (v10.7.0, Section 8.2).

Intent: the user decides one number (the budget) from their salary; the system
does the rest but always shows reasons. Cash competes openly as an asset; nothing
is automatic behind the user's back.

Algorithm:
  - long base = MONTHLY_LONG_TERM_SHARE (70 percent) of the budget.
  - distribute the long base across FORTRESS holdings proportionally to their
    target gaps (largest gap first); if no FORTRESS holding exists, propose one
    broad ETF with the full long base and a reason.
  - active pool = budget - long base. If the regime is bear, the active pool
    goes to cash with the regime reason. Else if any ALPHA holding or candidate
    has HIGH conviction, allocate to at most two (largest conviction first);
    else to cash with the no-idea reason.
  - bets: only if the user has SPECULATIVE holdings or explicitly enabled bets,
    the bets weight is below 2 percent, and a candidate has HIGH conviction;
    amount = min(10 percent of the budget, room to 2 percent); else zero.
  - all EUR rounded to 5. The cash line is always shown when cash receives
    anything, because cash is a decision, not a leftover.

Invariants: pure function; no I/O.
"""
from __future__ import annotations

from typing import Any

from quant.config import (
    ACTIVE_TRADE_FEE_EUR,
    ALPHA_CONVICTION_HIGH,
    BETS_MAX,
    MONTHLY_LONG_TERM_SHARE,
    SPARPLAN_BUY_FEE_EUR,
    TARGET_WEIGHTS_INVESTED,
)
from quant.engine.sizing import round_down

ROUND_STEP = 5.0
DEFAULT_BROAD_ETF = "EUNL.DE"
DEFAULT_BROAD_ETF_NAME = "iShares Core MSCI World"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_high(conviction: Any) -> bool:
    return _num(conviction, 0.0) >= ALPHA_CONVICTION_HIGH


def _leg(symbol, name, amount, kind, reason, fee):
    return {
        "symbol": symbol,
        "name": name,
        "amount_eur": round_down(amount, ROUND_STEP),
        "kind": kind,
        "reason": reason,
        "fee_eur": fee,
    }


def _buffett_note(holding: dict) -> str:
    """A short Buffett-quality suffix for a long-term leg reason (v10.7.6).

    Empty unless the holding carries a passing Buffett result, so callers that
    do not attach one (and the golden backtest) see the reason unchanged.
    """
    b = holding.get("buffett") or {}
    if not b.get("passes_filter"):
        return ""
    moat = b.get("moat")
    if moat:
        return f" It passes the Buffett quality filter ({moat} moat)."
    return " It passes the Buffett quality filter."


def allocate(
    budget_eur: float,
    holdings: list[dict] | None = None,
    regime: str | None = None,
    candidates: list[dict] | None = None,
    bets_enabled: bool = False,
) -> list[dict]:
    """Return the monthly split as a list of legs (long-term, active, bet, cash)."""
    budget = _num(budget_eur)
    if budget <= 0:
        return []
    holdings = holdings or []
    candidates = candidates or []
    legs: list[dict] = []

    long_base = budget * MONTHLY_LONG_TERM_SHARE
    fortress = [h for h in holdings if str(h.get("tier", "")).upper() == "FORTRESS"]

    if fortress:
        gaps = []
        for h in fortress:
            target = _num(h.get("target_weight"))
            if h.get("target_weight") is None:
                # Part 4.1: the ONE per-symbol target map.
                target = TARGET_WEIGHTS_INVESTED.get(str(h.get("symbol", "")), 0.0)
            current = _num(h.get("current_weight"))
            gaps.append((max(0.0, target - current), h))
        total_gap = sum(g for g, _ in gaps)
        if total_gap <= 0:
            # No holding is under target: the long base goes to the largest
            # target long-term holding (the core), never split equally.
            h = max(fortress, key=lambda x: _num(x.get("target_weight")))
            legs.append(_leg(
                h.get("symbol"), h.get("name"), long_base, "long_term",
                "long-term part is on target; regular buying is fine."
                + _buffett_note(h),
                SPARPLAN_BUY_FEE_EUR))
        else:
            # R10: the long base goes to the SINGLE largest-gap holding; ties
            # break alphabetically by symbol. The reason states the actual gap.
            gap, h = min(
                (g for g in gaps if g[0] > 0),
                key=lambda x: (-x[0], str(x[1].get("symbol", ""))))
            pct = _num(h.get("current_weight")) * 100
            target_val = _num(h.get("target_weight"))
            if h.get("target_weight") is None:
                target_val = TARGET_WEIGHTS_INVESTED.get(str(h.get("symbol", "")), 0.0)
            target_pct = target_val * 100
            reason = (
                f"{h.get('name')} is {pct:.0f} percent of invested vs "
                f"{target_pct:.0f} percent target, a gap of {gap * 100:.0f} points; "
                f"new long-term money goes here." + _buffett_note(h))
            legs.append(_leg(
                h.get("symbol"), h.get("name"), long_base, "long_term",
                reason, SPARPLAN_BUY_FEE_EUR))
    else:
        legs.append(_leg(
            DEFAULT_BROAD_ETF, DEFAULT_BROAD_ETF_NAME, long_base, "long_term",
            "no long-term holding yet; start with one broad fund.",
            SPARPLAN_BUY_FEE_EUR))

    active_pool = budget - long_base

    # Bets: gated, and carved OUT of the active pool so the split sums to B.
    invested = sum(_num(h.get("value_eur")) for h in holdings)
    spec_value = sum(
        _num(h.get("value_eur")) for h in holdings
        if str(h.get("tier", "")).upper() == "SPECULATIVE")
    bets_weight = (spec_value / invested) if invested > 0 else 0.0
    has_spec = spec_value > 0
    bet_candidate = next(
        (c for c in candidates
         if str(c.get("tier", "")).upper() == "SPECULATIVE" and _is_high(c.get("conviction"))),
        None)
    bet_amount = 0.0
    if (has_spec or bets_enabled) and bets_weight < BETS_MAX and bet_candidate is not None:
        room = max(0.0, BETS_MAX * invested - spec_value)
        bet_amount = round_down(min(0.10 * budget, room), ROUND_STEP)
        if bet_amount < ROUND_STEP:
            bet_amount = 0.0
    if bet_amount > 0:
        active_pool = max(0.0, active_pool - bet_amount)

    if active_pool > 0:
        if str(regime).lower() == "bear":
            legs.append(_leg(
                None, "cash", active_pool, "cash",
                "market is falling; new active money goes to cash until the "
                "regime recovers.", 0.0))
        else:
            high = [
                h for h in holdings
                if str(h.get("tier", "")).upper() == "ALPHA" and _is_high(h.get("conviction"))
            ] + [
                c for c in candidates
                if str(c.get("tier", "")).upper() == "ALPHA" and _is_high(c.get("conviction"))
            ]
            high.sort(key=lambda x: _num(x.get("conviction")), reverse=True)
            if high:
                chosen = high[:2]
                share = active_pool / len(chosen)
                for h in chosen:
                    pct = _num(h.get("current_weight")) * 100
                    target_pct = _num(h.get("target_weight")) * 100
                    gap = abs(_num(h.get("target_weight"))
                              - _num(h.get("current_weight"))) * 100
                    legs.append(_leg(
                        h.get("symbol"), h.get("name"), share, "active",
                        f"{h.get('name')} is {pct:.0f} percent of invested vs "
                        f"{target_pct:.0f} percent target, a gap of {gap:.0f} points; "
                        f"high conviction active idea.", ACTIVE_TRADE_FEE_EUR))
            else:
                legs.append(_leg(
                    None, "cash", active_pool, "cash",
                    "no high-conviction active idea this month; cash beats a "
                    "weak buy after fees.", 0.0))

    if bet_amount > 0:
        legs.append(_leg(
            bet_candidate.get("symbol"), bet_candidate.get("name"), bet_amount, "bet",
            "small high-risk bet within the 2 percent cap.", ACTIVE_TRADE_FEE_EUR))

    legs = [leg for leg in legs if leg["amount_eur"] > 0]
    # R10: the split must sum to the budget exactly. Rounding to 5 EUR steps
    # leaves a residual; it goes to the cash leg (created if needed), because
    # cash is the honest home for an unallocated remainder.
    allocated = sum(leg["amount_eur"] for leg in legs)
    residual = budget - allocated
    if residual > 1e-9:
        cash = next((leg for leg in legs if leg["kind"] == "cash"), None)
        if cash is not None:
            cash["amount_eur"] = cash["amount_eur"] + residual
        else:
            legs.append({
                "symbol": None, "name": "cash", "amount_eur": residual,
                "kind": "cash",
                "reason": "unallocated remainder; cash is the honest home.",
                "fee_eur": 0.0,
            })
    return legs

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
            current = _num(h.get("current_weight"))
            gaps.append((max(0.0, target - current), h))
        total_gap = sum(g for g, _ in gaps)
        if total_gap <= 0:
            # No gap: split equally.
            share = long_base / len(fortress)
            for h in fortress:
                legs.append(_leg(
                    h.get("symbol"), h.get("name"), share, "long_term",
                    "long-term part is on target; regular buying is fine.",
                    SPARPLAN_BUY_FEE_EUR))
        else:
            for gap, h in sorted(gaps, key=lambda x: x[0], reverse=True):
                amount = long_base * (gap / total_gap)
                if amount < ROUND_STEP:
                    continue
                pct = _num(h.get("current_weight")) * 100
                target_pct = _num(h.get("target_weight")) * 100
                legs.append(_leg(
                    h.get("symbol"), h.get("name"), amount, "long_term",
                    f"long-term part is {pct:.0f} percent of invested, "
                    f"target {target_pct:.0f} percent.",
                    SPARPLAN_BUY_FEE_EUR))
    else:
        legs.append(_leg(
            DEFAULT_BROAD_ETF, DEFAULT_BROAD_ETF_NAME, long_base, "long_term",
            "no long-term holding yet; start with one broad fund.",
            SPARPLAN_BUY_FEE_EUR))

    active_pool = budget - long_base
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
                    legs.append(_leg(
                        h.get("symbol"), h.get("name"), share, "active",
                        "high conviction active idea.", ACTIVE_TRADE_FEE_EUR))
            else:
                legs.append(_leg(
                    None, "cash", active_pool, "cash",
                    "no high-conviction active idea this month; cash beats a "
                    "weak buy after fees.", 0.0))

    # Bets: gated.
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
    if (has_spec or bets_enabled) and bets_weight < BETS_MAX and bet_candidate is not None:
        room = max(0.0, BETS_MAX * invested - spec_value)
        amount = min(0.10 * budget, room)
        if amount >= ROUND_STEP:
            legs.append(_leg(
                bet_candidate.get("symbol"), bet_candidate.get("name"), amount, "bet",
                "small high-risk bet within the 2 percent cap.", ACTIVE_TRADE_FEE_EUR))

    return [leg for leg in legs if leg["amount_eur"] > 0]

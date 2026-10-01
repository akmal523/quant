"""
fortress.py — Tier 1: Fortress (eternal holdings).

Intent (v10.6.2): Fortress assets are never sold, to avoid capital gains tax.
They are broad ETFs plus 3-5 core stocks accumulated via Sparplan (0 EUR buy).
Only the STRUCTURAL grade matters; the tactical grade and NLP sentiment are
ignored because a long-term holder does not trade the news. Rebalancing is
quarterly and only adjusts Sparplan amounts, never sells existing positions.

Invariants:
  - ``score_fortress_asset`` returns a dict with tier == "FORTRESS".
  - The signal is never a sell: only INCREASE_SPARPLAN or MAINTAIN.
  - ``liquidity_score`` is "LOW" (never recommended for emergency sale).
  - Pure functions (no I/O).

Dependencies: quant.config, quant.analytics.scoring.
"""
from __future__ import annotations

from quant.config import (
    FORTRESS_GRADE_FLOOR,
    FORTRESS_GRADE_FLOOR_MONTHS,
    FORTRESS_SPARPLAN_INCREASE_GRADE,
    TIER_CONSTRAINTS,
)

_TIER = "FORTRESS"


def fortress_signal(structural_grade: float) -> str:
    """Map a structural grade to a Fortress signal.

    Rule: grade > FORTRESS_SPARPLAN_INCREASE_GRADE -> INCREASE_SPARPLAN, else
    MAINTAIN. Never a sell. Pure.
    """
    if float(structural_grade) > FORTRESS_SPARPLAN_INCREASE_GRADE:
        return "INCREASE_SPARPLAN"
    return "MAINTAIN"


def is_structural_collapse(
    structural_grade: float,
    months_below: int,
) -> bool:
    """True only when the grade has been below the floor for long enough.

    Intent: the single exception to "never sell". A Fortress asset is only
    reconsidered when its structural grade sits below FORTRESS_GRADE_FLOOR for
    at least FORTRESS_GRADE_FLOOR_MONTHS consecutive months. Pure.
    """
    return (
        float(structural_grade) < FORTRESS_GRADE_FLOOR
        and int(months_below) >= FORTRESS_GRADE_FLOOR_MONTHS
    )


def score_fortress_asset(
    symbol: str,
    fundamentals: dict,
    stewardship: float | None = None,
) -> dict:
    """Score a Fortress asset on the structural grade alone.

    Intent: the Fortress scoring path. Tactical grade and NLP sentiment are
    deliberately absent. Invariants: returns a dict with tier "FORTRESS",
    signal in {INCREASE_SPARPLAN, MAINTAIN}, liquidity_score "LOW".
    """
    from quant.analytics.scoring import evaluate_structural_grade, stewardship_score_v2

    f = fundamentals or {}
    s_val = stewardship if stewardship is not None else stewardship_score_v2(f)
    structural_grade = evaluate_structural_grade(
        pe=f.get("PE"), peg=f.get("PEG"), roe=f.get("ROE"), stewardship_val=s_val,
    )
    constraints = TIER_CONSTRAINTS[_TIER]
    return {
        "symbol": symbol,
        "tier": _TIER,
        "structural_grade": round(float(structural_grade), 1),
        "tactical_grade": None,          # ignored for Fortress
        "nlp_sentiment": None,           # ignored for Fortress
        "signal": fortress_signal(structural_grade),
        "liquidity_score": constraints["liquidity_score"],
        "rebalance_frequency": constraints["rebalance_frequency"],
        "tax_implications": constraints["tax_implications"],
    }


def fortress_rebalance_recommendation(
    symbol: str,
    current_sparplan_eur: float,
    structural_grade: float,
) -> dict:
    """Quarterly Sparplan adjustment for a Fortress asset.

    Intent: Fortress rebalancing never sells; it only changes the monthly
    Sparplan amount. A grade above the increase threshold raises the amount by
    50 percent; otherwise the amount is maintained. Invariants: returns a dict
    with action in {INCREASE_SPARPLAN, MAINTAIN}; never a sell. Pure.
    """
    current = max(0.0, float(current_sparplan_eur))
    if fortress_signal(structural_grade) == "INCREASE_SPARPLAN":
        target = round(current * 1.5, 2)
        action = "INCREASE_SPARPLAN"
    else:
        target = round(current, 2)
        action = "MAINTAIN"
    return {
        "symbol": symbol,
        "tier": _TIER,
        "action": action,
        "current_sparplan_eur": round(current, 2),
        "target_sparplan_eur": target,
        "rebalance_frequency": TIER_CONSTRAINTS[_TIER]["rebalance_frequency"],
    }


def fortress_emergency_note(
    amount_needed: float,
    portfolio_value: float,
) -> str | None:
    """Return a tax warning when the cash need forces Fortress sales.

    Intent: Fortress assets are never recommended for sale. The single
    exception is a need exceeding 50 percent of the portfolio, where the system
    states the capital gains tax cost. Invariants: returns None when the need is
    at or below 50 percent of the portfolio; never raises. Pure.
    """
    if portfolio_value <= 0:
        return None
    if float(amount_needed) <= 0.5 * float(portfolio_value):
        return None
    return (
        "The amount needed exceeds half of the portfolio. Consider selling "
        "Fortress assets, but expect about 25 percent capital gains tax on the gain."
    )

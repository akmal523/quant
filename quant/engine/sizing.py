"""
sizing.py — Sizing rules for every advice (v10.7.0, Section 6).

Intent: the old system said "sell 200 EUR" of a 143 EUR position. These laws are
absolute and every advice path must route through them.

Laws:
  1. FORTRESS never sells. No sell advice may target a FORTRESS holding.
  2. Sell amount = min(drift, value - 1 EUR), rounded DOWN to 5 EUR steps.
     Below 25 EUR, no advice (fee relevance).
  3. A position below 100 EUR is untouchable (never emit a sell advice).
  4. Buy amounts round to 10 EUR steps.

Invariants: pure functions, no I/O.
"""
from __future__ import annotations

import math

from quant.config import (
    BUY_ROUND_STEP_EUR,
    SELL_MIN_EUR,
    SELL_ROUND_STEP_EUR,
    UNTOUCHABLE_POSITION_EUR,
)

FORTRESS = "FORTRESS"


def round_down(value: float, step: float) -> float:
    """Round ``value`` down to the nearest ``step`` (never above value)."""
    if step <= 0:
        return float(value)
    # Round the quotient first: 0.6 - 0.4 is 0.19999... in floating point, and
    # 20% of 1000 EUR must round to 200, not 195.
    return math.floor(round(float(value) / step, 9)) * step


def is_untouchable(position_value_eur: float) -> bool:
    """True when a position is too small for a sale to be worth the fee."""
    try:
        return float(position_value_eur) < UNTOUCHABLE_POSITION_EUR
    except (TypeError, ValueError):
        return True


def sell_amount(drift_eur: float, position_value_eur: float) -> float | None:
    """Suggested sell EUR, or None when the law forbids it.

    amount = min(drift, value - 1 EUR), rounded down to 5 EUR steps; None when
    the result is below 25 EUR.
    """
    try:
        drift = float(drift_eur)
        value = float(position_value_eur)
    except (TypeError, ValueError):
        return None
    if drift <= 0 or value <= 0:
        return None
    amount = min(drift, value - 1.0)
    amount = round_down(amount, SELL_ROUND_STEP_EUR)
    if amount < SELL_MIN_EUR:
        return None
    return amount


def can_sell(tier: str, position_value_eur: float, drift_eur: float) -> float | None:
    """The sell amount for a holding, or None when a law forbids it.

    Law 1: FORTRESS never sells. Law 3: positions below 100 EUR are untouchable.
    """
    if str(tier).upper() == FORTRESS:
        return None
    if is_untouchable(position_value_eur):
        return None
    return sell_amount(drift_eur, position_value_eur)


def round_buy(amount_eur: float) -> float:
    """Round a buy amount to 10 EUR steps (down)."""
    return round_down(amount_eur, BUY_ROUND_STEP_EUR)

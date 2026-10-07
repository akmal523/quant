"""
sizing.py — Sizing rules for every advice (v10.7.0, Section 6).

Intent: the old system said "sell 200 EUR" of a 143 EUR position. These laws are
absolute and every advice path must route through them.

Laws:
  1. FORTRESS never sells. No sell advice may target a FORTRESS holding.
  2. Sell amount = min(drift, value - 1 EUR), rounded DOWN to 5 EUR steps.
     Below 25 EUR, no advice (fee relevance).
  3. A position below 100 EUR is untouchable (never emit a sell advice).
  4. Cooldown blocks sells only; it never blocks savings-plan buys, new-money
     buys, or alert-driven sells.
  5. Buy amounts round to 10 EUR steps; active buys show the 1 EUR fee,
     savings-plan legs show 0 EUR.
  6. An active buy needs HIGH conviction; otherwise the money goes to cash.

Invariants: pure functions, no I/O.
"""
from __future__ import annotations

import math
from typing import Any

from quant.config import (
    ACTIVE_TRADE_FEE_EUR,
    BUY_ROUND_STEP_EUR,
    SELL_MIN_EUR,
    SELL_ROUND_STEP_EUR,
    SPARPLAN_BUY_FEE_EUR,
    UNTOUCHABLE_POSITION_EUR,
)

FORTRESS = "FORTRESS"


def round_down(value: float, step: float) -> float:
    """Round ``value`` down to the nearest ``step`` (never above value)."""
    if step <= 0:
        return float(value)
    return math.floor(float(value) / step) * step


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


def buy_fee(is_savings_plan: bool) -> float:
    """0 EUR for a savings-plan leg, 1 EUR for an active buy."""
    return SPARPLAN_BUY_FEE_EUR if is_savings_plan else ACTIVE_TRADE_FEE_EUR


def cooldown_blocks_sell(is_sell: bool) -> bool:
    """Cooldown blocks sells only (Law 4). Buys and alert sells are never blocked."""
    return bool(is_sell)


def cash_hurdle(conviction_high: bool) -> bool:
    """True when an active buy is allowed (HIGH conviction); else cash (Law 6)."""
    return bool(conviction_high)


def passes_fee_hurdle(expected_alpha_bps: Any, amount_eur: Any, fee_eur: Any) -> bool:
    """True when the expected alpha clears the fee (v10.7.4, Part 2.1).

    expected profit = expected_alpha_bps / 10000 * amount_eur. A buy is allowed
    when the expected profit is at least the fee (the boundary is inclusive, so
    50 bps on 200 EUR clears a 1 EUR fee). A missing alpha means no hurdle, so
    legacy callers keep working.
    """
    if expected_alpha_bps is None:
        return True
    try:
        expected = float(expected_alpha_bps) / 10000.0 * float(amount_eur)
        return expected >= float(fee_eur)
    except (TypeError, ValueError):
        return True

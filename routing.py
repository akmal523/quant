"""
routing.py — Signal Routing: Active Trade vs Sparplan (Phase 4, Module 1.3).

Intent: route capital allocation based on conviction type.
  - High Structural Grade (long-term hold) + low Tactical Grade (short-term
    noise) -> Sparplan (0 EUR buy, 1 EUR sell). Free accumulation.
  - High Tactical Grade (immediate breakout) -> Active Trade (1 EUR buy/sell).
  - Cannot beat the 2 EUR round-trip fee + cash yield -> Cash.

Phase 5 (v10.2):
  - INVERSE/LEVERAGED structure never routes to SPARPLAN (decay over time).
  - Missing ISIN emits an explicit "ISIN MISSING" instruction (not executable).
  - minimum_trade_size uses the symbol's expected alpha, not a constant 100 EUR.

Invariants:
  - route ∈ {SPARPLAN, ACTIVE, CASH, HOLD}
  - minimum_trade_size filter rejects signals below the fee hurdle.
  - Pure functions (no I/O).

Dependencies: config, optimizer.minimum_trade_size.
"""
from __future__ import annotations

from config import (
    SPARPLAN_STRUCT_MIN, ACTIVE_TACT_MIN,
    ROUND_TRIP_FEE_EUR, BROKER_CASH_APY,
)
from optimizer import minimum_trade_size
from taxonomy import PLAIN_STRUCTURE, INVERSE_STRUCTURE, LEVERAGED_STRUCTURE


def route_signal(
    structural_grade: float,
    tactical_grade: float,
    instrument_class: str = "EQUITY",
    structure: str = PLAIN_STRUCTURE,
) -> str:
    """Route a scored asset to SPARPLAN, ACTIVE, or HOLD.

    Intent: bifurcate execution by conviction type. ETFs/Core assets default to
    Sparplan (free execution). High-tactical equities route to Active Trade.
    Invariants: returns one of {SPARPLAN, ACTIVE, HOLD}.
    """
    # Inverse/leveraged products decay over time: never SPARPLAN, never CORE.
    if structure in (INVERSE_STRUCTURE, LEVERAGED_STRUCTURE):
        return "HOLD"

    # Core/ETF assets: long-term accumulation via free Sparplan.
    if instrument_class in ("ETF", "CASH"):
        return "SPARPLAN"

    # High tactical conviction -> immediate breakout -> Active Trade (1 EUR).
    if tactical_grade >= ACTIVE_TACT_MIN:
        return "ACTIVE"

    # High structural, low tactical -> long-term hold -> Sparplan accumulation.
    if structural_grade >= SPARPLAN_STRUCT_MIN:
        return "SPARPLAN"

    return "HOLD"


def minimum_capital_for_alpha(expected_alpha_bps: float) -> float:
    """Alias for optimizer.minimum_trade_size (fee hurdle)."""
    return minimum_trade_size(expected_alpha_bps, ROUND_TRIP_FEE_EUR)


def alpha_bps_from_active_score(active_score: float) -> float:
    """Map an active score to an expected alpha in basis points.

    Intent (Phase 5 / v10.2): the fee hurdle must vary with conviction, not be a
    constant 100 EUR. A score of 50 (neutral) maps to ~1 bps (floor); a score of
    100 maps to 5000 bps. Documented mapping:
        alpha_bps = max(active_score - 50.0, 1.0) * 100
    Invariants: returns >= 1.0; pure function (no I/O).
    """
    return max(active_score - 50.0, 1.0) * 100.0


def build_execution_instruction(
    symbol: str,
    route: str,
    current_price: float,
    capital_eur: float,
    expected_alpha_bps: float,
    isin: str = "",
    tr_ticker: str = "",
) -> dict:
    """Build a human-readable execution instruction for the daily briefing.

    Intent: produce exact trade instructions (e.g. "Sell 50 shares of AAPL via
    Active Trade") for the Streamlit dashboard and Telegram/Discord notifier.
    Phase 5 (v10.2): a missing ISIN makes the instruction unexecutable (the TR
    app searches by ISIN), so emit an explicit "ISIN MISSING" instruction.
    Invariants: returns dict with symbol, route, action, instruction, isin.
    """
    min_size = minimum_trade_size(expected_alpha_bps, ROUND_TRIP_FEE_EUR)
    fee_ok = capital_eur >= min_size

    if route == "ACTIVE":
        if not fee_ok:
            action = "CASH"
            instruction = (
                f"{symbol}: expected alpha {expected_alpha_bps:.0f} bps needs "
                f"min {min_size:.0f} EUR to clear the 2 EUR fee; capital "
                f"{capital_eur:.0f} EUR too small -> allocate to Cash "
                f"({BROKER_CASH_APY*100:.2f}% APY)."
            )
        elif not isin:
            action = "HOLD"
            instruction = (
                f"{symbol}: ISIN MISSING - resolve in broker app before execution"
            )
        else:
            shares = int(capital_eur / current_price) if current_price > 0 else 0
            action = "BUY"
            instruction = (
                f"Buy {shares} shares of {symbol} via Active Trade "
                f"(1 EUR fee). ISIN {isin} / {tr_ticker or symbol}."
            )
    elif route == "SPARPLAN":
        if not isin:
            action = "HOLD"
            instruction = (
                f"{symbol}: ISIN MISSING - resolve in broker app before execution"
            )
        else:
            action = "SPARPLAN"
            instruction = (
                f"Increase {symbol} Sparplan by {capital_eur:.0f} EUR "
                f"(0 EUR buy fee). ISIN {isin} / {tr_ticker or symbol}."
            )
    else:
        action = "HOLD"
        instruction = f"{symbol}: no action (HOLD)."

    return {
        "symbol": symbol,
        "route": route,
        "action": action,
        "instruction": instruction,
        "isin": isin,
        "tr_ticker": tr_ticker,
        "min_trade_size_eur": round(min_size, 2),
        "fee_hurdle_ok": fee_ok,
    }
"""
speculative.py — Tier 3: Speculative (high-risk bets).

Intent (v10.6.2): penny stocks, meme stocks, and recent IPOs for asymmetric
upside. Fundamentals are ignored (these companies may be worthless); only
momentum and volume matter. Hard cap: 2 percent of the portfolio. Stop-loss at
-50 percent, take-profit at +100 percent. Signals are generated daily but
executed weekly on Fridays.

Invariants:
  - ``score_speculative_asset`` returns a dict with tier == "SPECULATIVE".
  - ``max_allocation`` is the hard 2 percent cap.
  - Signals are in {BUY_SPECULATIVE, SELL_SPECULATIVE, HOLD_SPECULATIVE}.
  - Pure functions (no I/O).

Dependencies: quant.config.
"""
from __future__ import annotations

import pandas as pd

from quant.config import (
    SPECULATIVE_MAX_ALLOCATION,
    SPECULATIVE_MOMENTUM_BUY,
    SPECULATIVE_MOMENTUM_SELL,
    SPECULATIVE_STOP_LOSS,
    SPECULATIVE_TAKE_PROFIT,
    SPECULATIVE_VOLUME_SURGE,
    TIER_CONSTRAINTS,
)

_TIER = "SPECULATIVE"


def calculate_momentum(price_hist: pd.DataFrame | None, days: int = 90) -> float:
    """Return the percent price change over the last ``days`` sessions.

    Intent (v10.6.2): the primary speculative signal. Invariants: returns a
    float percent; 0.0 when history is missing or too short; pure.
    """
    if price_hist is None or "Close" not in getattr(price_hist, "columns", []):
        return 0.0
    close = price_hist["Close"].dropna()
    if len(close) < 2:
        return 0.0
    window = min(int(days), len(close) - 1)
    start = float(close.iloc[-1 - window])
    end = float(close.iloc[-1])
    if start <= 0:
        return 0.0
    return float((end / start - 1.0) * 100.0)


def volume_surge(price_hist: pd.DataFrame | None, lookback: int = 30) -> float:
    """Return current volume divided by the 30-day average volume.

    Intent (v10.6.2): a volume surge confirms a speculative move. Invariants:
    returns a float >= 0; 0.0 when volume is missing; pure.
    """
    if price_hist is None or "Volume" not in getattr(price_hist, "columns", []):
        return 0.0
    volume = price_hist["Volume"].dropna()
    if volume.empty:
        return 0.0
    avg = float(volume.tail(int(lookback)).mean())
    if avg <= 0:
        return 0.0
    return float(float(volume.iloc[-1]) / avg)


def speculative_signal(momentum_3m: float, volume_surge_ratio: float) -> str:
    """Map momentum and volume to a speculative signal.

    Rule: momentum > 50 and volume surge > 3 -> BUY_SPECULATIVE; momentum < -30
    -> SELL_SPECULATIVE; else HOLD_SPECULATIVE. Pure.
    """
    if momentum_3m > SPECULATIVE_MOMENTUM_BUY and volume_surge_ratio > SPECULATIVE_VOLUME_SURGE:
        return "BUY_SPECULATIVE"
    if momentum_3m < SPECULATIVE_MOMENTUM_SELL:
        return "SELL_SPECULATIVE"
    return "HOLD_SPECULATIVE"


def speculative_stop_take(pnl_pct: float) -> str | None:
    """Return a stop-loss or take-profit action for a speculative position.

    Intent (v10.6.2): cut losses at -50 percent, take profit at +100 percent.
    ``pnl_pct`` is a fraction (e.g. -0.5 for -50 percent). Invariants: returns
    "STOP_LOSS", "TAKE_PROFIT", or None; pure.
    """
    if pnl_pct <= SPECULATIVE_STOP_LOSS:
        return "STOP_LOSS"
    if pnl_pct >= SPECULATIVE_TAKE_PROFIT:
        return "TAKE_PROFIT"
    return None


def score_speculative_asset(symbol: str, price_hist: pd.DataFrame | None) -> dict:
    """Score a speculative asset on momentum and volume only.

    Intent (v10.6.2): fundamentals are ignored. Invariants: returns a dict
    with tier "SPECULATIVE", max_allocation 0.02, liquidity_score "VARIABLE",
    rebalance_frequency "DAILY_SIGNALS_WEEKLY_EXECUTION". Pure (no I/O).
    """
    momentum_3m = calculate_momentum(price_hist, days=90)
    surge = volume_surge(price_hist)
    constraints = TIER_CONSTRAINTS[_TIER]
    return {
        "symbol": symbol,
        "tier": _TIER,
        "momentum_3m": round(float(momentum_3m), 1),
        "volume_surge": round(float(surge), 2),
        "signal": speculative_signal(momentum_3m, surge),
        "max_allocation": SPECULATIVE_MAX_ALLOCATION,
        "liquidity_score": constraints["liquidity_score"],
        "rebalance_frequency": "DAILY_SIGNALS_WEEKLY_EXECUTION",
        "tax_implications": constraints["tax_implications"],
    }

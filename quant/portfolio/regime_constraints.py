"""
regime_constraints.py — Regime-Conditional Risk Limits (v10.4.0, Phase 3).

Intent: risk limits must depend on the market regime. In a Bear/Chop regime the
system hard-codes a 2% max single-stock weight and forbids leverage; in a Bull
regime the normal caps apply. This prevents the optimizer from concentrating
into volatile names exactly when the regime is hostile.

Invariants:
  - regime_from_bull_prob returns one of {bull, bear, chop}.
  - constraints_for_regime returns a dict with max_single_weight, max_leverage.
  - Pure functions (no I/O).

Dependencies: config.REGIME_CONSTRAINTS.
"""
from __future__ import annotations

from quant.config import REGIME_CONSTRAINTS

# HMM bull-probability thresholds. >= 0.60 is a bull regime; <= 0.40 is bear;
# the middle band is chop (uncertain).
BULL_THRESHOLD = 0.60
BEAR_THRESHOLD = 0.40


def regime_from_bull_prob(bull_prob: float) -> str:
    """Map an HMM bull probability to a discrete regime label.

    Intent: translate the continuous regime signal into the discrete keys used by
    REGIME_CONSTRAINTS. Invariants: returns one of {bull, bear, chop}.
    """
    if bull_prob is None:
        return "chop"
    if bull_prob >= BULL_THRESHOLD:
        return "bull"
    if bull_prob <= BEAR_THRESHOLD:
        return "bear"
    return "chop"


def constraints_for_regime(regime: str) -> dict:
    """Return the risk limits for a regime.

    Intent: single lookup so the optimizer and reporting agree on the caps.
    Invariants: always returns a dict with max_single_weight and max_leverage;
    unknown regimes fall back to the conservative chop limits.
    """
    return REGIME_CONSTRAINTS.get(regime, REGIME_CONSTRAINTS["chop"])


def apply_regime_constraints(max_weight: float, regime: str) -> float:
    """Cap a requested max single-asset weight by the regime limit.

    Intent: the optimizer's per-asset cap must never exceed the regime cap.
    Invariants: returns min(max_weight, regime cap); pure function (no I/O).
    """
    cap = constraints_for_regime(regime)["max_single_weight"]
    return float(min(max_weight, cap))

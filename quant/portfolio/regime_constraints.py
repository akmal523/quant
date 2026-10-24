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


# The review artifact stores the human words (rising/falling/mixed); the engine
# keys are bull/bear/chop. ``regime_key`` accepts either, so a consumer can never
# miss a bear market because of the vocabulary it was handed.
_REGIME_ALIASES = {
    "bull": "bull", "rising": "bull",
    "bear": "bear", "falling": "bear",
    "chop": "chop", "mixed": "chop",
}


def regime_key(regime) -> str | None:
    """The engine key (bull/bear/chop) for a label or a regime dict, else None."""
    label = regime.get("label") if isinstance(regime, dict) else regime
    return _REGIME_ALIASES.get(str(label or "").strip().lower())


def is_bear(regime) -> bool:
    """True when ``regime`` (label or dict, either vocabulary) is a bear market."""
    return regime_key(regime) == "bear"

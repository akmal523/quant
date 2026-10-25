"""
regime_constraints.py — the market-regime vocabulary (v10.4.0; v10.8.10).

Intent: one place normalises the two label vocabularies in use (engine keys
bull/bear/chop; the human words rising/falling/mixed), so every consumer (the
ONE list rules, the Holdings analysis alerts, the digest) recognises a bear
market the same way.

Invariants:
  - regime_key / is_bear accept a label or a regime dict in either vocabulary.
  - Pure functions (no I/O).
"""
from __future__ import annotations

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

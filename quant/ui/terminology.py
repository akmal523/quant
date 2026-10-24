"""
terminology.py — the ONE class-word map (v10.8.7, T3/A1).

Canonical UI words: **Long-term / Active / Speculative**. The descriptive form is
"Speculative (high risk)". No synonym ("Small bets", "FORTRESS") may appear in a
UI string or display-value constant; a lint test enforces it.

Invariants:
  - Pure: no I/O.
  - ``CLASS_LABELS`` is the single source for the class column and class table.
"""
from __future__ import annotations

CLASS_LONG_TERM = "Long-term"
CLASS_ACTIVE = "Active"
CLASS_SPECULATIVE = "Speculative"

CLASS_LABELS = {
    "FORTRESS": CLASS_LONG_TERM,
    "ALPHA": CLASS_ACTIVE,
    "SPECULATIVE": CLASS_SPECULATIVE,
}


def class_word(tier: str) -> str:
    """The canonical UI word for a tier key (falls back to the key)."""
    return CLASS_LABELS.get(str(tier).upper(), str(tier))

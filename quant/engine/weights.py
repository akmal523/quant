"""
weights.py — the ONE weight math (v10.8.7, T3/R-C3).

Intent: a single pair of functions produces instrument and class weights on the
invested-only basis. Every consumer (Holdings table, class table, decision drift,
invest plan) calls them; no local weight computation exists anywhere else.

Invariants:
  - Pure: no I/O.
  - Weights are fractions of the invested pool (sum to 1.0 when invested > 0).
  - Cash is not an entity (R-C1): the basis is the sum of invested values only.
"""
from __future__ import annotations

from quant.config import TARGET_WEIGHTS_INVESTED

CLASS_ORDER = ("FORTRESS", "ALPHA", "SPECULATIVE")


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def instrument_weights(holdings: list[dict] | None = None,
                       as_of=None) -> dict[str, float]:
    """The invested-only weight per symbol. Pure.

    Returns ``{symbol: fraction}``; empty when nothing is invested. ``as_of`` is
    accepted for a future date-based read; today the live holdings are the basis.
    """
    holdings = holdings or []
    total = sum(_num(h.get("value_eur")) for h in holdings)
    if total <= 0:
        return {}
    return {str(h.get("symbol")): _num(h.get("value_eur")) / total
            for h in holdings if h.get("symbol")}


def class_weights(holdings: list[dict] | None = None,
                  as_of=None) -> dict[str, float]:
    """The invested-only weight per class (FORTRESS / ALPHA / SPECULATIVE). Pure.

    Returns ``{class: fraction}``; all zeros when nothing is invested.
    """
    holdings = holdings or []
    total = sum(_num(h.get("value_eur")) for h in holdings)
    out = {c: 0.0 for c in CLASS_ORDER}
    if total <= 0:
        return out
    for h in holdings:
        tier = str(h.get("tier", "ALPHA")).upper()
        if tier in out:
            out[tier] += _num(h.get("value_eur"))
    return {c: v / total for c, v in out.items()}


def target_weight(symbol: str) -> float:
    """The per-symbol target fraction (the ONE target map). Pure."""
    return _num(TARGET_WEIGHTS_INVESTED.get(str(symbol), 0.0))

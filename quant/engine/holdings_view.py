"""
holdings_view.py — the live per-holding view (v10.8.3).

Intent: the five pages must show the user's real holdings the moment the table is
saved, without depending on a ``quant run`` review artifact. Before v10.8.3 the
pages read ``read_actions()``, which needs ``outputs/portfolio_audit.csv``; the
one-workflow never produces it, so Holdings and Full analysis were empty.

This module builds one list of holding dicts from the live position source
(:func:`quant.engine.positions.positions_now`) plus the tier map and the per-symbol
targets. It is the single input to the Portfolio table, the Full analysis table
and :func:`quant.engine.advice.build_advice`.

Invariants:
  - Pure read: no writes.
  - Never raises; returns [] on failure.
  - ``current_weight`` and ``target_weight`` are fractions of the invested pool.
"""
from __future__ import annotations

from typing import Any

from quant.config import TARGET_WEIGHTS_INVESTED

DEFAULT_TIER = "ALPHA"


def _display_name(symbol: str) -> str:
    try:
        from quant.data.names import display_name

        return display_name(symbol)
    except Exception:  # noqa: BLE001
        return symbol


def _tier_map() -> dict:
    try:
        from quant.portfolio.tier_manager import load_tiers_safe, tier_map

        return tier_map(load_tiers_safe()[0])
    except Exception:  # noqa: BLE001
        return {}


def holdings_view(conn: Any = None) -> list[dict]:
    """The live holdings, one dict per position.

    Fields: symbol, name, tier, value_eur, shares, cost_per_share, profit_eur,
    current_weight, target_weight, drift, estimated, as_of.
    """
    from quant.engine.positions import positions_now

    try:
        positions = positions_now(conn)
    except Exception:  # noqa: BLE001
        return []
    if not positions:
        return []

    tmap = _tier_map()
    total = sum(float(p.get("value_eur") or 0) for p in positions) or 1.0
    out: list[dict] = []
    for p in positions:
        sym = str(p.get("symbol") or "")
        if not sym:
            continue
        value = float(p.get("value_eur") or 0)
        current = value / total
        target = float(TARGET_WEIGHTS_INVESTED.get(sym, 0.0))
        out.append({
            "symbol": sym,
            "name": _display_name(sym),
            "tier": str(tmap.get(sym, DEFAULT_TIER)).upper(),
            "value_eur": value,
            "shares": float(p.get("shares") or 0),
            "cost_per_share": float(p.get("entry_eur") or 0),
            "profit_eur": float(p.get("profit_eur") or 0),
            "current_weight": current,
            "target_weight": target,
            "drift": current - target,
            "estimated": bool(p.get("estimated")),
            "as_of": p.get("as_of"),
        })
    return out


def class_totals(holdings: list[dict]) -> dict[str, float]:
    """The invested value per tier (FORTRESS / ALPHA / SPECULATIVE)."""
    out = {"FORTRESS": 0.0, "ALPHA": 0.0, "SPECULATIVE": 0.0}
    for h in holdings or []:
        tier = str(h.get("tier", DEFAULT_TIER)).upper()
        if tier in out:
            out[tier] += float(h.get("value_eur") or 0)
    return out

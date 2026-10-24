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
  - ``current_weight`` and ``target_weight`` are fractions of the invested pool;
    ``target_weight`` (and ``drift``) is None when no target is set.
"""
from __future__ import annotations

from typing import Any

from quant.engine.weights import instrument_weights

DEFAULT_TIER = "ALPHA"


def _display_name(symbol: str) -> str:
    try:
        from quant.data.names import display_name

        return display_name(symbol)
    except Exception:  # noqa: BLE001
        return symbol


def _tiers() -> tuple[dict, dict]:
    """``({symbol: tier}, {symbol: target fraction})`` from data/tiers.csv."""
    try:
        from quant.portfolio.tier_manager import load_tiers_safe, target_map, tier_map

        df = load_tiers_safe()[0]
        return tier_map(df), target_map(df)
    except Exception:  # noqa: BLE001
        return {}, {}


def holdings_view(conn: Any = None) -> list[dict]:
    """The live holdings, one dict per position.

    Fields: symbol, name, tier, value_eur, shares, cost_per_share, profit_eur,
    invested_eur, pnl_pct, current_weight, target_weight, drift, estimated, as_of.
    """
    from quant.engine.positions import positions_now

    positions = positions_now(conn)   # never raises
    if not positions:
        return []

    tmap, targets = _tiers()
    out: list[dict] = []
    for p in positions:
        sym = str(p.get("symbol") or "")
        if not sym:
            continue
        out.append({
            "symbol": sym,
            "name": _display_name(sym),
            "tier": str(tmap.get(sym, DEFAULT_TIER)).upper(),
            "value_eur": float(p.get("value_eur") or 0),
            "shares": float(p.get("shares") or 0),
            "cost_per_share": float(p.get("entry_eur") or 0),
            "profit_eur": float(p.get("profit_eur") or 0),
            "invested_eur": float(p.get("invested_eur") or 0),
            "pnl_pct": p.get("pnl_pct"),
            "estimated": bool(p.get("estimated")),
            "as_of": p.get("as_of"),
        })
    # v10.8.7 (T3/R-C3): the ONE weight math. No local value/total here.
    weights = instrument_weights(out)
    for h in out:
        h["current_weight"] = weights.get(h["symbol"], 0.0)
        # No target set -> None (no drift advice), never 0% (v10.8.10).
        h["target_weight"] = targets.get(h["symbol"])
        h["drift"] = (None if h["target_weight"] is None
                      else h["current_weight"] - h["target_weight"])
    return out

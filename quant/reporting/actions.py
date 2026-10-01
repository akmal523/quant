"""
actions.py — thin adapter over the ONE advice pipeline (v10.7.1, Part 1).

Intent: from v10.7.1 there is exactly ONE generator of user-facing advice,
:func:`quant.engine.advice.build_advice`. This module no longer computes its own
TRIM/BUY labels; it adapts the audit DataFrame into the advice pipeline's inputs
and maps the resulting Advice records back to the legacy dict shape that older
callers (the web briefing, the run status map, the doctor probe) still read.

Invariants:
  - Every action routes through :mod:`quant.engine.sizing`; a FORTRESS sell
    cannot be expressed.
  - A symbol with an actionable drift but no ISIN becomes BLOCKED with a remedy.
  - Pure function: reads the audit DataFrame + broker registry; no I/O writes.
"""
from __future__ import annotations

import pandas as pd

from quant.config import (
    LEGACY_TIER_MAPPING,
    MIN_TRADE_SIZE_EUR,
    REBALANCE_DRIFT_THRESHOLD,
    VALID_TIERS,
)
from quant.engine.advice import build_advice
from quant.execution.taxonomy import resolve_broker
from quant.ui import copy as ui_copy

_ACTION_WORD = {
    "sell_part": ui_copy.ADVICE_SELL_PART,
    "buy": ui_copy.ADVICE_BUY,
    "change_savings_plan": ui_copy.ADVICE_TOP_UP,
    "keep": ui_copy.ADVICE_KEEP,
    "to_cash": ui_copy.ADVICE_TO_CASH,
}
_STATUS_WORD = {
    "sell_part": ui_copy.STATUS_TRIM,
    "buy": ui_copy.STATUS_ADD,
    "change_savings_plan": ui_copy.STATUS_ADD,
    "keep": ui_copy.STATUS_ON_TRACK,
    "to_cash": ui_copy.STATUS_ON_TRACK,
}


def _pct(value) -> float:
    try:
        return float(str(value).rstrip("%") or 0) / 100.0
    except (TypeError, ValueError):
        return 0.0


def _tier(raw) -> str:
    tier = str(raw or "").upper()
    tier = LEGACY_TIER_MAPPING.get(tier, tier)
    return tier if tier in VALID_TIERS else "ALPHA"


def _tiers_map() -> dict:
    """Load the tiers.csv source of truth as ``{symbol: TIER}`` (never raises)."""
    try:
        from quant.portfolio.tier_manager import load_tiers, tier_map

        return tier_map(load_tiers())
    except Exception:  # noqa: BLE001
        return {}


def _resolve_tier(symbol: str, raw, tmap: dict) -> str:
    """tiers.csv wins; the audit's legacy Tier column is only a fallback.

    v10.7.2 (Part 5.1): the audit's Tier comes from the legacy ``classify_asset``
    (CORE/SATELLITE/ACTIVE/SECTOR), so a FORTRESS symbol could be misread as
    ALPHA and sold. Resolving from tiers.csv makes that impossible.
    """
    tier = tmap.get(str(symbol))
    if tier:
        return tier
    return _tier(raw)


def _holdings_from_audit(audit_df: pd.DataFrame, tmap: dict | None = None) -> list[dict]:
    """Adapt the audit DataFrame into build_advice's holding dicts."""
    tmap = tmap or {}
    holdings: list[dict] = []
    for _, r in audit_df.iterrows():
        symbol = str(r.get("Symbol", "")).strip()
        if not symbol:
            continue
        rec = str(r.get("Recommendation", "") or "")
        conviction = r.get("Conviction")
        if conviction is None or (isinstance(conviction, float) and conviction != conviction):
            # Legacy audits carry no conviction; a legacy BUY implies HIGH.
            conviction = 100.0 if rec.startswith("BUY") else 0.0
        cooldown = r.get("Cooldown_Until")
        if isinstance(cooldown, float) and cooldown != cooldown:
            cooldown = None
        value = r.get("Value_EUR")
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 0:
            # Legacy audits carry no Value_EUR; use a nominal so the drift in
            # EUR is meaningful for the sizing laws.
            value = 1000.0
        target = _pct(r.get("Target_Weight"))
        current = _pct(r.get("Current_Weight"))
        if "Current_Weight" not in r or r.get("Current_Weight") in (None, ""):
            # Legacy audits encode the direction in Drift = current - target.
            current = target + _pct(r.get("Drift"))
        holdings.append({
            "symbol": symbol,
            "name": str(r.get("Name", "") or symbol),
            "tier": _resolve_tier(symbol, r.get("Tier"), tmap),
            "value_eur": value,
            "current_weight": current,
            "target_weight": target,
            "conviction": float(conviction or 0),
            "cooldown_until": cooldown,
        })
    return holdings


def build_actions(audit_df: pd.DataFrame | None) -> list[dict]:
    """Build the canonical action list from the portfolio audit.

    Returns a list of dicts: symbol, action (dictionary word), amount_eur,
    reason, threshold, tier, target, drift, blocked, remedy, status, plus the
    advice fields (kind, company_name, from_where, fee_eur, tier_word, source).
    """
    if audit_df is None or audit_df.empty:
        return []

    tmap = _tiers_map()
    holdings = _holdings_from_audit(audit_df, tmap)
    advice, _rejected = build_advice(holdings, tiers=tmap)
    by_symbol = {a["symbol"]: a for a in advice if a.get("symbol")}

    actions: list[dict] = []
    for _, r in audit_df.iterrows():
        symbol = str(r.get("Symbol", ""))
        tier = _resolve_tier(symbol, r.get("Tier"), tmap)
        drift = str(r.get("Drift", ""))
        target = str(r.get("Target_Weight", ""))
        threshold = REBALANCE_DRIFT_THRESHOLD
        broker = resolve_broker(symbol)
        isin = broker.get("isin", "") if broker else ""
        if not isin:
            actions.append({
                "symbol": symbol, "action": "BLOCKED", "amount_eur": None,
                "reason": "ISIN missing", "threshold": threshold, "tier": tier,
                "target": target, "drift": drift, "blocked": True,
                "min_trade_eur": MIN_TRADE_SIZE_EUR,
                "status": ui_copy.STATUS_BLOCKED,
                "remedy": ui_copy.ACTION_BLOCKED_MANUAL.format(symbol=symbol),
                "kind": "blocked", "company_name": symbol, "from_where": None,
                "fee_eur": None, "tier_word": ui_copy.tier_word(tier), "source": "monitor",
            })
            continue
        a = by_symbol.get(symbol)
        kind = a["kind"] if a else "keep"
        if kind == "keep":
            # Legacy build_actions omitted within-threshold rows; keep that.
            continue
        rec = str(r.get("Recommendation", "") or "")
        cooldown = r.get("Cooldown_Until")
        if isinstance(cooldown, float) and cooldown != cooldown:
            cooldown = None
        # Status keeps the legacy semantics (WAITING on cooldown, ADD/TRIM by
        # drift); the ACTION WORD comes from the one advice pipeline.
        status = ui_copy.status_for(
            rec, cooldown_until=cooldown, drift_frac=_pct(drift), threshold=threshold)
        actions.append({
            "symbol": symbol,
            "action": _ACTION_WORD.get(kind, ui_copy.ADVICE_KEEP),
            "amount_eur": a.get("eur") if a else None,
            "reason": a.get("why") if a else "within its target band.",
            "threshold": threshold, "tier": tier, "target": target, "drift": drift,
            "blocked": False, "min_trade_eur": MIN_TRADE_SIZE_EUR,
            "status": status,
            "remedy": None,
            "kind": kind,
            "company_name": a.get("company_name") if a else symbol,
            "from_where": a.get("from_where") if a else None,
            "fee_eur": a.get("fee_eur") if a else None,
            "tier_word": ui_copy.tier_word(tier),
            "source": a.get("source") if a else "monitor",
        })
    return actions


def format_action_line(a: dict) -> str:
    """Format one action as a terse CLI line (spec 3.3)."""
    if a["blocked"]:
        return f"    BLOCKED   {a['symbol']:<9} {a['reason']}"
    amount = a.get("amount_eur")
    amount_str = f"{amount:.0f} EUR" if amount else ""
    return f"    {a['action']:<22} {a['symbol']:<9} {amount_str:<10} {a['reason']}"


def build_tier_actions(audit_df: pd.DataFrame | None) -> list[dict]:
    """Deprecated alias: the tier-aware list is now the one advice pipeline."""
    return build_actions(audit_df)

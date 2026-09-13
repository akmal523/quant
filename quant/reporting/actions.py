"""
actions.py — Canonical action list (v10.5.0, spec 3.3 / T3).

Intent: ONE source of truth for the actions shown in the CLI, the dashboard
Briefing page, and the hosted Published Briefing. Actions are derived from the
portfolio audit (drift vs tier target) plus the broker registry (ISIN routing).
No widget computes its own copy (R2).

Invariants:
  - Every action cites the threshold from quant.config that triggered it (R6).
  - A symbol with an actionable drift but no ISIN becomes BLOCKED with a remedy,
    never advice (R1).
  - Pure function: reads the audit DataFrame + broker registry; no I/O writes.

Dependencies: quant.config (REBALANCE_DRIFT_TIERS), quant.execution.taxonomy.
"""
from __future__ import annotations

import re

import pandas as pd

from quant.config import REBALANCE_DRIFT_TIERS
from quant.execution.taxonomy import resolve_broker
from quant.ui import copy as ui_copy

# Audit Recommendation format: "BUY 150 EUR (CORE drift -16.1% exceeds 10.0% threshold)"
_AMT_RE = re.compile(r"(BUY|SELL)\s+(\d+)\s+EUR")


def build_actions(audit_df: pd.DataFrame | None) -> list[dict]:
    """Build the canonical action list from the portfolio audit.

    Returns a list of dicts:
      symbol, action (BUY MORE | TRIM | BLOCKED), amount_eur, reason,
      threshold, tier, target, drift, blocked, remedy.
    """
    if audit_df is None or audit_df.empty:
        return []

    actions: list[dict] = []
    for _, r in audit_df.iterrows():
        rec = str(r.get("Recommendation", "") or "")
        m = _AMT_RE.search(rec)
        if not m:
            continue

        direction, amount = m.group(1), float(m.group(2))
        sym = str(r.get("Symbol", ""))
        tier = str(r.get("Tier", "ACTIVE"))
        drift = str(r.get("Drift", ""))
        target = str(r.get("Target_Weight", ""))
        threshold = REBALANCE_DRIFT_TIERS.get(tier, 0.05)

        broker = resolve_broker(sym)
        isin = broker.get("isin", "") if broker else ""
        # A3: the ONE status word, shared by the holdings table and the cards.
        cooldown_until = r.get("Cooldown_Until")
        if isinstance(cooldown_until, float) and cooldown_until != cooldown_until:
            cooldown_until = None  # NaN: no cooldown, not an invalid date

        if not isin:
            actions.append({
                "symbol": sym,
                "action": "BLOCKED",
                "amount_eur": None,
                "reason": "ISIN missing",
                "threshold": threshold,
                "tier": tier,
                "target": target,
                "drift": drift,
                "blocked": True,
                "status": ui_copy.status_for(rec, blocked=True),
                "remedy": ui_copy.ACTION_BLOCKED_MANUAL.format(symbol=sym),
            })
            continue

        action = "BUY MORE" if direction == "BUY" else "TRIM"
        reason = f"drift {drift} vs {tier} target {target} (threshold {threshold:.1%})"
        actions.append({
            "symbol": sym,
            "action": action,
            "amount_eur": amount,
            "reason": reason,
            "threshold": threshold,
            "tier": tier,
            "target": target,
            "drift": drift,
            "blocked": False,
            "status": ui_copy.status_for(rec, cooldown_until=cooldown_until),
            "remedy": None,
        })

    return actions


def format_action_line(a: dict) -> str:
    """Format one action as a terse CLI line (spec 3.3)."""
    if a["blocked"]:
        return f"    BLOCKED   {a['symbol']:<9} {a['reason']}"
    sign = "+" if a["action"] == "BUY MORE" else "-"
    return (
        f"    {a['action']:<9} {a['symbol']:<9} "
        f"{sign}{a['amount_eur']:.0f} EUR   {a['reason']}"
    )

"""
advice.py — the ONE generator of user-facing advice (v10.7.1, Part 1).

Intent: every surface that tells the user to do something — the briefing Actions
table, the ``quant run`` printout, the Overview steps, the weekly report, the
Monthly decision screen — renders from the same list of Advice records produced
here. No surface computes its own TRIM/BUY labels. Every sell and buy routes
through :mod:`quant.engine.sizing`, so the sizing laws are structurally
enforced: a FORTRESS sell cannot be expressed by any consumer.

Generation order (all enforced here):
  1. open alerts first (alert sells ignore cooldowns);
  2. FORTRESS holdings: never sell; a target gap over 10 points suggests a
     savings-plan leg change, else keep;
  3. ALPHA holdings: a drift sell only when every sizing law passes, else a
     rejected note; a buy only on HIGH conviction;
  4. SPECULATIVE holdings: stop-loss alerts only, no drift advice;
  5. a cash line when the regime is bear and a buy was considered.

Invariants:
  - Pure function; no I/O.
  - FORTRESS never yields ``sell_part``.
  - A considered-but-suppressed advice becomes a RejectedNote, never silence.
"""
from __future__ import annotations

from datetime import date
from typing import Any, TypedDict

from quant.config import (
    ACTIVE_TRADE_FEE_EUR,
    ALPHA_CONVICTION_HIGH,
    REBALANCE_DRIFT_THRESHOLD,
    SPARPLAN_BUY_FEE_EUR,
)
from quant.engine import sizing
from quant.ui import copy as ui_copy

FORTRESS_GAP_SUGGESTION = 0.10


class Advice(TypedDict):
    """One user-facing advice record."""

    kind: str            # buy | top_up | sell_part | keep | change_savings_plan | to_cash
    symbol: str | None
    company_name: str
    eur: float | None
    from_where: str | None   # cash | savings plan | position
    why: str
    fee_eur: float | None
    tier_word: str
    source: str          # alert | drift | monthly | monitor


class RejectedNote(TypedDict):
    """A considered-but-suppressed advice (the "Not this week" block)."""

    symbol: str
    considered_action: str
    plain_reason: str


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _tier_lookup(tiers: Any) -> dict:
    """Resolve a ``{symbol: TIER}`` map from a dict or a tiers DataFrame.

    v10.7.2 (Part 5.1): ``tiers.csv`` is the source of truth for the tier. The
    audit's legacy ``Tier`` column (CORE/SATELLITE/ACTIVE/SECTOR) must never
    decide whether a FORTRESS holding can be sold.
    """
    if tiers is None:
        return {}
    if isinstance(tiers, dict):
        return {str(k): str(v).upper() for k, v in tiers.items()}
    try:
        return {str(r["symbol"]): str(r["tier"]).upper() for _, r in tiers.iterrows()}
    except Exception:  # noqa: BLE001
        return {}


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _advice(kind, symbol, name, eur, from_where, why, fee, tier, source) -> Advice:
    return {
        "kind": kind,
        "symbol": symbol,
        "company_name": name,
        "eur": eur,
        "from_where": from_where,
        "why": why,
        "fee_eur": fee,
        "tier_word": ui_copy.tier_word(tier or "ALPHA"),
        "source": source,
    }


def _fortress_leg(plans, symbol, name, current, target) -> str:
    """The savings-plan leg suggestion for a FORTRESS gap."""
    leg = None
    if plans and plans.get("legs"):
        leg = next((x for x in plans["legs"] if x.get("symbol") == symbol), None)
    if leg:
        cur = _num(leg.get("amount_eur"))
        tgt = cur + 20.0
        return ui_copy.FORTRESS_LEG_SUGGESTION.format(
            name=name, current=f"{cur:.0f}", target=f"{tgt:.0f}",
            pct=f"{current * 100:.0f}", target_pct=f"{target * 100:.0f}")
    return (f"Consider raising the savings-plan leg for {name}; it is "
            f"{current * 100:.0f} percent of invested vs {target * 100:.0f} "
            f"percent target.")


def _reject_reason(tier, value, drift_eur, cooldown, as_of) -> str:
    """The plain reason a considered sell was suppressed."""
    if str(tier).upper() == "FORTRESS":
        return ui_copy.REJECT_FORTRESS
    if sizing.is_untouchable(value):
        return ui_copy.REJECT_TOO_SMALL.format(value=f"{value:.0f}")
    if cooldown is not None and _as_date(cooldown) and _as_date(cooldown) > as_of:
        return ui_copy.REJECT_COOLDOWN.format(date=ui_copy.fmt_date(cooldown))
    return ui_copy.REJECT_BELOW_MIN


def build_advice(
    holdings: list[dict] | None = None,
    tiers: dict | None = None,
    scores: dict | None = None,
    regime: Any = None,
    cooldowns: dict | None = None,
    open_alerts: list[dict] | None = None,
    plans: dict | None = None,
    as_of: date | None = None,
) -> tuple[list[Advice], list[RejectedNote]]:
    """Return (advice, rejected) for every surface to render."""
    as_of = as_of or date.today()
    holdings = holdings or []
    cooldowns = cooldowns or {}
    scores = scores or {}
    # v10.7.2 (Part 5.1): the tier comes from tiers.csv (the source of truth),
    # never from the audit's legacy Tier column. A FORTRESS holding can never be
    # sold, regardless of what the audit says.
    tmap = _tier_lookup(tiers)
    advice: list[Advice] = []
    rejected: list[RejectedNote] = []

    by_symbol = {str(h.get("symbol")): h for h in holdings}
    total = sum(_num(h.get("value_eur")) for h in holdings) or 1.0
    regime_label = regime.get("label") if isinstance(regime, dict) else regime
    buy_considered = False

    # 1. Open alerts first (alert sells ignore cooldowns).
    for alert in open_alerts or []:
        symbol = alert.get("symbol")
        holding = by_symbol.get(str(symbol), {})
        name = holding.get("name") or alert.get("name") or symbol or "Holding"
        action = str(alert.get("action", "review"))
        if action == "sell":
            advice.append(_advice(
                "sell_part", symbol, name, alert.get("amount_eur"),
                ui_copy.ADVICE_FROM_POSITION, alert.get("message", ""),
                alert.get("fee_eur"), holding.get("tier"), "alert"))
        else:
            advice.append(_advice(
                "keep", symbol, name, None, None, alert.get("message", ""),
                None, holding.get("tier"), "alert"))

    # 2-4. Per holding.
    for holding in holdings:
        symbol = str(holding.get("symbol", "")).strip()
        if not symbol:
            continue
        name = holding.get("name") or symbol
        # tiers.csv wins; the holding's tier is only a fallback.
        tier = tmap.get(symbol) or str(holding.get("tier", "")).upper()
        value = _num(holding.get("value_eur"))
        current = _num(holding.get("current_weight"))
        target = _num(holding.get("target_weight"))
        conviction = _num(holding.get("conviction"))
        cooldown = cooldowns.get(symbol) or holding.get("cooldown_until")
        drift_eur = abs(current - target) * total

        if tier == "FORTRESS":
            if (target - current) > FORTRESS_GAP_SUGGESTION:
                advice.append(_advice(
                    "change_savings_plan", symbol, name, None,
                    ui_copy.ADVICE_FROM_SAVINGS,
                    _fortress_leg(plans, symbol, name, current, target),
                    SPARPLAN_BUY_FEE_EUR, tier, "drift"))
            else:
                advice.append(_advice(
                    "keep", symbol, name, None, None,
                    "long-term holding; no action.", None, tier, "monitor"))
            continue

        if tier == "SPECULATIVE":
            advice.append(_advice(
                "keep", symbol, name, None, None,
                "small bet; stop-loss handled by alerts.", None, tier, "monitor"))
            continue

        # ALPHA: drift sell only when every law passes.
        if (current - target) > REBALANCE_DRIFT_THRESHOLD:
            if cooldown is not None and _as_date(cooldown) and _as_date(cooldown) > as_of:
                rejected.append({
                    "symbol": symbol, "considered_action": ui_copy.ADVICE_SELL_PART,
                    "plain_reason": ui_copy.REJECT_COOLDOWN.format(
                        date=ui_copy.fmt_date(cooldown)),
                })
            else:
                amount = sizing.can_sell(tier, value, drift_eur)
                if amount is None:
                    rejected.append({
                        "symbol": symbol, "considered_action": ui_copy.ADVICE_SELL_PART,
                        "plain_reason": _reject_reason(tier, value, drift_eur, cooldown, as_of),
                    })
                else:
                    advice.append(_advice(
                        "sell_part", symbol, name, amount,
                        ui_copy.ADVICE_FROM_POSITION,
                        f"it sits {abs(current - target) * 100:.0f} percent above its "
                        f"{target * 100:.0f} percent target.",
                        SPARPLAN_BUY_FEE_EUR, tier, "drift"))
        elif (target - current) > REBALANCE_DRIFT_THRESHOLD:
            buy_considered = True
            if conviction >= ALPHA_CONVICTION_HIGH:
                amount = sizing.round_buy(drift_eur)
                if amount >= 10:
                    advice.append(_advice(
                        "buy", symbol, name, amount, ui_copy.ADVICE_FROM_CASH,
                        f"high conviction; it sits {abs(target - current) * 100:.0f} "
                        f"percent below its {target * 100:.0f} percent target.",
                        ACTIVE_TRADE_FEE_EUR, tier, "drift"))
        else:
            advice.append(_advice(
                "keep", symbol, name, None, None,
                "within its target band.", None, tier, "monitor"))

    # 5. Cash line when the regime is bear and a buy was considered.
    if str(regime_label).lower() == "bear" and buy_considered:
        advice.append(_advice(
            "to_cash", None, "cash", None, ui_copy.ADVICE_FROM_CASH,
            ui_copy.CASH_REGIME_LINE, 0.0, "ALPHA", "monitor"))

    return advice, rejected

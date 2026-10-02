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
    ALPHA_CONVICTION_MEDIUM,
    BETS_MAX,
    REBALANCE_DRIFT_THRESHOLD,
    SPARPLAN_BUY_FEE_EUR,
    SPECULATIVE_TAKE_PROFIT,
    TARGET_WEIGHTS_INVESTED,
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


def _spec_change(holding: dict) -> float | None:
    """The price change fraction for a SPECULATIVE holding, or None (R5).

    Accepts either an explicit ``pnl_pct`` fraction or ``entry_price`` plus
    ``current_price``. Used only for the take-profit advisory.
    """
    if holding.get("pnl_pct") is not None:
        try:
            return float(holding.get("pnl_pct"))
        except (TypeError, ValueError):
            return None
    if holding.get("current_price") is None:
        return None
    entry = _num(holding.get("entry_price"))
    price = _num(holding.get("current_price"))
    if entry > 0:
        return price / entry - 1.0
    return None


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
    """The savings-plan top-up sentence for a FORTRESS gap (v10.7.3, Part 4.2).

    The SAME sentence is rendered by quant run and the Overview steps, so the
    two surfaces can never disagree. The label uses the "Name (TICKER)" form so
    the ticker is never doubled.
    """
    from quant.ui.search import label_for

    return ui_copy.STEP_TOP_UP.format(
        label=label_for(name, symbol),
        pct=f"{current * 100:.0f}", target=f"{target * 100:.0f}")


def _fortress_over_leg(symbol, name, current, target) -> str:
    """The plan-change sentence for a FORTRESS holding far OVER target (R2).

    A FORTRESS holding is never sold; when it drifts far above target the only
    honest lever is the savings-plan leg. The wording never implies a sale.
    """
    from quant.ui.search import label_for

    return ui_copy.FORTRESS_OVER_LEG.format(
        label=label_for(name, symbol),
        pct=f"{current * 100:.0f}", target=f"{target * 100:.0f}")


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
    system_lockdown: bool = False,
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
    invested = sum(_num(h.get("value_eur")) for h in holdings)
    spec_value = sum(
        _num(h.get("value_eur")) for h in holdings
        if (tmap.get(str(h.get("symbol", "")).strip())
            or str(h.get("tier", "")).upper()) == "SPECULATIVE")
    regime_label = regime.get("label") if isinstance(regime, dict) else regime
    is_bear = str(regime_label).lower() == "bear"
    buy_considered = False

    # 1. Open alerts first (alert sells ignore cooldowns, but FORTRESS never
    #    sells: a structural break on a long-term holding is a signal, not a
    #    sale).
    for alert in open_alerts or []:
        symbol = alert.get("symbol")
        holding = by_symbol.get(str(symbol), {})
        name = holding.get("name") or alert.get("name") or symbol or "Holding"
        action = str(alert.get("action", "review"))
        alert_tier = tmap.get(str(symbol)) or str(holding.get("tier", "")).upper()
        if action == "sell" and alert_tier != "FORTRESS":
            advice.append(_advice(
                "sell_part", symbol, name, alert.get("amount_eur"),
                ui_copy.ADVICE_FROM_POSITION, alert.get("message", ""),
                alert.get("fee_eur"), alert_tier, "alert"))
        else:
            advice.append(_advice(
                "keep", symbol, name, None, None, alert.get("message", ""),
                None, alert_tier, "alert"))

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
        if holding.get("target_weight") is None:
            # Part 4.1: TARGET_WEIGHTS_INVESTED is the ONE per-symbol target map.
            target = TARGET_WEIGHTS_INVESTED.get(symbol, 0.0)
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
            elif (current - target) > FORTRESS_GAP_SUGGESTION:
                # R2: far over target -> change the plan, never sell.
                advice.append(_advice(
                    "change_savings_plan", symbol, name, None,
                    ui_copy.ADVICE_FROM_SAVINGS,
                    _fortress_over_leg(symbol, name, current, target),
                    SPARPLAN_BUY_FEE_EUR, tier, "drift"))
            else:
                advice.append(_advice(
                    "keep", symbol, name, None, None,
                    "long-term holding; no action.", None, tier, "monitor"))
            continue

        if tier == "SPECULATIVE":
            # R5: advisories are notes, never forced sells. The stop-loss itself
            # is an alert (alerts.evaluate_alerts), not drift advice.
            why = "small bet; stop-loss handled by alerts."
            if invested > 0 and (spec_value / invested) > BETS_MAX:
                why = ui_copy.SPEC_CAP_VIOLATION.format(
                    pct=f"{spec_value / invested * 100:.0f}")
            else:
                change = _spec_change(holding)
                if (change is not None and change >= SPECULATIVE_TAKE_PROFIT
                        and conviction >= ALPHA_CONVICTION_HIGH):
                    why = ui_copy.SPEC_TAKE_PROFIT
            advice.append(_advice(
                "keep", symbol, name, None, None, why, None, tier, "monitor"))
            continue

        # ALPHA: drift sell only when every law passes.
        if (current - target) > REBALANCE_DRIFT_THRESHOLD:
            if system_lockdown:
                # Part 3.1: a system-wide lockdown pauses non-emergency sells.
                # Alert sells (emergency) are unaffected.
                rejected.append({
                    "symbol": symbol, "considered_action": ui_copy.ADVICE_SELL_PART,
                    "plain_reason": ui_copy.REJECT_LOCKDOWN,
                })
            elif cooldown is not None and _as_date(cooldown) and _as_date(cooldown) > as_of:
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
            if conviction >= ALPHA_CONVICTION_HIGH:
                buy_considered = True
                if is_bear:
                    # R9: the regime overrides micro signals. No ALPHA buy in
                    # bear; a HIGH-conviction candidate becomes a rejected note.
                    rejected.append({
                        "symbol": symbol, "considered_action": ui_copy.ADVICE_BUY,
                        "plain_reason": ui_copy.REJECT_BEAR_REGIME.format(name=name),
                    })
                elif sizing.is_untouchable(value):
                    # R4: the untouchable law applies to ACTIVE buys too.
                    rejected.append({
                        "symbol": symbol, "considered_action": ui_copy.ADVICE_BUY,
                        "plain_reason": ui_copy.REJECT_BUY_TOO_SMALL,
                    })
                else:
                    amount = sizing.round_buy(drift_eur)
                    if amount < 10:
                        rejected.append({
                            "symbol": symbol, "considered_action": ui_copy.ADVICE_BUY,
                            "plain_reason": ui_copy.REJECT_BUY_BELOW_MIN,
                        })
                    elif not sizing.passes_fee_hurdle(
                            holding.get("expected_alpha_bps"), amount,
                            ACTIVE_TRADE_FEE_EUR):
                        rejected.append({
                            "symbol": symbol, "considered_action": ui_copy.ADVICE_BUY,
                            "plain_reason": ui_copy.REJECT_FEE_HURDLE.format(
                                alpha=f"{_num(holding.get('expected_alpha_bps')):.0f}",
                                amount=f"{amount:.0f}",
                                fee=f"{ACTIVE_TRADE_FEE_EUR:.0f}"),
                        })
                    else:
                        advice.append(_advice(
                            "buy", symbol, name, amount, ui_copy.ADVICE_FROM_CASH,
                            f"high conviction; it sits {abs(target - current) * 100:.0f} "
                            f"percent below its {target * 100:.0f} percent target.",
                            ACTIVE_TRADE_FEE_EUR, tier, "drift"))
            elif conviction >= ALPHA_CONVICTION_MEDIUM:
                # R3: a considered buy with MEDIUM conviction is never silent.
                advice.append(_advice(
                    "keep", symbol, name, None, None,
                    "below target; waiting for high conviction.", None, tier, "monitor"))
                rejected.append({
                    "symbol": symbol, "considered_action": ui_copy.ADVICE_BUY,
                    "plain_reason": ui_copy.REJECT_MEDIUM_CONVICTION,
                })
            # LOW conviction: no advice generated (Part 2.2).
        else:
            advice.append(_advice(
                "keep", symbol, name, None, None,
                "within its target band.", None, tier, "monitor"))

    # 5. Cash line when the regime is bear and a buy was considered.
    if is_bear and buy_considered:
        advice.append(_advice(
            "to_cash", None, "cash", None, ui_copy.ADVICE_FROM_CASH,
            ui_copy.CASH_REGIME_LINE, 0.0, "ALPHA", "monitor"))

    return advice, rejected

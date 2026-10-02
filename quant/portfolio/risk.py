"""
risk.py — Asymmetric downside risk quantification.
Replaces linear correlation with empirical tail risk (VaR) and lower partial moments (Sortino).

Phase 4 (2.1): The risk-free rate is now the Trade Republic cash APY (2.5% from
16 Sep 2026; 2.25% before), converted to a daily yield. This is the REAL
opportunity cost of capital, not the theoretical US Treasury yield.
v10.5.1: the rate comes from quant/portfolio/cash_rate.py (dated schedule).
"""
import math
import warnings
from typing import TypedDict

import numpy as np
import pandas as pd

from quant.config import (
    LIQUIDITY_LOW_VOLUME_EUR,
    LIQUIDITY_MED_VOLUME_EUR,
    LIQUIDITY_SPREAD_WEIGHT,
    LIQUIDITY_VOLUME_REF_EUR,
    LIQUIDITY_VOLUME_WEIGHT,
    WEEKLY_VAR_HORIZON_DAYS,
)
from quant.portfolio.cash_rate import current_cash_apy

warnings.filterwarnings("ignore", category=RuntimeWarning)


class EmergencySellRecommendation(TypedDict):
    """One position in an emergency sell plan."""

    symbol: str
    tier: str
    value_eur: float
    liquidity_score: float
    pnl_eur: float
    priority: str
    warning: str | None


class EmergencySellPlan(TypedDict):
    """Result of ``emergency_sell_plan``."""

    recommendations: list[EmergencySellRecommendation]
    total_available: float
    fortress_warning: str | None
    shortfall: float


def daily_risk_free_rate(apy: float | None = None) -> float:
    """Convert broker cash APY to a daily risk-free rate.

    Intent: the daily hurdle an active trade must beat after fees & volatility.
    Formula: daily_rf = (1 + APY)^(1/365) - 1.
    Invariants: returns float in [0, 1); pure function (no I/O).
    """
    if apy is None:
        apy = current_cash_apy()
    if apy <= 0:
        return 0.0
    return float((1.0 + apy) ** (1.0 / 365.0) - 1.0)

def calculate_historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Эмпирический Value at Risk (VaR).
    Определяет максимальный ожидаемый дневной убыток на заданном интервале уверенности.
    """
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0

    percentile = (1.0 - confidence_level) * 100
    var = np.percentile(clean_returns, percentile)
    return float(var)

def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float | None = None,
    target_return: float = 0.0
) -> float:
    """
    Sortino ratio — downside-deviation-adjusted return.

    Phase 4 (2.1): risk_free_rate defaults to the Trade Republic daily cash
    yield (2.25% APY). An active trade must beat this hurdle after fees.
    """
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0

    if risk_free_rate is None:
        risk_free_rate = daily_risk_free_rate()

    # Isolate capital destruction (returns below target).
    downside_returns = clean_returns[clean_returns < target_return]

    if downside_returns.empty:
        return float('inf')

    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

    if downside_deviation == 0:
        return 0.0

    expected_return = clean_returns.mean()
    sortino = (expected_return - risk_free_rate) / downside_deviation

    return float(sortino)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe ratio — total-volatility-adjusted return.

    Phase 4 (2.1): uses the broker cash daily yield as the risk-free rate.
    Annualized by sqrt(periods_per_year).
    """
    clean_returns = returns.dropna()
    if clean_returns.empty or clean_returns.std() == 0:
        return 0.0

    if risk_free_rate is None:
        risk_free_rate = daily_risk_free_rate()

    excess = clean_returns.mean() - risk_free_rate
    return float(excess / clean_returns.std() * np.sqrt(periods_per_year))

def calculate_risk_penalty(returns: pd.Series, var_threshold: float = -0.05) -> float:
    """
    Вычисление штрафных баллов на базе экстремальных отклонений.
    Заменяет штраф за корреляцию Пирсона в композитном скоринге.
    """
    if returns.empty:
        return 0.0

    var_95 = calculate_historical_var(returns, 0.95)
    penalty = 0.0

    # var_95 выражен отрицательным числом. Штраф начисляется, если риск превышает порог.
    if var_95 < var_threshold:
        excess_risk = abs(var_95 - var_threshold)
        penalty = (excess_risk / abs(var_threshold)) * 15.0

    return min(penalty, 25.0)


# ── v10.6.2: Weekly risk (5-day VaR) ─────────────────────────────────────────

def weekly_var_95(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Scale the daily VaR to a 5-day horizon.

    Intent (v10.6.2): Alpha rebalances weekly, so the risk number that matters
    is the 5-day VaR, not the daily one. Under the square-root-of-time rule:
        VaR_weekly = VaR_daily * sqrt(WEEKLY_VAR_HORIZON_DAYS)
    Invariants: returns a float (negative for a loss); 0.0 on empty input; pure.
    """
    daily_var = calculate_historical_var(returns, confidence_level)
    return float(daily_var * math.sqrt(WEEKLY_VAR_HORIZON_DAYS))


# ── v10.6.2: Emergency liquidity scoring ─────────────────────────────────────

def estimate_spread_bps(price_hist: pd.DataFrame) -> float:
    """Heuristic bid-ask spread in basis points from the daily range.

    Intent (v10.6.2): no live order book is available, so the average true
    range over the last 14 sessions is used as a proxy. A wider range implies a
    wider spread. Invariants: returns a float in [0, 100]; 50.0 when the range
    cannot be computed; pure.
    """
    if price_hist is None or price_hist.empty:
        return 50.0
    if not {"High", "Low", "Close"}.issubset(price_hist.columns):
        return 50.0
    try:
        high = price_hist["High"].dropna()
        low = price_hist["Low"].dropna()
        close = price_hist["Close"].dropna()
        if high.empty or low.empty or close.empty:
            return 50.0
        tr = (high - low).tail(14).mean()
        price = float(close.iloc[-1])
        if price <= 0 or pd.isna(tr):
            return 50.0
        # The spread is a small fraction of the daily range; scale to bps.
        return float(max(0.0, min(100.0, (float(tr) / price) * 10000.0 * 0.1)))
    except Exception:  # noqa: BLE001
        return 50.0


def calculate_liquidity_score(symbol: str, price_hist: pd.DataFrame) -> float:
    """Score how quickly and cheaply an asset can be sold (0-100).

    Intent (v10.6.2): the emergency-liquidity ranking. Higher is easier to
    sell. Factors: average daily EUR volume (40 percent), spread (40 percent),
    and a time-to-execute penalty for thin names. Formula:
        volume_score = min(100, avg_daily_volume_eur / 100k * 100)
        spread_score = max(0, 100 - spread_bps)
        liquidity = 0.4*volume_score + 0.4*spread_score - time_penalty
    Invariants: returns a float in [0, 100]; 0.0 when no volume/close data; pure.
    """
    if price_hist is None or price_hist.empty:
        return 0.0
    if not {"Volume", "Close"}.issubset(price_hist.columns):
        return 0.0
    try:
        volume = price_hist["Volume"].dropna()
        close = price_hist["Close"].dropna()
        if volume.empty or close.empty:
            return 0.0
        # Per the v10.6.2 spec: EUR volume = shares * mean close, averaged.
        avg_daily_volume_eur = float((volume * close.mean()).tail(20).mean())
        if pd.isna(avg_daily_volume_eur) or avg_daily_volume_eur <= 0:
            return 0.0
        volume_score = min(100.0, avg_daily_volume_eur / LIQUIDITY_VOLUME_REF_EUR * 100.0)
        spread_bps = estimate_spread_bps(price_hist)
        spread_score = max(0.0, 100.0 - spread_bps)
        if avg_daily_volume_eur < LIQUIDITY_LOW_VOLUME_EUR:
            time_penalty = 30.0
        elif avg_daily_volume_eur < LIQUIDITY_MED_VOLUME_EUR:
            time_penalty = 15.0
        else:
            time_penalty = 0.0
        score = (
            volume_score * LIQUIDITY_VOLUME_WEIGHT
            + spread_score * LIQUIDITY_SPREAD_WEIGHT
            - time_penalty
        )
        return float(max(0.0, min(100.0, score)))
    except Exception:  # noqa: BLE001
        return 0.0


def prioritize_sells(holdings: list[dict], amount_needed: float) -> list[dict]:
    """Order holdings for an emergency sale.

    Intent (v10.6.2): sell the most liquid first, and within that prefer
    losers (tax-loss harvesting). Sort key: (liquidity_score, -pnl_eur) with
    liquidity high first and pnl low first. Accumulate until the amount needed
    is covered. Invariants: returns a list of the input dicts; never raises;
    empty list when nothing is needed. Pure.
    """
    if not holdings or amount_needed <= 0:
        return []

    def _key(h: dict) -> tuple[float, float]:
        liq = float(h.get("liquidity_score", 0) or 0)
        pnl = float(h.get("pnl_eur", 0) or 0)
        return (liq, -pnl)

    ordered = sorted(holdings, key=_key, reverse=True)
    recommendations: list[dict] = []
    cumulative = 0.0
    for h in ordered:
        if cumulative >= amount_needed:
            break
        recommendations.append(h)
        cumulative += float(h.get("current_value_eur", 0) or 0)
    return recommendations


def emergency_sell_recommendation(
    amount_needed: float,
    alpha_holdings: list[dict],
    tax_rate: float = 0.26375,
) -> list[dict]:
    """Tax-aware emergency sell order for Alpha holdings.

    Intent (v10.6.2): given a cash need, return the ordered sell list with the
    estimated tax effect per position. A gain carries tax; a loss is a tax
    saving (tax-loss harvest). Invariants: returns a list of dicts with
    ``tax_eur`` and ``tax_note`` added; never raises. Pure.
    """
    recommendations = prioritize_sells(alpha_holdings, amount_needed)
    out: list[dict] = []
    for h in recommendations:
        pnl = float(h.get("pnl_eur", 0) or 0)
        tax_eur = pnl * tax_rate
        if pnl < 0:
            note = "loss, tax-loss harvest"
        elif pnl > 0:
            note = "profit, taxable"
        else:
            note = "no gain or loss"
        out.append({**h, "tax_eur": round(tax_eur, 2), "tax_note": note})
    return out


# ── v10.6.3: Tier-aware emergency sell plan ───────────────────────────────────

def emergency_sell_plan(
    amount_needed: float,
    portfolio_df: pd.DataFrame | None,
    tiers_df: pd.DataFrame | None = None,
) -> EmergencySellPlan:
    """Tier-aware emergency sell plan.

    Intent (v10.6.3): sell ALPHA first (liquid reserve), then SPECULATIVE, and
    only as a last resort FORTRESS (with a capital gains tax warning). Returns
    ``{recommendations, total_available, fortress_warning, shortfall}``.
    Invariants: never raises; an empty portfolio yields an empty plan with the
    full shortfall; pure (no I/O).
    """
    amount_needed = float(amount_needed or 0)
    empty = {
        "recommendations": [], "total_available": 0.0,
        "fortress_warning": None, "shortfall": amount_needed,
    }
    if portfolio_df is None or portfolio_df.empty:
        return empty
    if "Current_Value_EUR" not in portfolio_df.columns:
        return empty

    df = portfolio_df.copy()
    if "Tier" not in df.columns:
        from quant.portfolio.tier_manager import tier_map
        tmap = tier_map(tiers_df)
        df["Tier"] = df["Symbol"].map(lambda s: tmap.get(str(s), "ALPHA"))

    def _rows(tier: str):
        sub = df[df["Tier"] == tier].copy()
        # R6: liquidity first, then losers (negative broker PnL) before winners
        # at equal liquidity. Stable and deterministic.
        liq = (pd.to_numeric(sub["Liquidity_Score"], errors="coerce").fillna(0.0)
               if "Liquidity_Score" in sub.columns else 0.0)
        pnl = (pd.to_numeric(sub["Broker_PnL_EUR"], errors="coerce").fillna(0.0)
               if "Broker_PnL_EUR" in sub.columns else 0.0)
        sub = sub.assign(_liq=liq, _pnl=pnl)
        sub = sub.sort_values(["_liq", "_pnl"], ascending=[False, True],
                              kind="mergesort")
        return sub

    recommendations: list[dict] = []
    total_available = 0.0
    fortress_warning = None

    for tier, priority in (("ALPHA", "HIGH"), ("SPECULATIVE", "LOW")):
        for _, row in _rows(tier).iterrows():
            if total_available >= amount_needed:
                break
            value = float(row.get("Current_Value_EUR", 0) or 0)
            recommendations.append({
                "symbol": str(row.get("Symbol", "")),
                "tier": tier,
                "value_eur": value,
                "liquidity_score": float(row.get("Liquidity_Score", 0) or 0),
                "pnl_eur": float(row.get("Broker_PnL_EUR", 0) or 0),
                "priority": priority,
            })
            total_available += value

    shortfall = max(0.0, amount_needed - total_available)
    if shortfall > 0:
        fortress = _rows("FORTRESS")
        if not fortress.empty:
            fortress_warning = (
                f"ALPHA and SPECULATIVE assets only provide {total_available:.2f} EUR. "
                f"To reach {amount_needed:.2f} EUR you must sell FORTRESS assets "
                f"(expected capital gains tax about 25 percent). "
                f"Consider if this is truly an emergency."
            )
            for _, row in fortress.iterrows():
                if total_available >= amount_needed:
                    break
                value = float(row.get("Current_Value_EUR", 0) or 0)
                recommendations.append({
                    "symbol": str(row.get("Symbol", "")),
                    "tier": "FORTRESS",
                    "value_eur": value,
                    "liquidity_score": 10.0,
                    "pnl_eur": float(row.get("Broker_PnL_EUR", 0) or 0),
                    "priority": "LAST_RESORT",
                    "warning": "FORTRESS asset: selling triggers capital gains tax.",
                })
                total_available += value

    return {
        "recommendations": recommendations,
        "total_available": total_available,
        "fortress_warning": fortress_warning,
        "shortfall": max(0.0, amount_needed - total_available),
    }


def prioritize_sells_with_tax(
    recommendations: list[dict],
    tax_rate: float = 0.25,
    freistellungsauftrag: float = 1000.0,
) -> list[dict]:
    """Reorder sell recommendations to minimize taxes.

    Intent (v10.6.3): sell losers first (tax-loss harvesting), then winners up
    to the Freistellungsauftrag (tax-free allowance), then the remaining
    winners. Adds ``tax_impact`` and ``tax_note`` to each recommendation.
    Invariants: returns a new list; never raises; pure.
    """
    def _pnl(r: dict) -> float:
        return float(r.get("pnl_eur", 0) or 0)

    def _liq(r: dict) -> float:
        return float(r.get("liquidity_score", 0) or 0)

    losers = sorted([r for r in recommendations if _pnl(r) < 0], key=_liq, reverse=True)
    winners = sorted([r for r in recommendations if _pnl(r) >= 0], key=_liq, reverse=True)

    remaining = float(freistellungsauftrag)
    out: list[dict] = []
    for rec in losers:
        r = dict(rec)
        r["tax_impact"] = round(_pnl(r) * tax_rate, 2)
        r["tax_note"] = "Losses you can use to lower tax: offsets gains"
        out.append(r)
    for rec in winners:
        r = dict(rec)
        pnl = _pnl(r)
        if remaining > 0:
            taxable = min(pnl, remaining)
            tax_free = pnl - taxable
            r["tax_impact"] = round(taxable * tax_rate, 2)
            r["tax_note"] = f"{tax_free:.2f} EUR tax-free (Freistellungsauftrag)"
            remaining -= taxable
        else:
            r["tax_impact"] = round(pnl * tax_rate, 2)
            r["tax_note"] = "Fully taxable"
        out.append(r)
    return out

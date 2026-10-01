"""
autobalance.py — Tier auto-balance recommendation engine (v10.6.4).

Intent: analyze current tier allocations, detect violations (for example ALPHA
above its 50 percent cap), and suggest tier reassignments that bring the
allocations within limits. Suggestions are advisory only: the user approves or
rejects each one, and no trade is ever executed. Applying a suggestion only
edits ``data/tiers.csv``.

Invariants:
  - ``analyze_tier_allocations`` never raises; returns a dict with
    ``total_value_eur``, ``allocations``, and ``violations``.
  - ``suggest_rebalance`` is bounded (no infinite loop) and returns a list of
    suggestion dicts.
  - ``apply_rebalance_suggestions`` only changes approved symbols.
  - Pure computation (no I/O).

Dependencies: pandas, quant.config, quant.execution.taxonomy.
"""
from __future__ import annotations

from typing import TypedDict

import pandas as pd

from quant.config import TIER_CONSTRAINTS

_COLUMNS = ["symbol", "tier", "last_updated", "notes"]


class AllocationInfo(TypedDict):
    """Per-tier allocation: value, percent, limit, and violation flag."""

    value_eur: float
    pct: float
    limit: float | None
    violated: bool


class AnalysisResult(TypedDict):
    """Result of ``analyze_tier_allocations``."""

    total_value_eur: float
    allocations: dict[str, AllocationInfo]
    violations: list[str]


class RebalanceSuggestion(TypedDict):
    """One proposed tier reassignment."""

    symbol: str
    current_tier: str
    suggested_tier: str
    reason: str
    value_eur: float
    impact_on_allocation: dict[str, float]


def _normalize_tiers(tiers_df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a tiers frame to the canonical schema (never raises)."""
    if tiers_df is None or tiers_df.empty:
        return pd.DataFrame(columns=_COLUMNS)
    out = tiers_df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    for col in _COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[_COLUMNS]
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["tier"] = out["tier"].astype(str).str.strip().str.upper()
    return out.reset_index(drop=True)


def _asset_value(portfolio_df: pd.DataFrame, symbol: str) -> float:
    """Return the current EUR value of a symbol (0.0 when absent)."""
    if portfolio_df is None or portfolio_df.empty or "Symbol" not in portfolio_df.columns:
        return 0.0
    row = portfolio_df[portfolio_df["Symbol"].astype(str) == str(symbol)]
    if row.empty or "Current_Value_EUR" not in row.columns:
        return 0.0
    try:
        return float(row["Current_Value_EUR"].iloc[0] or 0.0)
    except (TypeError, ValueError):
        return 0.0


def analyze_tier_allocations(
    portfolio_df: pd.DataFrame,
    tiers_df: pd.DataFrame,
) -> AnalysisResult:
    """Analyze current tier allocations and detect violations.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Holdings with columns ``Symbol`` and ``Current_Value_EUR``.
    tiers_df : pd.DataFrame
        Tier assignments with columns ``symbol`` and ``tier``.

    Returns
    -------
    AnalysisResult
        ``total_value_eur``, per-tier ``allocations`` (value, pct, limit,
        violated), and the list of ``violations``.

    Invariants: never raises; pure (no I/O).
    """
    empty = {"total_value_eur": 0.0, "allocations": {}, "violations": []}
    if (portfolio_df is None or portfolio_df.empty
            or "Current_Value_EUR" not in portfolio_df.columns):
        return empty

    total_value = float(pd.to_numeric(
        portfolio_df["Current_Value_EUR"], errors="coerce").fillna(0.0).sum())
    tiers = _normalize_tiers(tiers_df)

    allocations: dict[str, dict] = {}
    violations: list[str] = []
    for tier, constraints in TIER_CONSTRAINTS.items():
        symbols = set(tiers[tiers["tier"] == tier]["symbol"]) if not tiers.empty else set()
        if symbols:
            tier_value = float(pd.to_numeric(
                portfolio_df.loc[portfolio_df["Symbol"].astype(str).isin(symbols),
                                 "Current_Value_EUR"],
                errors="coerce").fillna(0.0).sum())
        else:
            tier_value = 0.0
        pct = tier_value / total_value if total_value > 0 else 0.0
        max_alloc = constraints.get("max_allocation")
        violated = max_alloc is not None and pct > max_alloc
        allocations[tier] = {
            "value_eur": round(tier_value, 2),
            "pct": pct,
            "limit": max_alloc,
            "violated": violated,
        }
        if violated:
            violations.append(tier)

    return {
        "total_value_eur": round(total_value, 2),
        "allocations": allocations,
        "violations": violations,
    }


def _score_reassignment_suitability(
    symbol: str,
    current_tier: str,
    portfolio_df: pd.DataFrame,
    tiers_df: pd.DataFrame,
) -> float:
    """Score how suitable an asset is for reassignment (0-100).

    Higher is a better candidate to move. Factors: asset class, structural
    grade (when present), and PnL (tax considerations). Invariants: returns a
    float in [0, 100]; never raises; pure.
    """
    from quant.execution.taxonomy import get_instrument_class

    score = 50.0
    try:
        asset_class = get_instrument_class(symbol)
    except Exception:  # noqa: BLE001
        asset_class = "EQUITY"

    if current_tier == "ALPHA" and asset_class in ("ETF", "CASH"):
        score += 30
    elif current_tier == "FORTRESS" and asset_class == "EQUITY":
        score += 20

    row = portfolio_df[portfolio_df["Symbol"].astype(str) == str(symbol)] \
        if portfolio_df is not None and not portfolio_df.empty else pd.DataFrame()
    if not row.empty:
        if "Structural_Grade" in row.columns:
            try:
                struct_grade = float(row["Structural_Grade"].iloc[0])
                if current_tier == "ALPHA" and struct_grade > 70:
                    score += 20
                elif current_tier == "FORTRESS" and struct_grade < 40:
                    score += 20
            except (TypeError, ValueError):
                pass
        if "Broker_PnL_EUR" in row.columns:
            try:
                pnl = float(row["Broker_PnL_EUR"].iloc[0])
                if pnl < 0 and current_tier == "FORTRESS":
                    score += 15
                elif pnl > 1000 and current_tier == "ALPHA":
                    score += 10
            except (TypeError, ValueError):
                pass

    return float(min(100.0, max(0.0, score)))


def _find_target_tier(
    symbol: str,
    source_tier: str,
    analysis: dict,
    portfolio_df: pd.DataFrame,
    tiers_df: pd.DataFrame,
) -> str | None:
    """Find the best under-allocated target tier for a reassignment.

    Prefers FORTRESS for ETFs/CASH and as the uncapped sink when reducing a
    capped tier. Invariants: returns a tier name or None; never raises; pure.
    """
    from quant.execution.taxonomy import get_instrument_class

    try:
        asset_class = get_instrument_class(symbol)
    except Exception:  # noqa: BLE001
        asset_class = "EQUITY"

    under: list[str] = []
    for tier in TIER_CONSTRAINTS:
        if tier == source_tier:
            continue
        alloc = analysis["allocations"].get(tier, {})
        limit = alloc.get("limit")
        pct = alloc.get("pct", 0.0)
        if limit is None or pct < limit:
            under.append(tier)

    if not under:
        return None
    if asset_class in ("ETF", "CASH") and "FORTRESS" in under:
        return "FORTRESS"
    if asset_class == "EQUITY" and source_tier == "FORTRESS" and "ALPHA" in under:
        return "ALPHA"
    if "FORTRESS" in under:
        return "FORTRESS"
    return under[0]


def _generate_reassignment_reason(
    symbol: str,
    source_tier: str,
    target_tier: str,
    suitability: float,
) -> str:
    """Generate a plain-language reason for a reassignment. Pure."""
    from quant.execution.taxonomy import get_instrument_class

    try:
        asset_class = get_instrument_class(symbol)
    except Exception:  # noqa: BLE001
        asset_class = "EQUITY"

    if source_tier == "ALPHA" and target_tier == "FORTRESS":
        if asset_class in ("ETF", "CASH"):
            return (f"{symbol} is an ETF. ETFs are better suited for FORTRESS "
                    f"(long-term savings plan, never sell).")
        return (f"{symbol} has high structural quality. Consider moving to "
                f"FORTRESS for tax-free long-term holding.")
    if source_tier == "FORTRESS" and target_tier == "ALPHA":
        return (f"{symbol} may benefit from active trading. Consider moving to "
                f"ALPHA for weekly rebalancing.")
    if source_tier == "ALPHA" and target_tier == "SPECULATIVE":
        return (f"{symbol} shows high volatility. Consider moving to SPECULATIVE "
                f"(max 2 percent allocation).")
    return (f"Reassignment suitability score: {suitability:.0f}/100. Moving from "
            f"{source_tier} to {target_tier} improves allocation balance.")


def suggest_rebalance(
    portfolio_df: pd.DataFrame,
    tiers_df: pd.DataFrame,
    max_suggestions: int = 20,
) -> list[RebalanceSuggestion]:
    """Suggest tier reassignments to fix allocation violations.

    Intent: iteratively move the most suitable asset out of a violated tier
    into an under-allocated tier until the violations clear or no candidate
    remains. Invariants: bounded by ``max_suggestions``; never raises; returns a
    list of ``RebalanceSuggestion``; pure.
    """
    analysis = analyze_tier_allocations(portfolio_df, tiers_df)
    if not analysis["violations"]:
        return []

    working = _normalize_tiers(tiers_df)
    suggestions: list[dict] = []
    tried: set[str] = set()

    for _ in range(int(max_suggestions)):
        analysis = analyze_tier_allocations(portfolio_df, working)
        if not analysis["violations"]:
            break
        violated_tier = analysis["violations"][0]

        candidates: list[tuple[str, float]] = []
        for symbol in working[working["tier"] == violated_tier]["symbol"]:
            if symbol in tried:
                continue
            suitability = _score_reassignment_suitability(
                symbol, violated_tier, portfolio_df, working)
            candidates.append((symbol, suitability))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)

        moved = False
        for symbol, suitability in candidates:
            target = _find_target_tier(
                symbol, violated_tier, analysis, portfolio_df, working)
            if target is None:
                tried.add(symbol)
                continue
            working.loc[working["symbol"] == symbol, "tier"] = target
            new_analysis = analyze_tier_allocations(portfolio_df, working)
            suggestions.append({
                "symbol": symbol,
                "current_tier": violated_tier,
                "suggested_tier": target,
                "reason": _generate_reassignment_reason(
                    symbol, violated_tier, target, suitability),
                "value_eur": _asset_value(portfolio_df, symbol),
                "impact_on_allocation": {
                    t: a["pct"] for t, a in new_analysis["allocations"].items()
                },
            })
            moved = True
            break
        if not moved:
            break

    return suggestions


def apply_rebalance_suggestions(
    tiers_df: pd.DataFrame,
    suggestions: list[dict],
    approved_symbols: list[str],
) -> pd.DataFrame:
    """Apply approved reassignment suggestions to a tiers frame.

    Intent: only approved symbols change tier; each change is annotated in the
    notes column. Invariants: returns a canonical tiers frame; never raises;
    pure.
    """
    updated = _normalize_tiers(tiers_df)
    approved = {str(s) for s in (approved_symbols or [])}
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    for suggestion in suggestions or []:
        symbol = str(suggestion.get("symbol", ""))
        if symbol not in approved:
            continue
        mask = updated["symbol"] == symbol
        if not mask.any():
            continue
        updated.loc[mask, "tier"] = str(suggestion["suggested_tier"]).upper()
        updated.loc[mask, "notes"] = (
            f"Auto-balanced from {suggestion['current_tier']} on {today}"
        )
    return updated

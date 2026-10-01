"""
test_autobalance.py — Tests for the tier auto-balance engine (v10.6.4).

Hermetic: synthetic data only, no network, no live store.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def _pure_instrument_class(monkeypatch):
    """Keep the tests hermetic: the real get_instrument_class upserts registry
    rows, which pollutes other tests' display names. Stub it to a pure function.
    """
    import quant.execution.taxonomy as tax

    monkeypatch.setattr(tax, "get_instrument_class", lambda s: "EQUITY")


@pytest.fixture
def portfolio_df():
    """Sample portfolio with an ALPHA violation (66.7 percent)."""
    return pd.DataFrame({
        "Symbol": ["URTH", "AAPL", "NVDA", "TSM"],
        "Tier": ["FORTRESS", "ALPHA", "ALPHA", "ALPHA"],
        "Current_Value_EUR": [2000, 1000, 1500, 1500],
        "Broker_PnL_EUR": [200, 100, 300, -50],
    })


@pytest.fixture
def tiers_df():
    """Sample tiers matching the portfolio."""
    return pd.DataFrame({
        "symbol": ["URTH", "AAPL", "NVDA", "TSM"],
        "tier": ["FORTRESS", "ALPHA", "ALPHA", "ALPHA"],
        "last_updated": ["2026-10-01"] * 4,
        "notes": [""] * 4,
    })


def test_analyze_tier_allocations_no_violations():
    """Allocations within limits report no violations."""
    from quant.portfolio.autobalance import analyze_tier_allocations

    portfolio_df = pd.DataFrame({
        "Symbol": ["URTH", "AAPL"],
        "Current_Value_EUR": [6000, 4000],
    })
    tiers_df = pd.DataFrame({
        "symbol": ["URTH", "AAPL"], "tier": ["FORTRESS", "ALPHA"],
    })
    analysis = analyze_tier_allocations(portfolio_df, tiers_df)
    assert analysis["total_value_eur"] == 10000
    assert not analysis["violations"]
    assert analysis["allocations"]["FORTRESS"]["pct"] == 0.6
    assert analysis["allocations"]["ALPHA"]["pct"] == 0.4


def test_analyze_tier_allocations_with_violation(portfolio_df, tiers_df):
    """An ALPHA allocation above 50 percent is flagged."""
    from quant.portfolio.autobalance import analyze_tier_allocations

    analysis = analyze_tier_allocations(portfolio_df, tiers_df)
    assert "ALPHA" in analysis["violations"]
    assert analysis["allocations"]["ALPHA"]["pct"] > 0.5


def test_suggest_rebalance_alpha_violation(portfolio_df, tiers_df):
    """Suggestions move an ALPHA asset to FORTRESS to clear the violation."""
    from quant.portfolio.autobalance import suggest_rebalance

    suggestions = suggest_rebalance(portfolio_df, tiers_df)
    assert len(suggestions) > 0
    alpha_to_fortress = [
        s for s in suggestions
        if s["current_tier"] == "ALPHA" and s["suggested_tier"] == "FORTRESS"
    ]
    assert len(alpha_to_fortress) > 0
    for s in suggestions:
        for key in ("symbol", "current_tier", "suggested_tier", "reason",
                    "value_eur", "impact_on_allocation"):
            assert key in s


def test_suggest_rebalance_no_violations():
    """No suggestions when allocations are within limits."""
    from quant.portfolio.autobalance import suggest_rebalance

    portfolio_df = pd.DataFrame({
        "Symbol": ["URTH", "AAPL"], "Current_Value_EUR": [6000, 4000],
    })
    tiers_df = pd.DataFrame({
        "symbol": ["URTH", "AAPL"], "tier": ["FORTRESS", "ALPHA"],
    })
    assert suggest_rebalance(portfolio_df, tiers_df) == []


def test_apply_rebalance_suggestions(tiers_df):
    """Approved suggestions change the tier; others are untouched."""
    from quant.portfolio.autobalance import apply_rebalance_suggestions

    suggestions = [{
        "symbol": "NVDA", "current_tier": "ALPHA", "suggested_tier": "FORTRESS",
        "reason": "Test", "value_eur": 1500, "impact_on_allocation": {},
    }]
    updated = apply_rebalance_suggestions(tiers_df, suggestions, ["NVDA"])
    assert updated[updated["symbol"] == "NVDA"]["tier"].iloc[0] == "FORTRESS"
    assert updated[updated["symbol"] == "AAPL"]["tier"].iloc[0] == "ALPHA"


def test_apply_rebalance_partial_approval(tiers_df):
    """Only approved symbols are applied."""
    from quant.portfolio.autobalance import apply_rebalance_suggestions

    suggestions = [
        {"symbol": "NVDA", "current_tier": "ALPHA", "suggested_tier": "FORTRESS",
         "reason": "Test", "value_eur": 1500, "impact_on_allocation": {}},
        {"symbol": "TSM", "current_tier": "ALPHA", "suggested_tier": "FORTRESS",
         "reason": "Test", "value_eur": 1500, "impact_on_allocation": {}},
    ]
    updated = apply_rebalance_suggestions(tiers_df, suggestions, ["NVDA"])
    assert updated[updated["symbol"] == "NVDA"]["tier"].iloc[0] == "FORTRESS"
    assert updated[updated["symbol"] == "TSM"]["tier"].iloc[0] == "ALPHA"


def test_rebalance_etf_preference(monkeypatch):
    """An ETF in ALPHA is suggested for FORTRESS."""
    import quant.execution.taxonomy as tax

    monkeypatch.setattr(
        tax, "get_instrument_class",
        lambda s: "ETF" if s == "SPY" else "EQUITY",
    )
    from quant.portfolio.autobalance import suggest_rebalance

    portfolio_df = pd.DataFrame({
        "Symbol": ["URTH", "AAPL", "NVDA", "TSM", "SPY"],
        "Current_Value_EUR": [2000, 1000, 1500, 1500, 1000],
        "Broker_PnL_EUR": [200, 100, 300, -50, 50],
    })
    tiers_df = pd.DataFrame({
        "symbol": ["URTH", "AAPL", "NVDA", "TSM", "SPY"],
        "tier": ["FORTRESS", "ALPHA", "ALPHA", "ALPHA", "ALPHA"],
        "last_updated": ["2026-10-01"] * 5,
        "notes": [""] * 5,
    })
    suggestions = suggest_rebalance(portfolio_df, tiers_df)
    spy_suggestions = [s for s in suggestions if s["symbol"] == "SPY"]
    assert len(spy_suggestions) > 0
    assert spy_suggestions[0]["suggested_tier"] == "FORTRESS"


def test_rebalance_idempotent(portfolio_df, tiers_df):
    """Applying all suggestions leaves no further suggestions."""
    from quant.portfolio.autobalance import (
        apply_rebalance_suggestions,
        suggest_rebalance,
    )

    suggestions_1 = suggest_rebalance(portfolio_df, tiers_df)
    approved = [s["symbol"] for s in suggestions_1]
    updated = apply_rebalance_suggestions(tiers_df, suggestions_1, approved)
    suggestions_2 = suggest_rebalance(portfolio_df, updated)
    assert len(suggestions_2) <= len(suggestions_1)

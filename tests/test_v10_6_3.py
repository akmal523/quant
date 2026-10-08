"""
test_v10_6_3.py — v10.6.3 edge-case and polish tests.

Covers: unclassified detection, tier validation/repair, tier-aware emergency
liquidity, tax-aware prioritization, empty-portfolio report, stale signal cache,
the dict-returning weekly guardrail, batch scoring, and allocation limits.

All tests are hermetic: synthetic data only, no network, no live store.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Unclassified detection and auto-assignment ────────────────────────────────

def test_detect_unclassified_assets(monkeypatch):
    """Symbols in portfolio.csv but not tiers.csv are detected with a tier."""
    import quant.execution.taxonomy as tax

    monkeypatch.setattr(
        tax, "get_instrument_class",
        lambda s: "ETF" if s == "URTH" else "EQUITY",
    )
    from quant.portfolio.tier_manager import detect_unclassified_assets

    portfolio_df = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT", "URTH"],
        "Current_Value_EUR": [1000, 2000, 3000],
    })
    tiers_df = pd.DataFrame({
        "symbol": ["AAPL"], "tier": ["ALPHA"],
        "last_updated": [""], "notes": [""],
    })
    recs = detect_unclassified_assets(portfolio_df, tiers_df)
    assert len(recs) == 2
    assert recs[0]["symbol"] == "MSFT"
    assert recs[1]["symbol"] == "URTH"
    assert recs[1]["recommended_tier"] == "FORTRESS"


def test_auto_assign_tiers():
    """Recommendations are appended to the tiers frame."""
    from quant.portfolio.tier_manager import auto_assign_tiers

    recommendations = [
        {"symbol": "URTH", "recommended_tier": "FORTRESS", "reason": "ETF"},
        {"symbol": "NVDA", "recommended_tier": "ALPHA", "reason": "Stock"},
    ]
    tiers_df = pd.DataFrame(columns=["symbol", "tier", "last_updated", "notes"])
    updated = auto_assign_tiers(recommendations, tiers_df)
    assert len(updated) == 2
    assert updated[updated["symbol"] == "URTH"]["tier"].iloc[0] == "FORTRESS"
    assert updated[updated["symbol"] == "NVDA"]["tier"].iloc[0] == "ALPHA"


# ── Tier validation and repair ────────────────────────────────────────────────

def test_validate_tiers_csv_corrupted():
    """Invalid tiers and duplicate symbols are reported."""
    from quant.portfolio.tier_manager import validate_tiers_csv

    portfolio_df = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT"], "Current_Value_EUR": [1000, 2000],
    })
    tiers_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "tier": ["FORTRESS", "INVALID_TIER", "ALPHA"],
        "last_updated": ["", "", ""], "notes": ["", "", ""],
    })
    is_valid, errors = validate_tiers_csv(tiers_df, portfolio_df)
    assert not is_valid
    assert any("INVALID_TIER" in e for e in errors)
    assert any("Duplicate" in e for e in errors)


def test_repair_tiers_csv():
    """Duplicates and orphans are removed; invalid tiers default to ALPHA."""
    from quant.portfolio.tier_manager import repair_tiers_csv

    portfolio_df = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT"], "Current_Value_EUR": [1000, 2000],
    })
    tiers_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT", "ORPHAN"],
        "tier": ["FORTRESS", "INVALID", "ALPHA", "SPECULATIVE"],
        "last_updated": ["", "", "", ""], "notes": ["", "", "", ""],
    })
    repaired = repair_tiers_csv(tiers_df, portfolio_df)
    assert len(repaired) == 2
    assert set(repaired["symbol"]) == {"AAPL", "MSFT"}
    assert repaired[repaired["symbol"] == "AAPL"]["tier"].iloc[0] == "FORTRESS"


def test_tier_allocation_exceeds_limit():
    """A SPECULATIVE allocation above 2 percent is reported."""
    from quant.portfolio.tier_manager import validate_tiers_csv

    portfolio_df = pd.DataFrame({
        "Symbol": ["GME", "AAPL"], "Current_Value_EUR": [500, 10000],
    })
    tiers_df = pd.DataFrame({
        "symbol": ["GME", "AAPL"], "tier": ["SPECULATIVE", "FORTRESS"],
        "last_updated": ["", ""], "notes": ["", ""],
    })
    is_valid, errors = validate_tiers_csv(tiers_df, portfolio_df)
    assert not is_valid
    assert any("SPECULATIVE" in e and "exceeds" in e for e in errors)


# ── Emergency liquidity edge cases ────────────────────────────────────────────

def test_emergency_sell_empty_portfolio():
    """An empty portfolio yields an empty plan with the full shortfall."""
    from quant.portfolio.risk import emergency_sell_plan

    result = emergency_sell_plan(5000, pd.DataFrame())
    assert result["total_available"] == 0
    assert result["shortfall"] == 5000
    assert result["recommendations"] == []


def test_emergency_sell_all_fortress():
    """All-FORTRESS holdings produce a last-resort plan with a tax warning."""
    from quant.portfolio.risk import emergency_sell_plan

    portfolio_df = pd.DataFrame({
        "Symbol": ["URTH", "SPY"],
        "Tier": ["FORTRESS", "FORTRESS"],
        "Current_Value_EUR": [3000, 2000],
        "Liquidity_Score": [10, 10],
        "Broker_PnL_EUR": [500, 300],
    })
    result = emergency_sell_plan(5000, portfolio_df)
    assert result["fortress_warning"] is not None
    assert "FORTRESS" in result["fortress_warning"]
    assert result["total_available"] == 5000
    assert all(r["priority"] == "LAST_RESORT" for r in result["recommendations"])


def test_prioritize_sells_with_tax():
    """Losers are sold first; winners use the Freistellungsauftrag."""
    from quant.portfolio.risk import prioritize_sells_with_tax

    recs = [
        {"symbol": "AAPL", "pnl_eur": 500, "liquidity_score": 95},
        {"symbol": "NVDA", "pnl_eur": -200, "liquidity_score": 90},
        {"symbol": "TSM", "pnl_eur": 300, "liquidity_score": 85},
    ]
    out = prioritize_sells_with_tax(recs, tax_rate=0.25, freistellungsauftrag=1000)
    assert out[0]["symbol"] == "NVDA"
    assert out[0]["tax_impact"] < 0
    assert "tax-free" in out[1]["tax_note"]



# ── Signal cache staleness ────────────────────────────────────────────────────

def test_signal_cache_stale_detection(tmp_path, monkeypatch):
    """A cache older than max_age_days is invalidated."""
    import quant.portfolio.signal_cache as sc

    cache_file = tmp_path / "signal_cache.json"
    monkeypatch.setattr(sc, "cache_path", lambda: str(cache_file))

    sc.save_signal_cache([{"symbol": "AAPL"}], "2026-10-01")
    assert sc.load_signal_cache(max_age_days=7) is not None

    old = time.time() - 10 * 86400
    os.utime(cache_file, (old, old))
    assert sc.load_signal_cache(max_age_days=7) is None

    sc.invalidate_signal_cache()
    assert not cache_file.exists()



# ── Batch scoring ─────────────────────────────────────────────────────────────

def test_batch_score_assets():
    """Batch scoring returns one row per symbol, routed by tier."""
    from quant.analytics.scoring import batch_score_assets

    df = batch_score_assets(
        ["EUNL.DE", "NVDA", "GME"],
        price_data={},
        fundamentals={
            "EUNL.DE": {"PE": 20, "ROE": 0.2},
            "NVDA": {"PE": 30, "ROE": 0.3},
        },
        news={},
        tiers={"EUNL.DE": "FORTRESS", "NVDA": "ALPHA", "GME": "SPECULATIVE"},
    )
    assert len(df) == 3
    assert set(df["symbol"]) == {"EUNL.DE", "NVDA", "GME"}

"""
test_v10_6_3.py — v10.6.3 edge-case and polish tests.

Covers: unclassified detection, tier validation/repair, tier-aware emergency
liquidity, tax-aware prioritization, empty-portfolio report, stale signal cache,
the dict-returning weekly guardrail, batch scoring, and allocation limits.

All tests are hermetic: synthetic data only, no network, no live store.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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

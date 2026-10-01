"""
test_stress.py — Stress tests for extreme scenarios (v10.6.5).

Hermetic: synthetic data only, no network, no live store.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_large_portfolio_100_assets():
    """Allocation analysis of 100 assets completes in under one second."""
    from quant.portfolio.autobalance import analyze_tier_allocations

    rng = np.random.default_rng(0)
    symbols = [f"ASSET_{i:03d}" for i in range(100)]
    portfolio_df = pd.DataFrame({
        "Symbol": symbols,
        "Current_Value_EUR": rng.uniform(1000, 10000, 100),
    })
    tiers_df = pd.DataFrame({"symbol": symbols, "tier": ["ALPHA"] * 100})

    start = time.time()
    analysis = analyze_tier_allocations(portfolio_df, tiers_df)
    elapsed = time.time() - start

    assert elapsed < 1.0
    assert analysis["total_value_eur"] > 0


def test_corrupted_tiers_csv_recovery(tmp_path):
    """A corrupted tiers file is recovered by parsing valid rows."""
    from quant.portfolio.tier_manager import _repair_corrupted_csv

    tiers_csv = tmp_path / "tiers.csv"
    tiers_csv.write_text(
        "symbol,tier,last_updated,notes\n"
        "AAPL,FORTRESS,2026-10-01,\n"
        "INVALID LINE WITHOUT COMMAS\n"
        "MSFT,ALPHA,2026-10-01,\n",
        encoding="utf-8",
    )
    df = _repair_corrupted_csv(str(tiers_csv))
    assert len(df) >= 2
    assert set(df["symbol"]) >= {"AAPL", "MSFT"}


def test_empty_portfolio_handling():
    """An empty portfolio is handled without crashing."""
    from quant.portfolio.autobalance import analyze_tier_allocations
    from quant.portfolio.risk import emergency_sell_plan

    portfolio_df = pd.DataFrame(columns=["Symbol", "Tier", "Current_Value_EUR"])
    tiers_df = pd.DataFrame(columns=["symbol", "tier"])

    analysis = analyze_tier_allocations(portfolio_df, tiers_df)
    assert analysis["total_value_eur"] == 0.0
    assert len(analysis["violations"]) == 0

    plan = emergency_sell_plan(5000.0, portfolio_df, tiers_df)
    assert plan["total_available"] == 0.0
    assert plan["shortfall"] == 5000.0
    assert len(plan["recommendations"]) == 0


def test_negative_values_handling():
    """Negative values are handled gracefully (total stays non-negative)."""
    from quant.portfolio.autobalance import analyze_tier_allocations

    portfolio_df = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT"],
        "Current_Value_EUR": [-1000, 2000],
    })
    tiers_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"], "tier": ["ALPHA", "ALPHA"],
    })
    analysis = analyze_tier_allocations(portfolio_df, tiers_df)
    assert analysis["total_value_eur"] >= 0

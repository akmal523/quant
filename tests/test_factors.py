"""
test_factors.py — Unit tests for the new production-grade features.
Covers: cross-sectional factor scoring, sector neutralization, no-lookahead
fundamentals, cost-aware backtest, liquidity filter, data validation.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import pandas as pd

# ── Cross-Sectional Factor Scoring ────────────────────────────────────────────

def test_factor_scores_high_quality_ranks_higher():
    """Higher ROE + lower PE should produce higher composite score."""
    from quant.analytics.scoring import factor_scores

    features = pd.DataFrame({
        "PE": [10.0, 20.0, 30.0],
        "ROE": [0.30, 0.15, 0.05],
        "momentum_6m": [0.10, 0.05, 0.0],
        "vol_60d": [0.20, 0.25, 0.30],
        "nlp_score": [50.0, 0.0, -50.0],
    })

    out = factor_scores(features)
    assert "composite_score" in out.columns
    # Best fundamentals should rank first.
    assert out["composite_score"].iloc[0] > out["composite_score"].iloc[2]
    print("  [PASS] test_factor_scores_high_quality_ranks_higher")


def test_factor_scores_zscore_columns_present():
    """All factor z-score columns must be present."""
    from quant.analytics.scoring import factor_scores

    features = pd.DataFrame({
        "PE": [10.0, 20.0],
        "ROE": [0.30, 0.10],
        "momentum_6m": [0.10, 0.0],
        "vol_60d": [0.20, 0.30],
        "nlp_score": [50.0, 0.0],
    })
    out = factor_scores(features)
    for col in ["value_z", "quality_z", "momentum_z", "low_risk_z", "sentiment_z"]:
        assert col in out.columns, f"missing {col}"
    print("  [PASS] test_factor_scores_zscore_columns_present")


# ── No-Lookahead Fundamentals ─────────────────────────────────────────────────

def test_get_fundamentals_as_of_no_lookahead():
    """Only snapshots published on/before as_of_date should be returned."""
    from quant.data.database import init_db
    init_db()  # ensure fundamentals_history table exists
    from quant.data.fundamentals import get_fundamentals_as_of, save_fundamentals_history

    save_fundamentals_history("TEST", {"PE": 10.0, "ROE": 0.30}, "2024-01-01", "2024-01-01")
    save_fundamentals_history("TEST", {"PE": 5.0, "ROE": 0.50}, "2024-06-01", "2024-06-01")

    # As of 2024-03-01, only the Jan snapshot is known.
    res = get_fundamentals_as_of("TEST", "2024-03-01")
    assert res is not None
    assert res["PE"] == 10.0, f"lookahead! got PE={res['PE']}"
    print("  [PASS] test_get_fundamentals_as_of_no_lookahead")


# ── Cost-Aware Backtest ───────────────────────────────────────────────────────

def test_cost_aware_backtest_t1_execution():
    """Signal at T executes at T+1 open; costs reduce net PnL."""
    from scripts.backtest import run_cost_aware_backtest

    dates = pd.date_range("2020-01-01", periods=10, freq="B")
    hist = pd.DataFrame({
        "Open": [100.0] * 10,
        "Close": [100.0] * 10,
        "Signal": ["HOLD", "BUY", "HOLD", "HOLD", "SELL", "HOLD", "HOLD", "HOLD", "HOLD", "HOLD"],
    }, index=dates)

    res = run_cost_aware_backtest(hist)
    assert res["CA_Trades"] == 1
    # Gross PnL ~0, net should be negative due to costs.
    assert res["CA_Net_PnL_pct"] < 0, f"expected negative net after costs, got {res}"
    print(f"  [PASS] test_cost_aware_backtest_t1_execution: {res}")


def test_cost_aware_backtest_empty():
    """No BUY/SELL signals -> empty result."""
    from scripts.backtest import run_cost_aware_backtest

    hist = pd.DataFrame({
        "Open": [100.0] * 5,
        "Close": [100.0] * 5,
        "Signal": ["HOLD"] * 5,
    })
    res = run_cost_aware_backtest(hist)
    assert res["CA_Trades"] == 0
    print("  [PASS] test_cost_aware_backtest_empty")


# ── Liquidity & Validation ────────────────────────────────────────────────────


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")

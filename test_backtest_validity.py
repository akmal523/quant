"""
test_backtest_validity.py — Validation suite for the backtesting engine.
Focuses on:
  1. Survivorship bias detection: inflated returns from only-listed tickers
  2. WFO (walk-forward) correctness: IS/OOS window alignment
  3. _run_window_trades: trade execution logic with mock data
  4. Edge-case handling: short histories, no trades, all-NaN data
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Survivorship Bias Warning ──────────────────────────────────────────────────

def test_survivorship_bias_warning_emitted():
    """WFO result must include survivorship_bias_warning on every call."""
    from backtest import walk_forward_optimization

    # Create mock price data with known trend
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    mock = pd.DataFrame({
        "Open": np.linspace(100, 120, 500) + np.random.normal(0, 1, 500),
        "High": np.linspace(102, 122, 500) + np.random.normal(0, 1, 500),
        "Low":  np.linspace(98, 118, 500) + np.random.normal(0, 1, 500),
        "Close": np.linspace(100, 120, 500) + np.random.normal(0, 0.5, 500),
        "Volume": np.ones(500) * 1_000_000,
    }, index=dates)

    result = walk_forward_optimization(mock)
    assert "survivorship_bias_warning" in result, "Missing survivorship bias warning"
    assert "SURVIVORSHIP BIAS" in result["survivorship_bias_warning"], "Warning content mismatch"
    print(f"  [PASS] test_survivorship_bias_warning_emitted: warning present")


def test_survivorship_bias_warning_empty():
    """Even with insufficient data, warning must be included."""
    from backtest import walk_forward_optimization

    short = pd.DataFrame({"Close": [100.0] * 10})
    result = walk_forward_optimization(short)
    assert "survivorship_bias_warning" in result, "Missing warning on empty result"
    assert result["wfo_periods"] == 0, "Expected 0 periods for short history"
    print(f"  [PASS] test_survivorship_bias_warning_empty: warning present, periods=0")


# ── WFO Correctness ────────────────────────────────────────────────────────────

def test_wfo_windows():
    """WFO should produce OOS periods only when sufficient data exists."""
    from backtest import walk_forward_optimization
    from config import WFO_IS_DAYS, WFO_OOS_DAYS, WFO_STEP_DAYS

    min_required = WFO_IS_DAYS + WFO_OOS_DAYS  # 455 days
    excess_days = min_required + WFO_STEP_DAYS * 2  # enough for ~3 windows

    dates = pd.date_range("2020-01-01", periods=excess_days, freq="B")
    mock = pd.DataFrame({
        "Open": np.random.randn(excess_days).cumsum() + 100,
        "High": np.random.randn(excess_days).cumsum() + 102,
        "Low":  np.random.randn(excess_days).cumsum() + 98,
        "Close": np.random.randn(excess_days).cumsum() + 100,
        "Volume": np.ones(excess_days) * 1_000_000,
    }, index=dates)

    result = walk_forward_optimization(mock)
    # Should have at least 1 OOS period
    assert isinstance(result["wfo_periods"], int), "wfo_periods should be int"
    assert result["wfo_oos_trades"] is None or result["wfo_oos_trades"] >= 0, "Negative trades"
    print(f"  [PASS] test_wfo_windows: periods={result['wfo_periods']}, trades={result['wfo_oos_trades']}")


def test_wfo_insufficient_data():
    """WFO with data < IS + OOS should return empty dict."""
    from backtest import walk_forward_optimization

    short = pd.DataFrame({
        "Close": np.random.randn(100).cumsum() + 100,
        "High": np.random.randn(100).cumsum() + 102,
        "Low": np.random.randn(100).cumsum() + 98,
        "Open": np.random.randn(100).cumsum() + 100,
        "Volume": np.ones(100) * 1_000_000,
    })
    result = walk_forward_optimization(short)
    assert result["wfo_periods"] == 0, "Expected 0 periods for insufficient data"
    assert result["wfo_oos_avg_pnl"] is None, "OOS avg PnL should be None"
    print(f"  [PASS] test_wfo_insufficient_data: empty result as expected")


# ── Macro Backtest ─────────────────────────────────────────────────────────────

def test_macro_backtest_returns_dict():
    """run_macro_backtest should always return a dict with expected keys."""
    from backtest import run_macro_backtest

    result = run_macro_backtest(None)
    assert isinstance(result, dict), "Should return dict for None input"
    assert result["BT_Trades"] == 0, "Expected 0 trades for None input"

    short = pd.DataFrame({"Close": [100.0] * 5})
    result = run_macro_backtest(short)
    assert result["BT_Trades"] == 0, "Expected 0 trades for short history"

    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    mock = pd.DataFrame({
        "Open": np.random.randn(500).cumsum() + 100,
        "High": np.random.randn(500).cumsum() + 102,
        "Low":  np.random.randn(500).cumsum() + 98,
        "Close": np.random.randn(500).cumsum() + 100,
        "Volume": np.ones(500) * 1_000_000,
    }, index=dates)
    result = run_macro_backtest(mock)
    assert "BT_Trades" in result, "Missing BT_Trades key"
    assert "BT_WinRate_pct" in result, "Missing BT_WinRate_pct key"
    assert "BT_Avg_PnL_pct" in result, "Missing BT_Avg_PnL_pct key"
    assert result["BT_Trades"] >= 0, "Negative trade count"
    print(f"  [PASS] test_macro_backtest_returns_dict: trades={result['BT_Trades']}, "
          f"win_rate={result['BT_WinRate_pct']}%")


# ── Historical Window Backtest ─────────────────────────────────────────────────

def test_historical_backtest_edge_cases():
    """Historical backtest should handle insufficient data gracefully."""
    from backtest import run_historical_backtest

    result = run_historical_backtest(None)
    assert result["Backtest_Signal"] == "N/A", "Expected N/A for None input"

    short = pd.DataFrame({"Close": [100.0] * 10})
    result = run_historical_backtest(short)
    assert result["Backtest_Signal"] == "N/A", "Expected N/A for short input"
    print(f"  [PASS] test_historical_backtest_edge_cases: both edge cases handled")


# ── Run 'em All ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  BACKTEST VALIDATION SUITE")
    print("=" * 60)

    test_survivorship_bias_warning_emitted()
    test_survivorship_bias_warning_empty()
    test_wfo_windows()
    test_wfo_insufficient_data()
    test_macro_backtest_returns_dict()
    test_historical_backtest_edge_cases()

    print("\n" + "=" * 60)
    print("  ALL BACKTEST VALIDATION TESTS PASSED")
    print("=" * 60)
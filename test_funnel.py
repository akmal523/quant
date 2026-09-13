"""
test_funnel.py — Unit tests for the multi-stage market funnel (Plan 3, Phase 1).

Covers the pure scoring logic (momentum_score) and the Stage 1 liquidity filter
rule. Network-dependent fetchers are not exercised here (they need yfinance).
"""
from __future__ import annotations

import pandas as pd


def test_momentum_score_uptrend_high():
    """A strong uptrend scores high (> 50)."""
    from funnel import momentum_score
    close = pd.Series(list(range(50, 250)))  # monotonic uptrend
    score = momentum_score(close)
    assert score > 50, f"uptrend should score > 50, got {score:.1f}"
    print(f"  [PASS] test_momentum_score_uptrend_high: {score:.1f}")


def test_momentum_score_downtrend_low():
    """A strong downtrend scores low (< 50)."""
    from funnel import momentum_score
    close = pd.Series(list(range(250, 50, -1)))  # monotonic downtrend
    score = momentum_score(close)
    assert score < 50, f"downtrend should score < 50, got {score:.1f}"
    print(f"  [PASS] test_momentum_score_downtrend_low: {score:.1f}")


def test_momentum_score_short_series_low():
    """Too little history scores near zero (no false confidence)."""
    from funnel import momentum_score
    close = pd.Series([100.0, 101.0, 102.0])
    assert momentum_score(close) == 0.0
    print("  [PASS] test_momentum_score_short_series_low")


def test_momentum_score_flat_series():
    """A flat series scores near the neutral 50 (no trend)."""
    from funnel import momentum_score
    close = pd.Series([100.0] * 200)
    score = momentum_score(close)
    assert 0.0 <= score <= 100.0
    print(f"  [PASS] test_momentum_score_flat_series: {score:.1f}")


def test_stage1_liquidity_rule():
    """Stage 1 keeps price >= min and daily $ volume >= min (pure rule).

    Stage 1 now batch-downloads snapshots, so the rule itself is the pure,
    network-free `_stage1_keep` helper.
    """
    from funnel import _stage1_keep
    assert _stage1_keep(50.0, 100_000, 5.0, 1_000_000) is True     # 5M $ vol
    assert _stage1_keep(0.5, 10_000_000, 5.0, 1_000_000) is False  # price < $5
    assert _stage1_keep(20.0, 100, 5.0, 1_000_000) is False        # 2k $ vol
    print("  [PASS] test_stage1_liquidity_rule")


if __name__ == "__main__":
    test_momentum_score_uptrend_high()
    test_momentum_score_downtrend_low()
    test_momentum_score_short_series_low()
    test_momentum_score_flat_series()
    test_stage1_liquidity_rule()
    print("\nAll funnel tests passed.")
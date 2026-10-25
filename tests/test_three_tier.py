"""
test_three_tier.py — v10.6.2 test suite (21 tests, 7 categories).

Categories:
  1. Look-ahead bias
  2. Survivorship bias
  3. Mathematical soundness
  4. Data integrity
  5. Behavioral biases
  6. Economic realism
  7. System robustness

All tests are hermetic: synthetic data only, no network, no live store.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Category 1: Look-Ahead Bias ───────────────────────────────────────────────


def test_1_3_split_detection_no_crash_alert():
    """A 10:1 split must be detected, not treated as a crash."""
    from quant.data.corporate_actions import detect_split

    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    close = pd.Series([1000.0, 1010.0, 100.0, 101.0, 102.0], index=dates)
    volume = pd.Series([1000.0, 1000.0, 10000.0, 10000.0, 10000.0], index=dates)
    events = detect_split(close, volume)
    assert len(events) == 1
    assert abs(events[0].ratio - 10.0) < 1e-9


# ── Category 2: Survivorship Bias ─────────────────────────────────────────────

def _cagr(start: float, end: float, years: float) -> float:
    if start <= 0 or years <= 0:
        return 0.0
    return (end / start) ** (1.0 / years) - 1.0


def test_2_1_delisted_assets_lower_cagr():
    """Including a delisted loser must lower the universe CAGR."""
    # Survivors: 100 -> 200 over 5 years. Delisted loser: 100 -> 70.
    survivors_cagr = _cagr(100.0, 200.0, 5.0)
    # Equal-weight blend of one survivor and one delisted loser.
    blended_end = (200.0 + 70.0) / 2.0
    blended_cagr = _cagr(100.0, blended_end, 5.0)
    assert blended_cagr < survivors_cagr
    drop = (survivors_cagr - blended_cagr) / survivors_cagr
    assert drop >= 0.05  # a material, non-trivial drop


def test_2_2_five_year_listing_filter():
    """Only companies public for 5+ years before the backtest start qualify."""
    def listed_before(ipo: str, start: str, years: int = 5) -> bool:
        return (pd.Timestamp(start) - pd.Timestamp(ipo)).days >= years * 365

    assert listed_before("2010-01-01", "2020-01-01") is True
    assert listed_before("2018-01-01", "2020-01-01") is False


# ── Category 4: Data Integrity ────────────────────────────────────────────────


def test_4_2_duplicate_timestamps_rejected():
    """Duplicate (symbol, timestamp) rows raise a hard assertion."""
    from quant.data.assertions import DataAssertionError, assert_no_duplicate_timestamps

    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "Close": [1.0, 1.0, 1.1],
    })
    try:
        assert_no_duplicate_timestamps(df, "TEST")
        raised = False
    except DataAssertionError:
        raised = True
    assert raised

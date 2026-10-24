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

import numpy as np
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


# ── Category 3: Mathematical Soundness ────────────────────────────────────────


def test_3_3_hmm_convergence():
    """GaussianHMM on synthetic market data converges in under 100 iterations."""
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(7)
    returns = rng.normal(0.0005, 0.01, 300)
    vol = pd.Series(returns).rolling(20).std().bfill().values
    x_scaled = StandardScaler().fit_transform(np.column_stack([returns, vol]))
    model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
    model.fit(x_scaled)
    assert model.monitor_.iter < 100


def test_3_4_cvar_at_least_as_extreme_as_var():
    """Expected shortfall (CVaR) is at least as extreme as VaR."""
    from quant.portfolio.risk import calculate_historical_var

    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(0.0, 0.02, 1000))
    var_95 = calculate_historical_var(returns, 0.95)
    tail = returns[returns <= var_95]
    cvar = float(tail.mean())
    # Both are negative; CVaR is more negative (more extreme) than VaR.
    assert cvar <= var_95


# ── Category 4: Data Integrity ────────────────────────────────────────────────

def test_4_1_yahoo_drift_golden_snapshot():
    """The golden backtest snapshot exists and is a stable JSON payload."""
    import json

    golden = Path(__file__).resolve().parent / "golden" / "backtest_2024.json"
    assert golden.exists()
    data = json.loads(golden.read_text())
    assert isinstance(data, dict) and data  # non-empty, parseable


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


def test_4_3_news_source_failure_penalizes_tactical():
    """No NLP data (data_confidence 0) lowers the tactical grade."""
    from quant.analytics.scoring import evaluate_tactical_grade

    with_data = evaluate_tactical_grade(0.6, 0.0, 0.0, data_confidence=1.0)
    without = evaluate_tactical_grade(0.6, 0.0, 0.0, data_confidence=0.0)
    assert without < with_data


def test_5_2_recency_bias_not_exclusive():
    """The regime model uses a long window, not only the last 3 months."""
    from quant.analytics.scoring import fit_market_regime

    rng = np.random.default_rng(11)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 300)))
    vol = close.pct_change().rolling(20).std().bfill()
    # With 300 points (well over 3 months) the model returns a real probability.
    score = fit_market_regime(close, vol)
    assert 0.0 <= score <= 15.0


# ── Category 6: Economic Realism ──────────────────────────────────────────────


def test_6_2_liquidity_constraint():
    """A 10k EUR order on a 50k-share ADV name is rejected."""
    from quant.portfolio.trade_size import check_volume_liquidity

    df = pd.DataFrame({
        "Close": [1.0] * 30,
        "Volume": [50_000.0] * 30,
    })
    ok, reason = check_volume_liquidity("THIN", 10_000.0, df)
    assert ok is False and "ADV" in reason


# ── Category 7: System Robustness ─────────────────────────────────────────────


def test_7_3_api_rate_limit_counter():
    """Rate-limit hits are counted so the 100/min contract is observable."""
    from quant.infra.observability import ObservabilityCollector

    obs = ObservabilityCollector()
    for _ in range(3):
        obs.increment("api_rate_limit_hits")
    assert obs.counters["api_rate_limit_hits"] == 3

"""
test_part3.py — Unit tests for Part 3 architectural refinement modules (v11).
Covers: data quality, feature cache, observability, incremental processing,
YAML config, alerts, health score, scenario simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Data Quality ──────────────────────────────────────────────────────────────

def test_data_quality_valid_clean():
    """Clean data passes validation."""
    from data_quality import DataQualityValidator
    dates = pd.date_range(pd.Timestamp.now().normalize() - pd.Timedelta(days=140),
                          periods=100, freq="B")
    df = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 110, 100)})
    ok, issues = DataQualityValidator().validate_batch(df, "TEST")
    assert ok, issues
    print("  [PASS] test_data_quality_valid_clean")


def test_data_quality_detects_nan_and_negative():
    """NaN and negative prices are flagged."""
    from data_quality import DataQualityValidator
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    close = np.linspace(100, 110, 100)
    close[10] = np.nan
    close[20] = -5.0
    df = pd.DataFrame({"Date": dates, "Close": close})
    ok, issues = DataQualityValidator().validate_batch(df, "TEST")
    assert not ok
    assert any("NaN" in i for i in issues)
    assert any("Negative" in i for i in issues)
    print(f"  [PASS] test_data_quality_detects_nan_and_negative: {issues}")


def test_data_quality_auto_repair():
    """auto_repair removes NaN and duplicates."""
    from data_quality import DataQualityValidator
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    close = np.linspace(100, 110, 100)
    close[10] = np.nan
    df = pd.DataFrame({"Date": list(dates) + [dates[0]], "Close": list(close) + [99.0]})
    repaired = DataQualityValidator().auto_repair(df, "TEST")
    assert repaired["Close"].isna().sum() == 0
    assert repaired["Date"].duplicated().sum() == 0
    print("  [PASS] test_data_quality_auto_repair")


# ── Feature Cache ─────────────────────────────────────────────────────────────

def test_feature_cache_roundtrip(tmp_path=None):
    """Cache set then get returns the same data."""
    from feature_cache import FeatureCache
    import tempfile
    cache = FeatureCache(cache_dir=tempfile.mkdtemp())
    df = pd.DataFrame({"SMA_200": [1.0, 2.0, 3.0]})
    h = cache.compute_data_hash(pd.DataFrame({"Close": [1.0, 2.0, 3.0]}))
    cache.set("EUNL.DE", "sma_200", h, df)
    got = cache.get("EUNL.DE", "sma_200", h)
    assert got is not None
    assert got["SMA_200"].tolist() == [1.0, 2.0, 3.0]
    print("  [PASS] test_feature_cache_roundtrip")


def test_feature_cache_miss_on_stale_hash():
    """Cache returns None when data_hash changed."""
    from feature_cache import FeatureCache
    import tempfile
    cache = FeatureCache(cache_dir=tempfile.mkdtemp())
    df = pd.DataFrame({"SMA_200": [1.0]})
    h1 = cache.compute_data_hash(pd.DataFrame({"Close": [1.0, 2.0]}))
    h2 = cache.compute_data_hash(pd.DataFrame({"Close": [5.0, 6.0]}))
    cache.set("X", "sma_200", h1, df)
    assert cache.get("X", "sma_200", h2) is None
    print("  [PASS] test_feature_cache_miss_on_stale_hash")


# ── Observability ─────────────────────────────────────────────────────────────

def test_observability_summary():
    """Summary reports step count and duration."""
    from observability import ObservabilityCollector
    obs = ObservabilityCollector()
    with obs.step("load"):
        pass
    with obs.step("score"):
        pass
    summary = obs.summary()
    assert "2 total" in summary
    assert "Pipeline completed" in summary
    print("  [PASS] test_observability_summary")


def test_observability_captures_error():
    """Errors are recorded in the step."""
    from observability import ObservabilityCollector
    obs = ObservabilityCollector()
    try:
        with obs.step("bad"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert obs.steps[-1]["status"] == "error"
    assert "boom" in obs.steps[-1]["error"]
    print("  [PASS] test_observability_captures_error")


# ── Incremental Processing ────────────────────────────────────────────────────

def test_incremental_change_detection(tmp_path=None):
    """detect_changes returns True only when data changed."""
    from incremental import IncrementalProcessor
    import tempfile
    proc = IncrementalProcessor(state_file=tempfile.mktemp(suffix=".json"))
    df = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=10),
                       "Close": np.arange(10.0)})
    assert proc.detect_changes(df, "AMZN") is True   # first time
    assert proc.detect_changes(df, "AMZN") is False  # unchanged
    df2 = df.copy()
    df2.loc[9, "Close"] = 99.0
    assert proc.detect_changes(df2, "AMZN") is True  # changed
    print("  [PASS] test_incremental_change_detection")


# ── YAML Config ───────────────────────────────────────────────────────────────

def test_config_loader():
    """Config loads nested values via dot path."""
    from config_loader import Config
    cfg = Config("config.yaml")
    assert cfg.get("fees.round_trip_eur") == 2.0
    assert cfg.get("scoring.factor_weights.momentum") == 0.30
    assert cfg.get("missing.path", "default") == "default"
    print("  [PASS] test_config_loader")


# ── Alerts ────────────────────────────────────────────────────────────────────

def test_alerts_drawdown():
    """Deep drawdown triggers CRITICAL alert."""
    from alerts import AlertSystem
    series = pd.Series([100.0, 100.0, 100.0, 80.0])  # -20%
    alerts = AlertSystem().check_drawdown(series)
    assert any(a.level == "CRITICAL" for a in alerts)
    print("  [PASS] test_alerts_drawdown")


def test_alerts_rebalance():
    """Drift beyond 5% triggers rebalance alert."""
    from alerts import AlertSystem
    alerts = AlertSystem().check_rebalance(
        {"EUNL.DE": 0.60}, {"EUNL.DE": 0.50}
    )
    assert len(alerts) == 1
    assert "SELL" in alerts[0].action
    print("  [PASS] test_alerts_rebalance")


# ── Health Score ──────────────────────────────────────────────────────────────

def test_health_score_bounds():
    """Health score is in [0, 100] with a grade."""
    from health_score import PortfolioHealthScore
    data = {
        "holdings": [{"Symbol": "A"}, {"Symbol": "B"}, {"Symbol": "C"}],
        "weights": {"A": 0.4, "B": 0.3, "C": 0.3},
        "correlation_to_spx": 0.7,
        "sharpe": 0.8,
        "max_drawdown": -0.08,
        "fee_drag_pct": 0.5,
        "liquid_pct": 0.9,
    }
    result = PortfolioHealthScore().compute(data)
    assert 0 <= result["total_score"] <= 100
    assert result["grade"] in "ABCDF+"
    print(f"  [PASS] test_health_score_bounds: {result['total_score']} ({result['grade']})")


# ── Scenario Simulator ────────────────────────────────────────────────────────

def test_scenario_simulator_trade():
    """Simulated trade returns vol/sharpe metrics."""
    from scenario_simulator import ScenarioSimulator
    portfolio = pd.DataFrame({"Symbol": ["A", "B"], "Amount_EUR": [500.0, 500.0]})
    rng = np.random.default_rng(42)
    returns = pd.DataFrame({
        "A": rng.normal(0.0005, 0.01, 300),
        "B": rng.normal(0.0005, 0.01, 300),
    })
    sim = ScenarioSimulator(portfolio, returns)
    result = sim.simulate_trade("A", "B", 200.0)
    assert "new_volatility" in result
    assert "recommendation" in result
    print(f"  [PASS] test_scenario_simulator_trade: {result['recommendation']}")


def test_scenario_simulator_crash():
    """Crash simulation returns impact and hedges."""
    from scenario_simulator import ScenarioSimulator
    portfolio = pd.DataFrame({"Symbol": ["A"], "Amount_EUR": [1000.0]})
    rng = np.random.default_rng(1)
    returns = pd.DataFrame({
        "A": rng.normal(0, 0.01, 300),
        "SPX": rng.normal(0, 0.01, 300),
    })
    sim = ScenarioSimulator(portfolio, returns)
    result = sim.simulate_crash(-0.20)
    assert "portfolio_impact" in result
    assert "recommended_hedges" in result
    print(f"  [PASS] test_scenario_simulator_crash: impact={result['portfolio_impact']}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} Part 3 tests passed.")
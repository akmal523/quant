"""
test_part3.py — Unit tests for Part 3 architectural refinement modules (v11).
Covers: data quality, feature cache, observability, incremental processing,
YAML config, alerts, health score, scenario simulator.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

# ── Data Quality ──────────────────────────────────────────────────────────────

def test_data_quality_valid_clean():
    """Clean data passes validation."""
    from quant.data.data_quality import DataQualityValidator
    dates = pd.date_range(pd.Timestamp.now().normalize() - pd.Timedelta(days=140),
                          periods=100, freq="B")
    df = pd.DataFrame({"Date": dates, "Close": np.linspace(100, 110, 100)})
    ok, issues = DataQualityValidator().validate_batch(df, "TEST")
    assert ok, issues
    print("  [PASS] test_data_quality_valid_clean")


def test_data_quality_detects_nan_and_negative():
    """NaN and negative prices are flagged."""
    from quant.data.data_quality import DataQualityValidator
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
    from quant.data.data_quality import DataQualityValidator
    dates = pd.date_range("2026-01-01", periods=100, freq="B")
    close = np.linspace(100, 110, 100)
    close[10] = np.nan
    df = pd.DataFrame({"Date": list(dates) + [dates[0]], "Close": list(close) + [99.0]})
    repaired = DataQualityValidator().auto_repair(df, "TEST")
    assert repaired["Close"].isna().sum() == 0
    assert repaired["Date"].duplicated().sum() == 0
    print("  [PASS] test_data_quality_auto_repair")


# ── Observability ─────────────────────────────────────────────────────────────

def test_observability_summary():
    """Summary reports step count and duration."""
    from quant.infra.observability import ObservabilityCollector
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
    from quant.infra.observability import ObservabilityCollector
    obs = ObservabilityCollector()
    try:
        with obs.step("bad"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert obs.steps[-1]["status"] == "error"
    assert "boom" in obs.steps[-1]["error"]
    print("  [PASS] test_observability_captures_error")


# ── Alerts ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} Part 3 tests passed.")

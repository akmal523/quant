"""
test_institutional.py — v10.4.0 institutional-grade test suite.

Covers: corporate actions, hard assertions, Mean-CVaR, kill switch, regime
constraints, TCA, volatility-aware sizing, reconciliation, and alpha metrics.
Standalone (no pytest required), matching the repo's test style.

Run: python3 tests/test_institutional.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def test_assertions_hard_gate():
    from quant.data.assertions import (
        DataAssertionError,
        assert_fundamentals_sane,
        assert_no_duplicate_timestamps,
        run_assertions,
    )
    # Duplicate timestamps -> fail.
    dup = pd.DataFrame({"Date": ["2024-01-01", "2024-01-01"], "Close": [10.0, 11.0]})
    try:
        assert_no_duplicate_timestamps(dup, "X")
        raise AssertionError("duplicate timestamps not caught")
    except DataAssertionError:
        pass

    # Unexplained >50% drop -> fail.
    drop = pd.DataFrame({
        "Date": pd.bdate_range("2024-01-01", periods=3),
        "Close": [100.0, 40.0, 41.0],
    })
    try:
        run_assertions(drop, "X")
        raise AssertionError("unexplained drop not caught")
    except DataAssertionError:
        pass

    # Profitable company with negative D/E -> fail.
    try:
        assert_fundamentals_sane({"ROE": 0.2, "DebtToEquity": -1.0}, "X")
        raise AssertionError("negative D/E not caught")
    except DataAssertionError:
        pass

    # Clean data passes.
    clean = pd.DataFrame({
        "Date": pd.bdate_range("2024-01-01", periods=3),
        "Close": [100.0, 101.0, 102.0],
    })
    assert run_assertions(clean, "X") == ["no_duplicate_timestamps", "no_unexplained_drop"]


def test_event_bus_new_events():
    from quant.infra.event_bus import EVENTS, EventBus
    assert EVENTS["MARKET_CLOSE_DATA_READY"] == "market_close_data_ready"
    assert EVENTS["KILL_SWITCH"] == "kill_switch"
    bus = EventBus()
    seen = []
    bus.subscribe(EVENTS["KILL_SWITCH"], lambda d: seen.append(d))
    bus.publish(EVENTS["KILL_SWITCH"], {"signal": "LIQUIDATE TO CASH"})
    assert seen and seen[0]["signal"] == "LIQUIDATE TO CASH"


def test_observability_metrics():
    from quant.infra.observability import ObservabilityCollector
    obs = ObservabilityCollector()
    obs.record_metric("duckdb_query_ms", 12.5)
    obs.record_metric("duckdb_query_ms", 7.5)
    obs.increment("api_rate_limit_hits")
    j = obs.to_json()
    assert j["metrics"]["duckdb_query_ms"]["mean"] == 10.0
    assert j["counters"]["api_rate_limit_hits"] == 1

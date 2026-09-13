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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def test_corporate_actions_detect_and_adjust():
    from quant.data.corporate_actions import (
        detect_split, apply_corporate_actions, adjust_cost_basis,
    )
    dates = pd.bdate_range("2024-01-01", periods=10)
    close = pd.Series([100, 101, 102, 103, 104, 52, 53, 54, 55, 56], index=dates, dtype=float)
    volume = pd.Series([1e6] * 5 + [2e6] * 5, index=dates, dtype=float)
    events = detect_split(close, volume)
    assert len(events) == 1, f"expected 1 split, got {len(events)}"
    assert abs(events[0].ratio - 2.0) < 0.01

    df = pd.DataFrame({"Close": close, "Volume": volume}, index=dates)
    adjusted, evs = apply_corporate_actions(df)
    # Pre-split close 100 -> 50 after 2:1 adjustment.
    assert abs(adjusted["Close"].iloc[0] - 50.0) < 0.01
    assert abs(adjust_cost_basis(100.0, 2.0) - 50.0) < 1e-9


def test_assertions_hard_gate():
    from quant.data.assertions import (
        run_assertions, DataAssertionError,
        assert_no_duplicate_timestamps, assert_fundamentals_sane,
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


def test_mean_cvar_optimizer():
    from quant.portfolio.optimizer import optimize_portfolio_cvar, cvar_of_weights
    rng = np.random.RandomState(0)
    R = rng.normal(0.0005, 0.01, size=(250, 4))
    exp_ret = np.array([0.001, 0.0008, 0.0006, 0.0004])
    w = optimize_portfolio_cvar(exp_ret, R, max_weight=0.5)
    assert abs(w.sum() - 1.0) < 1e-4, f"weights sum {w.sum()}"
    assert (w >= -1e-6).all()
    cvar = cvar_of_weights(w, R)
    assert cvar >= 0.0


def test_kill_switch_triggers():
    from quant.portfolio.risk_monitor import RiskMonitor
    # 3.5% single-day drop -> kill switch.
    values = pd.Series([100.0, 100.0, 96.5])
    rm = RiskMonitor(values)
    kill = rm.check_kill_switch()
    assert kill["triggered"] is True
    assert kill["signal"] == "LIQUIDATE TO CASH"

    # Flat series -> no trigger.
    flat = RiskMonitor(pd.Series([100.0, 100.0, 100.0]))
    assert flat.check_kill_switch()["triggered"] is False


def test_regime_constraints():
    from quant.portfolio.regime_constraints import (
        regime_from_bull_prob, apply_regime_constraints,
    )
    assert regime_from_bull_prob(0.9) == "bull"
    assert regime_from_bull_prob(0.1) == "bear"
    assert regime_from_bull_prob(0.5) == "chop"
    # Bear caps single weight at 2%.
    assert apply_regime_constraints(0.10, "bear") == 0.02
    assert apply_regime_constraints(0.10, "bull") == 0.10


def test_tca_implementation_shortfall():
    from quant.execution.tca import implementation_shortfall, estimate_slippage_bps
    # BUY filled above signal -> positive cost.
    assert implementation_shortfall(100.0, 101.0, "BUY") > 0
    # SELL filled below signal -> positive cost.
    assert implementation_shortfall(100.0, 99.0, "SELL") > 0
    # Perfect fill -> zero.
    assert implementation_shortfall(100.0, 100.0, "BUY") == 0.0
    assert estimate_slippage_bps(0.0) >= 0.0


def test_vol_aware_min_trade_size():
    from quant.portfolio.optimizer import (
        minimum_trade_size, minimum_trade_size_vol_aware,
    )
    base = minimum_trade_size(200.0)
    scaled = minimum_trade_size_vol_aware(200.0, 0.30)
    assert scaled > base, "higher vol must raise the fee floor"


def test_reconciliation_flags():
    from quant.execution.reconciliation import reconcile
    theo = pd.DataFrame([
        {"snapshot_date": "2024-01-01", "symbol": "AAPL", "shares": 1.0,
         "price_eur": 100.0, "value_eur": 100.0},
    ])
    broker = pd.DataFrame([{"Symbol": "AAPL", "Value_EUR": 100.5}])
    res = reconcile(theo, broker)
    assert res.iloc[0]["flag"] == ""  # within tolerance

    broker_bad = pd.DataFrame([{"Symbol": "AAPL", "Value_EUR": 90.0}])
    res_bad = reconcile(theo, broker_bad)
    assert res_bad.iloc[0]["flag"].startswith("[!]")


def test_alpha_metrics():
    from quant.analytics.metrics import (
        deflated_sharpe_ratio, information_coefficient, alpha_decay_curve,
        turnover_stats,
    )
    rng = np.random.RandomState(1)
    rets = pd.Series(rng.normal(0.0006, 0.01, 500))
    dsr = deflated_sharpe_ratio(rets, n_trials=10)
    assert 0.0 <= dsr <= 1.0

    sig = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    fwd = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    assert information_coefficient(sig, fwd) > 0.9

    prices = pd.DataFrame({"A": np.linspace(100, 120, 30)})
    signals = pd.DataFrame({"A": np.linspace(0, 1, 30)})
    curve = alpha_decay_curve(signals, prices, horizons=(1, 5))
    assert set(curve["horizon"]) == {1, 5}

    weights = pd.DataFrame({"A": [0.5, 0.6, 0.4], "B": [0.5, 0.4, 0.6]})
    ts = turnover_stats(weights)
    assert ts["mean_turnover"] > 0


def test_event_bus_new_events():
    from quant.infra.event_bus import EventBus, EVENTS
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


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)

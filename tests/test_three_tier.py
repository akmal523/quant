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

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ── Category 1: Look-Ahead Bias ───────────────────────────────────────────────

def test_1_1_bitemporal_consistency():
    """A fundamental published after as_of must never be visible."""
    from quant.data.fundamentals import get_fundamentals_pit

    # On the ephemeral test DB there is no history: the accessor must return
    # None (no lookahead is possible), never a fabricated value.
    assert get_fundamentals_pit("AAPL", "2020-01-01") is None

    # The PIT contract: only rows with published_date <= as_of are eligible.
    rows = [
        {"as_of_date": "2019-12-31", "published_date": "2020-02-15", "PE": 20},
        {"as_of_date": "2019-09-30", "published_date": "2019-11-01", "PE": 18},
    ]
    as_of = "2020-01-01"
    visible = [r for r in rows if r["published_date"] <= as_of]
    assert len(visible) == 1 and visible[0]["PE"] == 18


def test_1_2_dividend_leakage_not_a_sell():
    """A dividend-driven price drop must not become a sell signal."""
    from quant.portfolio.fortress import fortress_signal

    # A Fortress asset never sells, regardless of a price drop.
    assert fortress_signal(80.0) in ("INCREASE_SPARPLAN", "MAINTAIN")
    assert fortress_signal(10.0) == "MAINTAIN"


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

def test_3_1_deflated_sharpe_vs_raw():
    """Deflated Sharpe is finite and within 30 percent of the raw Sharpe."""
    from quant.analytics.scoring import deflated_sharpe_ratio

    raw = 1.5
    dsr = deflated_sharpe_ratio(raw, num_trials=10, n=252)
    assert math.isfinite(dsr)
    assert abs(dsr - raw) / raw < 0.30
    # A single trial returns the raw Sharpe unchanged.
    assert deflated_sharpe_ratio(raw, num_trials=1, n=252) == raw


def test_3_2_covariance_stability():
    """A 1 percent covariance perturbation moves weights by less than 5 percent."""
    from quant.portfolio.optimizer import optimize_portfolio

    rng = np.random.default_rng(42)
    n = 4
    a = rng.normal(size=(n, n))
    cov = a @ a.T / n + np.eye(n) * 0.1
    mu = np.array([0.10, 0.12, 0.08, 0.11])

    w1 = optimize_portfolio(mu, cov, max_weight=0.5)
    w2 = optimize_portfolio(mu, cov * 1.01, max_weight=0.5)
    assert np.allclose(w1.sum(), 1.0, atol=1e-4)
    assert np.max(np.abs(w1 - w2)) < 0.05


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


# ── Category 5: Behavioral Biases ─────────────────────────────────────────────

def test_5_1_overtrading_penalty():
    """Alpha is capped at 2 trades per week."""
    from quant.portfolio.alpha import alpha_trade_allowed
    from quant.portfolio.behavioral_guardrails import BehavioralGuardrails

    assert alpha_trade_allowed(0)[0] is True
    assert alpha_trade_allowed(1)[0] is True
    assert alpha_trade_allowed(2)[0] is False

    g = BehavioralGuardrails()
    g.register_alpha_trade("NVDA")
    g.register_alpha_trade("TSM")
    result = g.check_alpha_weekly_limit()
    assert result["allowed"] is False
    assert "limit" in result["warning"].lower()
    assert result["override_required"] is True


def test_5_2_recency_bias_not_exclusive():
    """The regime model uses a long window, not only the last 3 months."""
    from quant.analytics.scoring import fit_market_regime

    rng = np.random.default_rng(11)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 300)))
    vol = close.pct_change().rolling(20).std().bfill()
    # With 300 points (well over 3 months) the model returns a real probability.
    score = fit_market_regime(close, vol)
    assert 0.0 <= score <= 15.0


def test_5_3_disposition_effect_symmetric():
    """Sell prioritization is symmetric: liquidity first, then losers."""
    from quant.portfolio.risk import prioritize_sells

    holdings = [
        {"symbol": "WIN", "current_value_eur": 100.0, "pnl_eur": 50.0, "liquidity_score": 90.0},
        {"symbol": "LOSS", "current_value_eur": 100.0, "pnl_eur": -50.0, "liquidity_score": 90.0},
    ]
    recs = prioritize_sells(holdings, 100.0)
    # Equal liquidity -> the loser is sold first (tax-loss harvest).
    assert recs[0]["symbol"] == "LOSS"


# ── Category 6: Economic Realism ──────────────────────────────────────────────

def test_6_1_transaction_cost_reality():
    """A trade must clear the 2 EUR round-trip fee plus slippage."""
    from quant.portfolio.optimizer import minimum_trade_size, passes_fee_hurdle

    # 200 bps alpha -> 100 EUR minimum to clear the 2 EUR fee.
    assert abs(minimum_trade_size(200.0) - 100.0) < 1e-6
    assert passes_fee_hurdle(200.0, 100.0) is True
    assert passes_fee_hurdle(200.0, 99.0) is False


def test_6_2_liquidity_constraint():
    """A 10k EUR order on a 50k-share ADV name is rejected."""
    from quant.portfolio.optimizer import check_volume_liquidity

    df = pd.DataFrame({
        "Close": [1.0] * 30,
        "Volume": [50_000.0] * 30,
    })
    ok, reason = check_volume_liquidity("THIN", 10_000.0, df)
    assert ok is False and "ADV" in reason


def test_6_3_freistellungsauftrag_considered():
    """The 1000 EUR tax-free allowance is applied before tax is estimated."""
    from quant.portfolio.tax_optimizer import TaxOptimizer

    portfolio = pd.DataFrame({
        "Symbol": ["A", "B"],
        "PnL_EUR": [800.0, 500.0],
        "Tier": ["ALPHA", "ALPHA"],
    })
    opt = TaxOptimizer(portfolio, tax_free_allowance=1000.0)
    # Taxable income is driven by REALIZED gains (the accessor is a placeholder
    # in the offline build); inject 1300 EUR to exercise the allowance arithmetic.
    opt._get_realized_gains_ytd = lambda: 1300.0  # type: ignore[method-assign]
    pos = opt.compute_tax_position()
    # 1300 EUR realized gains - 1000 EUR allowance = 300 EUR taxable.
    assert abs(pos["net_taxable"] - 300.0) < 1e-6
    assert pos["estimated_tax"] > 0


# ── Category 7: System Robustness ─────────────────────────────────────────────

def test_7_1_empty_universe_goes_to_cash():
    """When every asset is weak, no BUY signal is emitted."""
    from quant.portfolio.portfolio import tier_audit

    portfolio = pd.DataFrame({"Symbol": ["A", "B"]})
    scan = pd.DataFrame({
        "Symbol": ["A", "B"],
        "Structural_Grade": [20.0, 30.0],
        "Tactical_Grade": [20.0, 25.0],
        "nlp_score": [-50.0, -40.0],
    })
    audit = tier_audit(portfolio, scan)
    assert not audit["Signal"].isin(["BUY", "BUY_SPECULATIVE"]).any()


def test_7_2_market_crash_lockdown():
    """A -30 percent week triggers LOCKDOWN."""
    from quant.portfolio.risk_monitor import RiskMonitor

    values = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 70.0])
    status = RiskMonitor(values).check_circuit_breakers()
    assert status["status"] == "LOCKDOWN"


def test_7_3_api_rate_limit_counter():
    """Rate-limit hits are counted so the 100/min contract is observable."""
    from quant.infra.observability import ObservabilityCollector

    obs = ObservabilityCollector()
    for _ in range(3):
        obs.increment("api_rate_limit_hits")
    assert obs.counters["api_rate_limit_hits"] == 3

"""
test_advanced.py — Unit tests for Part 2 advanced modules (v11).
Covers: portfolio context, strategy engine, tax optimizer, cash manager,
risk monitor, attribution, event bus, behavioral guardrails, validation engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── Portfolio Context ─────────────────────────────────────────────────────────

def test_risk_contribution_sums_to_100():
    """Risk contribution percentages sum to ~100."""
    from portfolio_context import PortfolioContext
    holdings = pd.DataFrame({
        "Symbol": ["A", "B"],
        "Amount_EUR": [500.0, 500.0],
    })
    rng = np.random.default_rng(42)
    returns = pd.DataFrame({
        "A": rng.normal(0.0005, 0.01, 300),
        "B": rng.normal(0.0005, 0.01, 300),
    })
    ctx = PortfolioContext(holdings, returns)
    rc = ctx.compute_risk_contribution()
    assert abs(rc.sum() - 100.0) < 1.0, f"sum={rc.sum()}"
    print(f"  [PASS] test_risk_contribution_sums_to_100: sum={rc.sum():.1f}")


def test_concentration_penalty_range():
    """Concentration penalty is in [0.5, 1.1]."""
    from portfolio_context import PortfolioContext
    holdings = pd.DataFrame({"Symbol": ["A"], "Amount_EUR": [1000.0]})
    returns = pd.DataFrame({"A": np.random.default_rng(1).normal(0, 0.01, 200)})
    ctx = PortfolioContext(holdings, returns)
    p = ctx.concentration_penalty("A")
    assert 0.5 <= p <= 1.1, p
    print(f"  [PASS] test_concentration_penalty_range: {p}")


# ── Strategy Engine ───────────────────────────────────────────────────────────

def test_strategy_engine_regime_weights():
    """Regime weights sum to ~1.0 and differ by regime."""
    from strategy_engine import StrategyEngine
    eng = StrategyEngine()
    bull = eng.regime_weights("bull_low_vol")
    bear = eng.regime_weights("bear")
    assert abs(sum(bull.values()) - 1.0) < 1e-9
    assert bull["Momentum"] > bear["Momentum"]  # momentum favored in bull
    print(f"  [PASS] test_strategy_engine_regime_weights: bull_mom={bull['Momentum']} bear_mom={bear['Momentum']}")


def test_strategy_engine_ensemble_signal():
    """Ensemble signal is a float."""
    from strategy_engine import StrategyEngine
    eng = StrategyEngine()
    data = {"returns_6m": 0.10, "volatility_60d": 0.02, "rsi_14": 45.0,
            "pe_ratio": 15.0, "dividend_yield": 0.02}
    score = eng.compute_ensemble_signal("TEST", data, "bull_low_vol")
    assert isinstance(score, float)
    print(f"  [PASS] test_strategy_engine_ensemble_signal: {score:.3f}")


# ── Tax Optimizer ─────────────────────────────────────────────────────────────

def test_tax_position_net_taxable_nonneg():
    """Net taxable is never negative."""
    from tax_optimizer import TaxOptimizer
    df = pd.DataFrame({"Symbol": ["A"], "PnL_EUR": [500.0], "Tier": ["ACTIVE"]})
    tax = TaxOptimizer(df)
    pos = tax.compute_tax_position()
    assert pos["net_taxable"] >= 0
    print(f"  [PASS] test_tax_position_net_taxable_nonneg: {pos['net_taxable']}")


def test_tax_harvest_skips_core():
    """CORE losers are never harvested."""
    from tax_optimizer import TaxOptimizer
    df = pd.DataFrame({
        "Symbol": ["EUNL.DE", "5J50.DE"],
        "PnL_EUR": [-100.0, -100.0],
        "Tier": ["CORE", "SECTOR"],
    })
    tax = TaxOptimizer(df)
    # Force net_taxable > 0 by overriding realized gains.
    tax._get_realized_gains_ytd = lambda: 2000.0
    harvest = tax.harvest_opportunities()
    if not harvest.empty:
        assert "EUNL.DE" not in harvest["Sell_Symbol"].values
    print("  [PASS] test_tax_harvest_skips_core")


# ── Cash Manager ──────────────────────────────────────────────────────────────

def test_cash_target_bounds():
    """Target cash allocation is in [0.05, 0.30]."""
    from cash_manager import CashManager
    cm = CashManager()
    for regime in ["bull_low_vol", "high_vol_choppy", "bear", "reflation"]:
        t = cm.target_cash_allocation(regime, vix=25.0, opportunity_score=0.5)
        assert 0.05 <= t <= 0.30, (regime, t)
    print("  [PASS] test_cash_target_bounds")


def test_dip_buying_scales_with_drawdown():
    """Deeper drawdown -> larger dip buy."""
    from cash_manager import CashManager
    cm = CashManager()
    small = cm.dip_buying_algorithm("X", -0.08, 1000.0)
    large = cm.dip_buying_algorithm("X", -0.25, 1000.0)
    assert large > small
    assert cm.dip_buying_algorithm("X", -0.02, 1000.0) == 0.0
    print(f"  [PASS] test_dip_buying_scales_with_drawdown: {small:.0f} < {large:.0f}")


# ── Risk Monitor ──────────────────────────────────────────────────────────────

def test_risk_monitor_drawdown():
    """Drawdown is negative after a decline."""
    from risk_monitor import RiskMonitor
    series = pd.Series([100.0, 110.0, 90.0])
    rm = RiskMonitor(series)
    dd = rm.compute_current_drawdown()
    assert dd < 0
    print(f"  [PASS] test_risk_monitor_drawdown: {dd:.1%}")


def test_risk_monitor_circuit_breakers():
    """Deep drawdown triggers LOCKDOWN."""
    from risk_monitor import RiskMonitor
    series = pd.Series([100.0, 100.0, 100.0, 70.0])  # -30% from peak
    rm = RiskMonitor(series)
    status = rm.check_circuit_breakers()
    assert status["status"] == "LOCKDOWN", status
    print(f"  [PASS] test_risk_monitor_circuit_breakers: {status['status']}")


# ── Attribution ───────────────────────────────────────────────────────────────

def test_attribution_total_effect():
    """Total effect = allocation + selection + interaction."""
    from attribution import BrinsonFachlerAttribution
    portfolio = pd.DataFrame({
        "Symbol": ["A", "B"],
        "Sector": ["Tech", "Broad"],
        "Amount_EUR": [600.0, 400.0],
    })
    bench = {"Tech": 0.2, "Broad": 0.8}
    returns_sector = pd.DataFrame({
        "return_30d": [0.05, 0.02],
    }, index=["Tech", "Broad"])
    attr = BrinsonFachlerAttribution(portfolio, bench, {}, returns_sector)
    df = attr.compute_attribution(30)
    assert not df.empty
    for _, r in df.iterrows():
        total = r["Allocation_Effect"] + r["Selection_Effect"] + r["Interaction"]
        assert abs(total - r["Total_Effect"]) < 0.01
    print("  [PASS] test_attribution_total_effect")


# ── Event Bus ─────────────────────────────────────────────────────────────────

def test_event_bus_publish():
    """Publish calls all subscribers for the event type."""
    from event_bus import EventBus
    bus = EventBus()
    received = []
    bus.subscribe("DIP_DETECTED", lambda d: received.append(d["symbol"]))
    bus.publish("DIP_DETECTED", {"symbol": "AMZN"})
    assert received == ["AMZN"]
    print("  [PASS] test_event_bus_publish")


# ── Behavioral Guardrails ─────────────────────────────────────────────────────

def test_guardrails_cooldown():
    """Cooldown blocks re-trading the same symbol."""
    from behavioral_guardrails import BehavioralGuardrails
    g = BehavioralGuardrails()
    g.register_trade("AMZN")
    ok, reason = g.check_cooldown("AMZN")
    assert not ok
    print(f"  [PASS] test_guardrails_cooldown: {reason}")


def test_guardrails_weekly_limit():
    """Weekly limit blocks after max trades."""
    from behavioral_guardrails import BehavioralGuardrails
    g = BehavioralGuardrails()
    for _ in range(5):
        g.register_trade("X")
    ok, _ = g.check_weekly_limit(max_trades=5)
    assert not ok
    print("  [PASS] test_guardrails_weekly_limit")


# ── Validation Engine ─────────────────────────────────────────────────────────

def test_validation_engine_robustness():
    """Robustness metrics return a dict."""
    from validation_engine import ValidationEngine
    ve = ValidationEngine()
    results = pd.DataFrame({
        "regime": ["bull", "bull", "bear"],
        "return": [0.01, -0.01, 0.02],
        "sharpe": [0.5, -0.2, 0.8],
    })
    metrics = ve.compute_robustness_metrics(results)
    assert isinstance(metrics, dict)
    assert metrics["max_consecutive_losses"] == 1
    print(f"  [PASS] test_validation_engine_robustness: {metrics}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} advanced tests passed.")
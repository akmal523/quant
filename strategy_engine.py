"""
strategy_engine.py — Regime-Aware Multi-Strategy Ensemble (Part 2, Upgrade #2).

Intent: blend signals from multiple strategies with weights that depend on the
current market regime. In high-vol/choppy markets the engine shifts from
momentum to mean-reversion; in bear markets value + risk-parity dominate.
No single strategy dominates forever.

Invariants:
  - STRATEGY_WEIGHTS_BY_REGIME weights sum to ~1.0 per regime.
  - compute_ensemble_signal returns a float (weighted blend).
  - Pure computation; no I/O.

Dependencies: strategies package.
"""
from __future__ import annotations

from strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    ValueStrategy,
    RiskParityStrategy,
)


class StrategyEngine:
    """Allocates capital across strategies based on current market regime."""

    STRATEGY_WEIGHTS_BY_REGIME = {
        "bull_low_vol": {
            "Momentum": 0.50,
            "Value": 0.10,
            "Mean Reversion": 0.10,
            "Risk Parity": 0.30,
        },
        "high_vol_choppy": {
            "Momentum": 0.10,
            "Value": 0.20,
            "Mean Reversion": 0.50,
            "Risk Parity": 0.20,
        },
        "bear": {
            "Momentum": 0.05,
            "Value": 0.40,
            "Mean Reversion": 0.15,
            "Risk Parity": 0.40,
        },
        "reflation": {
            "Momentum": 0.30,
            "Value": 0.40,
            "Mean Reversion": 0.10,
            "Risk Parity": 0.20,
        },
    }

    def __init__(self):
        self.strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            ValueStrategy(),
            RiskParityStrategy(),
        ]

    def compute_ensemble_signal(self, symbol: str, data: dict, regime: str) -> float:
        """Blend strategy signals weighted by the current regime."""
        weights = self.STRATEGY_WEIGHTS_BY_REGIME.get(
            regime, self.STRATEGY_WEIGHTS_BY_REGIME["bull_low_vol"]
        )
        ensemble = 0.0
        for strategy in self.strategies:
            signal = strategy.compute_signal(symbol, data)
            ensemble += signal * weights.get(strategy.name, 0.0)
        return float(ensemble)

    def regime_weights(self, regime: str) -> dict:
        """Return the strategy weight map for a regime (for reporting)."""
        return self.STRATEGY_WEIGHTS_BY_REGIME.get(
            regime, self.STRATEGY_WEIGHTS_BY_REGIME["bull_low_vol"]
        )
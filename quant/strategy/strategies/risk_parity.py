"""
strategies/risk_parity.py — Risk Parity strategy (Part 2, Upgrade #2).

Intent: weight assets inversely to volatility. Diversifier that works in any
regime. Signal = 1 / volatility.
"""
from __future__ import annotations

from quant.strategy.strategies.base import BaseStrategy


class RiskParityStrategy(BaseStrategy):
    name = "Risk Parity"
    preferred_regime = ["any"]

    def compute_signal(self, symbol: str, data: dict) -> float:
        vol = data.get("volatility_60d", 0.0) or 0.0
        if vol > 0:
            return float(1.0 / vol)
        return 0.0
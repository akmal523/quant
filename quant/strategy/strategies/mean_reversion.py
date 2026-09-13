"""
strategies/mean_reversion.py — Mean Reversion strategy (Part 2, Upgrade #2).

Intent: buy oversold assets (RSI < 30) expecting a bounce. Dominates in
choppy/high-vol regimes. Signal scales with how oversold the asset is.
"""
from __future__ import annotations

from quant.strategy.strategies.base import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    name = "Mean Reversion"
    preferred_regime = ["choppy", "high_vol"]

    def compute_signal(self, symbol: str, data: dict) -> float:
        rsi = data.get("rsi_14", 50.0)
        if rsi < 30:
            return float((30 - rsi) / 30.0)
        return 0.0
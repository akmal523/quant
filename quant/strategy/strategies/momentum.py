"""
strategies/momentum.py — Momentum strategy (Part 2, Upgrade #2).

Intent: buy assets with strong 6-month momentum. Dominates in bull/trending
regimes. Signal = z-scored 6-month return.
"""
from __future__ import annotations

import numpy as np

from quant.strategy.strategies.base import BaseStrategy


class MomentumStrategy(BaseStrategy):
    name = "Momentum"
    preferred_regime = ["bull_low_vol", "trending"]

    def compute_signal(self, symbol: str, data: dict) -> float:
        returns_6m = data.get("returns_6m", 0.0)
        # Cross-sectional z-score proxy: normalize by a rolling std if present.
        std = data.get("returns_6m_std", 1.0) or 1.0
        if std <= 0:
            std = 1.0
        return float(returns_6m / std)
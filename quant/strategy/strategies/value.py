"""
strategies/value.py — Value strategy (Part 2, Upgrade #2).

Intent: buy cheap assets (low PE, high dividend yield). Dominates in
reflation/bear regimes. Signal = z-scored negative PE + z-scored dividend yield.
"""
from __future__ import annotations

from quant.strategy.strategies.base import BaseStrategy


class ValueStrategy(BaseStrategy):
    name = "Value"
    preferred_regime = ["reflation", "bear"]

    def compute_signal(self, symbol: str, data: dict) -> float:
        pe = data.get("pe_ratio", 0.0) or 0.0
        dividend_yield = data.get("dividend_yield", 0.0) or 0.0
        # Lower PE is better -> negate. Normalize by 25 (typical PE scale).
        pe_score = -pe / 25.0 if pe > 0 else 0.0
        dy_score = dividend_yield / 0.05 if dividend_yield > 0 else 0.0
        return float(pe_score + dy_score)
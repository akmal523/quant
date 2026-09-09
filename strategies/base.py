"""
strategies/base.py — Abstract strategy interface (Part 2, Upgrade #2).

Intent: every strategy exposes a name, a compute_signal(symbol, data) -> float
scorer, and a list of regimes it prefers. The StrategyEngine blends these
signals with regime-dependent weights so no single strategy dominates forever.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    name: str = "Base"
    preferred_regime: list[str] = ["any"]

    @abstractmethod
    def compute_signal(self, symbol: str, data: dict) -> float:
        """Return a signal score for the asset. Higher = more attractive."""
        raise NotImplementedError
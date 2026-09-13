"""
cash_manager.py — Dynamic Cash Reserve & Dip Buying (Part 2, Upgrade #4).

Intent: treat cash as a strategic asset, not residual. Trade Republic pays
2.5% APY on cash (from 16 Sep 2026), so holding cash earns yield AND provides
dry powder to buy dips aggressively. Target cash allocation scales with regime,
VIX, and the opportunity set. Dip-buying deploys cash proportionally to
drawdown depth.

Invariants:
  - target_cash_allocation returns a fraction in [0.05, 0.30].
  - dip_buying_algorithm returns a non-negative EUR amount <= cash_available.
  - Pure computation; no I/O.

Dependencies: numpy, quant.portfolio.cash_rate.
"""
from __future__ import annotations

import numpy as np

from quant.portfolio.cash_rate import current_cash_apy


class CashManager:
    """Manages cash as a strategic asset, not residual."""

    def __init__(self, broker_cash_apy: float | None = None):
        self.apy = broker_cash_apy if broker_cash_apy is not None else current_cash_apy()
        self.daily_yield = (1.0 + self.apy) ** (1.0 / 365.0) - 1.0

    def target_cash_allocation(self, regime: str, vix: float,
                               opportunity_score: float) -> float:
        """Target cash % based on market conditions.

        Returns: target cash fraction in [0.05, 0.30].
        """
        base_cash = {
            "bull_low_vol": 0.05,
            "high_vol_choppy": 0.15,
            "bear": 0.25,
            "reflation": 0.10,
        }.get(regime, 0.10)

        # Increase cash when VIX is elevated.
        vix_adjustment = max(0.0, (vix - 20.0) / 40.0) * 0.10
        # Decrease cash when opportunities are excellent.
        opp_adjustment = -min(opportunity_score * 0.05, 0.10)

        return float(np.clip(
            base_cash + vix_adjustment + opp_adjustment, 0.05, 0.30
        ))

    def dip_buying_algorithm(self, symbol: str, drawdown_pct: float,
                             cash_available: float) -> float:
        """Aggressive DCA when assets drop from recent highs.

        Buy size scales with drawdown magnitude. Returns EUR to deploy.
        """
        if drawdown_pct > -0.05:
            return 0.0  # no dip to buy

        if drawdown_pct < -0.20:
            buy_pct = 0.20
        elif drawdown_pct < -0.15:
            buy_pct = 0.15
        elif drawdown_pct < -0.10:
            buy_pct = 0.10
        elif drawdown_pct < -0.05:
            buy_pct = 0.05
        else:
            buy_pct = 0.0

        return float(cash_available * buy_pct)
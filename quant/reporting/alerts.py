"""
alerts.py — Rule-Based Alert System (Part 3, Addition #2).

Intent: surface important events (drawdowns, rebalancing triggers, tax-loss
harvesting opportunities) that would otherwise be buried in daily output.

Invariants:
  - check_* methods return lists of Alert.
  - Alert has level in {CRITICAL, WARNING, INFO, OPPORTUNITY}.
  - Pure computation; no I/O.

Dependencies: pandas, dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Alert:
    level: str  # CRITICAL | WARNING | INFO | OPPORTUNITY
    message: str
    action: str
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())


class AlertSystem:
    """Rule-based alert system for important events."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def check_drawdown(self, portfolio_returns: pd.Series) -> list[Alert]:
        """Alert on significant drawdown."""
        if portfolio_returns.empty:
            return []
        current_dd = (portfolio_returns.iloc[-1] / portfolio_returns.cummax().iloc[-1]) - 1

        if current_dd < -0.15:
            return [Alert(
                level="CRITICAL",
                message=f"Portfolio drawdown {current_dd:.1%} exceeds 15%",
                action="Consider reducing risk or hedging",
            )]
        if current_dd < -0.08:
            return [Alert(
                level="WARNING",
                message=f"Portfolio drawdown {current_dd:.1%} exceeds 8%",
                action="Monitor closely, consider trimming winners",
            )]
        return []

    def check_rebalance(self, portfolio_weights: dict,
                        target_weights: dict) -> list[Alert]:
        """Alert when rebalancing is needed."""
        alerts = []
        for symbol, target in target_weights.items():
            current = portfolio_weights.get(symbol, 0)
            drift = abs(current - target)
            if drift > 0.05:
                direction = "BUY" if current < target else "SELL"
                alerts.append(Alert(
                    level="INFO",
                    message=f"{symbol} drifted {drift:.1%} from target",
                    action=f"Rebalance: {direction} {abs(drift)*100:.0f}%",
                ))
        return alerts

    def check_tax_opportunity(self, portfolio_df: pd.DataFrame) -> list[Alert]:
        """Alert on tax-loss harvesting opportunities."""
        alerts = []
        for _, row in portfolio_df.iterrows():
            pnl = row.get("PnL_EUR", 0)
            if pnl < -50:
                alerts.append(Alert(
                    level="OPPORTUNITY",
                    message=f"{row['Symbol']} has EUR {abs(pnl):.0f} unrealized loss",
                    action="Consider selling to harvest tax loss",
                ))
        return alerts

    def run_all_checks(self, portfolio_data: dict) -> list[Alert]:
        """Run all alert checks."""
        alerts = []
        alerts.extend(self.check_drawdown(portfolio_data.get("returns", pd.Series(dtype=float))))
        alerts.extend(self.check_rebalance(
            portfolio_data.get("weights", {}),
            portfolio_data.get("target_weights", {}),
        ))
        alerts.extend(self.check_tax_opportunity(portfolio_data.get("holdings", pd.DataFrame())))
        return alerts
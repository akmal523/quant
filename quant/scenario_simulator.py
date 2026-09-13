"""
scenario_simulator.py — What-If Scenario Analysis (Part 3, Addition #1).

Intent: answer "what if" questions about the portfolio — sell AMZN buy EUNL,
market drops 20%, add gold — before committing capital. Simulates trades and
crashes, computing volatility/Sharpe impact and recommending hedges.

Invariants:
  - simulate_trade returns a dict with old/new vol and Sharpe.
  - simulate_crash returns portfolio impact and recommended hedges.
  - Pure computation; no I/O.

Dependencies: numpy, pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ScenarioSimulator:
    """Run what-if scenarios on the portfolio."""

    def __init__(self, portfolio_df: pd.DataFrame, returns_matrix: pd.DataFrame):
        self.portfolio = portfolio_df
        self.returns = returns_matrix

    def _compute_weights(self) -> pd.Series:
        total = self.portfolio["Amount_EUR"].sum()
        if total <= 0:
            return pd.Series(dtype=float)
        return self.portfolio.set_index("Symbol")["Amount_EUR"] / total

    def _compute_portfolio_vol(self, weights: pd.Series) -> float:
        clean = self.returns.dropna(axis=1, how="all").dropna()
        if clean.shape[0] < 2 or clean.shape[1] < 2:
            return 0.0
        cov = clean.cov() * 252
        w = weights.reindex(cov.index).fillna(0.0)
        if w.sum() <= 0:
            return 0.0
        return float(np.sqrt(w.T @ cov @ w))

    def _compute_sharpe(self, weights: pd.Series) -> float:
        clean = self.returns.dropna(axis=1, how="all").dropna()
        if clean.shape[0] < 2 or clean.shape[1] < 2:
            return 0.0
        w = weights.reindex(clean.columns).fillna(0.0)
        port_ret = (clean.mean() * w).sum() * 252
        vol = self._compute_portfolio_vol(weights)
        return float(port_ret / vol) if vol > 0 else 0.0

    def simulate_trade(self, sell_symbol: str, buy_symbol: str,
                       amount_eur: float) -> dict:
        """Simulate selling one asset and buying another."""
        total = self.portfolio["Amount_EUR"].sum()
        if total <= 0:
            return {}

        current_weights = self._compute_weights()
        new_weights = current_weights.copy()
        new_weights[sell_symbol] = new_weights.get(sell_symbol, 0) - amount_eur / total
        new_weights[buy_symbol] = new_weights.get(buy_symbol, 0) + amount_eur / total

        old_vol = self._compute_portfolio_vol(current_weights)
        new_vol = self._compute_portfolio_vol(new_weights)
        old_sharpe = self._compute_sharpe(current_weights)
        new_sharpe = self._compute_sharpe(new_weights)

        return {
            "old_volatility": round(old_vol, 4),
            "new_volatility": round(new_vol, 4),
            "vol_change": round(new_vol - old_vol, 4),
            "old_sharpe": round(old_sharpe, 3),
            "new_sharpe": round(new_sharpe, 3),
            "fee_cost": 2.0,
            "recommendation": (
                "PROCEED" if new_sharpe > old_sharpe else "SKIP"
            ),
        }

    def simulate_crash(self, crash_pct: float = -0.20) -> dict:
        """Simulate a market crash and show portfolio impact."""
        weights = self._compute_weights()
        if weights.empty:
            return {}

        # Apply crash to all positions (uniform shock).
        portfolio_impact = crash_pct * weights.sum()
        hedges = self._find_best_hedges(crash_pct)

        return {
            "portfolio_impact": round(float(portfolio_impact), 4),
            "max_drawdown": round(crash_pct, 4),
            "recommended_hedges": hedges,
        }

    def _find_best_hedges(self, crash_pct: float) -> list[dict]:
        """Find assets that perform well in crashes (negative correlation)."""
        if "SPX" not in self.returns.columns:
            return []
        crash_performers = []
        for symbol in self.returns.columns:
            if symbol == "SPX":
                continue
            corr = self.returns[symbol].corr(self.returns["SPX"])
            if pd.notna(corr) and corr < -0.3:
                crash_performers.append({
                    "symbol": symbol,
                    "correlation": round(float(corr), 3),
                    "expected_return_in_crash": round(float(crash_pct * corr * -1), 4),
                })
        return sorted(
            crash_performers, key=lambda x: x["expected_return_in_crash"], reverse=True
        )[:5]
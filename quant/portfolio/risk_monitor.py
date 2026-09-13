"""
risk_monitor.py — Drawdown Circuit Breakers (Part 2, Upgrade #5).

Intent: portfolio-level circuit breakers that override all signals during a
crash. In a real drawdown (2008, 2020) the bot must halt new longs, raise cash,
and consider hedges — not continue normal operations while the portfolio
collapses.

Invariants:
  - check_circuit_breakers returns a dict with status in
    {NORMAL, CAUTION, ALERT, LOCKDOWN}.
  - compute_current_drawdown returns a negative fraction (or 0).
  - Pure computation; no I/O.

Dependencies: numpy, pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import (
    MAX_DAILY_DRAWDOWN, VOL_KILL_MULTIPLIER, TARGET_VOLATILITY,
)


class RiskMonitor:
    """Portfolio-level circuit breakers that override all signals."""

    def __init__(self, portfolio_returns: pd.Series):
        self.returns = portfolio_returns
        self.peak_value = portfolio_returns.cummax()

    def compute_current_drawdown(self) -> float:
        """Current drawdown from peak (negative fraction)."""
        if self.returns.empty:
            return 0.0
        current_value = self.returns.iloc[-1]
        peak = self.peak_value.iloc[-1]
        if peak <= 0:
            return 0.0
        return float((current_value - peak) / peak)

    def compute_var_95(self, lookback_days: int = 252) -> float:
        """95% Value at Risk (historical), daily."""
        returns_series = self.returns.pct_change().dropna()
        if returns_series.empty:
            return 0.0
        return float(np.percentile(returns_series.tail(lookback_days), 5))

    def compute_annualized_vol(self) -> float:
        """20-day rolling annualized volatility."""
        pct = self.returns.pct_change().dropna()
        if len(pct) < 2:
            return 0.0
        return float(pct.rolling(20).std().iloc[-1] * np.sqrt(252))

    def check_circuit_breakers(self) -> dict:
        """Return circuit breaker status and recommended actions."""
        dd = self.compute_current_drawdown()
        var = self.compute_var_95()
        vol = self.compute_annualized_vol()

        if dd < -0.20:
            return {
                "status": "LOCKDOWN",
                "message": "Portfolio drawdown >20%. HALT all new long positions.",
                "actions": [
                    "Cancel all pending BUY orders",
                    "Increase cash to 30%",
                    "Consider inverse ETF hedges (SH, SDS)",
                ],
                "drawdown": dd,
                "var_95": var,
                "vol": vol,
            }
        if dd < -0.12 or vol > 0.25:
            return {
                "status": "ALERT",
                "message": "Significant drawdown or elevated volatility.",
                "actions": [
                    "Reduce position sizes by 50%",
                    "Increase cash to 20%",
                    "Tighten stop-losses",
                ],
                "drawdown": dd,
                "var_95": var,
                "vol": vol,
            }
        if dd < -0.07 or vol > 0.20:
            return {
                "status": "CAUTION",
                "message": "Moderate risk — proceed carefully.",
                "actions": [
                    "Increase cash to 15%",
                    "Only highest-conviction trades",
                ],
                "drawdown": dd,
                "var_95": var,
                "vol": vol,
            }
        return {
            "status": "NORMAL",
            "message": "All clear.",
            "actions": [],
            "drawdown": dd,
            "var_95": var,
            "vol": vol,
        }

    def compute_daily_drawdown(self) -> float:
        """Single-day return of the portfolio value series (negative = loss).

        Intent: the kill switch reacts to a fast, single-day loss, distinct from
        the peak-to-trough drawdown used by the advisory circuit breakers.
        Invariants: returns a fraction; 0.0 if insufficient data.
        """
        if self.returns is None or len(self.returns) < 2:
            return 0.0
        prev = float(self.returns.iloc[-2])
        curr = float(self.returns.iloc[-1])
        if prev <= 0:
            return 0.0
        return (curr - prev) / prev

    def check_kill_switch(self) -> dict:
        """Hard circuit breaker: LIQUIDATE TO CASH and pause the scanner.

        Intent (v10.4.0, Phase 3): a hard stop, not an advisory status. If the
        portfolio breaches MAX_DAILY_DRAWDOWN or realized vol exceeds
        VOL_KILL_MULTIPLIER x target, the system must liquidate to cash and halt
        new signals. Pure computation (no I/O); the caller publishes the event.
        Invariants: returns dict with triggered (bool) and signal (str).
        """
        daily_dd = self.compute_daily_drawdown()
        vol = self.compute_annualized_vol()
        vol_limit = TARGET_VOLATILITY * VOL_KILL_MULTIPLIER

        reasons = []
        if daily_dd < -MAX_DAILY_DRAWDOWN:
            reasons.append(
                f"daily drawdown {daily_dd:.1%} < -{MAX_DAILY_DRAWDOWN:.1%}"
            )
        if vol > vol_limit:
            reasons.append(f"realized vol {vol:.1%} > {vol_limit:.1%}")

        triggered = len(reasons) > 0
        return {
            "triggered": triggered,
            "signal": "LIQUIDATE TO CASH" if triggered else "HOLD",
            "reason": "; ".join(reasons) if reasons else "within hard limits",
            "daily_drawdown": daily_dd,
            "vol": vol,
            "vol_limit": vol_limit,
        }
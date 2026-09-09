"""
behavioral_guardrails.py — Overtrading Guardrails (Part 2, Upgrade #8).

Intent: prevent common algorithmic trading mistakes that create fee drag and
psychological noise. Enforces per-symbol cooldowns, a weekly trade limit, and
position-size reduction after consecutive losses.

Invariants:
  - check_cooldown / check_weekly_limit / check_profit_run return (bool, reason).
  - register_trade sets a cooldown and increments the weekly counter.
  - Pure logic; no I/O (state held in-memory).

Dependencies: pandas.
"""
from __future__ import annotations

import pandas as pd


class BehavioralGuardrails:
    """Prevents common algorithmic trading mistakes."""

    def __init__(self):
        self.cooldown_registry: dict[str, pd.Timestamp] = {}
        self.weekly_trade_count = 0
        self.weekly_start = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
        self.trade_history: list[dict] = []

    def check_cooldown(self, symbol: str) -> tuple[bool, str]:
        """Prevent re-trading the same asset within the cooldown window."""
        if symbol in self.cooldown_registry:
            next_allowed = self.cooldown_registry[symbol]
            if pd.Timestamp.now() < next_allowed:
                days_remaining = (next_allowed - pd.Timestamp.now()).days
                return False, f"Cooldown active: {days_remaining} days remaining"
        return True, "OK"

    def check_weekly_limit(self, max_trades: int = 5) -> tuple[bool, str]:
        """Prevent more than N trades per week."""
        if self.weekly_trade_count >= max_trades:
            return False, f"Weekly trade limit reached ({max_trades})"
        return True, "OK"

    def check_profit_run(self, consecutive_losses: int = 3) -> tuple[bool, str]:
        """Reduce size after consecutive losses."""
        recent = self.trade_history[-consecutive_losses:]
        if len(recent) == consecutive_losses and all(t["pnl"] < 0 for t in recent):
            return False, f"Reduce size: {consecutive_losses} consecutive losses"
        return True, "OK"

    def register_trade(self, symbol: str, pnl: float = 0.0,
                       cooldown_days: int = 7) -> None:
        """Register a completed trade, set cooldown, and log PnL."""
        self.cooldown_registry[symbol] = pd.Timestamp.now() + pd.Timedelta(days=cooldown_days)
        self.weekly_trade_count += 1
        self.trade_history.append({"symbol": symbol, "pnl": pnl})
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

from quant.config import MAX_ALPHA_TRADES_PER_WEEK


class BehavioralGuardrails:
    """Prevents common algorithmic trading mistakes."""

    def __init__(self):
        self.cooldown_registry: dict[str, pd.Timestamp] = {}
        self.weekly_trade_count = 0
        self.weekly_start = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
        self.trade_history: list[dict] = []
        # v10.6.2: Alpha trades are capped separately (max 2 per week).
        self.alpha_trade_count = 0

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

    # ── v10.6.2: Alpha weekly cap (max 2 trades per week) ────────────────────

    def check_alpha_weekly_limit(
        self,
        trades_this_week: int | None = None,
        max_trades: int = MAX_ALPHA_TRADES_PER_WEEK,
        override: bool = False,
    ) -> dict:
        """Check the weekly Alpha trade cap.

        Intent (v10.6.3): the Alpha tier rebalances weekly; the cap prevents
        overtrading. Returns a dict
        ``{allowed, trades_remaining, warning, override_required}``. When
        ``trades_this_week`` is None the instance counter is used. Invariants:
        never raises; pure.
        """
        count = self.alpha_trade_count if trades_this_week is None else int(trades_this_week)
        remaining = max(0, int(max_trades) - count)
        if count >= int(max_trades) and not override:
            return {
                "allowed": False,
                "trades_remaining": 0,
                "warning": (
                    f"Weekly Alpha trade limit reached ({count}/{max_trades}). "
                    f"Overtrading increases transaction costs and reduces returns. "
                    f"Wait until next Friday or override with explicit confirmation."
                ),
                "override_required": True,
            }
        return {
            "allowed": True,
            "trades_remaining": remaining,
            "warning": None,
            "override_required": False,
        }

    def register_alpha_trade(self, symbol: str, pnl: float = 0.0) -> None:
        """Register an Alpha trade and increment the weekly Alpha counter."""
        self.alpha_trade_count += 1
        self.register_trade(symbol, pnl=pnl)


def track_weekly_trades(as_of: str) -> int:
    """Count trades executed this week (Monday to as_of) from trade_log.

    Intent (v10.6.3): the weekly cap needs a persistent count across sessions.
    Reads the DuckDB ``trade_log`` table. Invariants: returns an int; never
    raises; 0 when the table is missing or empty.
    """
    try:
        from quant.data.database import get_connection
        conn = get_connection()
        rows = conn.execute("SELECT ts FROM trade_log").fetchall()
    except Exception:  # noqa: BLE001
        return 0
    try:
        as_of_date = pd.Timestamp(as_of).normalize()
    except Exception:  # noqa: BLE001
        return 0
    week_start = as_of_date - pd.Timedelta(days=as_of_date.weekday())
    count = 0
    for (ts,) in rows:
        try:
            d = pd.Timestamp(ts).normalize()
        except Exception:  # noqa: BLE001
            continue
        if week_start <= d <= as_of_date:
            count += 1
    return count

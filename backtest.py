"""
backtest.py — 12-month lookback backtest. No lookahead bias.

Entry rule  : RSI(14) in [30, 65] AND Close ≥ EMA50 × 0.97 (12 months ago)
Stop-loss   : entry_price − ATR(14) × 2.5
Exit price  : today's Close
P&L         : (exit − entry) / entry × 100

No transaction costs, slippage, or taxes. Results are hypothetical.
"""
from __future__ import annotations
import pandas as pd
from indicators import rsi as calc_rsi, atr

_EMPTY = {"Backtest_PnL_pct": None, "Backtest_StopHit": False, "Backtest_Signal": "N/A"}


def run_historical_backtest(hist: pd.DataFrame) -> dict:
    """
    Apply the entry rule on data from ~252 trading days ago.
    Returns a result dict safe to merge into the main DataFrame.
    """
    if hist is None or len(hist) < 280:
        return _EMPTY

    try:
        # Slice history up to the signal date (252 bars ago)
        entry_idx  = max(0, len(hist) - 252)
        hist_past  = hist.iloc[: entry_idx + 1]

        if len(hist_past) < 20:
            return _EMPTY

        rsi_val     = calc_rsi(hist_past["Close"], 14)
        ema50       = hist_past["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
        entry_price = float(hist_past["Close"].iloc[-1])

        # Entry condition
        if not (
            rsi_val is not None
            and 30 <= rsi_val <= 65
            and entry_price >= float(ema50) * 0.97
        ):
            return {**_EMPTY, "Backtest_Signal": "NO_ENTRY"}

        # Stop-loss level
        atr_val    = atr(hist_past["High"], hist_past["Low"], hist_past["Close"], 14)
        stop_price = (entry_price - atr_val * 2.5) if atr_val else None

        # Check stop-hit over holding period
        holding  = hist.iloc[entry_idx:]
        stop_hit = False
        if stop_price is not None and not holding.empty:
            stop_hit = bool((holding["Low"] < stop_price).any())

        exit_price = float(hist["Close"].iloc[-1])
        pnl        = (exit_price - entry_price) / entry_price * 100 if entry_price else 0.0

        return {
            "Backtest_PnL_pct":  round(pnl, 2),
            "Backtest_StopHit":  stop_hit,
            "Backtest_Signal":   "BUY",
        }

    except Exception:
        return _EMPTY

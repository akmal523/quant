"""
backtest.py — Backtesting Suite v7.

New in this version:
  1. Walk-Forward Optimization (WFO): rolling IS/OOS windows prove strategy
     generalizes beyond the training period. Replaces single fixed-window test.
  2. COMMISSION_SLIPPAGE centralized in config.py (15 bps per round-trip).
  3. Survivorship bias advisory emitted on every WFO call — the universe contains
     only currently-listed instruments. Delisted/bankrupt companies are structurally
     absent, causing systematic overstatement of historical returns.

Functions:
  run_macro_backtest()       — full-history rolling simulation (win rate, avg PnL)
  run_historical_backtest()  — single 30-day window entry check
  walk_forward_optimization()— IS/OOS rolling validation (primary quality gate)
  _run_window_trades()       — shared trade-execution kernel
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import COMMISSION_SLIPPAGE, WFO_IS_DAYS, WFO_OOS_DAYS, WFO_STEP_DAYS
from indicators import rsi as calc_rsi, atr

logger = logging.getLogger(__name__)

# ── Survivorship Bias Warning ─────────────────────────────────────────────────
# This string is attached to every WFO result dict so callers and reports
# surface the advisory without requiring separate documentation.
_SB_WARNING = (
    "SURVIVORSHIP BIAS: Universe contains only currently-listed instruments. "
    "Companies that were delisted, went bankrupt, or were acquired during the "
    "backtest period are absent from SECTOR_UNIVERSE. Historical PnL figures "
    "are structurally overstated. For unbiased results, replace the universe "
    "with a point-in-time snapshot database (e.g., Compustat, QuantQuote)."
)


# ── Shared Trade Execution Kernel ─────────────────────────────────────────────

def _run_window_trades(window: pd.DataFrame) -> list[float]:
    """
    Execute the entry/exit logic on a given price window.
    Entry: RSI in [30, 65] AND price >= EMA50 * 0.97
    Exit:  Stop (price crosses ATR-based stop) OR RSI > 75

    Returns list of net PnL per trade (after COMMISSION_SLIPPAGE).
    """
    trades:      list[float] = []
    in_trade:    bool        = False
    entry_price: float       = 0.0
    stop_price:  float       = 0.0

    if len(window) < 60:
        return trades

    for i in range(50, len(window)):
        current_row = window.iloc[i]
        hist_past   = window.iloc[: i + 1]

        if not in_trade:
            rsi_val = calc_rsi(hist_past["Close"], 14)
            ema50   = hist_past["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
            price   = float(current_row["Close"])

            if rsi_val and 30 <= rsi_val <= 65 and price >= float(ema50) * 0.97:
                in_trade    = True
                entry_price = price
                atr_val     = atr(hist_past["High"], hist_past["Low"], hist_past["Close"], 14)
                stop_price  = (entry_price - atr_val * 2.5) if atr_val else (entry_price * 0.85)
        else:
            rsi_val = calc_rsi(hist_past["Close"], 14)

            if float(current_row["Low"]) < stop_price:
                exit_price = stop_price
            elif rsi_val and rsi_val > 75:
                exit_price = float(current_row["Close"])
            else:
                continue

            gross = (exit_price - entry_price) / entry_price
            trades.append(gross - COMMISSION_SLIPPAGE)
            in_trade = False

    return trades


# ── Walk-Forward Optimization ─────────────────────────────────────────────────

def walk_forward_optimization(hist: pd.DataFrame) -> dict:
    """
    Walk-Forward Optimization (WFO).

    Methodology:
      The dataset is divided into overlapping IS/OOS window pairs that roll
      forward by WFO_STEP_DAYS on each iteration.

      In-Sample  (IS):  strategy is evaluated — only windows with a detectable
                        positive edge (win_rate >= 45%) are accepted.
      Out-of-Sample (OOS): the strategy is applied blindly to unseen data.
                        This is the *only* performance metric that matters.

    Why WFO matters:
      A single fixed-window backtest can fit the strategy parameters to the
      historical noise of that specific period. WFO forces the strategy to
      prove its edge on data it has never seen — approximating live deployment.

    Survivorship bias caveat:
      Even with WFO, the universe itself is survivorship-biased. The bias
      operates at the data-selection level, not the window-selection level.
      WFO cannot correct for the absence of delisted instruments.
    """
    empty = {
        "wfo_oos_avg_pnl":    None,
        "wfo_oos_win_rate":   None,
        "wfo_periods":        0,
        "wfo_oos_trades":     0,
        "survivorship_bias_warning": _SB_WARNING,
    }

    min_required = WFO_IS_DAYS + WFO_OOS_DAYS
    if hist is None or len(hist) < min_required:
        return empty

    oos_trades_all: list[float] = []
    period_count:   int         = 0
    cursor:         int         = 0

    while cursor + WFO_IS_DAYS + WFO_OOS_DAYS <= len(hist):
        is_slice  = hist.iloc[cursor : cursor + WFO_IS_DAYS]
        oos_slice = hist.iloc[cursor + WFO_IS_DAYS : cursor + WFO_IS_DAYS + WFO_OOS_DAYS]

        # IS phase: confirm the strategy has a detectable edge
        is_trades = _run_window_trades(is_slice)
        if len(is_trades) < 2:
            cursor += WFO_STEP_DAYS
            continue

        is_win_rate = len([t for t in is_trades if t > 0]) / len(is_trades)
        if is_win_rate < 0.45:
            # No edge detected in this IS window — skip OOS deployment
            logger.debug("[WFO] IS win_rate %.1f%% below threshold — skipping OOS at cursor %d",
                         is_win_rate * 100, cursor)
            cursor += WFO_STEP_DAYS
            continue

        # OOS phase: blind test
        oos_trades = _run_window_trades(oos_slice)
        oos_trades_all.extend(oos_trades)
        period_count += 1

        cursor += WFO_STEP_DAYS

    if not oos_trades_all:
        return {**empty, "wfo_periods": period_count}

    return {
        "wfo_oos_avg_pnl":  round(float(np.mean(oos_trades_all)) * 100, 2),
        "wfo_oos_win_rate": round(
            len([t for t in oos_trades_all if t > 0]) / len(oos_trades_all) * 100, 1
        ),
        "wfo_periods":      period_count,
        "wfo_oos_trades":   len(oos_trades_all),
        "survivorship_bias_warning": _SB_WARNING,
    }


# ── Rolling Macro Backtest ────────────────────────────────────────────────────

def run_macro_backtest(hist: pd.DataFrame) -> dict:
    """
    Full-history rolling simulation.
    Calculates win rate and average PnL across all detected trade entries.
    Used as a quick sanity check; prefer WFO for strategy validation.
    """
    empty = {"BT_Trades": 0, "BT_WinRate_pct": 0.0, "BT_Avg_PnL_pct": 0.0}
    if hist is None or len(hist) < 200:
        return empty

    trades = _run_window_trades(hist)

    if not trades:
        return empty

    return {
        "BT_Trades":     len(trades),
        "BT_WinRate_pct": round(
            len([t for t in trades if t > 0]) / len(trades) * 100, 1
        ),
        "BT_Avg_PnL_pct": round(float(np.mean(trades)) * 100, 2),
    }


# ── Historical Window Backtest ────────────────────────────────────────────────

def run_historical_backtest(
    hist: pd.DataFrame,
    window_start: int = 260,
    window_end:   int = 230,
) -> dict:
    """
    Scans a 30-day historical window (~1 year ago) for the first valid entry.
    Tracks the trade from entry to present price or stop.

    Transaction costs (COMMISSION_SLIPPAGE) applied to PnL.
    """
    _EMPTY = {
        "Backtest_PnL_pct":   None,
        "Backtest_StopHit":   False,
        "Backtest_Signal":    "N/A",
        "Backtest_Entry_Date": None,
    }

    if hist is None or len(hist) < window_start + 20:
        return _EMPTY

    entry_idx = None
    for i in range(len(hist) - window_start, len(hist) - window_end):
        hist_past = hist.iloc[: i + 1]
        rsi_val   = calc_rsi(hist_past["Close"], 14)
        ema50     = hist_past["Close"].ewm(span=50, adjust=False).mean().iloc[-1]

        if rsi_val and 30 <= rsi_val <= 65 and float(hist_past["Close"].iloc[-1]) >= float(ema50) * 0.97:
            entry_idx = i
            break

    if entry_idx is None:
        return {**_EMPTY, "Backtest_Signal": "NO_WINDOW_ENTRY"}

    entry_price = float(hist.iloc[entry_idx]["Close"])
    atr_val     = atr(
        hist.iloc[: entry_idx + 1]["High"],
        hist.iloc[: entry_idx + 1]["Low"],
        hist.iloc[: entry_idx + 1]["Close"],
        14,
    )
    stop_price = (entry_price - atr_val * 2.5) if atr_val else None

    exit_price, stop_hit = float(hist["Close"].iloc[-1]), False
    if stop_price:
        for _, row in hist.iloc[entry_idx + 1 :].iterrows():
            if float(row["Low"]) < stop_price:
                exit_price, stop_hit = stop_price, True
                break

    gross_return = (exit_price - entry_price) / entry_price
    return {
        "Backtest_PnL_pct":    round((gross_return - COMMISSION_SLIPPAGE) * 100, 2),
        "Backtest_StopHit":    stop_hit,
        "Backtest_Signal":     "EXIT_STOP" if stop_hit else "EXIT_CURRENT",
        "Backtest_Entry_Date": str(hist.index[entry_idx].date()),
    }

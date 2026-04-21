"""
backtest.py — Backtesting Suite v6.
Contains:
1. Macro-regime rolling backtest (Multi-year simulation).
2. Historical window backtest (30-day entry window check).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from indicators import rsi as calc_rsi, atr

# Real-world friction (slippage + fees per round trip)
COMMISSION_SLIPPAGE = 0.0015 

def run_macro_backtest(hist: pd.DataFrame) -> dict:
    """
    Executes a rolling backtest over the entire historical dataset.
    Calculates Win Rate and Avg PnL across multiple years/regimes.
    """
    results_empty = {"BT_Trades": 0, "BT_WinRate_pct": 0.0, "BT_Avg_PnL_pct": 0.0}
    if hist is None or len(hist) < 200:
        return results_empty

    trades = []
    in_trade = False
    entry_price, stop_price = 0.0, 0.0
    
    for i in range(50, len(hist)):
        current_row = hist.iloc[i]
        hist_past = hist.iloc[: i + 1]

        if not in_trade:
            rsi_val = calc_rsi(hist_past["Close"], 14)
            ema50 = current_row["EMA50"] if "EMA50" in current_row else hist_past["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
            price = float(current_row["Close"])

            if rsi_val and 30 <= rsi_val <= 65 and price >= float(ema50) * 0.97:
                in_trade = True
                entry_price = price
                atr_val = atr(hist_past["High"], hist_past["Low"], hist_past["Close"], 14)
                stop_price = (entry_price - atr_val * 2.5) if atr_val else (entry_price * 0.85)
        else:
            if float(current_row["Low"]) < stop_price:
                exit_price = stop_price
            elif calc_rsi(hist_past["Close"], 14) > 75:
                exit_price = float(current_row["Close"])
            else:
                continue

            trades.append(((exit_price - entry_price) / entry_price) - COMMISSION_SLIPPAGE)
            in_trade = False

    if not trades: return results_empty
    return {
        "BT_Trades": len(trades),
        "BT_WinRate_pct": round((len([t for t in trades if t > 0]) / len(trades)) * 100, 1),
        "BT_Avg_PnL_pct": round(np.mean(trades) * 100, 2)
    }

def run_historical_backtest(hist: pd.DataFrame, window_start: int = 260, window_end: int = 230) -> dict:
    """
    Scans a 30-day window for the first valid entry signal and tracks the trade.
    Focuses on a specific historical point (approx. 1 year ago).
    """
    _EMPTY = {"Backtest_PnL_pct": None, "Backtest_StopHit": False, "Backtest_Signal": "N/A", "Backtest_Entry_Date": None}
    if hist is None or len(hist) < window_start + 20:
        return _EMPTY

    entry_idx = None
    for i in range(len(hist) - window_start, len(hist) - window_end):
        hist_past = hist.iloc[: i + 1]
        rsi_val = calc_rsi(hist_past["Close"], 14)
        ema50 = hist_past["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
        if rsi_val and 30 <= rsi_val <= 65 and float(hist_past["Close"].iloc[-1]) >= float(ema50) * 0.97:
            entry_idx = i
            break

    if entry_idx is None: return {**_EMPTY, "Backtest_Signal": "NO_WINDOW_ENTRY"}

    entry_price = float(hist.iloc[entry_idx]["Close"])
    atr_val = atr(hist.iloc[:entry_idx+1]["High"], hist.iloc[:entry_idx+1]["Low"], hist.iloc[:entry_idx+1]["Close"], 14)
    stop_price = (entry_price - atr_val * 2.5) if atr_val else None

    exit_price, stop_hit = float(hist["Close"].iloc[-1]), False
    if stop_price:
        for _, row in hist.iloc[entry_idx + 1:].iterrows():
            if float(row["Low"]) < stop_price:
                exit_price, stop_hit = stop_price, True
                break

    return {
        "Backtest_PnL_pct": round(((exit_price - entry_price) / entry_price - COMMISSION_SLIPPAGE) * 100, 2),
        "Backtest_StopHit": stop_hit,
        "Backtest_Signal": "EXIT_STOP" if stop_hit else "EXIT_CURRENT",
        "Backtest_Entry_Date": str(hist.index[entry_idx].date())
    }

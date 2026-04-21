import pandas as pd
import numpy as np
from indicators import rsi as calc_rsi, atr

_EMPTY = {
    "Backtest_PnL_pct": None, 
    "Backtest_StopHit": False, 
    "Backtest_Signal": "N/A",
    "Backtest_Entry_Date": None
}

# Real-world friction (slippage + fees)
COMMISSION_SLIPPAGE = 0.0015 

def run_historical_backtest(hist: pd.DataFrame, window_start: int = 260, window_end: int = 230) -> dict:
    """
    Scans a 30-day window for the first valid entry signal and tracks the trade.
    """
    if hist is None or len(hist) < window_start + 20:
        return _EMPTY

    # 1. Identify first valid entry within the window
    entry_idx = None
    for i in range(len(hist) - window_start, len(hist) - window_end):
        hist_past = hist.iloc[: i + 1]
        
        rsi_val = calc_rsi(hist_past["Close"], 14)
        ema50 = hist_past["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
        price = float(hist_past["Close"].iloc[-1])

        # Entry Rule
        if rsi_val and 30 <= rsi_val <= 65 and price >= float(ema50) * 0.97:
            entry_idx = i
            break

    if entry_idx is None:
        return {**_EMPTY, "Backtest_Signal": "NO_WINDOW_ENTRY"}

    # 2. Establish Entry Parameters
    entry_price = float(hist.iloc[entry_idx]["Close"])
    entry_date = hist.index[entry_idx]
    
    hist_at_entry = hist.iloc[: entry_idx + 1]
    atr_val = atr(hist_at_entry["High"], hist_at_entry["Low"], hist_at_entry["Close"], 14)
    stop_price = (entry_price - atr_val * 2.5) if atr_val else None

    # 3. Simulate Trade Path
    holding_period = hist.iloc[entry_idx + 1:]
    exit_price = float(hist["Close"].iloc[-1])
    stop_hit = False
    
    if stop_price is not None:
        for _, row in holding_period.iterrows():
            if float(row["Low"]) < stop_price:
                exit_price = stop_price
                stop_hit = True
                break

    # 4. Calculate Net Return (Gross - Friction)
    raw_return = (exit_price - entry_price) / entry_price
    net_return = raw_return - COMMISSION_SLIPPAGE

    return {
        "Backtest_PnL_pct": round(net_return * 100, 2),
        "Backtest_StopHit": stop_hit,
        "Backtest_Signal": "EXIT_STOP" if stop_hit else "EXIT_CURRENT",
        "Backtest_Entry_Date": str(entry_date.date())
    }

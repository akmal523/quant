"""
backtest.py — 12-month look-back backtest (no lookahead bias).
Currency normalisation (GBp → GBP) is applied before P&L calculation.
"""
import pandas as pd
from datetime import datetime, timedelta

from config     import ATR_MULTIPLIER
from indicators import ema, rsi, atr


def run_backtest(symbol: str, hist: pd.DataFrame,
                 currency: str = "USD") -> dict:
    """
    Simulate what signal would have been generated 12 months ago,
    then measure the resulting P&L to the current price.

    Parameters
    ----------
    symbol   : Yahoo Finance ticker (used only for labelling).
    hist     : Full OHLCV history from yf.download().
    currency : Raw yfinance currency string (e.g. "GBp", "EUR", "USD").
               Pence tickers are divided by 100 so P&L units match display units.

    Returns
    -------
    dict with keys: bt_signal, bt_price_entry, bt_price_now,
                    bt_pnl_pct, bt_stop_hit, bt_note.
    """
    result = {
        "bt_signal":      "N/A",
        "bt_price_entry": None,
        "bt_price_now":   None,
        "bt_pnl_pct":     None,
        "bt_stop_hit":    False,
        "bt_note":        "",
    }

    if hist is None or hist.empty or "Close" not in hist.columns:
        result["bt_note"] = "Insufficient history"
        return result

    close = hist["Close"].squeeze()
    high  = hist["High"].squeeze() if "High" in hist.columns else close.copy()
    low   = hist["Low"].squeeze()  if "Low"  in hist.columns else close.copy()

    # Normalise London pence tickers (GBp / GBX) to pounds.
    if currency in ("GBp", "GBX"):
        close = close / 100
        high  = high  / 100
        low   = low   / 100

    target_date = pd.Timestamp(datetime.now() - timedelta(days=365))
    past_dates  = close.index[close.index <= target_date]

    if len(past_dates) < 60:
        result["bt_note"] = "Insufficient history"
        return result

    cutoff_loc = close.index.get_loc(past_dates[-1])
    c_past = close.iloc[: cutoff_loc + 1]
    h_past = high.iloc[:  cutoff_loc + 1]
    l_past = low.iloc[:   cutoff_loc + 1]

    if len(c_past) < 20:
        result["bt_note"] = "Insufficient history"
        return result

    rsi_past   = rsi(c_past)
    ema50_past = float(ema(c_past, 50).iloc[-1])
    price_past = float(c_past.iloc[-1])
    atr_past   = atr(h_past, l_past, c_past)
    stop_past  = (price_past - ATR_MULTIPLIER * atr_past) if atr_past else None

    # Reconstruct the signal that would have been generated 12 months ago.
    if rsi_past and 30 <= rsi_past <= 65 and price_past >= ema50_past * 0.97:
        past_sig = "BUY"
    elif rsi_past and rsi_past > 72:
        past_sig = "SELL"
    else:
        past_sig = "HOLD"

    price_now = float(close.iloc[-1])
    pnl       = (price_now - price_past) / price_past * 100

    # Check if stop-loss was triggered during the holding period.
    stop_hit = False
    if stop_past and past_sig == "BUY":
        lows_since = low.iloc[cutoff_loc:]
        if not lows_since.empty and float(lows_since.min()) < stop_past:
            stop_hit = True

    if past_sig == "BUY":
        note = (
            f"BUY signal → {pnl:+.1f}% "
            f"({'Stop hit!' if stop_hit else 'Position open'})"
        )
    else:
        note = f"Signal was {past_sig}. Hypothetical hold: {pnl:+.1f}%"

    result.update({
        "bt_signal":      past_sig,
        "bt_price_entry": round(price_past, 4),
        "bt_price_now":   round(price_now,  4),
        "bt_pnl_pct":     round(pnl, 1),
        "bt_stop_hit":    stop_hit,
        "bt_note":        note,
    })
    return result

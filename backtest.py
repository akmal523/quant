"""
backtest.py — multi-entry 12-month look-back backtest (no lookahead bias).

Changes from v4.5:
- Multiple entry points: scans every monthly bar over the past 12 months for
  valid BUY signals, not just the single bar from exactly 12 months ago.
- ATR-based position sizing: each entry is weighted inversely to its ATR
  (volatility-adjusted), so high-vol entries get smaller size.
- Aggregate statistics returned per symbol:
    win_rate       — fraction of entries that closed positive
    avg_pnl_pct    — equal-weighted average P&L across entries
    atr_wtd_pnl    — ATR-position-sized weighted P&L
    sharpe_approx  — mean / std of trade P&Ls (annualised proxy)
    n_trades       — number of triggered entries
- Backwards-compatible dict keys retained for the main result row;
  multi-entry detail available under bt_trades (list of dicts).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import ATR_MULTIPLIER
from indicators import ema, rsi, atr


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY SIGNAL (same rule as live scanner for consistency)
# ─────────────────────────────────────────────────────────────────────────────

def _entry_signal(rsi_val, ema50_val, price) -> str:
    if rsi_val is None:
        return "HOLD"
    if 30 <= rsi_val <= 65 and price >= ema50_val * 0.97:
        return "BUY"
    if rsi_val > 72:
        return "SELL"
    return "HOLD"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(symbol: str, hist: pd.DataFrame,
                 currency: str = "USD") -> dict:
    """
    Scan monthly entry points over the past 12 months and aggregate results.

    Returns a flat dict for the output DataFrame row plus a bt_trades list.
    """
    empty = dict(
        bt_signal="N/A", bt_price_entry=None, bt_price_now=None,
        bt_pnl_pct=None, bt_stop_hit=False, bt_note="",
        bt_n_trades=0, bt_win_rate=None, bt_avg_pnl=None,
        bt_atr_wtd_pnl=None, bt_sharpe=None, bt_trades=[],
    )

    if hist is None or hist.empty or "Close" not in hist.columns:
        empty["bt_note"] = "Insufficient history"
        return empty

    close = hist["Close"].squeeze()
    high = hist["High"].squeeze() if "High" in hist.columns else close.copy()
    low  = hist["Low"].squeeze()  if "Low"  in hist.columns else close.copy()

    if currency in ("GBp", "GBX", "GBx"):
        close, high, low = close / 100, high / 100, low / 100

    now = pd.Timestamp(datetime.now())
    start_window = now - timedelta(days=365)

    # Resample to monthly end-of-month bars for entry scanning
    monthly_dates = close.resample("ME").last().index
    monthly_dates = monthly_dates[
        (monthly_dates >= start_window) & (monthly_dates < now)
    ]

    if len(monthly_dates) < 2:
        empty["bt_note"] = "Insufficient monthly bars"
        return empty

    price_now = float(close.iloc[-1])
    trades: list[dict] = []

    for entry_date in monthly_dates:
        past_dates = close.index[close.index <= entry_date]
        if len(past_dates) < 60:
            continue
        loc = close.index.get_loc(past_dates[-1])
        c_p = close.iloc[: loc + 1]
        h_p = high.iloc[: loc + 1]
        l_p = low.iloc[: loc + 1]
        if len(c_p) < 55:
            continue

        rsi_p   = rsi(c_p)
        ema50_p = float(ema(c_p, 50).iloc[-1])
        price_p = float(c_p.iloc[-1])
        atr_p   = atr(h_p, l_p, c_p)
        sig     = _entry_signal(rsi_p, ema50_p, price_p)

        if sig != "BUY":
            continue

        # Stop-loss check: did any daily low breach entry − ATR × multiplier?
        stop_level = (price_p - ATR_MULTIPLIER * atr_p) if atr_p else None
        lows_since = low.loc[low.index > entry_date]
        stop_hit = bool(
            stop_level is not None
            and not lows_since.empty
            and float(lows_since.min()) < stop_level
        )

        # ATR-based position sizing: size ∝ 1/ATR (normalised later)
        atr_size_weight = (1.0 / atr_p) if atr_p and atr_p > 0 else 1.0

        pnl = (price_now - price_p) / price_p * 100

        trades.append(dict(
            entry_date=str(entry_date.date()),
            entry_price=round(price_p, 4),
            atr=round(atr_p, 4) if atr_p else None,
            stop_level=round(stop_level, 4) if stop_level else None,
            stop_hit=stop_hit,
            pnl_pct=round(pnl, 2),
            atr_weight=atr_size_weight,
        ))

    result = dict(empty)
    result["bt_trades"] = trades
    result["bt_n_trades"] = len(trades)
    result["bt_price_now"] = round(price_now, 4)

    if not trades:
        result["bt_note"] = "No BUY signals in 12-month window"
        result["bt_signal"] = "N/A"
        return result

    pnls = np.array([t["pnl_pct"] for t in trades])
    weights = np.array([t["atr_weight"] for t in trades])
    weights /= weights.sum()  # normalise

    win_rate = float(np.mean(pnls > 0))
    avg_pnl  = float(np.mean(pnls))
    wtd_pnl  = float(np.dot(weights, pnls))
    sharpe   = (float(np.mean(pnls) / np.std(pnls)) * np.sqrt(12)
                if np.std(pnls) > 0 else None)   # monthly → annualised proxy

    # Representative "primary" entry = the earliest trade (12-month-ago equivalent)
    first = trades[0]
    stop_any = any(t["stop_hit"] for t in trades)

    note_parts = [
        f"{len(trades)} entr{'y' if len(trades)==1 else 'ies'}",
        f"avg P&L {avg_pnl:+.1f}%",
        f"win {win_rate*100:.0f}%",
    ]
    if sharpe is not None:
        note_parts.append(f"Sharpe {sharpe:.2f}")
    if stop_any:
        note_parts.append("stop hit on ≥1 trade")

    result.update(dict(
        bt_signal="BUY",
        bt_price_entry=first["entry_price"],
        bt_pnl_pct=round(avg_pnl, 1),
        bt_stop_hit=stop_any,
        bt_win_rate=round(win_rate, 3),
        bt_avg_pnl=round(avg_pnl, 2),
        bt_atr_wtd_pnl=round(wtd_pnl, 2),
        bt_sharpe=round(sharpe, 3) if sharpe is not None else None,
        bt_note=" | ".join(note_parts),
    ))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST-LEVEL AGGREGATE STATS
# ─────────────────────────────────────────────────────────────────────────────

def watchlist_backtest_summary(results: list[dict]) -> dict:
    """
    Given the list of per-asset result dicts produced by run_backtest,
    compute aggregate statistics across the whole watchlist.

    Returns a dict with:
      total_trades, win_rate, avg_pnl, sharpe, best, worst
    """
    all_pnls = []
    for r in results:
        for t in r.get("bt_trades", []):
            all_pnls.append(t["pnl_pct"])

    if not all_pnls:
        return dict(total_trades=0, win_rate=None, avg_pnl=None,
                    sharpe=None, best=None, worst=None)

    arr = np.array(all_pnls)
    sharpe = (float(np.mean(arr) / np.std(arr)) * np.sqrt(12)
              if np.std(arr) > 0 else None)
    return dict(
        total_trades=len(arr),
        win_rate=round(float(np.mean(arr > 0)), 3),
        avg_pnl=round(float(np.mean(arr)), 2),
        sharpe=round(sharpe, 3) if sharpe is not None else None,
        best=round(float(arr.max()), 2),
        worst=round(float(arr.min()), 2),
    )

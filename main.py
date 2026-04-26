"""
main.py — Orchestration engine for Quant-AI v7.

Architecture changes:
  1. Async I/O: Phase 1 (yfinance data) runs in a ThreadPoolExecutor via asyncio.
     Phase 2 (news) uses aiohttp concurrent batch fetch.
     Total scan time scales as O(1) instead of O(N) for network I/O.
  2. Stewardship v2: ICR (Interest Coverage Ratio) from income_stmt fed into scorer.
  3. NER pre-filter: ticker + company_name passed to analyze_news_context_v2.
  4. Position sizing: Kelly + Target-Vol recommendation appended to every result row.
  5. WFO: Walk-Forward OOS metrics replace single-window backtest as primary signal.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yfinance as yf

from universe import SECTOR_UNIVERSE
from config import TOP_GLOBAL, TOP_PER_SECTOR, MAX_ASYNC_WORKERS
from fundamentals import get_fundamentals
from currency import apply_fx_conversion
from indicators import add_all_indicators, rsi as calc_rsi
from news import get_headlines_batch_async, get_recent_headlines
from sentiment import analyze_news_context_v2
from scoring import (
    stewardship_score,
    technical_score_v2,
    composite_score_v3,
    stewardship_trade_signal,
    classify_horizon,
    position_size,
)
from backtest import run_historical_backtest, run_macro_backtest, walk_forward_optimization
from portfolio import load_portfolio, audit_portfolio
from reporting import print_terminal_report, export_excel, export_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Async Data Acquisition ────────────────────────────────────────────────────

def _fetch_ticker_blocking(symbol: str, display_name: str, sector_name: str) -> dict | None:
    """
    Synchronous data fetch for a single ticker.
    Runs inside a ThreadPoolExecutor so the event loop is not blocked.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist   = ticker.history(period="5y")

        if hist.empty:
            logger.warning("[Fetch] Empty history for %s — skipping.", symbol)
            return None

        hist_eur  = apply_fx_conversion(hist, from_currency=ticker.info.get("currency", "USD"))
        hist_ind  = add_all_indicators(hist_eur)
        f_data    = get_fundamentals(symbol)

        return {
            "symbol":       symbol,
            "display_name": display_name,
            "sector":       sector_name,
            "hist":         hist_ind,
            "f_data":       f_data,
            "info":         ticker.info,
        }
    except Exception as exc:
        logger.error("[Fetch] Failed %s: %s", symbol, exc)
        return None


async def _run_data_acquisition(sector_universe: dict) -> dict:
    """
    Phase 1: Parallel yfinance fetch for all tickers.

    Strategy: yfinance is blocking I/O (no native async support).
    We dispatch all fetches concurrently into a ThreadPoolExecutor,
    collecting results via asyncio.gather().
    """
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor(max_workers=MAX_ASYNC_WORKERS) as executor:
        tasks = []
        meta  = []   # (symbol, display_name) for later mapping

        for sector_name, instruments in sector_universe.items():
            for display_name, symbol in instruments.items():
                tasks.append(
                    loop.run_in_executor(
                        executor,
                        _fetch_ticker_blocking,
                        symbol, display_name, sector_name,
                    )
                )
                meta.append(symbol)

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    raw_data_map: dict = {}
    for sym, result in zip(meta, raw_results):
        if isinstance(result, Exception):
            logger.error("[Acquire] Exception for %s: %s", sym, result)
        elif result is not None:
            raw_data_map[sym] = result

    return raw_data_map


async def _run_news_acquisition(
    raw_data_map: dict,
    max_items: int = 32,
) -> dict[str, list[dict]]:
    """
    Phase 3a: Concurrent news fetch for all tickers via aiohttp.
    Falls back to sequential sync if aiohttp is unavailable.
    """
    fetch_specs = [
        (sym, d["display_name"], max_items)
        for sym, d in raw_data_map.items()
    ]
    return await get_headlines_batch_async(fetch_specs)


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def _async_pipeline() -> None:
    results: list[dict] = []

    # ── Phase 1: Parallel data acquisition ───────────────────────────────────
    logger.info("Phase 1: Async data acquisition (%d tickers)...",
                sum(len(v) for v in SECTOR_UNIVERSE.values()))
    raw_data_map = await _run_data_acquisition(SECTOR_UNIVERSE)
    logger.info("Phase 1 complete: %d tickers loaded.", len(raw_data_map))

    # ── Phase 2: Sector median calculation ───────────────────────────────────
    sector_metrics = []
    for sym, d in raw_data_map.items():
        sector_metrics.append({
            "Symbol": sym,
            "Sector": d["sector"],
            "RSI":    calc_rsi(d["hist"]["Close"], 14),
            "PE":     d["f_data"].get("PE"),
        })

    metrics_df     = pd.DataFrame(sector_metrics)
    sector_medians = metrics_df.groupby("Sector").median(numeric_only=True)

    # ── Phase 3a: Concurrent news fetch ──────────────────────────────────────
    logger.info("Phase 3a: Async news fetch...")
    headlines_map = await _run_news_acquisition(raw_data_map)

    # ── Phase 3b: Analysis & scoring ─────────────────────────────────────────
    logger.info("Phase 3b: Scoring %d tickers...", len(raw_data_map))
    for symbol, d in raw_data_map.items():
        hist     = d["hist"]
        f_data   = d["f_data"]
        info     = d["info"]
        sector   = d["sector"]

        # Technical
        curr_price   = float(hist["Close"].iloc[-1])
        curr_sma50   = float(hist["SMA50"].iloc[-1])
        curr_rsi     = calc_rsi(hist["Close"], 14)
        sec_rsi_med  = sector_medians.loc[sector]["RSI"] if sector in sector_medians.index else 50

        t_val = technical_score_v2(curr_rsi, curr_price, curr_sma50, hist["Close"], sec_rsi_med)

        # Stewardship v2: D/E + ICR + Payout
        s_val = stewardship_score(
            debt_to_equity = info.get("debtToEquity") or f_data.get("DebtToEquity"),
            payout_ratio   = info.get("payoutRatio"),
            dividend_yield = info.get("dividendYield"),
            icr            = f_data.get("ICR"),          # ← new: interest coverage
        )

        # NER-filtered FinBERT sentiment
        headlines = headlines_map.get(symbol, [])
        if not headlines:
            headlines = get_recent_headlines(symbol, d["display_name"], max_items=32)

        sent_data = analyze_news_context_v2(
            headlines,
            ticker       = symbol,          # ← passed for NER entity matching
            company_name = d["display_name"],
        )
        if not sent_data:
            sent_data = {"score": 0, "reasoning": "No news data.", "ner_filtered": 0}

        # Composite score
        total_score = composite_score_v3(
            pe              = f_data.get("PE"),
            peg             = f_data.get("PEG"),
            roe             = f_data.get("ROE"),
            stewardship_val = s_val,
            tech_val        = t_val,
            geo_sent_val    = sent_data["score"] / 4,   # Normalize [-100,+100] → [-25,+25]
            vol             = info.get("beta"),
        )

        # Signal & horizon
        upside = (
            (info.get("targetMeanPrice", curr_price) - curr_price) / curr_price * 100
        )
        signal  = stewardship_trade_signal(total_score, s_val, upside)
        horizon = classify_horizon(
            info.get("dividendYield"), info.get("beta"),
            f_data.get("ROE"), f_data.get("PE"),
        )

        # Historical backtest (30-day window, ~1yr ago)
        bt_res = run_historical_backtest(hist)

        # Walk-Forward Optimization (primary validation metric)
        wfo_res = walk_forward_optimization(hist)

        # Position sizing
        macro_bt   = run_macro_backtest(hist)
        n_trades   = macro_bt.get("BT_Trades", 0)
        win_rate   = (macro_bt.get("BT_WinRate_pct",  0) / 100) if n_trades > 0 else None
        avg_pnl    = (macro_bt.get("BT_Avg_PnL_pct",  0) / 100) if n_trades > 0 else None
        # Estimate avg_win and avg_loss from win_rate and avg_pnl
        # avg_pnl ≈ win_rate*avg_win - (1-win_rate)*avg_loss
        # Without per-trade granularity, approximate: avg_win ≈ avg_pnl / win_rate
        avg_win  = (avg_pnl / win_rate) if (win_rate and win_rate > 0 and avg_pnl) else None
        avg_loss = abs(avg_win) * 0.6 if avg_win else None   # Conservative estimate

        # Asset annualized volatility from daily returns std
        daily_vol = hist["Close"].pct_change().std()
        annual_vol = float(daily_vol * (252 ** 0.5)) if daily_vol > 0 else None

        sizing = position_size(win_rate, avg_win, avg_loss, annual_vol)

        results.append({
            # Identity
            "Symbol":              symbol,
            "Name":                d["display_name"],
            "Sector":              sector,
            "Price_EUR":           round(curr_price, 2),
            # Scores
            "Total_Score":         total_score,
            "Stewardship":         s_val,
            "Technical":           t_val,
            "FinBERT_Score":       round(sent_data.get("score", 0), 1),
            "NER_Filtered":        sent_data.get("ner_filtered", 0),
            # Signal
            "Signal":              signal,
            "Horizon":             horizon,
            "Upside_pct":          round(upside, 1),
            # Fundamentals
            "PE":                  f_data.get("PE"),
            "RSI":                 curr_rsi,
            "ICR":                 f_data.get("ICR"),
            "DebtToEquity":        f_data.get("DebtToEquity") or info.get("debtToEquity"),
            # Backtest (historical window)
            "Backtest_PnL":        bt_res.get("Backtest_PnL_pct", 0.0),
            "Backtest_Status":     bt_res.get("Backtest_Signal", "N/A"),
            "BT_WinRate":          macro_bt.get("BT_WinRate_pct", 0.0),
            "BT_Avg_PnL":          macro_bt.get("BT_Avg_PnL_pct", 0.0),
            "BT_Trades":           n_trades,
            # WFO (out-of-sample validation)
            "WFO_OOS_AvgPnL":      wfo_res.get("wfo_oos_avg_pnl"),
            "WFO_OOS_WinRate":     wfo_res.get("wfo_oos_win_rate"),
            "WFO_Periods":         wfo_res.get("wfo_periods", 0),
            # Position sizing
            "Kelly_Size_pct":      sizing["Kelly_Size_pct"],
            "TargetVol_Size_pct":  sizing["TargetVol_Size_pct"],
            "Rec_Size_pct":        sizing["Recommended_Size_pct"],
            # Context
            "Reasoning":           ", ".join(sent_data.get("drivers", [])) or "N/A",
        })

    scan_df = pd.DataFrame(results)

    # ── Phase 4: Portfolio audit ──────────────────────────────────────────────
    portfolio_df = load_portfolio()
    audit_df = audit_portfolio(portfolio_df, scan_df) if not portfolio_df.empty else None

    # ── Phase 5: Reporting ────────────────────────────────────────────────────
    # Surface survivorship bias warning once, prominently
    if results:
        from backtest import _SB_WARNING
        logger.warning("[BACKTEST] %s", _SB_WARNING)

    print_terminal_report(scan_df, audit_df)
    export_excel(scan_df, audit_df)
    export_csv(scan_df, audit_df)


def run_pipeline() -> None:
    asyncio.run(_async_pipeline())


if __name__ == "__main__":
    run_pipeline()

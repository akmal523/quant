"""
main.py — Orchestration engine for Quant-AI v6.
Implements hierarchical data fetching, stewardship-centric scoring, 
and window-based backtesting.
"""

import pandas as pd
import logging
from universe import SECTOR_UNIVERSE
from config import TOP_GLOBAL, TOP_PER_SECTOR
from fundamentals import get_fundamentals
from currency import apply_fx_conversion
from indicators import add_all_indicators, rsi as calc_rsi
from news import get_recent_headlines
from sentiment import analyze_news_context_v2
from scoring import (
    stewardship_score, 
    technical_score_v2, 
    composite_score_v3, 
    stewardship_trade_signal,
    classify_horizon
)
from backtest import run_historical_backtest
from portfolio import load_portfolio, audit_portfolio
from reporting import print_terminal_report, export_excel, export_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline():
    results = []
    
    # Phase 1: Data Acquisition & Pre-processing
    # Collect raw metrics for sector median calculations
    raw_data_map = {}
    
    for sector_name, instruments in SECTOR_UNIVERSE.items():
        logger.info(f"Scanning Sector: {sector_name}")
        for display_name, symbol in instruments.items():
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2y") # 2y for Z-score and backtest window
                
                if hist.empty: continue
                
                # Standardize to EUR
                hist = apply_fx_conversion(hist, from_currency=ticker.info.get("currency", "USD"))
                hist_with_ind = add_all_indicators(hist)
                
                # Fetch Fundamentals (Resilient Service)
                f_data = get_fundamentals(symbol)
                
                # Store for main processing
                raw_data_map[symbol] = {
                    "sector": sector_name,
                    "display_name": display_name,
                    "hist": hist_with_ind,
                    "f_data": f_data,
                    "info": ticker.info
                }
            except Exception as e:
                logger.error(f"Failed {symbol}: {e}")

    # Phase 2: Sector Median Calculation
    # Required for relative-value technical scoring
    sector_metrics = []
    for sym, d in raw_data_map.items():
        current_rsi = calc_rsi(d["hist"]["Close"], 14)
        sector_metrics.append({
            "Symbol": sym,
            "Sector": d["sector"],
            "RSI": current_rsi,
            "PE": d["f_data"].get("PE")
        })
    
    metrics_df = pd.DataFrame(sector_metrics)
    sector_medians = metrics_df.groupby("Sector").median(numeric_only=True)

    # Phase 3: Analysis & Scoring
    for symbol, d in raw_data_map.items():
        hist = d["hist"]
        f_data = d["f_data"]
        info = d["info"]
        sector = d["sector"]
        
        # 1. Technicals (Z-Score & Relative RSI)
        curr_price = float(hist["Close"].iloc[-1])
        curr_sma50 = float(hist["SMA50"].iloc[-1])
        curr_rsi = calc_rsi(hist["Close"], 14)
        
        sec_rsi_med = sector_medians.loc[sector]["RSI"] if sector in sector_medians.index else 50
        t_val = technical_score_v2(curr_rsi, curr_price, curr_sma50, hist["Close"], sec_rsi_med)
        
        # 2. Stewardship & Intrinsic Quality
        s_val = stewardship_score(
            debt_to_equity = info.get("debtToEquity"),
            payout_ratio   = info.get("payoutRatio"),
            dividend_yield = info.get("dividendYield")
        )
        
        # 3. GeoSentiment (Weighted & Batched)
        headlines = get_recent_headlines(symbol, d["display_name"], max_items=32)
        sent_data = analyze_news_context_v2(headlines)
        
        # 4. Final Composite
        total_score = composite_score_v3(
            pe              = f_data.get("PE"),
            peg             = f_data.get("PEG"),
            roe             = f_data.get("ROE"),
            stewardship_val = s_val,
            tech_val        = t_val,
            geo_sent_val    = (sent_data["score"] / 4), # Normalized to 25
            vol             = info.get("beta") # Beta as volatility proxy
        )
        
        # 5. Signal & Horizon
        upside = ((info.get("targetMeanPrice", curr_price) - curr_price) / curr_price) * 100
        signal = stewardship_trade_signal(total_score, s_val, upside)
        horizon = classify_horizon(info.get("dividendYield"), info.get("beta"), f_data.get("ROE"), f_data.get("PE"))
        
        # 6. Backtest (Window-based)
        bt_res = run_historical_backtest(hist)
        
        results.append({
            "Symbol": symbol,
            "Name": d["display_name"],
            "Sector": sector,
            "Price_EUR": round(curr_price, 2),
            "Total_Score": total_score,
            "Stewardship": s_val,
            "Technical": t_val,
            "Sentiment": round(sent_data["score"], 1),
            "Signal": signal,
            "Horizon": horizon,
            "Backtest_PnL": bt_res["Backtest_PnL_pct"],
            "Backtest_Status": bt_res["Backtest_Signal"],
            "PE": f_data.get("PE"),
            "RSI": curr_rsi,
            "Upside_pct": round(upside, 1)
        })

    scan_df = pd.DataFrame(results)

    # Phase 4: Portfolio Audit
    portfolio_df = load_portfolio()
    audit_df = audit_portfolio(portfolio_df, scan_df) if not portfolio_df.empty else None

    # Phase 5: Reporting
    print_terminal_report(scan_df, audit_df)
    export_excel(scan_df, audit_df)
    export_csv(scan_df, audit_df)

if __name__ == "__main__":
    run_pipeline()

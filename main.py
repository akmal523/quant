"""
main.py — Orchestration engine v8.4.
Resolved: Indentation errors, Multiprocessing DuckDB lock bypass.
"""
from __future__ import annotations
import logging
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import duckdb

# Local Module Imports
from database import get_connection
from currency import apply_fx_conversion, get_eur_rate
from portfolio import load_portfolio, audit_portfolio, print_audit_report
from indicators import add_all_indicators
from sec_edgar import fetch_latest_8k
from sentiment import init_worker, score_corporate_document
from risk import calculate_risk_penalty
from scoring import (
    evaluate_structural_grade,
    evaluate_tactical_grade,
    allocate_capital_regime,
    hmm_market_state_score,
    stewardship_score_v2,
    apply_fast_filter
)
from fundamentals import get_fundamentals
from universe import CORE_INDEX  
from universe import is_etf

# Suppress library noise
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)


def deduce_currency(symbol: str) -> str:
    """Detect native currency from Yahoo Finance ticker suffix."""
    if "." not in symbol: 
        return "USD"  # US stocks have no suffix (e.g., ECL, HMY)
    
    suffix = symbol.split(".")[-1].upper()
    eur_zones = {"DE", "PA", "AS", "MI", "MC", "BR", "VI", "HE"}
    
    if suffix in eur_zones: return "EUR"
    if suffix == "L": return "GBX"  # London trades in pence
    if suffix == "SW": return "CHF" # Swiss Francs (Roche)
    if suffix == "CO": return "DKK" # Danish Krone
    if suffix == "OL": return "NOK" # Norwegian Krone
    if suffix == "ST": return "SEK" # Swedish Krona
    if suffix == "TO": return "CAD" # Canadian Dollar
    if suffix == "AX": return "AUD" # Australian Dollar
    if suffix == "KS": return "KRW" # Korean Won
    return "USD"


def process_asset(symbol: str, df: pd.DataFrame, f_data: dict, raw_8k: str, sector: str, precalc_nlp: dict | None) -> dict | None:
    try:
        # --- 1. Technical & Indicators ---
        price_hist = df.drop(columns=["Symbol", "Sector"], errors="ignore")
        
        # [NEW] Translate all foreign prices to EUR dynamically
        native_ccy = deduce_currency(symbol)
        price_hist = apply_fx_conversion(price_hist, from_currency=native_ccy, to_currency="EUR")
        
        hist_ind = add_all_indicators(price_hist)
        garch_vol = hist_ind["GARCH_Vol"]
        
        # --- 2. Market State (HMM) ---
        hmm_prob_bull = hmm_market_state_score(hist_ind["Close"], garch_vol) / 15.0
        
        # --- 3. Risk Metrics ---
        returns = np.log(hist_ind["Close"] / hist_ind["Close"].shift(1)).dropna()
        var_penalty = calculate_risk_penalty(returns)
        
        # --- 4. Stewardship & Fundamentals ---
        s_val = stewardship_score_v2(f_data, sector)
        struct_grade = evaluate_structural_grade(
            pe=f_data.get("PE"), peg=f_data.get("PEG"), 
            roe=f_data.get("ROE"), stewardship_val=s_val
        )

        # --- 5. Sentiment Logic (Bypassing DB in workers) ---
        if precalc_nlp:
            nlp_data = precalc_nlp
            text_source = "DuckDB Cache"
        else:
            text_to_analyze = raw_8k
            text_source = "SEC 8-K"
            if not text_to_analyze:
                from news import fetch_news_headlines
                text_to_analyze = fetch_news_headlines(symbol)
                text_source = "News RSS"

            if text_to_analyze:
                nlp_data = score_corporate_document(text_to_analyze)
            else:
                nlp_data = {"score": 0.0, "reasoning": "No data found.", "doc_hash": None}
        
        # --- 6. Tactical Grade & Allocation ---
        tact_grade = evaluate_tactical_grade(
            hmm_prob_bull=hmm_prob_bull, 
            finbert_score=nlp_data.get("score", 0.0), 
            var_penalty=var_penalty
        )
        allocation = allocate_capital_regime(struct_grade, tact_grade, s_val)

        return {
            "Symbol": symbol,
            "Current_Price": round(hist_ind["Close"].iloc[-1], 2),
            "Structural_Grade": round(struct_grade, 1),
            "Tactical_Grade": round(tact_grade, 1),
            "Stewardship": s_val,
            "Horizon": allocation["Horizon"],
            "Signal": allocation["Signal"],
            "Active_Score": allocation["Active_Score"],
            "NLP_Reasoning": f"[{text_source}] {nlp_data.get('reasoning', 'N/A')}",
            "doc_hash": nlp_data.get("doc_hash"),
            "nlp_score": nlp_data.get("score")
        }

    except Exception as e:
        print(f"\n[FATAL WORKER CRASH] {symbol}: {str(e)}") # Forces output to your terminal
        return None


def main() -> None:
    logger.warning("Loading localized database...")
    conn = get_connection()
    
    try:
        market_data = conn.execute("SELECT * FROM market_history").df()
    except Exception:
        logger.error("market_history missing. Run data_updater.py.")
        return

    grouped_data = {symbol: df for symbol, df in market_data.groupby("Symbol")}
    port_df = load_portfolio("portfolio.csv")
    portfolio_symbols = set(port_df["Symbol"].unique()) if not port_df.empty else set()
    logger.warning(f"Loaded Portfolio: {list(portfolio_symbols)}")
    
    # --- 1. Filtering ---
    survivors = {}
    survivor_funds = {}
    for i, (sym, df) in enumerate(grouped_data.items()):
        f_data = get_fundamentals(sym)
        
        # Automatically let Portfolio items AND any ETF survive
        is_portfolio = sym in portfolio_symbols
        is_asset_etf = is_etf(sym)
        
        if is_portfolio or is_asset_etf or apply_fast_filter(f_data):
            survivors[sym] = df
            survivor_funds[sym] = f_data
    
    # --- 2. Async Text Fetch ---
    import asyncio
    from async_fetcher import fetch_all_texts_concurrently
    survivor_texts = asyncio.run(fetch_all_texts_concurrently(list(survivors.keys())))
    
    # --- 3. Pre-fetch NLP Scores (Prevents Worker DB Locks) ---
    import hashlib
    precalc_map = {}
    for sym, text in survivor_texts.items():
        if text:
            h = hashlib.sha256(text.encode('utf-8')).hexdigest()
            row = conn.execute("SELECT score FROM nlp_scores WHERE doc_hash = ?", [h]).fetchone()
            if row:
                precalc_map[sym] = {"score": row[0], "reasoning": "Cache Hit", "doc_hash": h}

    # --- 4. Multiprocessing Pool ---
    results = []
    cpu_cores = max(1, multiprocessing.cpu_count() - 1)
    ctx = multiprocessing.get_context("spawn")
    
    with ProcessPoolExecutor(max_workers=cpu_cores, initializer=init_worker, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                process_asset, sym, df, survivor_funds[sym], 
                survivor_texts[sym], 
                df['Sector'].iloc[0] if 'Sector' in df.columns else "Other",
                precalc_map.get(sym)
            ): sym for sym, df in survivors.items()
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res:
                doc_hash = res.pop("doc_hash", None)
                nlp_score = res.pop("nlp_score", None)
                if doc_hash and nlp_score is not None:
                    # Save results in main thread only
                    conn.execute("INSERT OR REPLACE INTO nlp_scores (doc_hash, score) VALUES (?, ?)", [doc_hash, nlp_score])
                results.append(res)

    # --- REPORTING ---
    if not results:
        print("No assets passed the filters or completed scoring.")
        return

    # --- 1. DATA PROCESSING ---
    final_df = pd.DataFrame(results)
    final_df.to_csv("outputs/market_scan_v8.csv", index=False)

    # --- 2. CURRENCY DISPLAY ---
    print(f"\n CURRENCY: 1 EUR = {get_eur_rate():.4f} USD")

    # --- 3. TOP 3 OPPORTUNITIES ---
    print("\n" + "="*40)
    print("TOP 3 BUY OPPORTUNITIES")
    print("="*40)
    # Filter for BUY signals, sort by Active_Score, and take top 3
    top_buys = final_df[final_df['Signal'] == 'BUY'].sort_values(by='Active_Score', ascending=False).head(3)
    if not top_buys.empty:
        print(top_buys[['Symbol', 'Active_Score', 'Current_Price', 'NLP_Reasoning']].to_string(index=False))
    else:
        print("No high-conviction BUY signals found.")

    # --- 4. FULL MARKET SCAN ---
    print("\n" + "="*100)
    print("FULL MARKET SCAN")
    print("="*100)
    print(final_df.to_string(index=False))

    # --- 5. PORTFOLIO AUDIT ---
    if not port_df.empty:
        audit_res = audit_portfolio(port_df, final_df)
        audit_res.to_csv("outputs/portfolio_audit.csv", index=False)
        
        print("\n" + "="*100)
        print("FULL PORTFOLIO AUDIT")
        print("="*100)
        # Show columns relevant for audit
        cols = ['Symbol', 'PnL_pct', 'Audit_Decision', 'Active_Score', 'Signal']
        print(audit_res[cols].to_string(index=False))
        print("\n")

if __name__ == "__main__":
    main()

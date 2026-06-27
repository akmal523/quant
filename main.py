"""
main.py — Orchestration engine v9.0.
Architecture:
  1. Load & sort market data from DuckDB
  2. Filter survivors (portfolio + ETFs + fundamentals screener)
  3. Async fetch SEC 8-K / news texts
  4. Score ALL texts in main process with single FinBERT instance
  5. Multiprocess technical + fundamental analysis (no FinBERT in workers)
  6. Data-confidence penalty prevents false BUY from missing NLP data
  7. Batch-write new NLP cache entries
  8. Report: top buys + stocks (non-ETF), full scan, portfolio audit
"""
from __future__ import annotations
import hashlib
import logging
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from database import get_connection, init_db
from currency import apply_fx_conversion, get_eur_rate
from portfolio import load_portfolio, audit_portfolio, account_effectiveness, print_effectiveness_report
from indicators import add_all_indicators
from sentiment import score_corporate_document
from sentiment import init_worker as _init_nlp_worker
from risk import calculate_risk_penalty
from config import WEIGHT_TECHNICAL
from scoring import (
    evaluate_structural_grade,
    evaluate_tactical_grade,
    allocate_capital_regime,
    hmm_market_state_score,
    stewardship_score_v2,
    apply_fast_filter,
)
from fundamentals import get_fundamentals
from universe import is_etf

# ── Logging ───────────────────────────────────────────────────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ── Currency Detection ─────────────────────────────────────────────────────────

def deduce_currency(symbol: str) -> str:
    if "." not in symbol:
        return "USD"
    suffix = symbol.split(".")[-1].upper()
    eur_zones = {"DE", "PA", "AS", "MI", "MC", "BR", "VI", "HE"}
    if suffix in eur_zones: return "EUR"
    if suffix == "L": return "GBX"
    if suffix == "SW": return "CHF"
    if suffix == "CO": return "DKK"
    if suffix == "OL": return "NOK"
    if suffix == "ST": return "SEK"
    if suffix == "TO": return "CAD"
    if suffix == "AX": return "AUD"
    if suffix == "KS": return "KRW"
    return "USD"


# ── Worker Process ────────────────────────────────────────────────────────────

def process_asset(symbol: str, df: pd.DataFrame, f_data: dict, sector: str, nlp_data: dict) -> dict | None:
    """
    Compute technical, fundamental, and risk metrics for one asset.
    NLP data is pre-computed in main process — no FinBERT in workers.
    data_confidence penalises the tactical grade when no real news exists.
    """
    try:
        price_hist = df.drop(columns=["Symbol", "Sector"], errors="ignore")
        native_ccy = deduce_currency(symbol)
        price_hist = apply_fx_conversion(price_hist, from_currency=native_ccy, to_currency="EUR")

        last_close = price_hist["Close"].iloc[-1] if "Close" in price_hist.columns else None
        if last_close is None or (isinstance(last_close, float) and (pd.isna(last_close) or np.isnan(last_close))):
            logger.warning("[SKIP] %s: No valid price data (NaN close)", symbol)
            return {
                "Symbol": symbol,
                "Is_ETF": is_etf(symbol),
                "Current_Price": 0.0,
                "Structural_Grade": 0.0,
                "Tactical_Grade": 0.0,
                "Stewardship": 18.0,
                "Horizon": "N/A",
                "Signal": "N/A",
                "Active_Score": 0.0,
                "NLP_Reasoning": nlp_data.get("reasoning", "No price data available"),
            }

        hist_ind = add_all_indicators(price_hist)
        garch_vol = hist_ind["GARCH_Vol"]
        hmm_prob_bull = hmm_market_state_score(hist_ind["Close"], garch_vol) / float(WEIGHT_TECHNICAL)

        returns = np.log(hist_ind["Close"] / hist_ind["Close"].shift(1)).dropna()
        var_penalty = calculate_risk_penalty(returns)

        asset_is_etf = is_etf(symbol)
        if asset_is_etf:
            s_val = 18.0
            struct_grade = 85.0
        else:
            s_val = stewardship_score_v2(f_data, sector)
            struct_grade = evaluate_structural_grade(
                pe=f_data.get("PE"), peg=f_data.get("PEG"),
                roe=f_data.get("ROE"), stewardship_val=s_val,
            )

        # data_confidence: 1.0 = real SEC/News data scored, 0.0 = no-data fallback
        data_confidence = nlp_data.get("data_confidence", 1.0)
        tact_grade = evaluate_tactical_grade(
            hmm_prob_bull=hmm_prob_bull,
            finbert_score=nlp_data.get("score", 0.0),
            var_penalty=var_penalty,
            data_confidence=data_confidence,
        )
        allocation = allocate_capital_regime(struct_grade, tact_grade, s_val)

        return {
            "Symbol": symbol,
            "Is_ETF": asset_is_etf,
            "Current_Price": round(hist_ind["Close"].iloc[-1], 2),
            "Structural_Grade": round(struct_grade, 1),
            "Tactical_Grade": round(tact_grade, 1),
            "Stewardship": s_val,
            "Horizon": allocation["Horizon"],
            "Signal": allocation["Signal"],
            "Active_Score": allocation["Active_Score"],
            "NLP_Reasoning": nlp_data.get("reasoning", "N/A"),
            "doc_hash": nlp_data.get("doc_hash"),
            "nlp_score": nlp_data.get("score"),
        }

    except Exception as e:
        logger.exception("[WORKER CRASH] %s: %s", symbol, str(e))
        return None


# ── Main Orchestration ─────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Loading localized database...")
    conn = get_connection()
    init_db()

    try:
        market_data = conn.execute("SELECT * FROM market_history ORDER BY Date ASC").df()
    except Exception:
        logger.error("market_history missing. Run data_updater.py.")
        return

    market_data['Date'] = pd.to_datetime(market_data['Date'])
    market_data = market_data.sort_values(['Symbol', 'Date'])
    grouped_data = {symbol: df for symbol, df in market_data.groupby("Symbol")}

    port_df = load_portfolio("portfolio.csv")
    portfolio_symbols = set(port_df["Symbol"].unique()) if not port_df.empty else set()
    logger.info("Loaded Portfolio: %s", list(portfolio_symbols))

    # ── Step 1: Filter survivors ────────────────────────────────────────────
    survivors: dict[str, pd.DataFrame] = {}
    survivor_funds: dict[str, dict] = {}
    for sym, df in grouped_data.items():
        f_data = get_fundamentals(sym)
        is_portfolio = sym in portfolio_symbols
        is_asset_etf = is_etf(sym)
        if is_portfolio or is_asset_etf or apply_fast_filter(f_data):
            survivors[sym] = df
            survivor_funds[sym] = f_data
    logger.info("Survivors after filtering: %d", len(survivors))

    # ── Step 2: Async text fetch ────────────────────────────────────────────
    import asyncio
    from async_fetcher import fetch_all_texts_concurrently
    survivor_texts = asyncio.run(fetch_all_texts_concurrently(list(survivors.keys())))

    # ── Step 3: NLP scoring in MAIN process (single FinBERT) ────────────────
    _init_nlp_worker()

    nlp_cache_rows: list[tuple] = []
    nlp_data_map: dict[str, dict] = {}

    for sym, text in survivor_texts.items():
        if not text:
            # No data found — neutral score with ZERO confidence penalty
            nlp_data_map[sym] = {
                "score": 0.0,
                "reasoning": "No SEC/News data available — low confidence neutral score",
                "doc_hash": None,
                "data_confidence": 0.0,
            }
            logger.debug("[NLP] %s: no text data, data_confidence=0.0", sym)
            continue

        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        row = conn.execute("SELECT score FROM nlp_scores WHERE doc_hash = ?", [h]).fetchone()
        if row:
            nlp_data_map[sym] = {
                "score": row[0],
                "reasoning": "Cache Hit",
                "doc_hash": h,
                "data_confidence": 1.0,
            }
            logger.debug("[NLP] %s: cache hit (score=%.1f)", sym, row[0])
        else:
            nlp_result = score_corporate_document(text)
            nlp_data_map[sym] = {
                "score": nlp_result["score"],
                "reasoning": nlp_result["reasoning"],
                "doc_hash": nlp_result["doc_hash"],
                "data_confidence": 1.0,
            }
            if nlp_result["doc_hash"]:
                nlp_cache_rows.append((nlp_result["doc_hash"], nlp_result["score"]))
            logger.debug("[NLP] %s: scored in main (score=%.1f)", sym, nlp_result["score"])

    # ── Step 4: Multiprocessing (no FinBERT in workers) ──────────────────────
    results = []
    cpu_cores = min(4, max(1, multiprocessing.cpu_count() - 1))
    logger.info("Processing %d survivors with %d workers...", len(survivors), cpu_cores)

    with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
        futures = {
            executor.submit(
                process_asset, sym, df, survivor_funds[sym],
                df['Sector'].iloc[0] if 'Sector' in df.columns else "Other",
                nlp_data_map.get(sym, {"score": 0.0, "reasoning": "No NLP data", "data_confidence": 0.0}),
            ): sym for sym, df in survivors.items()
        }

        for future in as_completed(futures):
            res = future.result()
            if res:
                doc_hash = res.pop("doc_hash", None)
                nlp_score_val = res.pop("nlp_score", None)
                if doc_hash and nlp_score_val is not None:
                    nlp_cache_rows.append((doc_hash, nlp_score_val))
                results.append(res)

    if nlp_cache_rows:
        conn.execute("BEGIN TRANSACTION")
        for h, s in nlp_cache_rows:
            conn.execute("INSERT OR REPLACE INTO nlp_scores (doc_hash, score) VALUES (?, ?)", [h, s])
        conn.execute("COMMIT")
        logger.info("NLP cache: %d new entries saved", len(nlp_cache_rows))

    # Ensure all portfolio symbols appear in results, even if missing from DB
    scanned_symbols = {r["Symbol"] for r in results}
    for sym in portfolio_symbols:
        if sym not in scanned_symbols:
            logger.warning("[PORTFOLIO] %s: No market data found — adding placeholder", sym)
            results.append({
                "Symbol": sym,
                "Is_ETF": is_etf(sym),
                "Current_Price": 0.0,
                "Structural_Grade": 0.0,
                "Tactical_Grade": 0.0,
                "Stewardship": 18.0,
                "Horizon": "N/A",
                "Signal": "N/A",
                "Active_Score": 0.0,
                "NLP_Reasoning": "No market data available for this symbol",
            })

    if not results:
        logger.warning("No assets passed the filters or completed scoring.")
        return

    # ── Step 5: Reporting ────────────────────────────────────────────────────
    final_df = pd.DataFrame(results)
    final_df.to_csv("outputs/market_scan_v8.csv", index=False)

    print(f"\n CURRENCY: 1 EUR = {get_eur_rate():.4f} USD")

    # --- TOP 3 BUY OPPORTUNITIES (Inc. ETFs) ---
    print("\n" + "=" * 40)
    print("TOP 3 BUY OPPORTUNITIES")
    print("=" * 40)
    buys_with_price = final_df[(final_df['Signal'] == 'BUY') & (final_df['Current_Price'].notna())]
    top_buys = buys_with_price.sort_values(by='Active_Score', ascending=False).head(3)
    if not top_buys.empty:
        print(top_buys[['Symbol', 'Active_Score', 'Current_Price', 'NLP_Reasoning']].to_string(index=False))
    else:
        print("No high-conviction BUY signals found.")

    # --- TOP 3 STOCKS (Non-ETF) ---
    print("\n" + "=" * 40)
    print("TOP 3 STOCKS (Non-ETF)")
    print("=" * 40)
    stock_buys = final_df[(final_df['Signal'] == 'BUY') & (final_df['Is_ETF'] == False) & (final_df['Current_Price'].notna())]
    top_stocks = stock_buys.sort_values(by='Active_Score', ascending=False).head(3)
    if not top_stocks.empty:
        print(top_stocks[['Symbol', 'Active_Score', 'Current_Price', 'NLP_Reasoning']].to_string(index=False))
    else:
        print("No stock BUY signals found (only ETFs).")

    # --- FULL MARKET SCAN ---
    print("\n" + "=" * 145)
    print("FULL MARKET SCAN")
    print("=" * 145)
    display_cols = ['Symbol', 'Is_ETF', 'Current_Price', 'Structural_Grade', 'Tactical_Grade',
                    'Stewardship', 'Horizon', 'Signal', 'Active_Score', 'NLP_Reasoning']
    print(final_df[display_cols].to_string(index=False))

    # --- PORTFOLIO AUDIT ---
    if not port_df.empty:
        audit_res = audit_portfolio(port_df, final_df)
        audit_res.to_csv("outputs/portfolio_audit.csv", index=False)

        print("\n" + "=" * 120)
        print("FULL PORTFOLIO AUDIT")
        print("=" * 120)
        cols = ['Symbol', 'PnL_pct', 'PnL_EUR', 'Audit_Decision', 'Active_Score', 'Signal']
        print(audit_res[cols].to_string(index=False))
        print("\n")

        eff = account_effectiveness(audit_res, port_df)
        print_effectiveness_report(eff)


if __name__ == "__main__":
    main()

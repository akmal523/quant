"""
main.py — Three-phase pipeline:
  1. SCAN    — yfinance data, EUR-normalised, all indicators
  2. SCORE   — FinBERT sentiment + composite scoring + horizon classification
  3. REPORT  — terminal table, Excel, CSV, portfolio audit
"""
from __future__ import annotations
import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

import config
from universe   import get_market_universe, symbol_to_sector
from currency   import apply_fx_conversion
from indicators import add_all_indicators
import news        as news_module
import sentiment   as sentiment_module
from scoring    import (
    calculate_sector_medians,
    execute_scoring_pipeline,
    classify_horizon,
    geo_score,
)
from backtest   import run_historical_backtest
import portfolio as portfolio_module
import reporting

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")


# ─── Reasoning String ─────────────────────────────────────────────────────────

def _reasoning(row: pd.Series) -> str:
    parts: list[str] = []
    signal  = row.get("Signal", "")
    rsi     = row.get("RSI")
    fb      = row.get("FinBERT_Score", 0.0) or 0.0
    score   = row.get("Total_Score", 0)
    drivers = str(row.get("Risk_Drivers", "") or "")

    if signal == "SELL":
        if score < 40: parts.append("Weak structure")
        if rsi and float(rsi) > 70:
            parts.append(f"Overbought RSI={float(rsi):.0f}")
    if float(fb) < -30:
        parts.append(f"Bearish news ({float(fb):+.0f})")
    elif float(fb) > 30:
        parts.append(f"Bullish news ({float(fb):+.0f})")
    if row.get("Volatility_Penalty", 0):
        parts.append("High-vol penalty")
    if row.get("Fundamental_Score", 0) >= 30:
        parts.append("Strong fundamentals")
    if drivers:
        parts.append(drivers[:60])

    return " | ".join(parts) if parts else "Neutral backdrop"


# ─── Phase 1: Scan ────────────────────────────────────────────────────────────

def scan_universe() -> list[dict]:
    universe  = get_market_universe()
    total     = len(universe)
    results: list[dict] = []

    print(f"\n  Scanning {total} assets...\n")

    for i, (name, symbol) in enumerate(universe.items(), 1):
        print(f"  [{i:>3}/{total}]  {symbol:<14}  {name[:38]}", end="\r")

        try:
            t   = yf.Ticker(symbol)
            inf = t.info or {}

            # Skip if yfinance returned no meaningful price data
            if not (inf.get("regularMarketPrice") or inf.get("currentPrice")):
                continue

            hist = t.history(period=config.HIST_PERIOD, auto_adjust=True)
            if hist.empty or len(hist) < 60:
                continue

            # ── Currency normalisation → EUR ─────────────────────────────────
            src_ccy = inf.get("currency", "USD")
            hist    = apply_fx_conversion(hist, src_ccy, "EUR")
            hist    = add_all_indicators(hist)
            curr    = hist.iloc[-1]

            # ── News + FinBERT ────────────────────────────────────────────────
            headlines = news_module.get_recent_headlines(symbol, name)
            sent      = sentiment_module.analyze_news_context(headlines)

            # ── Derived metrics ───────────────────────────────────────────────
            vol        = float(hist["Close"].pct_change().dropna().std() * np.sqrt(252))
            div        = float(inf.get("dividendYield") or 0.0)
            roe        = inf.get("returnOnEquity")
            pe         = inf.get("trailingPE")
            peg        = inf.get("pegRatio")
            country    = inf.get("country", "Unknown")
            sector     = symbol_to_sector(symbol) or inf.get("sector", "Unknown")

            close_eur  = float(curr["Close"])
            sma50      = float(curr["SMA50"]) if pd.notna(curr.get("SMA50")) else None
            rsi_val    = float(curr["RSI"])   if pd.notna(curr.get("RSI"))   else None
            atr_val    = float(curr["ATR"])   if pd.notna(curr.get("ATR"))   else None

            # Analyst consensus upside
            target = inf.get("targetMeanPrice")
            upside: float | None = None
            if target and close_eur > 0:
                upside = (float(target) - close_eur) / close_eur * 100

            geo = geo_score(country, sector, symbol, headlines)

            row: dict = {
                "Symbol":          symbol,
                "Name":            name,
                "Sector":          sector,
                "Country":         country,
                "Close_EUR":       round(close_eur, 4),
                "PE":              pe,
                "PEG":             peg,
                "ROE":             roe,
                "Dividend_Yield":  round(div, 4),
                "RSI":             rsi_val,
                "ATR":             atr_val,
                "SMA50_EUR":       sma50,
                "Price_vs_SMA50":  (close_eur / sma50) if sma50 else None,
                "Volatility":      round(vol, 4),
                "Geo_Risk":        geo,
                "FinBERT_Score":   sent["score"],
                "Risk_Drivers":    " // ".join(sent["drivers"]),
                "Headline_Count":  sent["headline_count"],
                "Upside_pct":      round(upside, 2) if upside is not None else None,
                "Horizon":         classify_horizon(div, vol, roe, pe),
            }
            row.update(run_historical_backtest(hist))
            results.append(row)

        except Exception as exc:
            logging.debug("[%s] Skipped: %s", symbol, exc)
            continue

    print(f"\n\n  Scan complete: {len(results)}/{total} assets processed.\n")
    return results


# ─── Phase 2: Score ───────────────────────────────────────────────────────────

def score_results(results: list[dict]) -> pd.DataFrame:
    df      = pd.DataFrame(results)
    s_meds  = calculate_sector_medians(df)

    for idx, row in df.iterrows():
        sector  = row["Sector"]
        med     = s_meds.loc[sector] if sector in s_meds.index else None
        scored  = execute_scoring_pipeline(row.to_dict(), med)
        for k, v in scored.items():
            df.at[idx, k] = v

    df["Reasoning"] = df.apply(_reasoning, axis=1)
    return df.sort_values(["Sector", "Total_Score"], ascending=[True, False])


# ─── Phase 3: Report ──────────────────────────────────────────────────────────

def report(df: pd.DataFrame) -> None:
    portfolio_df = portfolio_module.load_portfolio("portfolio.csv")
    audit_df     = portfolio_module.audit(portfolio_df, df)

    reporting.print_terminal_report(df, audit_df if not audit_df.empty else None)

    os.makedirs("outputs", exist_ok=True)
    reporting.export_csv(df, audit_df if not audit_df.empty else None)
    reporting.export_excel(df, audit_df if not audit_df.empty else None)


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    results = scan_universe()

    if not results:
        print("CRITICAL: No assets processed. Check network / VPN.")
        return

    df = score_results(results)
    report(df)


if __name__ == "__main__":
    main()

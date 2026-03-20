outputs/
========

This folder is created automatically on the first run of main.py.
All generated files land here so the project root stays clean.

─────────────────────────────────────────────────────────────────────────────
FILES GENERATED EACH RUN
─────────────────────────────────────────────────────────────────────────────

watchlist_v45.csv
  Plain-text CSV of the full results table.
  Useful for further analysis in Excel, Python (pandas), or Google Sheets.
  Every column described in run_summary.txt is present here.

watchlist_v45.xlsx
  Formatted Excel workbook with:
    • Conditional row colouring: green=BUY, red=SELL, yellow=HOLD.
    • Gold highlight + bold for the top-3 picks.
    • Frozen header row.
    • Auto-sized columns (capped at 48 chars).
  Open in Excel or LibreOffice Calc.

run_summary.txt
  Human-readable summary of the latest run:
    • Top-3 picks with full detail (price, signal, AI reasoning, backtest).
    • Backtest table for all tickers that had a BUY signal 12 months ago.
    • Full column glossary explaining every metric in the CSV/XLSX.
  Overwritten on each run — reflects the most recent execution only.

─────────────────────────────────────────────────────────────────────────────
HOW THE SCORES ARE COMPUTED
─────────────────────────────────────────────────────────────────────────────

The composite Score (0–100) has three components:

  Fundamental  (max 40)
    Evaluated from P/E, PEG ratio, and Return on Equity.
    ETFs and funds score 0 here (no fundamental data available).

  Technical    (max 30)
    Evaluated from the 14-period RSI and the % distance to the analyst
    consensus target (Upside_Pct).

  GeoSentiment (max 30)
    Geo-risk score derived from the company's headquarters country and
    news keyword scan (sanctions, tariffs, conflict, etc.), combined with
    the Gemini AI sentiment score (−100 to +100 mapped onto 0–15 pts).
    High annualised volatility (>60%) deducts 3 pts.

Trade signal:
  BUY   — adj. score ≥ 65 AND upside > 5 %
  HOLD  — adj. score ≥ 50
  SELL  — RSI > 72 OR upside < −10 %
  (adj. score = score − 15 when AI sentiment < −50)

─────────────────────────────────────────────────────────────────────────────
DISCLAIMER
─────────────────────────────────────────────────────────────────────────────

All output is informational only and does not constitute financial advice.
Backtest results are hypothetical and do not account for transaction costs,
slippage, taxes, or liquidity constraints.

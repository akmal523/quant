"""
Trade Republic Watchlist — Quant-AI Edition v4.5
=================================================
Three-phase pipeline:
  Phase 1 — Parallel news fetch for all tickers.
  Phase 2 — Batch Gemini AI sentiment scoring.
  Phase 3 — Technical + fundamental analysis, scoring, backtest.

Requirements:
    pip install yfinance pandas numpy scipy openpyxl google-genai feedparser python-dotenv

Usage:
    python main.py
"""
# Load .env before any config import so os.getenv() sees the values.
import os
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY", "").strip(' "\'')

import calibrate
import time
import warnings
from datetime import datetime

import numpy  as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 45)
pd.set_option("display.width", 220)

from config     import (
    ATR_MULTIPLIER, BATCH_SIZE, CONVERT_TO_EUR,
    GEMINI_MODEL, OUTPUT_DIR, VERSION,
)
from tickers    import TICKER_MAP
from currency   import get_eur_rate
from macro      import load_macro
from news       import fetch_news_for_ticker
from gemini_ai  import probe_gemini, get_ai_sentiment_batch
from analyzer   import analyze
from reporting  import send_email_report, write_excel
from utils      import fv, fmt_sentiment


def _ensure_output_dir():
    """Create the outputs/ directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _write_run_summary(df: pd.DataFrame, csv_path: str, xlsx_path: str):
    """
    Write a human-readable run summary to outputs/run_summary.txt.
    Overwrites the previous summary so the file always reflects the latest run.
    """
    path = os.path.join(OUTPUT_DIR, "run_summary.txt")
    top3_idx = df["Score"].nlargest(3).index

    lines = [
        f"Trade Republic Quant-AI  v{VERSION}",
        f"Run date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Universe : {len(df)} instruments",
        f"Model    : {GEMINI_MODEL}",
        f"Stop-Loss: {ATR_MULTIPLIER}×ATR",
        "=" * 70,
        "",
        "OUTPUT FILES",
        f"  {csv_path}   — full results table (all metrics)",
        f"  {xlsx_path}  — formatted Excel with conditional colouring",
        "",
        "COLUMN GLOSSARY",
        "  Score             Composite 0–100 (Fundamental 40 + Technical 30 + GeoSentiment 30)",
        "  Score_Fundamental P/E, PEG, ROE sub-score (max 40)",
        "  Score_Technical   RSI + analyst-target upside sub-score (max 30)",
        "  Score_GeoSentiment Geo risk + AI sentiment sub-score (max 30)",
        "  Signal            BUY / HOLD / SELL derived from Score, RSI, Upside, AI Sentiment",
        "  TOP_PICK          Top 3 instruments by Score in this run",
        "  AI_Sentiment      Gemini score −100 (very bearish) to +100 (very bullish)",
        "  AI_Summary        One-line Gemini rationale for the sentiment score",
        "  Entry_Fmt         Entry price (current price or EMA50 fallback for ETFs)",
        "  Target_Fmt        Target price (70% analyst consensus + 30% 60-bar swing high)",
        "  Stop_Fmt          Stop-loss = Entry − ATR × " + str(ATR_MULTIPLIER),
        "  Upside_Pct        % distance from current price to analyst consensus target",
        "  RSI               14-period Relative Strength Index",
        "  CAGR_5Y_pct       5-year compound annual growth rate",
        "  Vol_Ann_pct       Annualised volatility (252-day rolling std × √252)",
        "  Geo_Risk          Low / Med / High geo-political risk label",
        "  Geo_Score         Numeric geo-risk score 1–10 (lower = safer)",
        "  Corr_*            Monthly Pearson correlation vs macro series",
        "                    (CPI proxy=TIP, CB rate proxy=^IRX, OIL=CL=F,",
        "                     GOLD=GC=F, SP500=^GSPC)",
        "  bt_signal         Signal that would have been generated 12 months ago",
        "  bt_price_entry    Price 12 months ago (backtest entry)",
        "  bt_price_now      Latest price (backtest exit)",
        "  bt_pnl_pct        Hypothetical 12-month P&L %",
        "  bt_stop_hit       True if the stop-loss was breached during the holding period",
        "  bt_note           Human-readable backtest outcome description",
        "",
        "=" * 70,
        "TOP 3 PICKS THIS RUN",
        "=" * 70,
    ]

    for rank, idx in enumerate(top3_idx, 1):
        row = df.loc[idx]
        bt  = f"{float(row['bt_pnl_pct']):+.1f}%" if pd.notna(row["bt_pnl_pct"]) else "N/A"
        up  = f"{float(row['Upside_Pct']):+.1f}%" if pd.notna(row["Upside_Pct"])  else "N/A"
        lines += [
            f"\n  #{rank}  {row['Asset']} ({row['Ticker']}) [{row['Currency']}]",
            f"       Score  : {row['Score']}/100"
            f"  [F={row['Score_Fundamental']}"
            f"  T={row['Score_Technical']}"
            f"  G={row['Score_GeoSentiment']}]",
            f"       Signal : {row['Signal']}",
            f"       Entry  : {row['Entry_Fmt']}   "
            f"Target: {row['Target_Fmt']}   "
            f"Stop ({ATR_MULTIPLIER}×ATR): {row['Stop_Fmt']}",
            f"       Upside : {up}",
            f"       AI     : {fmt_sentiment(row['AI_Sentiment'])}  → {row['AI_Summary']}",
            f"       BT 12m : {bt}  ({row['bt_note']})",
            f"       Geo    : {row['Geo_Risk']} ({row['Geo_Score']}/10)",
        ]
        if row["Note"]:
            lines.append(f"       Note   : {row['Note']}")

    lines += [
        "",
        "=" * 70,
        "BACKTEST — 12-month BUY signals",
        "=" * 70,
    ]
    bt_df = df[df["bt_signal"] == "BUY"].sort_values("bt_pnl_pct", ascending=False)
    if not bt_df.empty:
        for _, row in bt_df.iterrows():
            sw  = "  ⚠ Stop hit" if row["bt_stop_hit"] else ""
            bt  = f"{float(row['bt_pnl_pct']):+.1f}%" if pd.notna(row["bt_pnl_pct"]) else "N/A"
            lines.append(
                f"  {row['Ticker']:<14} "
                f"Entry={fv(row['bt_price_entry'])}  "
                f"Now={fv(row['bt_price_now'])}  P&L={bt}{sw}"
            )
    else:
        lines.append("  No BUY signals triggered 12 months ago.")

    lines += [
        "",
        "=" * 70,
        "NOTES",
        "=" * 70,
        "  • Results are informational only — not financial advice.",
        "  • AI sentiment is generated by Gemini based on recent news headlines.",
        "  • Backtest uses a simple RSI+EMA50 entry rule — no transaction costs.",
        "  • Stop-loss is calculated as Entry − ATR(14) × " + str(ATR_MULTIPLIER) + ".",
        "  • Geo risk score is heuristic (country base + news keyword bump).",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Run summary: {path}")


def main(config=None) -> pd.DataFrame:
    _ensure_output_dir()
    print("=" * 80)
    print(f"  Trade Republic — Quant-AI Edition v{VERSION}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
          f"{len(TICKER_MAP)} instruments  |  ATR×{ATR_MULTIPLIER}")

    gemini_ok = bool(probe_gemini())
    print(f"  {GEMINI_MODEL:<36}: {'OK' if gemini_ok else 'FAILED — see [DEBUG] above'}")
    print(f"  Batch size       : {BATCH_SIZE} tickers/request  "
          f"(~{-(-len(TICKER_MAP) // BATCH_SIZE)} API calls)")
    print(f"  USD→EUR convert  : {'ON' if CONVERT_TO_EUR else 'OFF'}")
    print("=" * 80)

    if CONVERT_TO_EUR:
        get_eur_rate()

    print("\nLoading macro series...")
    macro = load_macro()
    print(f"  Loaded: {list(macro.keys())}\n")

    # ── PHASE 1: Fetch news for all tickers ───────────────────────────────────
    print("─" * 60)
    print("PHASE 1 — Market data & news fetch")
    print("─" * 60)
    ticker_news: dict = {}
    for i, (name, sym) in enumerate(TICKER_MAP.items(), 1):
        print(f"  [{i:02d}/{len(TICKER_MAP)}] {sym:<18}", end=" news=", flush=True)
        news = fetch_news_for_ticker(name, sym)
        ticker_news[sym] = news
        print(len(news))

    # ── PHASE 2: Batch Gemini sentiment ───────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"PHASE 2 — Gemini batch sentiment  (batch={BATCH_SIZE})")
    print("─" * 60)
    items = [
        {"name": name, "sym": sym, "news": ticker_news[sym]}
        for name, sym in TICKER_MAP.items()
    ]
    ai_scores: dict = {}
    total_batches = -(-len(items) // BATCH_SIZE)

    for b_start in range(0, len(items), BATCH_SIZE):
        batch      = items[b_start: b_start + BATCH_SIZE]
        batch_syms = [it["sym"] for it in batch]
        batch_num  = b_start // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches}: {', '.join(batch_syms)}",
              end=" → ", flush=True)
        result = get_ai_sentiment_batch(batch)
        ai_scores.update(result)
        scores_str = "  ".join(f"{s}:{v[0]:+d}" for s, v in result.items())
        print(scores_str)
        time.sleep(0.2)

    # ── PHASE 3: Full technical analysis ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print("PHASE 3 — Technical analysis & scoring")
    print("─" * 60)
    records = []
    for i, (name, sym) in enumerate(TICKER_MAP.items(), 1):
        ai_score, ai_summary = ai_scores.get(sym, (0, "No News"))
        print(f"  [{i:02d}/{len(TICKER_MAP)}] {sym:<18} {name}  [AI={ai_score:+d}]")
        records.append(analyze(
            name, sym, macro,
            prefetched_ai=(ai_score, ai_summary),
            prefetched_news=ticker_news.get(sym, []),
            config=config,
        ))
    df = pd.DataFrame(records)

    # Coerce numeric columns.
    for col in ["Score", "PE", "PEG", "ROE", "Upside_Pct", "CAGR_5Y_pct",
                "Vol_Ann_pct", "Entry_Price", "Target_Price", "Stop_Loss",
                "EMA50", "ATR", "RSI", "bt_pnl_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def _coerce_sentiment(v):
        """Keep 'No News' strings; coerce numbers; drop NaN."""
        if isinstance(v, str):
            return v
        try:
            f = float(v)
            return f if not np.isnan(f) else None
        except Exception:
            return v

    df["AI_Sentiment"] = df["AI_Sentiment"].apply(_coerce_sentiment)

    # TOP-3 PICKS.
    top3_idx = df["Score"].nlargest(3).index
    df["TOP_PICK"] = ""
    for rank, idx in enumerate(top3_idx, 1):
        df.at[idx, "TOP_PICK"] = f"TOP PICK #{rank}"

    # ── CONSOLE OUTPUT ────────────────────────────────────────────────────────
    disp = df[[
        "Asset", "Ticker", "Currency", "Score", "Signal", "TOP_PICK",
        "AI_Sentiment", "Entry_Fmt", "Target_Fmt", "Stop_Fmt",
        "Upside_Pct", "RSI", "bt_pnl_pct", "Geo_Risk", "Note",
    ]].copy()
    disp["Upside_Pct"] = disp["Upside_Pct"].apply(
        lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A")
    disp["bt_pnl_pct"] = disp["bt_pnl_pct"].apply(
        lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A")
    disp.rename(columns={
        "Entry_Fmt": "Entry", "Target_Fmt": "Target",
        "Stop_Fmt": "Stop-Loss", "bt_pnl_pct": "12m BT",
    }, inplace=True)

    print("\n" + "=" * 200)
    print("FINAL TRADE TABLE  (sorted by Score)")
    print("=" * 200)
    print(disp.sort_values("Score", ascending=False).to_string(index=False))

    # TOP-3 detail.
    print("\n" + "=" * 80)
    print("TOP 3 PICKS")
    print("=" * 80)
    for rank, idx in enumerate(top3_idx, 1):
        row = df.loc[idx]
        bt  = f"{float(row['bt_pnl_pct']):+.1f}%" if pd.notna(row["bt_pnl_pct"]) else "N/A"
        up  = f"{float(row['Upside_Pct']):+.1f}%" if pd.notna(row["Upside_Pct"])  else "N/A"
        print(f"\n  #{rank}  {row['Asset']} ({row['Ticker']}) [{row['Currency']}]")
        print(f"       Score:    {row['Score']}/100  "
              f"[F={row['Score_Fundamental']} T={row['Score_Technical']} "
              f"G={row['Score_GeoSentiment']}]")
        print(f"       Signal:   {row['Signal']}")
        print(f"       Entry:    {row['Entry_Fmt']}   "
              f"Target: {row['Target_Fmt']}   "
              f"Stop ({ATR_MULTIPLIER}×ATR): {row['Stop_Fmt']}")
        print(f"       Upside:   {up}")
        print(f"       AI Sent:  {fmt_sentiment(row['AI_Sentiment'])}  → {row['AI_Summary']}")
        print(f"       Backtest: {bt}  ({row['bt_note']})")
        print(f"       Geo Risk: {row['Geo_Risk']} ({row['Geo_Score']}/10)")
        if row["Note"]:
            print(f"       Note:     {row['Note']}")

    # Backtest summary.
    print("\n" + "=" * 80)
    print("BACKTEST — 12-month BUY signals")
    print("=" * 80)
    bt_df = df[df["bt_signal"] == "BUY"].sort_values("bt_pnl_pct", ascending=False)
    if not bt_df.empty:
        for _, row in bt_df.iterrows():
            sw  = " ⚠ Stop hit" if row["bt_stop_hit"] else ""
            bt  = f"{float(row['bt_pnl_pct']):+.1f}%" if pd.notna(row["bt_pnl_pct"]) else "N/A"
            print(f"  {row['Ticker']:<14} "
                  f"Entry={fv(row['bt_price_entry'])}  "
                  f"Now={fv(row['bt_price_now'])}  P&L={bt}{sw}")
    else:
        print("  No BUY signals triggered 12 months ago.")

    # Price warnings.
    warned = df[df["Note"].str.contains("Price Warning", na=False)]
    if not warned.empty:
        print("\n" + "=" * 80)
        print("PRICE WARNINGS (deviation from EMA50 > 150%)")
        print("=" * 80)
        for _, row in warned.iterrows():
            print(f"  {row['Ticker']:<14} {row['Note']}")

    # ── SAVE OUTPUTS ──────────────────────────────────────────────────────────
    tag       = f"watchlist_v{VERSION.replace('.', '')}"
    csv_path  = os.path.join(OUTPUT_DIR, f"{tag}.csv")
    xlsx_path = os.path.join(OUTPUT_DIR, f"{tag}.xlsx")

    df.to_csv(csv_path, index=False)
    print(f"\nCSV  saved: {csv_path}")
    write_excel(df, xlsx_path)
    _write_run_summary(df, csv_path, xlsx_path)

    if input("\nSend HTML email report? [y/N]: ").strip().lower() == "y":
        send_email_report(df, csv_path, xlsx_path)

    return df


if __name__ == "__main__":
    # 1. Run calibration to update calibrated_weights.json (used by scoring.py).
    #    Returns a config dict with indicator period overrides for analyze().
    try:
        optimized_settings = calibrate.run_calibration()
    except Exception as e:
        print(f"Calibration failed: {e}. Using defaults.")
        optimized_settings = {}

    # 2. Start the main pipeline with the calibrated settings.
    main(config=optimized_settings)

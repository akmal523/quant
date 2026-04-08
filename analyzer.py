"""
analyzer.py — per-asset analysis orchestrator.
Pulls together price history, fundamentals, AI sentiment, macro correlations,
geo risk, scoring, and backtest into a single result dict.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from config     import ATR_MULTIPLIER, PRICE_WARN_PCT
from currency   import get_eur_rate, currency_symbol, format_price
from indicators import ema, rsi, atr, safe_float
from macro      import corr_with_macro
from news       import fetch_rss_news
from gemini_ai  import get_ai_sentiment
from scoring    import geo_score_fallback, deep_score, trade_signal
from backtest   import run_backtest


def analyze(name: str, symbol: str, macro: dict,
            prefetched_ai:   tuple | None = None,
            prefetched_news: list  | None = None) -> dict:
    """
    Run the full analysis pipeline for a single instrument.

    Parameters
    ----------
    name            : Human-readable asset name.
    symbol          : Yahoo Finance ticker.
    macro           : Pre-loaded macro series dict (from macro.load_macro).
    prefetched_ai   : (score, summary) tuple from the batch AI phase, or None.
    prefetched_news : News list from the batch news phase, or None.

    Returns
    -------
    Flat dict of all metrics used downstream (DataFrame row).
    """
    r = dict(
        Asset=name, Ticker=symbol, Sector="N/A", Country="N/A",
        Type="Equity", Currency="N/A", CurrencySymbol="",
        Entry_Price=None, EMA50=None, Target_Price=None,
        Stop_Loss=None, ATR=None, Resistance=None,
        Entry_Fmt="—", Target_Fmt="—", Stop_Fmt="—",
        PE=None, PEG=None, ROE=None, PB=None, DivYield_pct=None,
        Analyst_Target=None, Upside_Pct=None,
        RSI=None, CAGR_5Y_pct=None, Vol_Ann_pct=None,
        AI_Sentiment=None, AI_Summary="",
        Corr_CPI=None, Corr_CB_Rate=None,
        Corr_Oil=None, Corr_Gold=None, Corr_SP500=None,
        Geo_Score=None, Geo_Risk="N/A",
        Score=None, Score_Fundamental=None,
        Score_Technical=None, Score_GeoSentiment=None,
        bt_signal="N/A", bt_price_entry=None, bt_price_now=None,
        bt_pnl_pct=None, bt_stop_hit=False, bt_note="",
        Signal="N/A", TOP_PICK="", Note="",
    )
    try:
        tk   = yf.Ticker(symbol)
        info = tk.info or {}
        if not info or info.get("quoteType") is None:
            r["Note"] = "No data from yfinance"
            return r

        qt     = info.get("quoteType", "")
        is_etf = qt in ("ETF", "MUTUALFUND")
        r["Type"]    = "ETF/Fund" if is_etf else "Equity"
        r["Sector"]  = info.get("sector", "ETF/Fund" if is_etf else "N/A")
        r["Country"] = info.get("country", "N/A")

        # ── Currency detection ────────────────────────────────────────────────
        # yfinance may return "GBp" or "GBX" for pence-denominated London tickers.
        raw_currency = (
            info.get("currency")
            or info.get("financialCurrency")
            or "USD"
        )
        is_pence = raw_currency in ("GBp", "GBX", "GBx")
        display_currency = "GBP" if is_pence else raw_currency.upper()

        r["Currency"]       = display_currency
        r["CurrencySymbol"] = currency_symbol(display_currency)

        # ── News ─────────────────────────────────────────────────────────────
        if prefetched_news is not None:
            news = prefetched_news
        else:
            news = []
            try:
                yf_news = tk.news or []
                news = [n for n in yf_news if n.get("title")]
            except Exception:
                pass
            if not news:
                print(f"    [News] yfinance returned 0 — trying Google News RSS...")
                news = fetch_rss_news(symbol, company_name=name)
                if news:
                    print(f"    [News] RSS returned {len(news)} headlines for {symbol}")
                else:
                    print(f"    [News] RSS also returned 0 for {symbol}")

        # ── AI Sentiment ─────────────────────────────────────────────────────
        if prefetched_ai is not None:
            ai_score, ai_summary = prefetched_ai
        else:
            ai_score, ai_summary = get_ai_sentiment(name, news)
        r["AI_Sentiment"] = ai_score
        r["AI_Summary"]   = ai_summary

        # ── Fundamentals (equities only) ──────────────────────────────────────
        if not is_etf:
            r["PE"]  = safe_float(info.get("trailingPE"))
            r["PEG"] = safe_float(info.get("pegRatio"))
            r["PB"]  = safe_float(info.get("priceToBook"))
            r["ROE"] = safe_float(info.get("returnOnEquity"))
            div = info.get("dividendYield")
            r["DivYield_pct"] = round(float(div) * 100, 2) if div else None

            tgt = info.get("targetMeanPrice")
            cpx = info.get("currentPrice") or info.get("regularMarketPrice")

            # Pence → pounds for price fields.
            if is_pence:
                if cpx: cpx = float(cpx) / 100
                if tgt: tgt = float(tgt) / 100

            if tgt and cpx:
                r["Analyst_Target"] = round(float(tgt), 4)
                r["Upside_Pct"]     = round(
                    (float(tgt) - float(cpx)) / float(cpx) * 100, 1)
                r["Entry_Price"]    = round(float(cpx), 4)

        # ── Historical price data ────────────────────────────────────────────
        hist = yf.download(
            symbol, start="2018-01-01",
            auto_adjust=True, progress=False,
        )
        if hist.empty or "Close" not in hist.columns:
            r["Note"] = "No price history"
            return r

        close_raw = hist["Close"].squeeze()
        high_raw  = hist["High"].squeeze() if "High" in hist.columns else close_raw
        low_raw   = hist["Low"].squeeze()  if "Low"  in hist.columns else close_raw

        # Pence → pounds normalisation on the full series.
        close = close_raw / 100 if is_pence else close_raw
        high  = high_raw  / 100 if is_pence else high_raw
        low   = low_raw   / 100 if is_pence else low_raw

        e50 = float(ema(close, 50).iloc[-1])
        r["EMA50"] = round(e50, 4)
        if r["Entry_Price"] is None:
            r["Entry_Price"] = round(e50, 4)

        # Price anomaly warning.
        if r["Entry_Price"] and e50 and e50 > 0:
            deviation = abs(r["Entry_Price"] - e50) / e50
            if deviation > PRICE_WARN_PCT:
                r["Note"] = (
                    r["Note"] + " | Price Warning: "
                    f"{deviation*100:.0f}% from EMA50"
                ).strip(" | ")

        # ATR and stop-loss.
        atr_val = atr(high, low, close)
        if atr_val:
            r["ATR"]       = round(atr_val, 4)
            r["Stop_Loss"] = round(r["Entry_Price"] - ATR_MULTIPLIER * atr_val, 4)

        # Resistance: 60-bar swing high.
        r["Resistance"] = round(float(high.tail(60).max()), 4)

        # Target: 70 % analyst consensus + 30 % resistance level.
        if r["Analyst_Target"] and r["Resistance"]:
            r["Target_Price"] = round(
                0.70 * r["Analyst_Target"] + 0.30 * r["Resistance"], 4)
        elif r["Analyst_Target"]:
            r["Target_Price"] = r["Analyst_Target"]
        else:
            r["Target_Price"] = r["Resistance"]

        def fp(v):
            return format_price(v, display_currency)

        r["Entry_Fmt"]  = fp(r["Entry_Price"])
        r["Target_Fmt"] = fp(r["Target_Price"])
        r["Stop_Fmt"]   = fp(r["Stop_Loss"])

        # RSI.
        rsi_val = rsi(close)
        r["RSI"] = round(rsi_val, 1) if rsi_val else None

        # 5-year CAGR.
        c5 = close[close.index >= datetime.now() - timedelta(days=5 * 365)]
        if len(c5) > 50 and float(c5.iloc[0]) > 0:
            r["CAGR_5Y_pct"] = round(
                ((float(c5.iloc[-1]) / float(c5.iloc[0])) ** 0.2 - 1) * 100, 1)

        # Annualised volatility (252 trading-day window).
        ret = close.pct_change().dropna().tail(252)
        if len(ret) > 20:
            r["Vol_Ann_pct"] = round(float(ret.std()) * np.sqrt(252) * 100, 1)

        # Macro correlations.
        corrs = corr_with_macro(close, macro)
        r["Corr_CPI"]     = corrs.get("TIPS_INF")
        r["Corr_CB_Rate"] = corrs.get("FED_PROXY")
        r["Corr_Oil"]     = corrs.get("OIL")
        r["Corr_Gold"]    = corrs.get("GOLD")
        r["Corr_SP500"]   = corrs.get("SP500")

        # Geo risk.
        gs = geo_score_fallback(info, news, symbol)
        r["Geo_Score"] = gs
        r["Geo_Risk"]  = "Low" if gs <= 3 else ("Med" if gs <= 6 else "High")

        # Composite score.
        total, bd = deep_score(
            r["PE"], r["PEG"], r["ROE"], r["RSI"],
            r["Upside_Pct"], gs, r["Vol_Ann_pct"], r["AI_Sentiment"],
        )
        r["Score"]              = total
        r["Score_Fundamental"]  = bd["Fundamental"]
        r["Score_Technical"]    = bd["Technical"]
        r["Score_GeoSentiment"] = bd["GeoSentiment"]

        # Trade signal.
        r["Signal"] = trade_signal(total, r["Upside_Pct"], r["RSI"], r["AI_Sentiment"])

        # Backtest — pass raw_currency so run_backtest normalises correctly.
        bt = run_backtest(
            symbol, hist,
            currency=raw_currency,
            pe=r["PE"],
            peg=r["PEG"],
            roe=r["ROE"],
            geo=gs,
            ai_sentiment=r["AI_Sentiment"],
        )
        r.update(bt)

    except Exception as e:
        r["Note"] = str(e)[:100]
    return r

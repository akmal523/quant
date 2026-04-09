"""
analyzer.py — per-asset analysis orchestrator.

Changes from v4.5:
- Pulls new indicators: MACD, Bollinger Bands, volume trend, EMA200.
- Pulls new fundamental fields: EV/EBITDA, FCF yield, debt/equity, revenue growth YoY.
- Passes all new fields to deep_score() and trade_signal() in scoring.py.
- Macro correlations now include rolling 6-month correlations and ROC signals
  from the updated macro.py.
- Backtest result dict now includes multi-entry stats (win rate, Sharpe).
- Result dict keys are strictly additive — downstream reporting.py and CSV
  output see all existing keys plus new ones; nothing renamed.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from config import ATR_MULTIPLIER, PRICE_WARN_PCT
from currency import get_fx_rates, currency_symbol, format_price, eur_equivalent
from indicators import (
    ema, ema200, rsi, atr, safe_float,
    macd as calc_macd,
    bollinger as calc_bollinger,
    volume_trend as calc_volume_trend,
)
from macro import corr_with_macro, macro_roc
from news import fetch_rss_news
from gemini_ai import get_ai_sentiment
from scoring import geo_score_fallback, deep_score, trade_signal
from backtest import run_backtest


def analyze(name, symbol, macro, prefetched_ai=None, prefetched_news=None, config=None):
    """
    Run the full analysis pipeline for a single instrument.

    Parameters
    ----------
    name            : Human-readable asset name.
    symbol          : Yahoo Finance ticker.
    macro           : Pre-loaded macro series dict (from macro.load_macro).
    prefetched_ai   : (score, summary) tuple from batch AI phase, or None.
    prefetched_news : News list from batch news phase, or None.
    config          : Optional dict of calibrated parameter overrides.

    Returns
    -------
    Flat dict of all metrics (DataFrame row).
    """
    config = config or {}
    # Calibrated indicator periods — passed to indicator functions below.
    rsi_period  = config.get("rsi_period",  14)
    ema_fast    = config.get("ema_fast",     12)
    ema_slow    = config.get("ema_slow",     26)
    macd_signal = config.get("macd_signal",   9)

    r = dict(
        Asset=name, Ticker=symbol, Sector="N/A", Country="N/A",
        Type="Equity", Currency="N/A", CurrencySymbol="",
        Entry_Price=None, EMA50=None, EMA200=None, Target_Price=None,
        Stop_Loss=None, ATR=None, Resistance=None,
        Entry_Fmt="—", Target_Fmt="—", Stop_Fmt="—",
        # Technical
        RSI=None, CAGR_5Y_pct=None, Vol_Ann_pct=None,
        MACD_Line=None, MACD_Signal=None, MACD_Histogram=None,
        MACD_Bullish_Cross=False, MACD_Bearish_Cross=False,
        BB_Upper=None, BB_Mid=None, BB_Lower=None,
        BB_PctB=None, BB_Bandwidth=None,
        Vol_Trend_Weak=False, Vol_Ratio=None,
        # Fundamentals
        PE=None, PEG=None, ROE=None, PB=None, DivYield_pct=None,
        EV_EBITDA=None, FCF_Yield=None, Debt_Equity=None, Revenue_Growth=None,
        Analyst_Target=None, Upside_Pct=None,
        # AI / Sentiment
        AI_Sentiment=None, AI_Summary="",
        # Macro correlations — full history
        Corr_CPI=None, Corr_CB_Rate=None,
        Corr_Oil=None, Corr_Gold=None, Corr_SP500=None,
        # Macro correlations — rolling 6-month
        RollCorr_Oil=None, RollCorr_Gold=None, RollCorr_SP500=None,
        # Macro ROC signals
        ROC1M_Oil=None, ROC3M_Oil=None,
        ROC1M_Rates=None, ROC3M_Rates=None,
        # Geo / Scores
        Geo_Score=None, Geo_Risk="N/A",
        Score=None, Score_Fundamental=None,
        Score_Technical=None, Score_GeoSentiment=None,
        # Backtest
        bt_signal="N/A", bt_price_entry=None, bt_price_now=None,
        bt_pnl_pct=None, bt_stop_hit=False, bt_note="",
        bt_n_trades=0, bt_win_rate=None, bt_avg_pnl=None,
        bt_atr_wtd_pnl=None, bt_sharpe=None,
        # Output
        Signal="N/A", TOP_PICK="", Note="",
    )

    try:
        # Warm up FX cache (no-op if already loaded)
        get_fx_rates()

        tk = yf.Ticker(symbol)
        info = tk.info or {}

        if not info or info.get("quoteType") is None:
            r["Note"] = "No data from yfinance"
            return r

        qt = info.get("quoteType", "")
        is_etf = qt in ("ETF", "MUTUALFUND")
        r["Type"] = "ETF/Fund" if is_etf else "Equity"
        r["Sector"] = info.get("sector", "ETF/Fund" if is_etf else "N/A")
        r["Country"] = info.get("country", "N/A")

        # ── Currency ─────────────────────────────────────────────────────
        raw_currency = (
            info.get("currency")
            or info.get("financialCurrency")
            or "USD"
        )
        is_pence = raw_currency in ("GBp", "GBX", "GBx")
        display_currency = "GBP" if is_pence else raw_currency.upper()
        r["Currency"] = display_currency
        r["CurrencySymbol"] = currency_symbol(display_currency)

        # ── News ─────────────────────────────────────────────────────────
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
                print(f"  [News] yfinance 0 — trying Google News RSS...")
                news = fetch_rss_news(symbol, company_name=name)
                if news:
                    print(f"  [News] RSS returned {len(news)} for {symbol}")

        # ── AI Sentiment ─────────────────────────────────────────────────
        if prefetched_ai is not None:
            ai_score, ai_summary = prefetched_ai
        else:
            ai_score, ai_summary = get_ai_sentiment(name, news)
        r["AI_Sentiment"] = ai_score
        r["AI_Summary"] = ai_summary

        # ── Fundamentals (equities only) ─────────────────────────────────
        if not is_etf:
            r["PE"]  = safe_float(info.get("trailingPE"))
            r["PEG"] = safe_float(info.get("pegRatio"))
            r["PB"]  = safe_float(info.get("priceToBook"))
            r["ROE"] = safe_float(info.get("returnOnEquity"))

            div = info.get("dividendYield")
            r["DivYield_pct"] = round(float(div) * 100, 2) if div else None

            # EV/EBITDA
            r["EV_EBITDA"] = safe_float(info.get("enterpriseToEbitda"))

            # Free cash flow yield = FCF / market cap
            fcf    = safe_float(info.get("freeCashflow"))
            mktcap = safe_float(info.get("marketCap"))
            if fcf is not None and mktcap and mktcap > 0:
                r["FCF_Yield"] = round(fcf / mktcap, 4)

            # Debt/equity — yfinance returns as percentage (e.g. 120 = 1.2×), normalise
            r["Debt_Equity"] = safe_float(info.get("debtToEquity"))
            if r["Debt_Equity"] is not None:
                r["Debt_Equity"] = round(r["Debt_Equity"] / 100, 3)

            # Revenue growth YoY
            r["Revenue_Growth"] = safe_float(info.get("revenueGrowth"))

            tgt = info.get("targetMeanPrice")
            cpx = info.get("currentPrice") or info.get("regularMarketPrice")
            if is_pence:
                if cpx: cpx = float(cpx) / 100
                if tgt: tgt = float(tgt) / 100
            if tgt and cpx:
                r["Analyst_Target"] = round(float(tgt), 4)
                r["Upside_Pct"] = round(
                    (float(tgt) - float(cpx)) / float(cpx) * 100, 1)
                r["Entry_Price"] = round(float(cpx), 4)

        # ── Historical price data ────────────────────────────────────────
        hist = yf.download(
            symbol, start="2018-01-01",
            auto_adjust=True, progress=False,
        )
        if hist.empty or "Close" not in hist.columns:
            r["Note"] = "No price history"
            return r

        close_raw = hist["Close"].squeeze()
        high_raw  = hist["High"].squeeze() if "High"   in hist.columns else close_raw
        low_raw   = hist["Low"].squeeze()  if "Low"    in hist.columns else close_raw
        vol_raw   = hist["Volume"].squeeze() if "Volume" in hist.columns else None

        close = close_raw / 100 if is_pence else close_raw
        high  = high_raw  / 100 if is_pence else high_raw
        low   = low_raw   / 100 if is_pence else low_raw

        # EMA50
        e50 = float(ema(close, 50).iloc[-1])
        r["EMA50"] = round(e50, 4)
        if r["Entry_Price"] is None:
            r["Entry_Price"] = round(e50, 4)

        # EMA200
        e200 = ema200(close)
        r["EMA200"] = round(e200, 4) if e200 else None

        # Price anomaly warning
        if r["Entry_Price"] and e50 and e50 > 0:
            deviation = abs(r["Entry_Price"] - e50) / e50
            if deviation > PRICE_WARN_PCT:
                r["Note"] = (
                    r["Note"] + " | Price Warning: "
                    f"{deviation*100:.0f}% from EMA50"
                ).strip(" | ")

        # ATR and stop-loss
        atr_val = atr(high, low, close)
        if atr_val:
            r["ATR"]       = round(atr_val, 4)
            r["Stop_Loss"] = round(r["Entry_Price"] - ATR_MULTIPLIER * atr_val, 4)

        # Resistance: 60-bar swing high
        r["Resistance"] = round(float(high.tail(60).max()), 4)

        # Target price
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

        # RSI — uses calibrated period
        rsi_val = rsi(close, period=rsi_period)
        r["RSI"] = round(rsi_val, 1) if rsi_val else None

        # MACD — uses calibrated fast/slow/signal periods
        m = calc_macd(close, fast=ema_fast, slow=ema_slow, signal=macd_signal)
        r["MACD_Line"]          = round(m["macd_line"],   4) if m["macd_line"]   else None
        r["MACD_Signal"]        = round(m["signal_line"], 4) if m["signal_line"] else None
        r["MACD_Histogram"]     = m["histogram"]
        r["MACD_Bullish_Cross"] = m["bullish_cross"]
        r["MACD_Bearish_Cross"] = m["bearish_cross"]

        # Bollinger Bands
        bb = calc_bollinger(close)
        r["BB_Upper"]     = bb["bb_upper"]
        r["BB_Mid"]       = bb["bb_mid"]
        r["BB_Lower"]     = bb["bb_lower"]
        r["BB_PctB"]      = bb["bb_pct_b"]
        r["BB_Bandwidth"] = bb["bb_bandwidth"]

        # Volume trend
        if vol_raw is not None:
            vt = calc_volume_trend(close, vol_raw)
            r["Vol_Trend_Weak"] = vt["weak_signal"]
            r["Vol_Ratio"]      = vt["vol_ratio"]

        # 5-year CAGR
        c5 = close[close.index >= datetime.now() - timedelta(days=5 * 365)]
        if len(c5) > 50 and float(c5.iloc[0]) > 0:
            r["CAGR_5Y_pct"] = round(
                ((float(c5.iloc[-1]) / float(c5.iloc[0])) ** 0.2 - 1) * 100, 1)

        # Annualised volatility
        ret = close.pct_change().dropna().tail(252)
        if len(ret) > 20:
            r["Vol_Ann_pct"] = round(float(ret.std()) * np.sqrt(252) * 100, 1)

        # Macro correlations (full + rolling) and ROC
        corrs = corr_with_macro(close, macro)
        r["Corr_CPI"]     = corrs.get("full_TIPS_INF")
        r["Corr_CB_Rate"] = corrs.get("full_FED_PROXY")
        r["Corr_Oil"]     = corrs.get("full_OIL")
        r["Corr_Gold"]    = corrs.get("full_GOLD")
        r["Corr_SP500"]   = corrs.get("full_SP500")
        r["RollCorr_Oil"]   = corrs.get("roll_OIL")
        r["RollCorr_Gold"]  = corrs.get("roll_GOLD")
        r["RollCorr_SP500"] = corrs.get("roll_SP500")

        roc = macro_roc(macro)
        r["ROC1M_Oil"]   = roc.get("roc1m_OIL")
        r["ROC3M_Oil"]   = roc.get("roc3m_OIL")
        r["ROC1M_Rates"] = roc.get("roc1m_FED_PROXY")
        r["ROC3M_Rates"] = roc.get("roc3m_FED_PROXY")

        # Geo risk
        gs = geo_score_fallback(info, news, symbol)
        r["Geo_Score"] = gs
        r["Geo_Risk"]  = "Low" if gs <= 3 else ("Med" if gs <= 6 else "High")

        # Composite score
        total, bd = deep_score(
            pe=r["PE"], peg=r["PEG"], roe=r["ROE"],
            sector=r["Sector"],
            ev_ebitda=r["EV_EBITDA"],
            fcf_yield=r["FCF_Yield"],
            debt_equity=r["Debt_Equity"],
            revenue_growth=r["Revenue_Growth"],
            rsi_val=r["RSI"], upside=r["Upside_Pct"],
            #macd_bullish_cross=r["MACD_Bullish_Cross"],
            bb_pct_b=r["BB_PctB"],
            geo=gs, vol=r["Vol_Ann_pct"], ai_sentiment=r["AI_Sentiment"],
            is_etf=is_etf,
        )
        r["Score"]              = total
        r["Score_Fundamental"]  = bd["Fundamental"]
        r["Score_Technical"]    = bd["Technical"]
        r["Score_GeoSentiment"] = bd["GeoSentiment"]

        # Trade signal
        r["Signal"] = trade_signal(
            score=total, upside=r["Upside_Pct"], rsi_val=r["RSI"],
            ai_sentiment=r["AI_Sentiment"],
            macd_bullish_cross=r["MACD_Bullish_Cross"],
            macd_bearish_cross=r["MACD_Bearish_Cross"],
            bb_pct_b=r["BB_PctB"],
            ema200_val=r["EMA200"],
            price=r["Entry_Price"],
        )

        # Backtest
        bt = run_backtest(symbol, hist, currency=raw_currency)
        r.update({k: v for k, v in bt.items() if k != "bt_trades"})

    except Exception as e:
        r["Note"]   = str(e)[:100]
        r["Signal"] = "N/A"

    return r

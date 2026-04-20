"""
scoring.py — Composite scoring, horizon classification, sector medians, trade signal.

Score architecture (0–100):
  Fundamental   max 40  — PE, PEG, ROE; sector-relative PE bonus
  Technical     max 30  — RSI(14), Price vs SMA50
  GeoSentiment  max 30  — geo-risk heuristic + FinBERT score [-100,+100]
  Volatility penalty    — up to −3 pts for annualised vol > 60%
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from universe import GEO_BASE, GEO_KEYWORDS


# ─── Investment Horizon Classification ───────────────────────────────────────

def classify_horizon(div_yield, vol, roe, pe) -> str:
    """
    Classify asset into one of three investment horizon buckets.

    RETIREMENT (20yr+)        — low vol, dividend payer, sound fundamentals
    BUSINESS COLLATERAL (10yr) — stable growth, low beta proxy, suitable as
                                 collateral for Lombard / corporate loans
    SPECULATIVE (short-term)  — high momentum / volatility, news-driven
    """
    div = float(div_yield) if div_yield else 0.0
    v   = float(vol)       if vol       else 1.0
    r   = float(roe)       if roe       else 0.0
    p   = float(pe)        if pe        else 999.0

    if div >= 0.02 and v < 0.25 and r > 0.10 and 0 < p < 30:
        return "RETIREMENT (20yr+)"
    if v < 0.35 and r > 0.05 and 0 < p < 40:
        return "BUSINESS COLLATERAL (10yr)"
    return "SPECULATIVE (short-term)"


# ─── Geo Risk ─────────────────────────────────────────────────────────────────

def geo_score(country: str, sector: str, symbol: str, headlines: list[dict]) -> int:
    """
    Heuristic geo-risk score in [1, 10]. Higher = riskier.
    Components:
      • Country base score (GEO_BASE table)
      • News keyword bump (max +3)
      • Sector override (+1 for Energy / Industrials)
      • Symbol-level overrides (Samsung, Raiffeisen)
    """
    base = GEO_BASE.get(country, 4)

    text = " ".join(
        (h.get("title", "") + " " + h.get("summary", "")).lower()
        for h in (headlines or [])
    )
    hits = sum(1 for kw in GEO_KEYWORDS if kw in text)
    bump = min(hits * 0.5, 3.0)

    if symbol in ("005930.KS",):           # Samsung — Korean peninsula risk
        base = max(base, 7)
    if symbol in ("RBI.VI",):              # Raiffeisen — Russia/Ukraine exposure
        base = max(base, 8)
    if sector in ("Energy", "Industrials"):
        base = min(base + 1, 10)

    return min(int(round(base + bump)), 10)


# ─── Sector Medians ───────────────────────────────────────────────────────────

def calculate_sector_medians(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-sector medians for PE, PEG, ROE.

    yfinance occasionally returns the *string* "Infinity" for PE (e.g. when
    earnings are zero/negative and the ratio overflows).  pandas cannot compute
    median on a mixed object/numeric column, so we:
      1. coerce every value to float64  (strings → NaN via errors='coerce')
      2. replace ±inf with NaN
    before grouping — so the median is always computed on clean float data.
    """
    cols      = ["PE", "PEG", "ROE"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.DataFrame()

    work = df[["Sector"] + available].copy()
    for c in available:
        work[c] = pd.to_numeric(work[c], errors="coerce")    # "Infinity" → NaN
        work[c] = work[c].where(np.isfinite(work[c].fillna(0)), other=np.nan)

    return work.groupby("Sector")[available].median()


# ─── Composite Score (0–100) ──────────────────────────────────────────────────

def composite_score(
    pe,
    peg,
    roe,
    rsi_val,
    price_vs_sma50,
    geo_val,
    vol,
    finbert_score,
    sector_medians: pd.Series | None = None,
) -> tuple[int, dict]:
    """
    Three-bucket composite score.
    Returns (total_score: int, breakdown: dict).
    All inputs accept None gracefully.
    """
    bd: dict = {}

    # ── Fundamental (max 40) ─────────────────────────────────────────────────
    f = 0

    pe_f  = float(pe)  if pe  is not None else None
    peg_f = float(peg) if peg is not None else None
    roe_f = float(roe) if roe is not None else None

    if pe_f is not None and pe_f > 0:
        f += 15 if pe_f < 20 else (10 if pe_f < 30 else 4)

    if peg_f is not None and peg_f > 0:
        f += 15 if peg_f < 1 else (8 if peg_f < 2 else 3)

    if roe_f is not None:
        f += 10 if roe_f > 0.20 else (6 if roe_f > 0.10 else (3 if roe_f > 0 else 0))

    # Sector-relative bonus: asset PE ≤ 80% of sector median → value discount
    if sector_medians is not None and pe_f is not None and pe_f > 0:
        med_pe = sector_medians.get("PE") if hasattr(sector_medians, "get") else None
        if med_pe and float(med_pe) > 0 and pe_f < float(med_pe) * 0.80:
            f = min(f + 5, 40)

    bd["Fundamental_Score"] = min(f, 40)

    # ── Technical (max 30) ───────────────────────────────────────────────────
    t = 0

    if rsi_val is not None:
        rv = float(rsi_val)
        # 30–60: accumulation zone; 60–70: extended; >70: overbought
        t += 15 if 30 <= rv <= 60 else (8 if rv <= 70 else 2)

    if price_vs_sma50 is not None:
        pv = float(price_vs_sma50)
        # Price in 0.97–1.15× SMA50 = healthy uptrend
        t += 15 if 0.97 <= pv <= 1.15 else (8 if pv < 0.97 else 3)

    bd["Technical_Score"] = min(t, 30)

    # ── GeoSentiment (max 30) ────────────────────────────────────────────────
    g = 0

    if geo_val is not None:
        g += 15 if int(geo_val) <= 3 else (10 if int(geo_val) <= 6 else 3)

    if finbert_score is not None and isinstance(finbert_score, (int, float)):
        # Map [-100, +100] → [0, 15]
        sent_pts = int((float(finbert_score) + 100) / 200 * 15)
        g += max(0, min(sent_pts, 15))

    # Volatility penalty: annualised vol > 60% → −3 pts from geo-sentiment bucket
    vol_penalty = 0
    if vol is not None and float(vol) > 0.60:
        vol_penalty = 3
        g = max(0, g - vol_penalty)

    bd["GeoSentiment_Score"] = min(g, 30)
    bd["Volatility_Penalty"] = vol_penalty

    total = bd["Fundamental_Score"] + bd["Technical_Score"] + bd["GeoSentiment_Score"]
    return min(total, 100), bd


# ─── Trade Signal ─────────────────────────────────────────────────────────────

def trade_signal(score: int, rsi_val, finbert_score, upside=None) -> str:
    """
    Derive BUY / HOLD / SELL.

    Adjusted score rule: if FinBERT score < −50 (confirmed bearish news),
    effective score is reduced by 15 pts to prevent buying into bad news.
    """
    rsi = float(rsi_val)      if rsi_val      is not None          else 50.0
    fb  = float(finbert_score) if isinstance(finbert_score, (int, float)) else 0.0

    adj = max(0, score - 15) if fb < -50 else score

    if adj >= 65 and (upside is None or float(upside) > 5):
        return "BUY"
    if adj >= 50:
        return "HOLD"
    if rsi > 72 or (upside is not None and float(upside) < -10):
        return "SELL"
    return "HOLD"


# ─── Pipeline Wrapper ─────────────────────────────────────────────────────────

def execute_scoring_pipeline(row: dict, sector_medians: pd.Series | None) -> dict:
    """
    Run composite_score + trade_signal on a single asset data row.
    Returns a dict of new columns to merge back into the main DataFrame.
    """
    total, bd = composite_score(
        pe              = row.get("PE"),
        peg             = row.get("PEG"),
        roe             = row.get("ROE"),
        rsi_val         = row.get("RSI"),
        price_vs_sma50  = row.get("Price_vs_SMA50"),
        geo_val         = row.get("Geo_Risk"),
        vol             = row.get("Volatility"),
        finbert_score   = row.get("FinBERT_Score"),
        sector_medians  = sector_medians,
    )
    signal = trade_signal(
        total,
        row.get("RSI"),
        row.get("FinBERT_Score"),
        row.get("Upside_pct"),
    )
    return {
        "Total_Score":        total,
        "Fundamental_Score":  bd.get("Fundamental_Score", 0),
        "Technical_Score":    bd.get("Technical_Score",   0),
        "GeoSentiment_Score": bd.get("GeoSentiment_Score",0),
        "Volatility_Penalty": bd.get("Volatility_Penalty",0),
        "Signal":             signal,
    }

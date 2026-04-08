"""
scoring.py — geo-risk scoring, composite deep-score (0–100), trade signal.

Changes from v4.5:
- All bucket scoring replaced with continuous sigmoid/linear functions —
  no more hard step-thresholds that cause cliff edges.
- Fundamental bucket now includes:
    - Sector-relative PE (vs sector median, not absolute threshold)
    - EV/EBITDA (capital-structure-neutral valuation)
    - Free cash flow yield (catches negative-PE growth companies)
    - Debt/equity ratio (risk penalty)
    - Revenue growth YoY
- Bucket weights are dynamic:
    - Equities:  Fundamental 40 / Technical 30 / GeoSentiment 30
    - ETFs/Funds: Fundamental 0  / Technical 50 / GeoSentiment 50
      (ETFs have no meaningful PE/ROE — reallocate weight to technical/macro)
- trade_signal: also uses MACD crossover and Bollinger %B as confirmation
"""

import math
from tickers import GEO_BASE, GEO_KEYWORDS

# ─────────────────────────────────────────────────────────────────────────────
# SECTOR PE MEDIANS  (approximate trailing PE medians, update periodically)
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_PE_MEDIAN = {
    "Technology":             28.0,
    "Communication Services": 22.0,
    "Consumer Discretionary": 24.0,
    "Consumer Staples":       20.0,
    "Health Care":            22.0,
    "Financials":             13.0,
    "Industrials":            20.0,
    "Materials":              16.0,
    "Energy":                 11.0,
    "Utilities":              16.0,
    "Real Estate":            35.0,   # REIT P/E is structurally high
    "ETF/Fund":               20.0,   # fallback only, not used for ETFs
    "N/A":                    20.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTINUOUS SCORING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float, center: float, steepness: float = 1.0) -> float:
    """
    Logistic function centred on `center`, returns value in (0, 1).
    Higher x → higher score. steepness controls how quickly it transitions.
    """
    try:
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))
    except OverflowError:
        return 1.0 if x > center else 0.0


def _linear_clamp(x: float, lo: float, hi: float) -> float:
    """Map x linearly onto [0, 1] between lo (→0) and hi (→1), clamped."""
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


# ─────────────────────────────────────────────────────────────────────────────
# GEO RISK
# ─────────────────────────────────────────────────────────────────────────────

def geo_score_fallback(info: dict, news: list, symbol: str) -> int:
    """
    Heuristic geo-risk score (1–10). Higher = riskier.
    Unchanged from v4.5 — geo risk is categorical, steps are appropriate here.
    """
    country = info.get("country", "Unknown")
    base = GEO_BASE.get(country, 4)
    text = " ".join(
        (n.get("title", "") + " " + n.get("summary", "")).lower()
        for n in (news or [])
    )
    hits = sum(1 for kw in GEO_KEYWORDS if kw in text)
    bump = min(hits * 0.5, 3)

    if symbol == "005930.KS":
        base = max(base, 7)
    if symbol == "RBI.VI":
        base = max(base, 8)
    if info.get("sector") in ("Energy", "Industrials"):
        base = min(base + 1, 10)

    return min(int(round(base + bump)), 10)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE (0–100)
# ─────────────────────────────────────────────────────────────────────────────

def deep_score(
    # fundamentals
    pe, peg, roe, sector,
    ev_ebitda, fcf_yield, debt_equity, revenue_growth,
    # technical
    rsi_val, upside,
    # macd / bollinger (optional, used as confirmation only)
    macd_bullish_cross=False, bb_pct_b=None,
    # macro/sentiment
    geo=None, vol=None, ai_sentiment=None,
    # asset type
    is_etf: bool = False,
) -> tuple[int, dict]:
    """
    Compute a composite score in [0, 100].

    Bucket weights:
      Equity:   Fundamental 40 / Technical 30 / GeoSentiment 30
      ETF/Fund: Fundamental  0 / Technical 50 / GeoSentiment 50

    All sub-scores use continuous functions to avoid threshold cliff edges.
    Returns (total_score, breakdown_dict).
    """
    bd: dict = {}

    sector_key = sector if sector in SECTOR_PE_MEDIAN else "N/A"
    sector_median_pe = SECTOR_PE_MEDIAN[sector_key]

    # ── Fundamental bucket ────────────────────────────────────────────────
    if is_etf:
        bd["Fundamental"] = 0
    else:
        f_pts = 0.0

        # Sector-relative PE (max 12 pts)
        # Score peaks when PE is 30% below sector median; decays above median.
        if pe is not None and pe > 0:
            pe_ratio = float(pe) / sector_median_pe   # 1.0 = at median
            # sigmoid centred at 0.7 (30% discount) — score > 0.5 when below median
            f_pts += 12 * _sigmoid(-pe_ratio, -0.85, steepness=4)

        # PEG (max 8 pts) — continuous: PEG < 1 is good, PEG > 3 is bad
        if peg is not None:
            f_pts += 8 * _sigmoid(-float(peg), -1.5, steepness=1.5)

        # EV/EBITDA (max 8 pts) — lower is cheaper; < 8 very cheap, > 20 expensive
        if ev_ebitda is not None and ev_ebitda > 0:
            f_pts += 8 * _sigmoid(-float(ev_ebitda), -12, steepness=0.3)

        # FCF yield (max 8 pts) — higher is better; 0%→0 pts, 8%+→full pts
        if fcf_yield is not None:
            f_pts += 8 * _linear_clamp(float(fcf_yield) * 100, 0, 8)

        # ROE (max 6 pts)
        if roe is not None:
            f_pts += 6 * _linear_clamp(float(roe), 0.0, 0.25)

        # Revenue growth YoY (max 5 pts) — negative growth scores 0
        if revenue_growth is not None:
            f_pts += 5 * _linear_clamp(float(revenue_growth), 0.0, 0.20)

        # Debt/equity penalty (subtracts up to 7 pts) — D/E > 2 is penalised
        if debt_equity is not None and debt_equity > 0:
            penalty = 7 * _linear_clamp(float(debt_equity), 0.5, 3.0)
            f_pts -= penalty

        bd["Fundamental"] = min(max(round(f_pts), 0), 40)

    # ── Technical bucket ─────────────────────────────────────────────────
    t_pts = 0.0
    max_tech = 50 if is_etf else 30

    # RSI — sweet spot 35–55; score decays toward 30 (oversold) and 70 (overbought)
    if rsi_val is not None:
        rv = float(rsi_val)
        # Bell-shape peaking at RSI = 45
        rsi_norm = math.exp(-0.5 * ((rv - 45) / 15) ** 2)
        t_pts += (max_tech * 0.50) * rsi_norm

    # Analyst-target upside — continuous from 0% to 40%
    if upside is not None:
        t_pts += (max_tech * 0.35) * _linear_clamp(float(upside), -5, 40)

    # MACD bullish crossover confirmation (+bonus, not primary signal)
    if macd_bullish_cross:
        t_pts += max_tech * 0.08

    # Bollinger %B — score highest near 0.3 (near lower band = value entry)
    # %B > 0.8 (near upper band) gets a small penalty
    if bb_pct_b is not None:
        b = float(bb_pct_b)
        if b < 0.5:
            t_pts += (max_tech * 0.07) * (1 - b / 0.5)
        else:
            t_pts -= (max_tech * 0.04) * min((b - 0.5) / 0.5, 1)

    bd["Technical"] = min(max(round(t_pts), 0), max_tech)

    # ── Geo + Sentiment bucket ────────────────────────────────────────────
    max_geo = 50 if is_etf else 30
    g_pts = 0.0

    if geo is not None:
        # Continuous: geo 1 → full score, geo 10 → 0
        g_pts += (max_geo * 0.50) * _linear_clamp(-float(geo), -10, -1)

    if ai_sentiment is not None and isinstance(ai_sentiment, (int, float)):
        # Sentiment maps from [−100, +100] to [0, max_geo*0.50]
        g_pts += (max_geo * 0.50) * _linear_clamp(float(ai_sentiment), -100, 100)

    # Volatility penalty: > 60% annualised → graduated reduction
    if vol is not None:
        excess = max(0.0, float(vol) - 60)
        g_pts -= min(excess * 0.05, max_geo * 0.15)

    bd["GeoSentiment"] = min(max(round(g_pts), 0), max_geo)

    total = bd["Fundamental"] + bd["Technical"] + bd["GeoSentiment"]
    return min(total, 100), bd


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def trade_signal(
    score: int, upside, rsi_val, ai_sentiment,
    macd_bullish_cross: bool = False,
    macd_bearish_cross: bool = False,
    bb_pct_b: float | None = None,
    ema200_val: float | None = None,
    price: float | None = None,
) -> str:
    """
    Derive BUY / HOLD / SELL.

    Logic:
    - Strong negative AI sentiment (< −50) applies a −15 pt penalty.
    - BUY requires score ≥ 65 AND upside > 5%.
      MACD bullish crossover or Bollinger %B < 0.25 each lower the BUY
      threshold by 3 pts (confirmation lowers the bar slightly).
    - Price below EMA200 disqualifies BUY (no buying into downtrend).
    - MACD bearish crossover or RSI > 72 can trigger SELL regardless of score.
    - Bollinger %B > 0.95 (extreme upper band) adds a SELL signal.
    """
    rsi_val = float(rsi_val) if rsi_val is not None else 50
    ai_num = float(ai_sentiment) if isinstance(ai_sentiment, (int, float)) else 0

    adj_score = max(0, score - 15) if ai_num < -50 else score

    # Confirmation bonuses lower BUY threshold
    buy_threshold = 65
    if macd_bullish_cross:
        buy_threshold -= 3
    if bb_pct_b is not None and bb_pct_b < 0.25:
        buy_threshold -= 3

    # Long-term trend filter
    above_ema200 = True
    if ema200_val is not None and price is not None and price > 0:
        above_ema200 = price >= ema200_val * 0.97  # 3% tolerance

    # SELL conditions
    hard_sell = (
        rsi_val > 72
        or (upside is not None and float(upside) < -10)
        or macd_bearish_cross
        or (bb_pct_b is not None and bb_pct_b > 0.95)
    )
    if hard_sell:
        return "SELL"

    if adj_score >= buy_threshold and above_ema200 and (upside is None or float(upside) > 5):
        return "BUY"

    if adj_score >= 50:
        return "HOLD"

    return "HOLD"

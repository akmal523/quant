"""
scoring.py — geo-risk scoring, composite deep-score (0–100), trade signal.
"""
from tickers import GEO_BASE, GEO_KEYWORDS


# ─────────────────────────────────────────────────────────────────────────────
# GEO RISK
# ─────────────────────────────────────────────────────────────────────────────
def geo_score_fallback(info: dict, news: list, symbol: str) -> int:
    """
    Heuristic geo-risk score (1–10) based on country, sector, and news keywords.
    Higher = riskier.
    """
    country = info.get("country", "Unknown")
    base    = GEO_BASE.get(country, 4)

    text = " ".join(
        (n.get("title", "") + " " + n.get("summary", "")).lower()
        for n in (news or [])
    )
    hits = sum(1 for kw in GEO_KEYWORDS if kw in text)
    bump = min(hits * 0.5, 3)

    # Explicit overrides for known high-risk domiciles.
    if symbol == "005930.KS":   # Samsung — Korea geopolitical risk
        base = max(base, 7)
    if symbol == "RBI.VI":      # Raiffeisen — Russia/Ukraine exposure
        base = max(base, 8)

    if info.get("sector") in ("Energy", "Industrials"):
        base = min(base + 1, 10)

    return min(int(round(base + bump)), 10)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE  (0–100)
# ─────────────────────────────────────────────────────────────────────────────
def deep_score(pe, peg, roe, rsi_val, upside, geo, vol,
               ai_sentiment) -> tuple[int, dict]:
    """
    Compute a composite score broken into three buckets:

      Fundamental  (max 40) — PE, PEG, ROE
      Technical    (max 30) — RSI, analyst-target upside
      GeoSentiment (max 30) — geo risk, AI sentiment, volatility penalty

    Returns (total_score, breakdown_dict).
    """
    bd: dict = {}

    # ── Fundamental (40) ─────────────────────────────────────────────────────
    f = 0
    if pe is not None:
        f += 15 if 0 < float(pe) < 20 else (10 if float(pe) < 30 else 5)
    if peg is not None:
        f += 15 if float(peg) < 1 else (8 if float(peg) < 2 else 3)
    if roe is not None:
        r = float(roe)
        f += 10 if r > 0.20 else (6 if r > 0.10 else (3 if r > 0 else 0))
    bd["Fundamental"] = min(f, 40)

    # ── Technical (30) ───────────────────────────────────────────────────────
    t = 0
    if rsi_val is not None:
        rv = float(rsi_val)
        t += 15 if 30 <= rv <= 60 else (8 if rv <= 70 else 3)
    if upside is not None:
        up = float(upside)
        t += 15 if up > 25 else (10 if up > 10 else (5 if up > 0 else 0))
    bd["Technical"] = min(t, 30)

    # ── Geo + Sentiment (30) ─────────────────────────────────────────────────
    g = 0
    if geo is not None:
        g += 15 if geo <= 3 else (10 if geo <= 6 else 3)
    # ai_sentiment == 0 with "No News" summary → treat as neutral, not bearish.
    if ai_sentiment is not None and isinstance(ai_sentiment, (int, float)):
        sent_pts = int((float(ai_sentiment) + 100) / 200 * 15)
        g += sent_pts
    if vol is not None and float(vol) > 60:
        g = max(0, g - 3)  # slight penalty for very high annualised volatility
    bd["GeoSentiment"] = min(g, 30)

    total = sum(bd.values())
    return min(total, 100), bd


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
def trade_signal(score: int, upside, rsi_val, ai_sentiment) -> str:
    """
    Derive BUY / HOLD / SELL from composite score, upside, RSI, and AI sentiment.

    Strong negative AI sentiment (< −50) reduces the effective score by 15 pts
    to prevent buying into confirmed bad news.
    """
    rsi_val = float(rsi_val) if rsi_val is not None else 50
    ai_num  = float(ai_sentiment) if isinstance(ai_sentiment, (int, float)) else 0
    adj_score = max(0, score - 15) if ai_num < -50 else score

    if adj_score >= 65 and (upside is None or float(upside) > 5):
        return "BUY"
    if adj_score >= 50:
        return "HOLD"
    if rsi_val > 72 or (upside is not None and float(upside) < -10):
        return "SELL"
    return "HOLD"

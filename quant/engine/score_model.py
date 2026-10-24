"""
score_model.py — the 0-100 score behind Research, ideas, swaps and the score floor.

Intent (v10.8.10): one small, explainable model for a private long-term investor.
It replaces a composite that scored missing data at 12/100 (so every ETF sat at
~27, "below the 40 floor", and nothing could reach high confidence), used one
month of momentum (noise), counted stewardship twice and read debt/equity in the
wrong unit.

Components (each 0-100, or None when it cannot be computed):

  technical    trend for a monthly buyer: 12-minus-1-month momentum and the
               price against its 200-day average (needs ~1 year of closes;
               falls back to the 200-day average alone with >= 200 closes).
  fundamental  stocks only: valuation (P/E, PEG), quality (ROE) and balance
               sheet (debt/equity, interest cover). Needs 3 of the 5 inputs.
  sentiment    recency-weighted news tone (half-life 7 days, last 30 days),
               needs >= SENTIMENT_MIN_DOCS headlines.

The composite is the weighted mean over the components that are present.
A component that does not apply (fundamentals of an ETF) is not "missing";
confidence is the present share of the applicable weight.

Invariants: pure functions, no I/O.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any

WEIGHTS = {"technical": 0.50, "fundamental": 0.35, "sentiment": 0.15}
SENTIMENT_HALF_LIFE_DAYS = 7.0
SENTIMENT_WINDOW_DAYS = 30
SENTIMENT_MIN_DOCS = 3
HIGH_CONFIDENCE = 0.8
MEDIUM_CONFIDENCE = 0.5
REGIME_BAND = 0.02        # index more than 2% above/below its 200-day average
REGIME_DAYS = 3           # ... on each of the last 3 closes (no one-day flips)


def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _clamp(x: float) -> float:
    return round(max(0.0, min(100.0, x)), 1)


def technical_score(closes: list[float]) -> float | None:
    """Trend score from daily closes (oldest first)."""
    px = [c for c in (_num(c) for c in closes) if c is not None and c > 0]
    if len(px) < 200:
        return None
    sma200 = sum(px[-200:]) / 200.0
    trend = math.tanh((px[-1] / sma200 - 1.0) * 5.0)        # +-10% vs the average
    if len(px) >= 253:
        mom = px[-22] / px[-253] - 1.0                       # 12 months, skip the last
        return _clamp(50.0 + 25.0 * trend + 25.0 * math.tanh(mom * 2.0))
    return _clamp(50.0 + 50.0 * trend)


def _debt_to_equity(raw: Any) -> float | None:
    """Yahoo reports debt/equity in percent (150 = 1.5x); normalize to a ratio."""
    de = _num(raw)
    if de is None:
        return None
    return de / 100.0 if de > 10.0 else de


def interest_cover(ebit: Any, interest_expense: Any) -> float | None:
    """EBIT / |interest expense|; None when either is unknown or interest is 0."""
    e, ie = _num(ebit), _num(interest_expense)
    if e is None or ie is None or ie == 0:
        return None
    return e / abs(ie)


def fundamental_score(f: dict | None) -> float | None:
    """Points for valuation, quality and balance sheet, scaled to 0-100."""
    if not f:
        return None
    pe, peg, roe = _num(f.get("pe")), _num(f.get("peg")), _num(f.get("roe"))
    de = _debt_to_equity(f.get("debt_to_equity"))
    icr = interest_cover(f.get("ebit"), f.get("interest_expense"))
    parts: list[tuple[float, float]] = []                    # (points, max)
    if pe is not None:
        parts.append((0 if pe <= 0 else 25 if pe <= 15 else 15 if pe <= 25
                      else 5 if pe <= 40 else 0, 25))
    if peg is not None:
        parts.append((0 if peg <= 0 else 15 if peg <= 1 else 8 if peg <= 2 else 0, 15))
    if roe is not None:
        parts.append((25 if roe >= 0.20 else 15 if roe >= 0.10 else 5 if roe > 0 else 0, 25))
    if de is not None:
        parts.append((20 if de <= 0.5 else 12 if de <= 1.0 else 5 if de <= 2.0 else 0, 20))
    if icr is not None:
        parts.append((15 if icr >= 8 else 8 if icr >= 3 else 0, 15))
    if len(parts) < 3:
        return None
    return _clamp(100.0 * sum(p for p, _ in parts) / sum(m for _, m in parts))


def sentiment_score(evidence: list[tuple[Any, Any]], today: date) -> float | None:
    """Recency-weighted tone of ``[(published_date, score -1..1), ...]``."""
    total = weight = 0.0
    n = 0
    for when, score in evidence or []:
        s = _num(score)
        if s is None or when is None:
            continue
        d = when if isinstance(when, date) else date.fromisoformat(str(when)[:10])
        age = (today - d).days
        if age < 0 or age > SENTIMENT_WINDOW_DAYS:
            continue
        w = 0.5 ** (age / SENTIMENT_HALF_LIFE_DAYS)
        total += w * max(-1.0, min(1.0, s))
        weight += w
        n += 1
    if n < SENTIMENT_MIN_DOCS or weight == 0:
        return None
    return _clamp(50.0 + 50.0 * total / weight)


def applicable(instrument_class: str | None) -> tuple[str, ...]:
    """Components that apply: fundamentals only exist for single stocks."""
    if str(instrument_class or "EQUITY").upper() == "EQUITY":
        return ("technical", "fundamental", "sentiment")
    return ("technical", "sentiment")


def composite(components: dict, instrument_class: str | None = None) -> dict:
    """Weighted mean of the present, applicable components + a confidence tier.

    Returns ``{composite, confidence, present, total, components}``;
    ``composite`` is None when no component is present.
    """
    names = applicable(instrument_class)
    present = {n: components.get(n) for n in names if components.get(n) is not None}
    total_w = sum(WEIGHTS[n] for n in names)
    present_w = sum(WEIGHTS[n] for n in present)
    value = (sum(WEIGHTS[n] * float(v) for n, v in present.items()) / present_w
             if present_w else None)
    share = present_w / total_w if total_w else 0.0
    confidence = ("high" if share >= HIGH_CONFIDENCE
                  else "medium" if share >= MEDIUM_CONFIDENCE else "low")
    return {"composite": None if value is None else round(value, 1),
            "confidence": confidence, "present": len(present), "total": len(names),
            "components": {n: components.get(n) for n in names}}


def trend_regime(closes: list[float]) -> tuple[str, float, str] | None:
    """``(bull|bear|chop, deviation, confidence)`` from index closes, or None."""
    px = [c for c in (_num(c) for c in closes) if c is not None and c > 0]
    if len(px) < 200 + REGIME_DAYS:
        return None
    devs = []
    for k in range(REGIME_DAYS):
        end = len(px) - k
        sma = sum(px[end - 200:end]) / 200.0
        devs.append(px[end - 1] / sma - 1.0)
    if all(d > REGIME_BAND for d in devs):
        label = "bull"
    elif all(d < -REGIME_BAND for d in devs):
        label = "bear"
    else:
        label = "chop"
    dev = devs[0]
    confidence = "high" if abs(dev) >= 0.05 else "medium" if abs(dev) >= REGIME_BAND else "low"
    return label, dev, confidence

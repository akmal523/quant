"""
scoring.py — geo-risk scoring, composite deep-score (0–100), trade signal.

Weight loading
--------------
At import time this module attempts to read `calibrated_weights.json` from the
same directory.  That file is written by `calibrate.py`.  If the file is absent
or malformed the original hardcoded values are used unchanged.

To inspect which source is active:
    from scoring import WEIGHTS_SOURCE, ACTIVE_WEIGHTS, ACTIVE_CAPS
    print(WEIGHTS_SOURCE)   # 'default' | 'calibrated (logreg)' | etc.

To force the defaults without deleting the JSON:
    import scoring
    scoring.ACTIVE_WEIGHTS = dict(scoring._DEFAULT_WEIGHTS)
    scoring.ACTIVE_CAPS    = dict(scoring._DEFAULT_CAPS)

Normalisation contract
----------------------
Each raw input is converted to a normalised sub-score ∈ [0, 1] using the
same step-function thresholds that existed in the original code.  The weight
multiplied against each normalised sub-score is what calibrate.py optimises.
With ACTIVE_WEIGHTS == _DEFAULT_WEIGHTS the output is bit-identical to the
original hardcoded scoring.py.
"""

import json
import os

from tickers import GEO_BASE, GEO_KEYWORDS

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT WEIGHTS  (original hardcoded values)
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_WEIGHTS: dict[str, float] = {
    # Fundamental bucket — cap 40
    "pe":          15.0,
    "peg":         15.0,
    "roe":         10.0,
    # Technical bucket — cap 30
    "rsi":         15.0,
    "upside":      15.0,
    # GeoSentiment bucket — cap 30
    "geo":         15.0,
    "sentiment":   15.0,
    "vol_penalty":  3.0,   # subtracted from GeoSentiment score
}

_DEFAULT_CAPS: dict[str, int] = {
    "fundamental": 40,
    "technical":   30,
    "geosent":     30,
}

# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT LOADING
# ─────────────────────────────────────────────────────────────────────────────

_WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "calibrated_weights.json")

_EXPECTED_WEIGHT_KEYS = set(_DEFAULT_WEIGHTS)
_EXPECTED_CAP_KEYS    = set(_DEFAULT_CAPS)


def _load_weights() -> tuple[dict, dict, str]:
    """
    Try to load calibrated_weights.json.

    Returns
    -------
    (weights, caps, source_label)
    Falls back to defaults silently if the file is absent, or with a printed
    warning if the file exists but is malformed.
    """
    if not os.path.exists(_WEIGHTS_FILE):
        return dict(_DEFAULT_WEIGHTS), dict(_DEFAULT_CAPS), "default"

    try:
        with open(_WEIGHTS_FILE) as fh:
            data = json.load(fh)

        raw_w = data.get("weights", {})
        raw_c = data.get("caps",    {})

        # Validate keys
        missing_w = _EXPECTED_WEIGHT_KEYS - set(raw_w)
        missing_c = _EXPECTED_CAP_KEYS   - set(raw_c)
        if missing_w or missing_c:
            raise ValueError(
                f"calibrated_weights.json is missing keys: "
                f"weights={missing_w}  caps={missing_c}"
            )

        weights = {k: float(v) for k, v in raw_w.items()}
        caps    = {k: int(v)   for k, v in raw_c.items()}

        method  = data.get("method", "calibrated")
        ts      = data.get("generated_at", "")
        label   = f"calibrated ({method})" + (f" @ {ts[:10]}" if ts else "")
        return weights, caps, label

    except Exception as exc:
        print(
            f"[scoring] WARNING: could not load {_WEIGHTS_FILE} ({exc}).  "
            "Using hardcoded defaults.  Run `python calibrate.py` to regenerate."
        )
        return dict(_DEFAULT_WEIGHTS), dict(_DEFAULT_CAPS), f"default (load failed: {exc})"


ACTIVE_WEIGHTS, ACTIVE_CAPS, WEIGHTS_SOURCE = _load_weights()


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
    if symbol == "005930.KS":   # Samsung — Korean geopolitical risk
        base = max(base, 7)
    if symbol == "RBI.VI":      # Raiffeisen — Russia/Ukraine exposure
        base = max(base, 8)
    if info.get("sector") in ("Energy", "Industrials"):
        base = min(base + 1, 10)

    return min(int(round(base + bump)), 10)


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION HELPERS
# Each function converts a raw metric to a normalised sub-score ∈ [0, 1].
# The tier boundaries are unchanged from the original code; only the maximum
# points per tier are now parameterised via ACTIVE_WEIGHTS.
#
# Normalised values:
#   PE   : 1.0 / 0.667 / 0.333  (15/15, 10/15, 5/15)
#   PEG  : 1.0 / 0.533 / 0.200  (15/15, 8/15,  3/15)
#   ROE  : 1.0 / 0.6   / 0.3 / 0.0
#   RSI  : 1.0 / 0.533 / 0.200
#   Up%  : 1.0 / 0.667 / 0.333 / 0.0
#   Geo  : 1.0 / 0.667 / 0.200
#   Sent : linear (−100,+100) → (0,1)
#   Vol  : 1.0 if > 60 %, else 0.0
# ─────────────────────────────────────────────────────────────────────────────

def _pe_norm(pe: float) -> float:
    return 1.0 if 0 < pe < 20 else (10 / 15 if pe < 30 else 5 / 15)

def _peg_norm(peg: float) -> float:
    return 1.0 if peg < 1 else (8 / 15 if peg < 2 else 3 / 15)

def _roe_norm(roe: float) -> float:
    return 1.0 if roe > 0.20 else (0.6 if roe > 0.10 else (0.3 if roe > 0 else 0.0))

def _rsi_norm(rsi: float) -> float:
    return 1.0 if 30 <= rsi <= 60 else (8 / 15 if rsi <= 70 else 3 / 15)

def _upside_norm(upside: float) -> float:
    return 1.0 if upside > 25 else (10 / 15 if upside > 10 else (5 / 15 if upside > 0 else 0.0))

def _geo_norm(geo: int) -> float:
    return 1.0 if geo <= 3 else (10 / 15 if geo <= 6 else 3 / 15)

def _sent_norm(ai_sentiment: float) -> float:
    return (float(ai_sentiment) + 100) / 200   # maps (−100,+100) → (0,1)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE  (0–100)
# ─────────────────────────────────────────────────────────────────────────────

def deep_score(
    pe, peg, roe, rsi_val, upside, geo, vol, ai_sentiment, sector=None, **kwargs
) -> tuple[int, dict]:
    """
    Compute a composite score broken into three buckets.

    Parameters are identical to the original deep_score() — no caller changes
    are required.  Weights are drawn from ACTIVE_WEIGHTS / ACTIVE_CAPS which
    are either the hardcoded defaults or values loaded from calibrated_weights.json.

    Returns
    -------
    (total_score, breakdown_dict)
      total_score    : int, 0–100
      breakdown_dict : {"Fundamental": int, "Technical": int, "GeoSentiment": int}
    """
    W = ACTIVE_WEIGHTS
    C = ACTIVE_CAPS
    bd: dict[str, int] = {}

    # ── Fundamental  (default cap: 40) ───────────────────────────────────────
    f = 0.0
    if pe is not None:
        f += W["pe"] * _pe_norm(float(pe))
    if peg is not None:
        f += W["peg"] * _peg_norm(float(peg))
    if roe is not None:
        f += W["roe"] * _roe_norm(float(roe))
    bd["Fundamental"] = min(int(round(f)), C["fundamental"])

    # ── Technical  (default cap: 30) ─────────────────────────────────────────
    t = 0.0
    if rsi_val is not None:
        t += W["rsi"] * _rsi_norm(float(rsi_val))
    if upside is not None:
        t += W["upside"] * _upside_norm(float(upside))
    bd["Technical"] = min(int(round(t)), C["technical"])

    # ── Geo + Sentiment  (default cap: 30) ───────────────────────────────────
    g = 0.0
    if geo is not None:
        g += W["geo"] * _geo_norm(int(geo))

    # ai_sentiment == 0 with a "No News" summary → treat as neutral, not bearish.
    if ai_sentiment is not None and isinstance(ai_sentiment, (int, float)):
        g += W["sentiment"] * _sent_norm(float(ai_sentiment))

    if vol is not None and float(vol) > 60:
        g = max(0.0, g - W["vol_penalty"])

    bd["GeoSentiment"] = min(int(round(g)), C["geosent"])

    total = sum(bd.values())
    return min(total, 100), bd


# ─────────────────────────────────────────────────────────────────────────────
# TRADE SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

def trade_signal(score: int, upside, rsi_val, ai_sentiment, macd_bullish_cross=False, macd_bearish_cross=False) -> str:
    """
    Derive STRONG BUY / BUY / HOLD / SELL from composite score, upside, RSI, AI sentiment, and MACD.
    Strong negative AI sentiment (< -50) reduces the effective score by 15 pts.
    """
    # Нормализация данных
    rsi_val = float(rsi_val) if rsi_val is not None else 50.0
    ai_num = float(ai_sentiment) if isinstance(ai_sentiment, (int, float, str)) and str(ai_sentiment).replace('.', '', 1).lstrip('-').isdigit() else 0.0
    
    # Штраф за негативный фон
    adj_score = max(0, score - 15) if ai_num < -50 else score

    # 1. Жесткие условия выхода (приоритет риск-менеджмента)
    if rsi_val > 72 or (upside is not None and float(upside) < -10):
        return "SELL"

    # 2. Катализатор сильной покупки (техническое подтверждение при допустимом фундаменте)
    if macd_bullish_cross and adj_score >= 50:
        return "STRONG BUY"

    # 3. Стандартная покупка
    if adj_score >= 65 and (upside is None or float(upside) > 5):
        return "BUY"

    # 4. Удержание
    if adj_score >= 50:
        return "HOLD"

    return "HOLD"

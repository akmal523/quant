"""
scoring.py — Unified Scoring Engine v9.0.
Integrates HMM stochastic models, Stewardship fundamentals, Data Confidence, and Bifurcated Horizons.
Key change in v9.0: tactical grade now penalises assets with no NLP data source to prevent
false BUY signals from default neutral scores.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import warnings
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from config import (
    WEIGHT_STEWARDSHIP, WEIGHT_TECHNICAL,
    FILTER_MAX_PE, FILTER_MIN_ROE,
    STRUCT_MAX_PE, STRUCT_MAX_PEG, STRUCT_MIN_ROE,
    STW_GEN_MAX_DE, STW_GEN_MID_DE, STW_GEN_MIN_ROE, STW_GEN_HI_ROE, STW_GEN_MIN_ICR,
    STW_FIN_MIN_PB, STW_FIN_MAX_PB, STW_FIN_MIN_ICR,
    MIN_STRUCT_GRADE_FOR_BUY, MIN_TACT_GRADE_FOR_BUY,
    SENTIMENT_NO_DATA_PENALTY,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ── Core Models ───────────────────────────────────────────────────────────────

def fit_market_regime(
    hist_close: pd.Series,
    vol: pd.Series,
    max_points: float = WEIGHT_TECHNICAL,
) -> float:
    """
    Fit GaussianHMM ONCE on a broad market index (SPY/URTH) to derive the
    macro "Probability of Bull Market". Per-asset HMM is statistically flawed
    (micro-caps lack independent regimes) and costs ~1s/asset.
    Intent: single macro regime multiplier applied to all asset tactical scores.
    Invariants: returns float in [0, max_points]; neutral 0.5*max_points on failure.
    Dependencies: hmmlearn, sklearn. Pure function (no I/O).
    """
    if len(hist_close) < 252 or vol.isna().all():
        return max_points / 2.0

    returns = np.log(hist_close / hist_close.shift(1)).dropna()
    common_index = returns.index.intersection(vol.dropna().index)
    if len(common_index) < 252:
        return max_points / 2.0

    X = pd.DataFrame({
        "Returns": returns[common_index],
        "Volatility": vol[common_index],
    })
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X_scaled)
        posterior_probs = model.predict_proba(X_scaled)
        state_means = model.means_[:, 0]
        bull_state_idx = np.argmax(state_means)
        current_bull_prob = posterior_probs[-1, bull_state_idx]
        return float(current_bull_prob * max_points)
    except Exception:
        return max_points / 2.0


def hmm_market_state_score(
    hist_close: pd.Series,
    garch_vol: pd.Series,
    max_points: float = WEIGHT_TECHNICAL,
) -> float:
    if len(hist_close) < 252 or garch_vol.isna().all():
        return max_points / 2.0

    returns = np.log(hist_close / hist_close.shift(1)).dropna()
    common_index = returns.index.intersection(garch_vol.dropna().index)

    if len(common_index) < 252:
        return max_points / 2.0

    X = pd.DataFrame({
        "Returns": returns[common_index],
        "Volatility": garch_vol[common_index],
    })

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X_scaled)

        posterior_probs = model.predict_proba(X_scaled)
        state_means = model.means_[:, 0]
        bull_state_idx = np.argmax(state_means)
        current_bull_prob = posterior_probs[-1, bull_state_idx]

        return float(current_bull_prob * max_points)
    except Exception:
        return max_points / 2.0


# ── Stewardship ───────────────────────────────────────────────────────────────

def stewardship_score_v2(f_data: dict, sector: str = "Technology") -> float:
    score = 0.0
    pb = f_data.get("PB")
    de = f_data.get("DebtToEquity")
    roe = f_data.get("ROE")
    icr = f_data.get("ICR")

    # Explicit None checks — 0 is a valid value, not "missing"
    if pb is None:
        pb = 2.0
    if de is None:
        de = 2.0
    if roe is None:
        roe = 0.0
    if icr is None:
        icr = 0.0

    if sector in ["Financials", "Financial Services"]:
        if pb < STW_FIN_MIN_PB:
            score += 15
        elif pb < STW_FIN_MAX_PB:
            score += 10
        if icr > STW_FIN_MIN_ICR:
            score += 15
        elif icr > 1.5:
            score += 7
    else:
        if de < STW_GEN_MAX_DE:
            score += 12
        elif de < STW_GEN_MID_DE:
            score += 7
        if roe > STW_GEN_HI_ROE:
            score += 10
        elif roe > STW_GEN_MIN_ROE:
            score += 5
        if icr > STW_GEN_MIN_ICR:
            score += 8

    return min(score, float(WEIGHT_STEWARDSHIP))


# ── Structural & Tactical Grades ──────────────────────────────────────────────

def evaluate_structural_grade(pe: float | None, peg: float | None, roe: float | None, stewardship_val: float) -> float:
    if pe is None and roe is None:
        return 85.0

    grade = stewardship_val * 1.5

    def is_valid(val):
        return val is not None and not (isinstance(val, float) and math.isnan(val))

    pe_v = float(pe) if is_valid(pe) else 999.0
    peg_v = float(peg) if is_valid(peg) else 9.0
    roe_v = float(roe) if is_valid(roe) else 0.0

    if 0 < pe_v < 15.0:
        grade += 25
    elif 0 < pe_v < 25.0:
        grade += 15

    if 0 < peg_v < STRUCT_MAX_PEG:
        grade += 15
    if roe_v > 0.25:
        grade += 25
    elif roe_v > 0.15:
        grade += 15

    return float(min(100.0, grade))


def evaluate_tactical_grade(
    hmm_prob_bull: float,    # [0, 1] normalised by caller
    finbert_score: float,    # Raw FinBERT [-100, +100]
    var_penalty: float,      # [0, 25]
    data_confidence: float = 1.0,  # 1.0 = real NLP data, 0.0 = no-data fallback
) -> float:
    """
    Tactical grade: HMM (60%) + Sentiment (25%) - Risk (-15%) - NoDataPenalty.
    When data_confidence < 1.0 (no SEC/News available), the tactical grade is
    reduced by up to SENTIMENT_NO_DATA_PENALTY points to prevent false BUY
    signals from default neutral scores.
    """
    grade = hmm_prob_bull * 60.0
    normalized_sentiment = (finbert_score + 100) / 5.0
    grade += normalized_sentiment
    grade -= var_penalty

    # Data-confidence penalty: when no real NLP data exists, reduce conviction
    if data_confidence < 1.0:
        penalty = (1.0 - data_confidence) * SENTIMENT_NO_DATA_PENALTY
        grade -= penalty

    return float(max(0.0, min(100.0, grade)))


def etf_tactical_grade(
    hmm_prob_bull: float,    # [0, 1] normalised by caller
    momentum_z: float,       # cross-sectional 12m momentum z-score
) -> float:
    """
    Continuous tactical grade for ETFs (Phase 5 / v10.2).

    Intent: the old ETF tactical grade was binary (99.4 vs 59.4) because the
    regime bull prob saturated at 0.99. Blend regime tilt with momentum so a
    saturated regime no longer forces every uptrending ETF to the same value.
        tactical = 50 + 25 * regime_tilt + 25 * tanh(momentum_z)
    where regime_tilt = (bull_prob - 0.5) * 2 (scaled to [-1, 1]).
    Invariants: returns float in [0, 100]; pure function (no I/O).
    """
    regime_tilt = (hmm_prob_bull - 0.5) * 2.0
    tactical = 50.0 + 25.0 * regime_tilt + 25.0 * math.tanh(momentum_z)
    return float(max(0.0, min(100.0, tactical)))


# ── Horizon Synchronization ───────────────────────────────────────────────────

def allocate_capital_regime(structural_grade: float, tactical_grade: float, stewardship_val: float) -> dict:
    if stewardship_val < (WEIGHT_STEWARDSHIP / 2) or structural_grade < 50:
        horizon = "SPECULATIVE"
        signal = "BUY" if tactical_grade >= MIN_TACT_GRADE_FOR_BUY else "SELL"
        active_score = tactical_grade
    elif structural_grade >= MIN_STRUCT_GRADE_FOR_BUY and tactical_grade >= 60:
        horizon = "CORE (12-Month)"
        signal = "BUY"
        active_score = (structural_grade * 0.4) + (tactical_grade * 0.6)
    else:
        horizon = "HOLD"
        signal = "HOLD"
        active_score = structural_grade

    return {"Horizon": horizon, "Signal": signal, "Active_Score": round(active_score, 1)}


# ── Position Sizing ───────────────────────────────────────────────────────────

def kelly_position_size(win_rate: float, avg_win: float, avg_loss: float) -> float:
    from config import KELLY_FRACTION, MAX_POSITION_PCT
    if avg_loss <= 0 or avg_win <= 0 or not (0.0 < win_rate < 1.0):
        return 0.0
    b = avg_win / avg_loss
    p = win_rate
    fractional = ((b * p - (1.0 - p)) / b) * KELLY_FRACTION
    return float(np.clip(fractional, 0.0, MAX_POSITION_PCT))

def target_volatility_size(asset_annual_vol: float) -> float:
    from config import TARGET_VOLATILITY, MAX_POSITION_PCT
    if asset_annual_vol <= 0:
        return float(MAX_POSITION_PCT)
    size = TARGET_VOLATILITY / asset_annual_vol
    return float(np.clip(size, 0.0, MAX_POSITION_PCT))

def position_size(
    win_rate:         float | None,
    avg_win:          float | None,
    avg_loss:         float | None,
    asset_annual_vol: float | None,
) -> dict:
    from config import KELLY_FRACTION, TARGET_VOLATILITY, MAX_POSITION_PCT
    kelly = kelly_position_size(win_rate or 0.0, avg_win or 0.0, avg_loss or 0.0)
    tv = target_volatility_size(asset_annual_vol or 0.30)
    final = min(kelly, tv) if kelly > 0 else tv

    return {
        "Kelly_Size_pct":        round(kelly * 100, 2),
        "TargetVol_Size_pct":    round(tv    * 100, 2),
        "Recommended_Size_pct":  round(final * 100, 2),
    }


# ── Cross-Sectional Factor Model ──────────────────────────────────────────────
# Replaces hardcoded absolute thresholds with universe-relative z-scores.
# Adapts to market conditions; robust to regime shifts.

# Factor weights (must sum to 1.0).
FACTOR_WEIGHTS = {
    "value":     0.25,
    "quality":   0.25,
    "momentum":  0.20,
    "low_risk":  0.15,
    "sentiment": 0.15,
}

# ── ETF Factor Model (Phase 5 / v10.2) ────────────────────────────────────────
# ETFs bypass Fundamentals/NLP, so they need a different factor set. Weights
# must sum to 1.0. This kills the degenerate 93.6 tie by ranking ETFs
# cross-sectionally on trend / relative strength / low-vol / momentum.
ETF_FACTOR_WEIGHTS = {
    "trend":    0.30,
    "rel_strength": 0.30,
    "low_vol":  0.20,
    "momentum": 0.20,
}


def zscore(series: pd.Series) -> pd.Series:
    """Standardize a factor across the universe. NaN-safe."""
    std = series.std()
    if std is None or std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def factor_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional factor z-scores and composite score.

    Intent: replace absolute thresholds (FILTER_MAX_PE etc.) with relative
    ranking. Each factor is z-scored across the tradable universe, then combined
    with FACTOR_WEIGHTS. Higher composite = more attractive.
    Invariants: input must have columns PE, ROE, momentum_6m, vol_60d, nlp_score.
    Returns a copy with value_z/quality_z/momentum_z/low_risk_z/sentiment_z/
    composite_score added.
    """
    out = features.copy()

    # Value: lower PE/PEG is better -> negate.
    value = -out["PE"].fillna(out["PE"].median())
    # Quality: higher ROE is better.
    quality = out["ROE"].fillna(out["ROE"].median())
    # Momentum: higher 6m return is better.
    momentum = out["momentum_6m"].fillna(0.0)
    # Low risk: lower vol is better -> negate.
    low_risk = -out["vol_60d"].fillna(out["vol_60d"].median())
    # Sentiment: higher NLP score is better.
    sentiment = out["nlp_score"].fillna(0.0)

    out["value_z"] = zscore(value)
    out["quality_z"] = zscore(quality)
    out["momentum_z"] = zscore(momentum)
    out["low_risk_z"] = zscore(low_risk)
    out["sentiment_z"] = zscore(sentiment)

    out["composite_score"] = (
        FACTOR_WEIGHTS["value"] * out["value_z"]
        + FACTOR_WEIGHTS["quality"] * out["quality_z"]
        + FACTOR_WEIGHTS["momentum"] * out["momentum_z"]
        + FACTOR_WEIGHTS["low_risk"] * out["low_risk_z"]
        + FACTOR_WEIGHTS["sentiment"] * out["sentiment_z"]
    )
    return out


def etf_factor_scores(features: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional factor z-scores for the ETF subset.

    Intent (Phase 5 / v10.2): ETFs bypass Fundamentals/NLP, so they need a
    different factor set (trend / rel_strength / low_vol / momentum). This
    populates the dashboard Z-score section for ETFs and kills the 93.6 tie.
    Invariants: input must have columns trend_strength, rel_strength_6m,
    vol_60d, ret_12m. Returns a copy with trend_z/rel_strength_z/low_vol_z/
    momentum_z/composite_score added.
    """
    out = features.copy()

    # Trend: higher close-vs-200SMA strength is better.
    trend = out["trend_strength"].fillna(0.0)
    # Relative strength: higher benchmark-relative 6m return is better.
    rel_strength = out["rel_strength_6m"].fillna(0.0)
    # Low vol: lower 60d vol is better -> negate.
    low_vol = -out["vol_60d"].fillna(out["vol_60d"].median())
    # Momentum: higher 12m return is better.
    momentum = out["ret_12m"].fillna(0.0)

    out["trend_z"] = zscore(trend)
    out["rel_strength_z"] = zscore(rel_strength)
    out["low_vol_z"] = zscore(low_vol)
    out["momentum_z"] = zscore(momentum)

    out["composite_score"] = (
        ETF_FACTOR_WEIGHTS["trend"] * out["trend_z"]
        + ETF_FACTOR_WEIGHTS["rel_strength"] * out["rel_strength_z"]
        + ETF_FACTOR_WEIGHTS["low_vol"] * out["low_vol_z"]
        + ETF_FACTOR_WEIGHTS["momentum"] * out["momentum_z"]
    )
    return out


def etf_quality_score(features: pd.DataFrame) -> pd.DataFrame:
    """Compute a cross-sectional structural grade (0-100) for the ETF subset.

    Intent (Phase 5 / v10.2): replace the hardcoded 85.0 ETF structural grade
    with a cross-sectional ranking so EIMI.L, CSPX.L, EUNL.DE rank differently.
    Uses the same ETF factor z-scores as etf_factor_scores, mapped to 0-100.
    Invariants: input must have columns trend_strength, rel_strength_6m,
    vol_60d, ret_12m. Returns a copy with etf_quality added.
    """
    out = etf_factor_scores(features)
    # Map the weighted z-sum to 0-100 via a logistic squash. A composite of 0
    # (universe average) maps to 50; strong positive z maps toward 100.
    composite = out["composite_score"]
    out["etf_quality"] = 50.0 + 50.0 * (composite / (1.0 + composite.abs()))
    return out


def sector_neutral_rank(features: pd.DataFrame) -> pd.DataFrame:
    """Neutralize sector bias by ranking composite_score within each sector.

    Intent: prevent the portfolio from concentrating in one sector (e.g. cheap
    financials or high-momentum tech). Adds sector_rank (1 = best in sector).
    Invariants: requires composite_score and Sector columns.
    """
    out = features.copy()
    out["sector_rank"] = (
        out.groupby("Sector")["composite_score"]
        .rank(ascending=False, method="min")
    )
    return out


# ── Fast Filter ───────────────────────────────────────────────────────────────

def apply_fast_filter(f_data: dict) -> bool:
    if not f_data:
        return False
    pe = f_data.get("PE")
    roe = f_data.get("ROE")

    if pe is None or pe <= 0 or pe > FILTER_MAX_PE:
        return False
    if roe is None or roe < FILTER_MIN_ROE:
        return False
    return True

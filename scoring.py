"""
scoring.py — Unified Scoring Engine v8.1.
Integrates HMM stochastic models, Stewardship fundamentals, and Bifurcated Horizons.
"""
from __future__ import annotations

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
    MIN_STRUCT_GRADE_FOR_BUY, MIN_TACT_GRADE_FOR_BUY
)

warnings.filterwarnings("ignore", category=UserWarning)


# ── Core Models ───────────────────────────────────────────────────────────────

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
        "Volatility": garch_vol[common_index]
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


def stewardship_score_v2(f_data: dict, sector: str = "Technology") -> float:
    score = 0.0
    pb = f_data.get("PB") or 2.0
    de = f_data.get("DebtToEquity") or 2.0
    roe = f_data.get("ROE") or 0.0
    icr = f_data.get("ICR") or 0.0

    if sector in ["Financials", "Financial Services"]:
        if pb < STW_FIN_MIN_PB: score += 15
        elif pb < STW_FIN_MAX_PB: score += 10
        if icr > STW_FIN_MIN_ICR: score += 15
        elif icr > 1.5: score += 7
    else:
        if de < STW_GEN_MAX_DE: score += 12
        elif de < STW_GEN_MID_DE: score += 7
        if roe > STW_GEN_HI_ROE: score += 10
        elif roe > STW_GEN_MIN_ROE: score += 5
        if icr > STW_GEN_MIN_ICR: score += 8
        
    return min(score, float(WEIGHT_STEWARDSHIP))

def stewardship_score(
    debt_to_equity: float | None,
    payout_ratio:   float | None,
    dividend_yield: float | None,
    icr:            float | None = None,
) -> float:
    s_score = 0.0

    de_raw = float(debt_to_equity) if debt_to_equity is not None else None
    if de_raw is None:
        de = 2.0
    elif de_raw > 5:
        de = de_raw / 100.0
    else:
        de = de_raw

    if de < 0.5:
        s_score += 12
    elif de < 1.0:
        s_score += 7
    elif de < 2.0:
        s_score += 3

    if icr is not None:
        icr_f = float(icr)
        if icr_f >= 5.0:
            s_score += 10
        elif icr_f >= 3.0:
            s_score += 6
        elif icr_f >= 1.5:
            s_score += 3
    else:
        s_score += 5

    payout = float(payout_ratio) if payout_ratio is not None else 1.0
    div_y  = float(dividend_yield) if dividend_yield else 0.0

    if div_y > 0:
        if 0.30 <= payout <= 0.70:
            s_score += 8
        elif payout < 0.90:
            s_score += 4
    else:
        if payout < 0.20:
            s_score += 4

    return min(s_score, float(WEIGHT_STEWARDSHIP))


# ── Bifurcated Vectors ────────────────────────────────────────────────────────

def evaluate_structural_grade(pe: float | None, peg: float | None, roe: float | None, stewardship_val: float) -> float:
    grade = stewardship_val * 1.5
    
    # Use .get() or check for math.isnan
    import math
    def is_valid(val):
        return val is not None and not (isinstance(val, float) and math.isnan(val))

    pe_v = float(pe) if is_valid(pe) else 999.0
    peg_v = float(peg) if is_valid(peg) else 9.0
    roe_v = float(roe) if is_valid(roe) else 0.0

    if 0 < pe_v < STRUCT_MAX_PE: grade += 20
    if 0 < peg_v < STRUCT_MAX_PEG: grade += 15
    if roe_v > STRUCT_MIN_ROE: grade += 20

    return float(min(100.0, grade))

def evaluate_tactical_grade(
    hmm_prob_bull: float,
    finbert_score: float,
    var_penalty:   float,
) -> float:
    grade = hmm_prob_bull * 60.0
    normalized_sentiment = (finbert_score + 100) / 5.0
    grade += normalized_sentiment
    grade -= var_penalty

    return float(max(0.0, min(100.0, grade)))


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
    if avg_loss <= 0 or avg_win <= 0 or not (0.0 < win_rate < 1.0):
        return 0.0
    b = avg_win / avg_loss
    p = win_rate
    fractional = ((b * p - (1.0 - p)) / b) * KELLY_FRACTION
    return float(np.clip(fractional, 0.0, MAX_POSITION_PCT))

def target_volatility_size(asset_annual_vol: float) -> float:
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
    kelly = kelly_position_size(win_rate or 0.0, avg_win or 0.0, avg_loss or 0.0)
    tv = target_volatility_size(asset_annual_vol or 0.30)
    final = min(kelly, tv) if kelly > 0 else tv

    return {
        "Kelly_Size_pct":        round(kelly * 100, 2),
        "TargetVol_Size_pct":    round(tv    * 100, 2),
        "Recommended_Size_pct":  round(final * 100, 2),
    }

def apply_fast_filter(f_data: dict) -> bool:
    if not f_data: return False
    pe = f_data.get("PE")
    roe = f_data.get("ROE")
    
    if pe is None or pe <= 0 or pe > FILTER_MAX_PE: return False
    if roe is None or roe < FILTER_MIN_ROE: return False
    return True

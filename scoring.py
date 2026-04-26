"""
scoring.py — Scoring engine v7.

Changes from v6:
  - Weights rebalanced: Technical 25→15, Stewardship 20→30.
    Rationale: RSI/oscillators produce short-term noise that punishes fundamentally
    sound businesses during market-wide corrections (observed: synchronous SELL signals
    across all sectors). Stewardship is the non-negotiable quality floor.
  - Stewardship v2: adds ICR (Interest Coverage Ratio) as mandatory debt-serviceability
    gate. In a high-rate environment, the ability to service debt from operating profit
    matters more than the nominal size of the debt.
  - Position sizing: Fractional Kelly + Target Volatility. System emits concrete
    capital allocation recommendations, not binary BUY/SELL signals alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    WEIGHT_FUNDAMENTALS,
    WEIGHT_STEWARDSHIP,
    WEIGHT_TECHNICAL,
    WEIGHT_SENTIMENT,
    KELLY_FRACTION,
    TARGET_VOLATILITY,
    MAX_POSITION_PCT,
)


# ── Technical ─────────────────────────────────────────────────────────────────

def calculate_z_score(current_val: float, historical_series: pd.Series) -> float:
    if historical_series is None or historical_series.empty:
        return 0.0
    mean = historical_series.mean()
    std  = historical_series.std()
    return (current_val - mean) / std if std > 0 else 0.0


def technical_score_v2(
    rsi_val: float | None,
    price: float | None,
    sma50: float | None,
    hist_close: pd.Series,
    sector_rsi_median: float | None,
) -> float:
    """
    Technical score — max WEIGHT_TECHNICAL pts (15).
    Intentionally constrained: technical signals are supporting context,
    not the primary thesis.

    Two equal sub-components (7.5 pts each):
      1. Price Z-score relative to 50-day history.
      2. RSI relative to sector median (sector-aware oversold detection).
    """
    half    = WEIGHT_TECHNICAL / 2
    t_score = 0.0

    # 1. Z-Score relative to SMA50 window
    if price and sma50:
        z = calculate_z_score(price, hist_close.tail(50))
        if -0.5 <= z <= 1.5:
            t_score += half
        elif -1.0 <= z <= 2.0:
            t_score += half * 0.5

    # 2. Sector-relative RSI
    if rsi_val and sector_rsi_median:
        relative_rsi = rsi_val - sector_rsi_median
        if -15 <= relative_rsi <= -5:
            t_score += half          # Asset is 'cooler' than sector peers
        elif -20 <= relative_rsi <= 5:
            t_score += half * 0.5
    elif rsi_val:
        if 30 <= rsi_val <= 60:      # Static fallback when sector data absent
            t_score += half

    return t_score


# ── Stewardship v2 ────────────────────────────────────────────────────────────

def stewardship_score(
    debt_to_equity: float | None,
    payout_ratio:   float | None,
    dividend_yield: float | None,
    icr:            float | None = None,
) -> float:
    """
    Stewardship v2 — evaluates balance-sheet integrity and debt serviceability.
    Max: WEIGHT_STEWARDSHIP pts (30).

    Sub-components:
      D/E Solvency   : 0–12 pts  — leverage level
      ICR Coverage   : 0–10 pts  — can operating profit service the debt?  ← NEW
      Dividend Policy: 0–8  pts  — payout sustainability
    """
    s_score = 0.0

    # 1. Leverage (D/E) — 12 pts
    de = float(debt_to_equity) if debt_to_equity is not None else 2.0
    if de < 0.5:
        s_score += 12    # Pristine: minimal leverage
    elif de < 1.0:
        s_score += 7     # Manageable
    elif de < 2.0:
        s_score += 3     # Elevated but functional

    # 2. Interest Coverage Ratio — 10 pts
    # In a rising-rate environment, the ability to service debt from EBIT
    # is the true measure of solvency, not the nominal D/E ratio.
    if icr is not None:
        icr_f = float(icr)
        if icr_f >= 5.0:
            s_score += 10    # Comfortable — strong operating cushion
        elif icr_f >= 3.0:
            s_score += 6     # Adequate coverage
        elif icr_f >= 1.5:
            s_score += 3     # Minimum viable — monitor in stress scenarios
        # ICR < 1.5: 0 pts — operating profit cannot cover interest payments
    else:
        # ICR data unavailable: partial credit (avoid penalizing missing data equally)
        s_score += 5

    # 3. Dividend policy — 8 pts
    payout = float(payout_ratio) if payout_ratio is not None else 1.0
    div_y  = float(dividend_yield) if dividend_yield else 0.0

    if div_y > 0:
        if 0.30 <= payout <= 0.70:
            s_score += 8     # Sustainable: distributes cash without exhausting it
        elif payout < 0.90:
            s_score += 4     # Acceptable — room for dividend cut before crisis
    else:
        if payout < 0.20:
            s_score += 4     # Growth reinvestor — acceptable alternative profile

    return min(s_score, float(WEIGHT_STEWARDSHIP))


# ── Composite ─────────────────────────────────────────────────────────────────

def composite_score_v3(
    pe:              float | None,
    peg:             float | None,
    roe:             float | None,
    stewardship_val: float,
    tech_val:        float,
    geo_sent_val:    float,
    vol:             float | None,
) -> float:
    """
    Rebalanced composite score (100 pts):
      Fundamentals (PE/PEG/ROE) : WEIGHT_FUNDAMENTALS = 30
      Stewardship (D/E+ICR)     : WEIGHT_STEWARDSHIP  = 30  ↑ from 20
      Technical (Z/RSI)         : WEIGHT_TECHNICAL    = 15  ↓ from 25
      GeoSentiment (NER+BERT)   : WEIGHT_SENTIMENT    = 25
    """
    # Fundamentals: 30 pts (10 pts per gate)
    f_score = 0
    if pe  and float(pe)  < 20:    f_score += 10
    if peg and float(peg) < 1.2:   f_score += 10
    if roe and float(roe) > 0.15:  f_score += 10

    total = f_score + stewardship_val + tech_val + geo_sent_val

    # Beta penalty: high-volatility assets get a structural 5-pt haircut
    if vol and float(vol) > 0.50:
        total -= 5

    return float(max(0.0, min(100.0, total)))


# ── Signals ───────────────────────────────────────────────────────────────────

def stewardship_trade_signal(
    adj_score:       float,
    stewardship_val: float,
    upside:          float | None,
) -> str:
    """
    Signal with quality floor.
    Floor threshold raised to 8 pts (was 5) — tighter solvency requirement.
    """
    if stewardship_val < 8:
        return "SELL" if adj_score < 40 else "HOLD"
    if adj_score >= 70 and (upside is None or float(upside) > 5):
        return "BUY"
    if adj_score >= 50:
        return "HOLD"
    return "SELL"


def classify_horizon(
    div_yield: float | None,
    vol:       float | None,
    roe:       float | None,
    pe:        float | None,
) -> str:
    div = float(div_yield) if div_yield else 0.0
    v   = float(vol)       if vol       else 1.0
    r   = float(roe)       if roe       else 0.0
    p   = float(pe)        if pe        else 999.0

    if div >= 0.02 and v < 0.25 and r > 0.10 and 0 < p < 30:
        return "RETIREMENT (20yr+)"
    if v < 0.35 and r > 0.05 and 0 < p < 40:
        return "BUSINESS COLLATERAL (10yr)"
    return "SPECULATIVE (short-term)"


# ── Position Sizing ───────────────────────────────────────────────────────────

def kelly_position_size(
    win_rate: float,
    avg_win:  float,
    avg_loss: float,
) -> float:
    """
    Fractional Kelly criterion.

    Full Kelly:  f* = (b·p − (1−p)) / b
    Applied:     f  = f* × KELLY_FRACTION  (quarter-Kelly by default)

    where:
      b = avg_win / avg_loss   (win/loss ratio from backtest)
      p = win_rate             (empirical win probability from backtest)

    Quarter-Kelly: The fractional multiplier (0.25) protects against the
    inevitable estimation error in p and b derived from limited backtest samples.
    A full-Kelly bet sized on noisy estimates destroys capital at geometric speed.

    Returns: fraction of capital to deploy, clipped to [0, MAX_POSITION_PCT].
    """
    if avg_loss <= 0 or avg_win <= 0 or not (0.0 < win_rate < 1.0):
        return 0.0
    b           = avg_win / avg_loss
    p           = win_rate
    full_kelly  = (b * p - (1.0 - p)) / b
    fractional  = full_kelly * KELLY_FRACTION
    return float(np.clip(fractional, 0.0, MAX_POSITION_PCT))


def target_volatility_size(asset_annual_vol: float) -> float:
    """
    Target Volatility sizing: size = TARGET_VOLATILITY / asset_volatility.

    Scales position down when the asset is more volatile than the portfolio target,
    and up (to the cap) when it is less volatile.

    Returns: fraction of capital to deploy, clipped to [0, MAX_POSITION_PCT].
    """
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
    """
    Conservative position size = min(Kelly, Target-Volatility).

    Both methods act as independent, orthogonal guards:
      - Kelly constrains loss at the *strategy edge* level.
      - Target-Vol constrains loss at the *portfolio volatility* level.
    Taking the minimum ensures both constraints are satisfied simultaneously.
    """
    kelly = kelly_position_size(
        win_rate or 0.0,
        avg_win  or 0.0,
        avg_loss or 0.0,
    )
    tv    = target_volatility_size(asset_annual_vol or 0.30)
    final = min(kelly, tv) if kelly > 0 else tv

    return {
        "Kelly_Size_pct":        round(kelly * 100, 2),
        "TargetVol_Size_pct":    round(tv    * 100, 2),
        "Recommended_Size_pct":  round(final * 100, 2),
    }

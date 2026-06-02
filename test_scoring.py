"""
test_scoring.py — Comprehensive unit tests for the scoring engine.
Covers: HMM, stewardship, structural/tactical grades, capital allocation, position sizing.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from config import WEIGHT_STEWARDSHIP  # needed in test scope

# ── Stewardship & Structural Grade ─────────────────────────────────────────────

def test_stewardship_score_v2_general():
    """General sector: low D/E + high ROE + high ICR = max score."""
    from scoring import stewardship_score_v2
    from config import WEIGHT_STEWARDSHIP

    f_data = {
        "PB": 2.0,
        "DebtToEquity": 0.3,   # below STW_GEN_MAX_DE 0.5 → +12
        "ROE": 0.25,            # above STW_GEN_HI_ROE 0.20 → +10
        "ICR": 8.0,             # above STW_GEN_MIN_ICR 5.0 → +8
    }
    score = stewardship_score_v2(f_data, sector="Technology")
    assert 0 <= score <= WEIGHT_STEWARDSHIP, f"Score {score} out of range [0, {WEIGHT_STEWARDSHIP}]"
    assert score == 30.0, f"Expected 30.0 (max), got {score}"
    print(f"  [PASS] test_stewardship_score_v2_general: score={score}")


def test_stewardship_score_v2_financials():
    """Financials sector: PB in range + high ICR."""
    from scoring import stewardship_score_v2
    from config import WEIGHT_STEWARDSHIP

    f_data = {
        "PB": 1.2,              # between STW_FIN_MIN_PB 1.0 and STW_FIN_MAX_PB 1.5 → +10
        "DebtToEquity": 5.0,    # ignored for financials
        "ROE": 0.05,            # ignored for financials
        "ICR": 4.0,             # above STW_FIN_MIN_ICR 3.0 → +15
    }
    score = stewardship_score_v2(f_data, sector="Financials")
    assert score == 25.0, f"Expected 25.0 (10+15), got {score}"
    print(f"  [PASS] test_stewardship_score_v2_financials: score={score}")


def test_stewardship_score_v2_missing_data():
    """Missing fundamentals should fall back to defaults without crashing."""
    from scoring import stewardship_score_v2

    score = stewardship_score_v2({}, sector="Technology")
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    print(f"  [PASS] test_stewardship_score_v2_missing_data: score={score}")


def test_evaluate_structural_grade_etf_bypass():
    """ETF bypass: PE=None + ROE=None → returns 85.0."""
    from scoring import evaluate_structural_grade

    grade = evaluate_structural_grade(pe=None, peg=None, roe=None, stewardship_val=0.0)
    assert grade == 85.0, f"Expected 85.0 for ETF, got {grade}"
    print(f"  [PASS] test_evaluate_structural_grade_etf_bypass: grade={grade}")


def test_evaluate_structural_grade_deep_value():
    """Low PE + low PEG + high ROE should produce near-max grade."""
    from scoring import evaluate_structural_grade

    grade = evaluate_structural_grade(pe=12.0, peg=1.0, roe=0.30, stewardship_val=WEIGHT_STEWARDSHIP)
    # stewardship 30 * 1.5 = 45 + PE bonus 25 + PEG bonus 15 + ROE bonus 25 = 110, clipped to 100
    assert grade == 100.0, f"Expected 100.0 for deep value, got {grade}"
    print(f"  [PASS] test_evaluate_structural_grade_deep_value: grade={grade}")


def test_evaluate_structural_grade_clipping():
    """Grade should never exceed 100 or go below 0."""
    from scoring import evaluate_structural_grade

    grade = evaluate_structural_grade(pe=5.0, peg=0.5, roe=0.40, stewardship_val=30.0)
    assert grade <= 100.0, f"Grade {grade} exceeds 100 cap"
    assert grade >= 0.0, f"Grade {grade} below 0 floor"
    print(f"  [PASS] test_evaluate_structural_grade_clipping: grade={grade}")


# ── Tactical Grade ─────────────────────────────────────────────────────────────

def test_evaluate_tactical_grade_bullish():
    """Strong bull HMM + positive sentiment + no risk penalty → high grade."""
    from scoring import evaluate_tactical_grade

    grade = evaluate_tactical_grade(
        hmm_prob_bull=1.0,    # max bullish
        finbert_score=50.0,    # very positive
        var_penalty=0.0,       # no penalty
    )
    # 1.0 * 60 = 60 + (50+100)/5 = 30 + 0 = 90
    assert grade == 90.0, f"Expected 90.0, got {grade}"
    print(f"  [PASS] test_evaluate_tactical_grade_bullish: grade={grade}")


def test_evaluate_tactical_grade_bearish():
    """Weak HMM + negative sentiment + high penalty → floor at 0."""
    from scoring import evaluate_tactical_grade

    grade = evaluate_tactical_grade(
        hmm_prob_bull=0.0,     # max bear
        finbert_score=-100.0,   # very negative
        var_penalty=25.0,       # max penalty
    )
    # 0 * 60 = 0 + (-100+100)/5 = 0 - 25 = -25 → clipped to 0
    assert grade == 0.0, f"Expected 0.0 (floor), got {grade}"
    print(f"  [PASS] test_evaluate_tactical_grade_bearish: grade={grade}")


def test_evaluate_tactical_grade_clipping():
    """Grade should always be in [0, 100]."""
    from scoring import evaluate_tactical_grade

    for _ in range(20):
        grade = evaluate_tactical_grade(
            hmm_prob_bull=np.random.uniform(0, 1),
            finbert_score=np.random.uniform(-100, 100),
            var_penalty=np.random.uniform(0, 25),
        )
        assert 0.0 <= grade <= 100.0, f"Grade {grade} out of [0, 100]"
    print(f"  [PASS] test_evaluate_tactical_grade_clipping: 20 random grades all in [0,100]")


# ── Capital Allocation ────────────────────────────────────────────────────────

def test_allocate_capital_core_buy():
    """High structural + high tactical → CORE (12-Month) BUY."""
    from scoring import allocate_capital_regime

    result = allocate_capital_regime(
        structural_grade=85.0,
        tactical_grade=80.0,
        stewardship_val=25.0,
    )
    assert result["Horizon"] == "CORE (12-Month)", f"Expected CORE, got {result['Horizon']}"
    assert result["Signal"] == "BUY", f"Expected BUY, got {result['Signal']}"
    # 85*0.4 + 80*0.6 = 34 + 48 = 82.0
    assert result["Active_Score"] == 82.0, f"Expected 82.0, got {result['Active_Score']}"
    print(f"  [PASS] test_allocate_capital_core_buy: {result}")


def test_allocate_capital_speculative():
    """Low stewardship floor → SPECULATIVE."""
    from scoring import allocate_capital_regime
    from config import WEIGHT_STEWARDSHIP

    result = allocate_capital_regime(
        structural_grade=80.0,
        tactical_grade=75.0,
        stewardship_val=5.0,  # below WEIGHT_STEWARDSHIP / 2 = 15
    )
    assert result["Horizon"] == "SPECULATIVE", f"Expected SPECULATIVE, got {result['Horizon']}"
    assert result["Active_Score"] == 75.0, f"Expected 75.0 (tactical_grade), got {result['Active_Score']}"
    print(f"  [PASS] test_allocate_capital_speculative: {result}")


def test_allocate_capital_structural_floor():
    """Structural grade < 50 → SPECULATIVE regardless of stewardship."""
    from scoring import allocate_capital_regime

    result = allocate_capital_regime(
        structural_grade=45.0,
        tactical_grade=70.0,
        stewardship_val=25.0,  # above floor, but structural < 50
    )
    assert result["Horizon"] == "SPECULATIVE", f"Expected SPECULATIVE, got {result['Horizon']}"
    print(f"  [PASS] test_allocate_capital_structural_floor: {result}")


def test_allocate_capital_hold():
    """Meets stewardship floor but structural below buy limit → HOLD."""
    from scoring import allocate_capital_regime

    result = allocate_capital_regime(
        structural_grade=65.0,  # < MIN_STRUCT_GRADE_FOR_BUY 75
        tactical_grade=70.0,
        stewardship_val=20.0,   # >= floor
    )
    assert result["Horizon"] == "HOLD", f"Expected HOLD, got {result['Horizon']}"
    assert result["Signal"] == "HOLD", f"Expected HOLD, got {result['Signal']}"
    assert result["Active_Score"] == 65.0, f"Expected 65.0 (structural_grade), got {result['Active_Score']}"
    print(f"  [PASS] test_allocate_capital_hold: {result}")


# ── Position Sizing ────────────────────────────────────────────────────────────

def test_kelly_position_size():
    """Kelly with 60% win rate, 2:1 reward/risk."""
    from scoring import kelly_position_size
    from config import KELLY_FRACTION, MAX_POSITION_PCT

    size = kelly_position_size(win_rate=0.60, avg_win=2.0, avg_loss=1.0)
    # full Kelly = (2*0.6 - 0.4) / 2 = 0.4, quarter-Kelly = 0.1
    expected = 0.10
    assert abs(size - expected) < 0.005, f"Expected {expected}, got {size}"
    assert 0.0 <= size <= MAX_POSITION_PCT, f"Size {size} out of range"
    print(f"  [PASS] test_kelly_position_size: size={size}")


def test_kelly_position_size_edge_cases():
    """Zero or negative inputs should return 0."""
    from scoring import kelly_position_size

    assert kelly_position_size(0.0, 1.0, 1.0) == 0.0, "Zero win rate"
    assert kelly_position_size(0.6, 0.0, 1.0) == 0.0, "Zero avg win"
    assert kelly_position_size(0.6, 1.0, 0.0) == 0.0, "Zero avg loss"
    assert kelly_position_size(-0.5, 1.0, 1.0) == 0.0, "Negative win rate"
    assert kelly_position_size(1.1, 1.0, 1.0) == 0.0, "Win rate > 1"
    print(f"  [PASS] test_kelly_position_size_edge_cases: all edge cases return 0")


def test_target_volatility_size():
    """Target volatility sizing: vol=30% → size=50% clipped to MAX_POSITION_PCT 10%."""
    from scoring import target_volatility_size
    from config import TARGET_VOLATILITY, MAX_POSITION_PCT

    size = target_volatility_size(asset_annual_vol=0.30)
    expected = min(TARGET_VOLATILITY / 0.30, MAX_POSITION_PCT)
    assert abs(size - expected) < 0.005, f"Expected {expected}, got {size}"
    assert size <= MAX_POSITION_PCT, f"Size {size} exceeds MAX_POSITION_PCT"
    print(f"  [PASS] test_target_volatility_size: size={size}")


# ── Fast Filter ───────────────────────────────────────────────────────────────

def test_apply_fast_filter():
    from scoring import apply_fast_filter

    assert apply_fast_filter({"PE": 15.0, "ROE": 0.25}) is True, "Qualified ticker"
    assert apply_fast_filter({"PE": 30.0, "ROE": 0.25}) is False, "PE too high"
    assert apply_fast_filter({"PE": 15.0, "ROE": 0.05}) is False, "ROE too low"
    assert apply_fast_filter({"PE": -5.0, "ROE": 0.25}) is False, "Negative PE"
    assert apply_fast_filter({}) is False, "Empty dict"
    assert apply_fast_filter(None) is False, "None input"
    print(f"  [PASS] test_apply_fast_filter: all cases correct")


# ── HMM ────────────────────────────────────────────────────────────────────────

def test_hmm_market_state_score_short_history():
    """History < 252 days → returns mid-point (max_points/2)."""
    from scoring import hmm_market_state_score
    from config import WEIGHT_TECHNICAL

    short_close = pd.Series(np.random.randn(200).cumsum() + 100)
    short_vol = pd.Series(np.abs(np.random.randn(200)))
    score = hmm_market_state_score(short_close, short_vol)
    expected = WEIGHT_TECHNICAL / 2.0
    assert score == expected, f"Expected {expected} for short history, got {score}"
    print(f"  [PASS] test_hmm_market_state_score_short_history: score={score}")


def test_hmm_market_state_score_nan_vol():
    """All-NaN volatility → returns mid-point."""
    from scoring import hmm_market_state_score
    from config import WEIGHT_TECHNICAL

    close = pd.Series(np.random.randn(500).cumsum() + 100)
    nan_vol = pd.Series([np.nan] * 500)
    score = hmm_market_state_score(close, nan_vol)
    assert score == WEIGHT_TECHNICAL / 2.0, f"Expected mid-point for NaN vol, got {score}"
    print(f"  [PASS] test_hmm_market_state_score_nan_vol: score={score}")


# ── Risk ───────────────────────────────────────────────────────────────────────

def test_calculate_risk_penalty():
    from risk import calculate_risk_penalty

    # Normal returns → no penalty
    normal_returns = pd.Series(np.random.normal(0.001, 0.01, 500))
    penalty = calculate_risk_penalty(normal_returns)
    assert 0.0 <= penalty <= 25.0, f"Penalty {penalty} out of range"
    print(f"  [PASS] test_calculate_risk_penalty (normal): {penalty:.2f}")

    # High-risk returns → penalty > 0
    crash_returns = pd.Series(np.random.normal(-0.005, 0.03, 500))
    crash_penalty = calculate_risk_penalty(crash_returns)
    assert crash_penalty > 0, f"Expected penalty > 0 for high-risk series, got {crash_penalty}"
    print(f"  [PASS] test_calculate_risk_penalty (high-risk): {crash_penalty:.2f}")

    # Empty series → 0 penalty
    assert calculate_risk_penalty(pd.Series([], dtype=float)) == 0.0, "Empty series"
    print(f"  [PASS] test_calculate_risk_penalty (empty): 0.0")


# ── Run All ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SCORING ENGINE UNIT TESTS")
    print("=" * 60)

    # Stewardship & Structural
    test_stewardship_score_v2_general()
    test_stewardship_score_v2_financials()
    test_stewardship_score_v2_missing_data()
    test_evaluate_structural_grade_etf_bypass()
    test_evaluate_structural_grade_deep_value()
    test_evaluate_structural_grade_clipping()

    # Tactical
    test_evaluate_tactical_grade_bullish()
    test_evaluate_tactical_grade_bearish()
    test_evaluate_tactical_grade_clipping()

    # Capital Allocation
    test_allocate_capital_core_buy()
    test_allocate_capital_speculative()
    test_allocate_capital_structural_floor()
    test_allocate_capital_hold()

    # Position Sizing
    test_kelly_position_size()
    test_kelly_position_size_edge_cases()
    test_target_volatility_size()

    # Fast Filter
    test_apply_fast_filter()

    # HMM
    test_hmm_market_state_score_short_history()
    test_hmm_market_state_score_nan_vol()

    # Risk
    test_calculate_risk_penalty()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
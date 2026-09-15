"""
test_scoring.py — Comprehensive unit tests for the scoring engine.
Covers: HMM, stewardship, structural/tactical grades, data confidence, capital allocation, position sizing.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from quant.config import WEIGHT_STEWARDSHIP

# ── Stewardship & Structural Grade ─────────────────────────────────────────────

def test_stewardship_score_v2_general():
    """General sector: low D/E + high ROE + high ICR = max score."""
    from quant.analytics.scoring import stewardship_score_v2
    f_data = {"PB": 2.0, "DebtToEquity": 0.3, "ROE": 0.25, "ICR": 8.0}
    score = stewardship_score_v2(f_data, sector="Technology")
    assert 0 <= score <= WEIGHT_STEWARDSHIP
    assert score == 30.0, f"Expected 30.0, got {score}"
    print(f"  [PASS] test_stewardship_score_v2_general: score={score}")

def test_stewardship_score_v2_financials():
    """Financials sector: PB in range + high ICR."""
    from quant.analytics.scoring import stewardship_score_v2
    f_data = {"PB": 1.2, "DebtToEquity": 5.0, "ROE": 0.05, "ICR": 4.0}
    score = stewardship_score_v2(f_data, sector="Financials")
    assert score == 25.0, f"Expected 25.0, got {score}"
    print(f"  [PASS] test_stewardship_score_v2_financials: score={score}")

def test_stewardship_score_v2_missing_data():
    """Missing fundamentals should fall back to defaults without crashing."""
    from quant.analytics.scoring import stewardship_score_v2
    score = stewardship_score_v2({}, sector="Technology")
    assert isinstance(score, float)
    print(f"  [PASS] test_stewardship_score_v2_missing_data: score={score}")

def test_stewardship_score_v2_zero_values():
    """PB=0 or D/E=0 should NOT be treated as missing (falsy bug fix)."""
    from quant.analytics.scoring import stewardship_score_v2
    s1 = stewardship_score_v2({"PB": 0.0, "DebtToEquity": 0.5, "ROE": 0.25, "ICR": 8.0}, sector="Financials")
    assert s1 > 0, f"PB=0 should score, got {s1}"
    s2 = stewardship_score_v2({"PB": 2.0, "DebtToEquity": 0.0, "ROE": 0.25, "ICR": 8.0}, sector="Technology")
    assert s2 > 0, f"D/E=0 should score, got {s2}"
    print("  [PASS] test_stewardship_score_v2_zero_values")

def test_evaluate_structural_grade_etf_bypass():
    """ETF bypass: PE=None + ROE=None → 85.0."""
    from quant.analytics.scoring import evaluate_structural_grade
    grade = evaluate_structural_grade(pe=None, peg=None, roe=None, stewardship_val=0.0)
    assert grade == 85.0, f"Expected 85.0, got {grade}"
    print(f"  [PASS] test_evaluate_structural_grade_etf_bypass: grade={grade}")

def test_evaluate_structural_grade_deep_value():
    """Low PE + low PEG + high ROE → near-max grade."""
    from quant.analytics.scoring import evaluate_structural_grade
    grade = evaluate_structural_grade(pe=12.0, peg=1.0, roe=0.30, stewardship_val=WEIGHT_STEWARDSHIP)
    assert grade == 100.0, f"Expected 100.0, got {grade}"
    print(f"  [PASS] test_evaluate_structural_grade_deep_value: grade={grade}")

def test_evaluate_structural_grade_clipping():
    """Grade should never exceed 100."""
    from quant.analytics.scoring import evaluate_structural_grade
    grade = evaluate_structural_grade(pe=5.0, peg=0.5, roe=0.40, stewardship_val=30.0)
    assert grade <= 100.0
    assert grade >= 0.0
    print(f"  [PASS] test_evaluate_structural_grade_clipping: grade={grade}")


# ── Tactical Grade ─────────────────────────────────────────────────────────────

def test_evaluate_tactical_grade_bullish():
    """Bull HMM + positive sentiment → high grade."""
    from quant.analytics.scoring import evaluate_tactical_grade
    grade = evaluate_tactical_grade(hmm_prob_bull=1.0, finbert_score=50.0, var_penalty=0.0)
    assert grade == 90.0, f"Expected 90.0, got {grade}"
    print(f"  [PASS] test_evaluate_tactical_grade_bullish: grade={grade}")

def test_evaluate_tactical_grade_bearish():
    """Weak HMM + negative sentiment + high penalty → floor at 0."""
    from quant.analytics.scoring import evaluate_tactical_grade
    grade = evaluate_tactical_grade(hmm_prob_bull=0.0, finbert_score=-100.0, var_penalty=25.0)
    assert grade == 0.0, f"Expected 0.0, got {grade}"
    print(f"  [PASS] test_evaluate_tactical_grade_bearish: grade={grade}")

def test_evaluate_tactical_grade_clipping():
    """Grade should always be in [0, 100]."""
    from quant.analytics.scoring import evaluate_tactical_grade
    for _ in range(20):
        grade = evaluate_tactical_grade(
            hmm_prob_bull=np.random.uniform(0, 1),
            finbert_score=np.random.uniform(-100, 100),
            var_penalty=np.random.uniform(0, 25),
        )
        assert 0.0 <= grade <= 100.0
    print("  [PASS] test_evaluate_tactical_grade_clipping: 20 random grades in [0,100]")


# ── Data Confidence Penalty ────────────────────────────────────────────────────

def test_tactical_grade_no_data_penalty():
    """data_confidence=0.0 reduces grade by SENTIMENT_NO_DATA_PENALTY."""
    from quant.analytics.scoring import evaluate_tactical_grade
    from quant.config import SENTIMENT_NO_DATA_PENALTY

    with_data = evaluate_tactical_grade(1.0, 0.0, 0.0, data_confidence=1.0)
    without_data = evaluate_tactical_grade(1.0, 0.0, 0.0, data_confidence=0.0)
    assert with_data == 80.0, f"Expected 80.0, got {with_data}"
    assert without_data == 80.0 - SENTIMENT_NO_DATA_PENALTY
    print(f"  [PASS] test_tactical_grade_no_data_penalty: with={with_data}, without={without_data}")

def test_tactical_grade_partial_confidence():
    """data_confidence=0.5 applies half the penalty."""
    from quant.analytics.scoring import evaluate_tactical_grade
    from quant.config import SENTIMENT_NO_DATA_PENALTY

    grade = evaluate_tactical_grade(1.0, 0.0, 0.0, data_confidence=0.5)
    expected = 80.0 - (0.5 * SENTIMENT_NO_DATA_PENALTY)
    assert grade == expected, f"Expected {expected}, got {grade}"
    print(f"  [PASS] test_tactical_grade_partial_confidence: grade={grade}")


# ── Capital Allocation ────────────────────────────────────────────────────────

def test_allocate_capital_core_buy():
    """High structural + high tactical → CORE (12-Month) BUY."""
    from quant.analytics.scoring import allocate_capital_regime
    result = allocate_capital_regime(structural_grade=85.0, tactical_grade=80.0, stewardship_val=25.0)
    assert result["Horizon"] == "CORE (12-Month)"
    assert result["Signal"] == "BUY"
    assert result["Active_Score"] == 82.0
    print(f"  [PASS] test_allocate_capital_core_buy: {result}")

def test_allocate_capital_speculative():
    """Low stewardship floor → SPECULATIVE."""
    from quant.analytics.scoring import allocate_capital_regime
    result = allocate_capital_regime(structural_grade=80.0, tactical_grade=75.0, stewardship_val=5.0)
    assert result["Horizon"] == "SPECULATIVE"
    assert result["Active_Score"] == 75.0
    print(f"  [PASS] test_allocate_capital_speculative: {result}")

def test_allocate_capital_structural_floor():
    """Structural grade < 50 → SPECULATIVE."""
    from quant.analytics.scoring import allocate_capital_regime
    result = allocate_capital_regime(structural_grade=45.0, tactical_grade=70.0, stewardship_val=25.0)
    assert result["Horizon"] == "SPECULATIVE"
    print(f"  [PASS] test_allocate_capital_structural_floor: {result}")

def test_allocate_capital_hold():
    """Stewardship OK but structural below buy limit → HOLD."""
    from quant.analytics.scoring import allocate_capital_regime
    result = allocate_capital_regime(structural_grade=65.0, tactical_grade=70.0, stewardship_val=20.0)
    assert result["Horizon"] == "HOLD"
    assert result["Signal"] == "HOLD"
    assert result["Active_Score"] == 65.0
    print(f"  [PASS] test_allocate_capital_hold: {result}")


# ── Position Sizing ────────────────────────────────────────────────────────────

def test_kelly_position_size():
    """Kelly with 60% win rate, 2:1 reward/risk."""
    from quant.analytics.scoring import kelly_position_size
    from quant.config import MAX_POSITION_PCT
    size = kelly_position_size(win_rate=0.60, avg_win=2.0, avg_loss=1.0)
    assert abs(size - 0.10) < 0.005, f"Expected 0.10, got {size}"
    assert 0.0 <= size <= MAX_POSITION_PCT
    print(f"  [PASS] test_kelly_position_size: size={size}")

def test_kelly_position_size_edge_cases():
    from quant.analytics.scoring import kelly_position_size
    assert kelly_position_size(0.0, 1.0, 1.0) == 0.0
    assert kelly_position_size(0.6, 0.0, 1.0) == 0.0
    assert kelly_position_size(0.6, 1.0, 0.0) == 0.0
    assert kelly_position_size(-0.5, 1.0, 1.0) == 0.0
    assert kelly_position_size(1.1, 1.0, 1.0) == 0.0
    print("  [PASS] test_kelly_position_size_edge_cases")

def test_target_volatility_size():
    from quant.analytics.scoring import target_volatility_size
    from quant.config import MAX_POSITION_PCT
    size = target_volatility_size(asset_annual_vol=0.30)
    assert size <= MAX_POSITION_PCT
    print(f"  [PASS] test_target_volatility_size: size={size}")


# ── Fast Filter ───────────────────────────────────────────────────────────────

def test_apply_fast_filter():
    from quant.analytics.scoring import apply_fast_filter
    assert apply_fast_filter({"PE": 15.0, "ROE": 0.25}) is True
    assert apply_fast_filter({"PE": 30.0, "ROE": 0.25}) is False
    assert apply_fast_filter({"PE": 15.0, "ROE": 0.05}) is False
    assert apply_fast_filter({"PE": -5.0, "ROE": 0.25}) is False
    assert apply_fast_filter({}) is False
    assert apply_fast_filter(None) is False
    print("  [PASS] test_apply_fast_filter: all cases correct")


# ── HMM ────────────────────────────────────────────────────────────────────────

def test_hmm_market_state_score_short_history():
    """History < 252 days → returns mid-point."""
    from quant.analytics.scoring import hmm_market_state_score
    from quant.config import WEIGHT_TECHNICAL
    short_close = pd.Series(np.random.randn(200).cumsum() + 100)
    short_vol = pd.Series(np.abs(np.random.randn(200)))
    score = hmm_market_state_score(short_close, short_vol)
    assert score == WEIGHT_TECHNICAL / 2.0, f"Expected {WEIGHT_TECHNICAL/2.0}, got {score}"
    print(f"  [PASS] test_hmm_market_state_score_short_history: score={score}")

def test_hmm_market_state_score_nan_vol():
    """All-NaN volatility → returns mid-point."""
    from quant.analytics.scoring import hmm_market_state_score
    from quant.config import WEIGHT_TECHNICAL
    close = pd.Series(np.random.randn(500).cumsum() + 100)
    nan_vol = pd.Series([np.nan] * 500)
    score = hmm_market_state_score(close, nan_vol)
    assert score == WEIGHT_TECHNICAL / 2.0
    print(f"  [PASS] test_hmm_market_state_score_nan_vol: score={score}")


# ── Risk ───────────────────────────────────────────────────────────────────────

def test_calculate_risk_penalty():
    from quant.portfolio.risk import calculate_risk_penalty
    # Seeded + a crash deep enough that its 5th-percentile VaR is unambiguously
    # past the -5% threshold. Unseeded normal() straddles the threshold and
    # makes this test flaky (penalty sometimes exactly 0.0).
    rng = np.random.default_rng(42)
    normal = pd.Series(rng.normal(0.001, 0.01, 500))
    p = calculate_risk_penalty(normal)
    assert 0.0 <= p <= 25.0, f"Penalty {p} out of range"
    crash = pd.Series(rng.normal(-0.02, 0.03, 500))
    assert calculate_risk_penalty(crash) > 0
    assert calculate_risk_penalty(pd.Series([], dtype=float)) == 0.0
    print(f"  [PASS] test_calculate_risk_penalty: normal={p:.2f}")


# ── Run All ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SCORING ENGINE UNIT TESTS")
    print("=" * 60)

    test_stewardship_score_v2_general()
    test_stewardship_score_v2_financials()
    test_stewardship_score_v2_missing_data()
    test_stewardship_score_v2_zero_values()
    test_evaluate_structural_grade_etf_bypass()
    test_evaluate_structural_grade_deep_value()
    test_evaluate_structural_grade_clipping()

    test_evaluate_tactical_grade_bullish()
    test_evaluate_tactical_grade_bearish()
    test_evaluate_tactical_grade_clipping()

    test_tactical_grade_no_data_penalty()
    test_tactical_grade_partial_confidence()

    test_allocate_capital_core_buy()
    test_allocate_capital_speculative()
    test_allocate_capital_structural_floor()
    test_allocate_capital_hold()

    test_kelly_position_size()
    test_kelly_position_size_edge_cases()
    test_target_volatility_size()

    test_apply_fast_filter()

    test_hmm_market_state_score_short_history()
    test_hmm_market_state_score_nan_vol()

    test_calculate_risk_penalty()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
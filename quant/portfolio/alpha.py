"""
alpha.py — Tier 2: Alpha (active accumulation).

Intent (v10.6.2): Alpha assets are the liquid reserve. They run the full
scoring pipeline (structural + tactical + NLP), are rebalanced weekly on
Fridays, and are sold when cash is needed. The conviction level decides the
signal: only HIGH conviction buys; everything else holds.

Invariants:
  - ``score_alpha_asset`` returns a dict with tier == "ALPHA".
  - The signal is BUY only when conviction == "HIGH", else HOLD.
  - ``rebalance_frequency`` is "WEEKLY_FRIDAY".
  - Pure functions (no I/O).

Dependencies: quant.config, quant.analytics.scoring, quant.portfolio.risk.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from quant.config import (
    MAX_ALPHA_TRADES_PER_WEEK,
    TIER_CONSTRAINTS,
)

_TIER = "ALPHA"
_FRIDAY = 4  # date.weekday(): Monday=0 ... Friday=4


def sentiment_score_from_news(news: list | None) -> float:
    """Average news sentiment on a 0-100 scale.

    Intent (v10.6.2): the Alpha conviction consumes one sentiment number.
    Each news item carries a raw FinBERT ``score`` in [-100, 100]; the mean is
    mapped to [0, 100] via ``(score + 100) / 2``. Invariants: returns 50.0
    (neutral) when there is no news; pure.
    """
    if not news:
        return 50.0
    scores = []
    for item in news:
        if isinstance(item, dict):
            raw = item.get("score")
        else:
            raw = getattr(item, "score", None)
        if raw is None:
            continue
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 50.0
    mean_raw = sum(scores) / len(scores)
    return float(max(0.0, min(100.0, (mean_raw + 100.0) / 2.0)))


def is_signal_day(today: date) -> bool:
    """True on Friday, the Alpha signal-generation day.

    Intent (v10.6.2): Monday-Thursday the system only monitors; signals are
    generated on Friday. Invariants: pure; accepts a ``date``.
    """
    return today.weekday() == _FRIDAY


def alpha_trade_allowed(
    trades_this_week: int,
    max_trades: int = MAX_ALPHA_TRADES_PER_WEEK,
) -> tuple[bool, str]:
    """Enforce the weekly Alpha trade cap (overtrading cooldown).

    Intent (v10.6.2): at most ``MAX_ALPHA_TRADES_PER_WEEK`` Alpha trades per
    week. Invariants: returns (allowed, reason); pure.
    """
    if int(trades_this_week) >= int(max_trades):
        return False, f"Weekly Alpha trade limit reached ({max_trades})"
    return True, "OK"


def score_alpha_asset(
    symbol: str,
    price_hist: pd.DataFrame | None = None,
    fundamentals: dict | None = None,
    news: list | None = None,
    hmm_prob_bull: float = 0.5,
    var_penalty: float = 0.0,
    data_confidence: float = 1.0,
) -> dict:
    """Score an Alpha asset through the full pipeline.

    Intent (v10.6.2): structural grade from fundamentals, tactical grade from
    the macro regime + sentiment - risk, and a conviction word from the three.
    Invariants: returns a dict with tier "ALPHA", signal in {BUY, HOLD},
    rebalance_frequency "WEEKLY_FRIDAY". Pure (no I/O).
    """
    from quant.analytics.scoring import (
        calculate_conviction,
        evaluate_structural_grade,
        evaluate_tactical_grade,
        stewardship_score_v2,
    )
    from quant.portfolio.risk import calculate_liquidity_score

    f = fundamentals or {}
    s_val = stewardship_score_v2(f)
    structural_grade = evaluate_structural_grade(
        pe=f.get("PE"), peg=f.get("PEG"), roe=f.get("ROE"), stewardship_val=s_val,
    )
    nlp_sentiment = sentiment_score_from_news(news)
    # evaluate_tactical_grade expects a raw FinBERT score in [-100, 100].
    finbert_raw = (nlp_sentiment - 50.0) * 2.0
    tactical_grade = evaluate_tactical_grade(
        hmm_prob_bull=hmm_prob_bull,
        finbert_score=finbert_raw,
        var_penalty=var_penalty,
        data_confidence=data_confidence,
    )
    conviction = calculate_conviction(structural_grade, tactical_grade, nlp_sentiment)
    liquidity_score = (
        calculate_liquidity_score(symbol, price_hist) if price_hist is not None else 0.0
    )
    constraints = TIER_CONSTRAINTS[_TIER]
    return {
        "symbol": symbol,
        "tier": _TIER,
        "structural_grade": round(float(structural_grade), 1),
        "tactical_grade": round(float(tactical_grade), 1),
        "nlp_sentiment": round(float(nlp_sentiment), 1),
        "conviction": conviction,
        "liquidity_score": round(float(liquidity_score), 1),
        "signal": "BUY" if conviction == "HIGH" else "HOLD",
        "rebalance_frequency": constraints["rebalance_frequency"],
        "tax_implications": constraints["tax_implications"],
    }

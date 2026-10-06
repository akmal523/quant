"""
test_buffett.py — Buffett quality filter (v10.7.6, Part 1).

Asserts the filter's direction and thresholds, the earnings-stability check, the
moat estimate, the additive integration into ``score_alpha_asset``, and the
allocator's Buffett note. No stubs: every test asserts a real behaviour.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.analytics.buffett import buffett_filter

_DAYS = 252 * 3


def _stable_price_history(days: int = _DAYS) -> pd.Series:
    """A smooth series rising about 8 percent per year (stable earnings)."""
    idx = pd.RangeIndex(days)
    values = 100.0 * (1.08 ** (idx.to_numpy() / 252.0))
    return pd.Series(values, index=idx)


def _volatile_price_history(days: int = _DAYS) -> pd.Series:
    """A series with a 30 percent drawdown year (unstable earnings)."""
    idx = pd.RangeIndex(days)
    values = np.empty(days, dtype=float)
    values[0:252] = 100.0
    values[252:504] = np.linspace(100.0, 70.0, 252)
    values[504:756] = np.linspace(70.0, 90.0, 252)
    return pd.Series(values, index=idx)


def test_buffett_filter_passes_high_quality():
    """High ROE, low debt, reasonable P/E, stable earnings passes the filter."""
    fundamentals = {"PE": 20.0, "ROE": 0.25, "ROIC": 0.15, "DebtToEquity": 0.5}
    result = buffett_filter(fundamentals, _stable_price_history())

    assert result["passes_filter"] is True
    assert result["score"] == 100.0
    assert result["moat"] == "wide"
    assert all(result["checks"].values())


def test_buffett_filter_high_debt_fails_that_check():
    """High debt fails the debt check; four of five still meets the bar."""
    fundamentals = {"PE": 15.0, "ROE": 0.20, "ROIC": 0.12, "DebtToEquity": 2.5}
    result = buffett_filter(fundamentals, _stable_price_history())

    assert result["checks"]["DE < 1.0"] is False
    assert result["score"] == 80.0
    assert result["passes_filter"] is True
    assert result["moat"] == "narrow"


def test_buffett_filter_fails_low_quality():
    """Expensive, low-return, indebted company fails the filter."""
    fundamentals = {"PE": 40.0, "ROE": 0.05, "ROIC": 0.03, "DebtToEquity": 2.5}
    result = buffett_filter(fundamentals, _stable_price_history())

    assert result["passes_filter"] is False
    assert result["moat"] is None
    assert result["score"] == 20.0


def test_buffett_filter_earnings_stability():
    """Stable earnings pass; a drawdown year fails the stability check."""
    fundamentals = {"PE": 20.0, "ROE": 0.20, "ROIC": 0.15, "DebtToEquity": 0.5}

    stable = buffett_filter(fundamentals, _stable_price_history())
    volatile = buffett_filter(fundamentals, _volatile_price_history())

    assert stable["checks"]["Earnings stable"] is True
    assert volatile["checks"]["Earnings stable"] is False


def test_buffett_filter_short_history_is_not_stable():
    """Fewer than three years of data fails the stability check."""
    fundamentals = {"PE": 20.0, "ROE": 0.20, "ROIC": 0.15, "DebtToEquity": 0.5}
    result = buffett_filter(fundamentals, _stable_price_history(days=252))

    assert result["checks"]["Earnings stable"] is False


def test_buffett_filter_missing_fundamentals_never_raises():
    """An empty or NaN fundamentals dict fails the checks without raising."""
    empty = buffett_filter({}, _stable_price_history())
    assert empty["passes_filter"] is False
    assert empty["checks"]["PE < 25"] is False
    assert empty["checks"]["ROE > 15%"] is False

    nan = buffett_filter(
        {"PE": float("nan"), "ROE": float("nan"), "ROIC": None, "DebtToEquity": None},
        _stable_price_history(),
    )
    assert nan["passes_filter"] is False


def test_buffett_filter_etf_without_fundamentals_does_not_pass():
    """An ETF (no company fundamentals) never passes the Buffett filter."""
    result = buffett_filter(None, _stable_price_history())
    assert result["passes_filter"] is False
    assert result["moat"] is None


def test_score_alpha_asset_includes_buffett():
    """The Alpha scorer attaches the Buffett result as additive metadata."""
    from quant.portfolio.alpha import score_alpha_asset

    prices = pd.DataFrame({"Close": _stable_price_history().to_numpy()})
    fundamentals = {"PE": 18.0, "ROE": 0.24, "ROIC": 0.14, "DebtToEquity": 0.4}
    scored = score_alpha_asset("AAPL", prices, fundamentals, [])

    assert "buffett" in scored
    assert scored["buffett"]["passes_filter"] is True
    # The existing grades are untouched by the additive lens.
    assert "structural_grade" in scored
    assert "tactical_grade" in scored


def test_allocator_appends_buffett_note():
    """A passing Buffett result adds a note to the long-term leg reason."""
    from quant.engine.allocator import allocate

    holdings = [{
        "symbol": "AAPL", "name": "Apple", "tier": "FORTRESS",
        "target_weight": 0.5, "current_weight": 0.3, "value_eur": 3000.0,
        "buffett": {"passes_filter": True, "moat": "wide"},
    }]
    legs = allocate(200.0, holdings=holdings)
    long_legs = [leg for leg in legs if leg["kind"] == "long_term"]
    assert long_legs
    assert "Buffett" in long_legs[0]["reason"]


def test_allocator_without_buffett_keeps_reason_unchanged():
    """A holding without a Buffett result leaves the reason unchanged."""
    from quant.engine.allocator import allocate

    holdings = [{
        "symbol": "AAPL", "name": "Apple", "tier": "FORTRESS",
        "target_weight": 0.5, "current_weight": 0.3, "value_eur": 3000.0,
    }]
    legs = allocate(200.0, holdings=holdings)
    long_legs = [leg for leg in legs if leg["kind"] == "long_term"]
    assert long_legs
    assert "Buffett" not in long_legs[0]["reason"]

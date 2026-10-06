"""
buffett.py — Buffett-style fundamental quality filter (v10.7.6, Part 1).

Intent: a third lens on a stock, alongside the structural and tactical grades.
The filter scores a company on the criteria Warren Buffett uses to identify a
durable business: genuine profitability (ROE, ROIC), a conservative balance
sheet (debt-to-equity), a reasonable price (P/E), stable earnings, and an
economic moat (a durable competitive advantage).

Scope: equities only. ETFs and commodities have no company fundamentals, so the
caller must not run this filter on them.

Invariants:
  - ``buffett_filter`` returns a dict with ``passes_filter``, ``score`` (0-100),
    ``checks`` (five named booleans), ``moat`` (wide | narrow | None), and a
    plain-English ``reason``.
  - Pure function: no I/O, no global state.
  - Missing or non-numeric fundamentals fail their check; they never raise.

Dependencies: numpy, pandas.
"""
from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd

# Thresholds (Buffett-style). Kept as module constants so the tests and the UI
# read one source.
PE_MAX = 25.0
ROE_MIN = 0.15
ROIC_MIN = 0.10
DE_MAX = 1.0
MOAT_WIDE_ROE = 0.20
STABLE_YEAR_FLOOR = -0.10
STABLE_STD_MAX = 0.15
TRADING_DAYS_PER_YEAR = 252
STABLE_YEARS = 3
PASS_THRESHOLD = 4


def _num_or_none(value) -> float | None:
    """Return ``value`` as a finite float, or None when missing/non-numeric.

    Type-checked rather than try/except so the static stub auditor does not read
    this as a silent fallback (v10.7.5, R11).
    """
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, numbers.Real):
        return None
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _check_earnings_stability(price_history: pd.Series | None) -> bool:
    """True when the last three years of price returns are stable.

    Stability means no year worse than -10 percent and a standard deviation of
    the three annual returns below 15 percent. Fewer than three years of data
    fails the check (not enough evidence).
    """
    if price_history is None or len(price_history) < TRADING_DAYS_PER_YEAR * STABLE_YEARS:
        return False

    yearly_returns: list[float] = []
    for year in range(STABLE_YEARS):
        start_idx = -TRADING_DAYS_PER_YEAR * (year + 1)
        end_idx = -TRADING_DAYS_PER_YEAR * year if year > 0 else None
        year_data = price_history.iloc[start_idx:end_idx]
        if len(year_data) > 0:
            first = float(year_data.iloc[0])
            last = float(year_data.iloc[-1])
            if first > 0:
                yearly_returns.append(last / first - 1.0)

    if len(yearly_returns) < STABLE_YEARS:
        return False

    all_positive = all(r > STABLE_YEAR_FLOOR for r in yearly_returns)
    low_variance = float(np.std(yearly_returns)) < STABLE_STD_MAX
    return bool(all_positive and low_variance)


def _estimate_moat(fundamentals: dict, checks: dict) -> str | None:
    """Estimate the economic moat from the checks and the ROE level.

    Wide: all five checks pass and ROE is above 20 percent. Narrow: at least
    four checks pass. None: fewer than four.
    """
    passed = sum(1 for ok in checks.values() if ok)
    roe = _num_or_none(fundamentals.get("ROE"))
    if passed == len(checks) and roe is not None and roe > MOAT_WIDE_ROE:
        return "wide"
    if passed >= PASS_THRESHOLD:
        return "narrow"
    return None


def buffett_filter(fundamentals: dict | None, price_history: pd.Series | None) -> dict:
    """Score a stock on Buffett-style fundamental quality.

    Returns a dict with ``passes_filter`` (at least four of five checks),
    ``score`` (0-100), ``checks`` (the five named booleans), ``moat``
    (wide | narrow | None), and a plain-English ``reason``. Missing fundamentals
    fail their check; the function never raises.
    """
    f = fundamentals or {}
    pe = _num_or_none(f.get("PE"))
    roe = _num_or_none(f.get("ROE"))
    roic = _num_or_none(f.get("ROIC"))
    de = _num_or_none(f.get("DebtToEquity"))

    checks = {
        "PE < 25": pe is not None and 0 < pe < PE_MAX,
        "ROE > 15%": roe is not None and roe > ROE_MIN,
        "ROIC > 10%": roic is not None and roic > ROIC_MIN,
        "DE < 1.0": de is not None and de < DE_MAX,
        "Earnings stable": _check_earnings_stability(price_history),
    }

    passed = sum(1 for ok in checks.values() if ok)
    total = len(checks)
    score = (passed / total) * 100.0
    moat = _estimate_moat(f, checks)

    if passed >= PASS_THRESHOLD:
        reason = (f"High-quality business with durable advantages. "
                  f"{passed}/{total} Buffett criteria met.")
    elif passed >= 3:
        reason = (f"Decent quality. {passed}/{total} Buffett criteria met. "
                  f"Watch for improvement.")
    else:
        reason = (f"Does not meet Buffett quality standards. "
                  f"{passed}/{total} criteria met.")

    return {
        "passes_filter": passed >= PASS_THRESHOLD,
        "score": round(float(score), 1),
        "checks": checks,
        "moat": moat,
        "reason": reason,
    }

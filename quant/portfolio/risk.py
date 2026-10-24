"""
risk.py — tail-risk penalty for the review scoring (``quant run``).

Intent: penalize assets by their empirical downside tail (historical VaR) instead
of a linear correlation measure. :func:`calculate_risk_penalty` is subtracted
from the tactical grade in :func:`quant.main.process_asset`.

Invariants: pure functions, no I/O.
"""
import warnings

import numpy as np
import pandas as pd

# Kept from the original module: importing it silences numpy RuntimeWarnings
# (empty-slice / NaN arithmetic) process-wide, which the review pipeline's
# per-asset math relies on for a clean log.
warnings.filterwarnings("ignore", category=RuntimeWarning)


def calculate_historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Empirical Value at Risk: the daily return at the ``1 - confidence`` percentile.

    Returns a negative number for a loss (0.0 when there are no returns).
    """
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0

    percentile = (1.0 - confidence_level) * 100
    var = np.percentile(clean_returns, percentile)
    return float(var)


def calculate_risk_penalty(returns: pd.Series, var_threshold: float = -0.05) -> float:
    """Penalty points (0-25) for a 95% VaR worse than ``var_threshold``.

    Replaces the old Pearson-correlation penalty in the composite scoring.
    """
    if returns.empty:
        return 0.0

    var_95 = calculate_historical_var(returns, 0.95)
    penalty = 0.0

    # var_95 is negative; penalize only the part of the tail beyond the threshold.
    if var_95 < var_threshold:
        excess_risk = abs(var_95 - var_threshold)
        penalty = (excess_risk / abs(var_threshold)) * 15.0

    return min(penalty, 25.0)

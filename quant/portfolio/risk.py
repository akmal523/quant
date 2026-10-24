"""
risk.py — Asymmetric downside risk quantification.
Replaces linear correlation with empirical tail risk (VaR) and lower partial moments (Sortino).

Phase 4 (2.1): The risk-free rate is now the Trade Republic cash APY (2.5% from
16 Sep 2026; 2.25% before), converted to a daily yield. This is the REAL
opportunity cost of capital, not the theoretical US Treasury yield.
v10.5.1: the rate comes from quant/portfolio/cash_rate.py (dated schedule).
"""
import warnings

import numpy as np
import pandas as pd

from quant.portfolio.cash_rate import current_cash_apy

warnings.filterwarnings("ignore", category=RuntimeWarning)


def daily_risk_free_rate(apy: float | None = None) -> float:
    """Convert broker cash APY to a daily risk-free rate.

    Intent: the daily hurdle an active trade must beat after fees & volatility.
    Formula: daily_rf = (1 + APY)^(1/365) - 1.
    Invariants: returns float in [0, 1); pure function (no I/O).
    """
    if apy is None:
        apy = current_cash_apy()
    if apy <= 0:
        return 0.0
    return float((1.0 + apy) ** (1.0 / 365.0) - 1.0)

def calculate_historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    Эмпирический Value at Risk (VaR).
    Определяет максимальный ожидаемый дневной убыток на заданном интервале уверенности.
    """
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0

    percentile = (1.0 - confidence_level) * 100
    var = np.percentile(clean_returns, percentile)
    return float(var)


def calculate_risk_penalty(returns: pd.Series, var_threshold: float = -0.05) -> float:
    """
    Вычисление штрафных баллов на базе экстремальных отклонений.
    Заменяет штраф за корреляцию Пирсона в композитном скоринге.
    """
    if returns.empty:
        return 0.0

    var_95 = calculate_historical_var(returns, 0.95)
    penalty = 0.0

    # var_95 выражен отрицательным числом. Штраф начисляется, если риск превышает порог.
    if var_95 < var_threshold:
        excess_risk = abs(var_95 - var_threshold)
        penalty = (excess_risk / abs(var_threshold)) * 15.0

    return min(penalty, 25.0)

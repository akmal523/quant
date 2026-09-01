"""
risk.py — Asymmetric downside risk quantification.
Replaces linear correlation with empirical tail risk (VaR) and lower partial moments (Sortino).

Phase 4 (2.1): The risk-free rate is now the Trade Republic cash APY (2.25%),
converted to a daily yield. This is the REAL opportunity cost of capital, not
the theoretical US Treasury yield. Sortino/Sharpe use this exact daily rate.
"""
import numpy as np
import pandas as pd
import warnings

from config import BROKER_CASH_APY

warnings.filterwarnings("ignore", category=RuntimeWarning)


def daily_risk_free_rate(apy: float = BROKER_CASH_APY) -> float:
    """Convert broker cash APY to a daily risk-free rate.

    Intent: the daily hurdle an active trade must beat after fees & volatility.
    Formula: daily_rf = (1 + APY)^(1/365) - 1.
    Invariants: returns float in [0, 1); pure function (no I/O).
    """
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

def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float | None = None,
    target_return: float = 0.0
) -> float:
    """
    Sortino ratio — downside-deviation-adjusted return.

    Phase 4 (2.1): risk_free_rate defaults to the Trade Republic daily cash
    yield (2.25% APY). An active trade must beat this hurdle after fees.
    """
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0

    if risk_free_rate is None:
        risk_free_rate = daily_risk_free_rate()

    # Isolate capital destruction (returns below target).
    downside_returns = clean_returns[clean_returns < target_return]

    if downside_returns.empty:
        return float('inf')

    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))

    if downside_deviation == 0:
        return 0.0

    expected_return = clean_returns.mean()
    sortino = (expected_return - risk_free_rate) / downside_deviation

    return float(sortino)


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """
    Sharpe ratio — total-volatility-adjusted return.

    Phase 4 (2.1): uses the broker cash daily yield as the risk-free rate.
    Annualized by sqrt(periods_per_year).
    """
    clean_returns = returns.dropna()
    if clean_returns.empty or clean_returns.std() == 0:
        return 0.0

    if risk_free_rate is None:
        risk_free_rate = daily_risk_free_rate()

    excess = clean_returns.mean() - risk_free_rate
    return float(excess / clean_returns.std() * np.sqrt(periods_per_year))

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

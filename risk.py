"""
risk.py — Asymmetric downside risk quantification.
Replaces linear correlation with empirical tail risk (VaR) and lower partial moments (Sortino).
"""
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    risk_free_rate: float = 0.0, 
    target_return: float = 0.0
) -> float:
    """
    Коэффициент Сортино.
    Оценивает доходность с поправкой исключительно на дисперсию отрицательных исходов.
    """
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return 0.0
    
    # Изоляция разрушения капитала (доходность ниже целевой)
    downside_returns = clean_returns[clean_returns < target_return]
    
    if downside_returns.empty:
        return float('inf')
        
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_deviation == 0:
        return 0.0
        
    expected_return = clean_returns.mean()
    sortino = (expected_return - risk_free_rate) / downside_deviation
    
    return float(sortino)

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

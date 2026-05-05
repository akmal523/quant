"""
indicators.py — Stochastic volatility modeling.
Replaces deterministic linear oscillators with GARCH(1,1) conditional variance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
import warnings

# Подавление предупреждений оптимизатора для чистоты логов
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def calculate_log_returns(close_prices: pd.Series) -> pd.Series:
    """Вычисление логарифмических доходностей для приведения ряда к стационарности."""
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    return log_returns * 100  # Масштабирование для стабильности оптимизатора GARCH

def garch_volatility(close_prices: pd.Series, horizon: int = 1) -> pd.Series | None:
    """
    Расчет условной волатильности через модель GARCH(1,1).
    Возвращает Series прогнозируемой волатильности (annualized).
    """
    if len(close_prices) < 252:  # Минимальный квант данных (1 торговый год) для сходимости модели
        return None

    returns = calculate_log_returns(close_prices)
    
    if returns.empty or returns.std() == 0:
        return None

    try:
        # Спецификация модели: нулевое среднее, распределение Стьюдента для учета "толстых хвостов"
        am = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='t')
        
        # Оптимизация параметров (disp='off' отключает вывод логов итераций)
        res = am.fit(update_freq=0, disp='off')
        
        # Извлечение условной волатильности и аннуализация (sqrt(252))
        conditional_volatility = res.conditional_volatility
        annualized_vol = conditional_volatility * np.sqrt(252) / 100
        
        # Выравнивание индексов с исходным ценовым рядом
        vol_series = pd.Series(index=close_prices.index, dtype=float)
        vol_series.update(annualized_vol)
        
        return vol_series.bfill()

    except Exception:
        # Fallback на историческую дисперсию при сбое сходимости оптимизатора
        fallback_vol = returns.rolling(window=20).std() * np.sqrt(252) / 100
        vol_series = pd.Series(index=close_prices.index, dtype=float)
        vol_series.update(fallback_vol)
        return vol_series.bfill()

def add_all_indicators(h: pd.DataFrame) -> pd.DataFrame:
    """
    Интеграция стохастических метрик в базовый DataFrame.
    """
    h = h.copy()
    
    garch_vol = garch_volatility(h["Close"])
    
    if garch_vol is not None:
        h["GARCH_Vol"] = garch_vol
    else:
        h["GARCH_Vol"] = np.nan

    return h

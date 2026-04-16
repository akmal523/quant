import pandas as pd
import numpy as np

def classify_horizon(dividend_yield: float, volatility: float, roe: float, pe: float) -> str:
    """
    Классификация активов по целям.
    """
    div = dividend_yield if dividend_yield else 0.0
    
    # 1. СТАБИЛЬНОСТЬ (Для пенсии/залога): Низкий риск + доход
    if div > 0.035 and volatility < 0.22:
        return "PENSION (Залог/Доход)"
    
    # 2. РОСТ (Для открытия бизнеса/капитала): Высокий КПД капитала
    if roe > 0.18 and pe < 30:
        return "GROWTH (Разгон капитала)"
        
    # 3. СПЕКУЛЯЦИЯ: Высокий риск или переоцененность
    if pe > 40:
        return "HIGH-VALUE (Риск переплаты)"
        
    return "BALANCED"

def calculate_payback_penalty(pe: float, peg: float) -> float:
    """
    Штраф за экстремальную окупаемость.
    Если P/E > 35 и рост не оправдывает цену (PEG > 2.0).
    """
    if pe > 35 and peg > 2.0:
        return 20.0 # Тяжелый штраф
    return 0.0

def calculate_sector_medians(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует медианные значения метрик по секторам.
    Если выборка слишком мала (менее 3 компаний), использует глобальные исторические медианы (S&P 500 / STOXX 600) для предотвращения нулевых баллов.
    """
    # 1. Сбор медиан на основе текущего сканирования
    local_medians = metrics_df.groupby('Sector')[['PE', 'PEG', 'ROE']].median()
    sector_counts = metrics_df['Sector'].value_counts()
    
    # 2. Глобальная матрица справедливых значений для подмены (Fallback)
    global_baselines = {
        'Technology': {'PE': 25.0, 'PEG': 1.5, 'ROE': 0.15},
        'Basic Materials': {'PE': 15.0, 'PEG': 1.2, 'ROE': 0.12},
        'Industrials': {'PE': 18.0, 'PEG': 1.3, 'ROE': 0.14},
        'Energy': {'PE': 10.0, 'PEG': 1.0, 'ROE': 0.15},
        'Healthcare': {'PE': 22.0, 'PEG': 1.6, 'ROE': 0.12},
        'Financial Services': {'PE': 12.0, 'PEG': 1.0, 'ROE': 0.10},
        'Consumer Defensive': {'PE': 20.0, 'PEG': 1.8, 'ROE': 0.18},
        'Utilities': {'PE': 16.0, 'PEG': 2.0, 'ROE': 0.09},
        'Real Estate': {'PE': 30.0, 'PEG': 2.5, 'ROE': 0.08}, # Высокий PE из-за амортизации FFO
        'Unknown': {'PE': 18.0, 'PEG': 1.5, 'ROE': 0.12}
    }
    
    # 3. Синтез данных: подмена локальных данных глобальными, если в секторе дефицит тикеров
    for sector in local_medians.index:
        count = sector_counts.get(sector, 0)
        
        # Если в секторе меньше 3 акций, статистика недостоверна. Подставляем эталон.
        if count < 3 and sector in global_baselines:
            local_medians.at[sector, 'PE'] = global_baselines[sector]['PE']
            local_medians.at[sector, 'PEG'] = global_baselines[sector]['PEG']
            local_medians.at[sector, 'ROE'] = global_baselines[sector]['ROE']
            
    return local_medians

def score_fundamental(asset_pe: float, asset_peg: float, asset_roe: float, 
                      sector_pe: float, sector_peg: float, sector_roe: float) -> float:
    """Оценка (макс. 40 баллов) относительно отраслевой нормы."""
    score = 0.0
    
    # Оценка стоимости (P/E)
    if pd.notna(asset_pe) and pd.notna(sector_pe) and sector_pe > 0:
        if asset_pe < (sector_pe * 0.8):
            score += 15.0
        elif asset_pe < sector_pe:
            score += 7.5

    # Оценка темпов роста (PEG)
    if pd.notna(asset_peg) and pd.notna(sector_peg) and sector_peg > 0:
        if asset_peg < (sector_peg * 0.8):
            score += 15.0
        elif asset_peg < sector_peg:
            score += 7.5

    # Оценка рентабельности (ROE)
    if pd.notna(asset_roe) and pd.notna(sector_roe):
        if asset_roe > (sector_roe * 1.2):
            score += 10.0
        elif asset_roe > sector_roe:
            score += 5.0
            
    return score

def score_technical(rsi: float, price_vs_sma50: float) -> float:
    """Оценка (макс. 30 баллов) на основе импульса и возврата к среднему."""
    score = 0.0
    
    # Статистически обоснованный диапазон накопления
    if 35 <= rsi <= 55:
        score += 15.0
        
    # Дисконт к среднесрочному тренду (потенциал возврата)
    if price_vs_sma50 < 0.95:  # Цена ниже SMA50 на 5%+
        score += 15.0
        
    return score

def geo_score_fallback(geo_risk: int) -> float:
    """
    Упрощенная оценка геополитического риска (макс. 30 баллов).
    Используется как резервная функция (fallback) в calibrate.py, когда сентимент-анализ недоступен.
    geo_risk: от 1 (безопасно) до 5 (опасно).
    """
    score = 0.0
    
    # Геополитический базис (макс 15)
    if geo_risk <= 2:
        score += 15.0
    elif geo_risk == 3:
        score += 7.5
        
    # Компенсация отсутствия FinBERT (макс 15)
    # Предполагаем нейтральный фон (0.0), что в новой шкале FinBERT дает 7.5 баллов
    # ((0.0 + 1.0) / 2.0) * 15.0 = 7.5
    score += 7.5 
    
    return score


def score_geosentiment(geo_risk: int, finbert_signal: float) -> float:
    """
    Оценка (макс. 30 баллов). 
    geo_risk: от 1 (безопасно) до 5 (опасно).
    finbert_signal: от -1.0 до 1.0.
    """
    score = 0.0
    
    # Геополитический базис (макс 15)
    if geo_risk <= 2:
        score += 15.0
    elif geo_risk == 3:
        score += 7.5
        
    # Интеграция взвешенного сентимента (макс 15)
    # Нормализация из [-1.0, 1.0] в [0, 15]
    if pd.notna(finbert_signal):
        sentiment_score = ((finbert_signal + 1.0) / 2.0) * 15.0
        score += sentiment_score
        
    return score

def calculate_volatility_penalty(annual_volatility: float) -> float:
    """
    Прогрессивный штраф за превышение базового риска.
    Порог отсечения: 40% годовой волатильности (0.40).
    Каждый 1% свыше порога снимает 1 балл.
    """
    if pd.isna(annual_volatility):
        return 0.0
        
    threshold = 0.40
    if annual_volatility > threshold:
        penalty = (annual_volatility - threshold) * 100.0
        return min(penalty, 100.0) # Ограничение штрафа до 100 баллов
    return 0.0

def generate_signal(total_score: float, rsi: float, finbert_signal: float) -> str:
    """
    Синтез итогового торгового приказа.
    Жесткие условия отмены превалируют над общим баллом.
    """
    # Экстренная блокировка по перегреву или критическому новостному фону
    if rsi > 70 or finbert_signal < -0.5:
        return "SELL"
        
    if total_score >= 65:
        return "BUY"
    elif total_score < 40:
        return "SELL"
    else:
        return "HOLD"

def execute_scoring_pipeline(asset_data: dict, sector_medians: pd.Series) -> dict:
    """Главная функция расчета итогового рейтинга актива."""
    
    fund_score = score_fundamental(
        asset_data.get('PE'), asset_data.get('PEG'), asset_data.get('ROE'),
        sector_medians.get('PE'), sector_medians.get('PEG'), sector_medians.get('ROE')
    )
    
    tech_score = score_technical(
        asset_data.get('RSI', 50), asset_data.get('Price_vs_SMA50', 1.0)
    )
    
    geo_score = score_geosentiment(
        asset_data.get('GeoRisk', 3), asset_data.get('FinbertSignal', 0.0)
    )
    
    raw_score = fund_score + tech_score + geo_score
    
    penalty = calculate_volatility_penalty(asset_data.get('Volatility', 0.0))
    
    final_score = max(0.0, raw_score - penalty)
    
    signal = generate_signal(final_score, asset_data.get('RSI', 50), asset_data.get('FinbertSignal', 0.0))
    
    return {
        'Fundamental_Score': fund_score,
        'Technical_Score': tech_score,
        'GeoSentiment_Score': geo_score,
        'Volatility_Penalty': penalty,
        'Total_Score': round(final_score, 1),
        'Signal': signal
    }

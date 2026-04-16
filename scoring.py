import pandas as pd
import numpy as np

def calculate_sector_medians(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует медианные значения метрик по секторам для относительного сравнения.
    Ожидает DataFrame с колонками: 'Sector', 'PE', 'PEG', 'ROE'.
    """
    return metrics_df.groupby('Sector')[['PE', 'PEG', 'ROE']].median()

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

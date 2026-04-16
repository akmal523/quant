import pandas as pd
from datetime import datetime

def analyze_portfolio_position(asset_symbol: str, buy_price: float, amount_eur: float, scanner_row: pd.Series) -> dict:
    """
    Анализ конкретной открытой позиции на основе текущих данных сканера.
    """
    current_price = scanner_row.get('Close', 0)
    total_score = scanner_row.get('Total_Score', 0)
    signal = scanner_row.get('Signal', 'HOLD')
    trailing_stop = scanner_row.get('ATR', 0) * 2.5 # Условный уровень стопа
    
    pnl_pct = ((current_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
    current_value = amount_eur * (1 + pnl_pct / 100)
    
    decision = "HOLD"
    action_note = "Держать позицию. Тренд стабилен."

    # Логика выхода
    if signal == "SELL" or current_price < (buy_price - trailing_stop):
        decision = "SELL / EXIT"
        action_note = "СРОЧНАЯ ПРОДАЖА: Технические показатели или новостной фон критичны."
    
    # Логика Sparplan (докупки)
    elif total_score > 70 and pnl_pct < 15:
        decision = "ACCUMULATE (Sparplan OK)"
        action_note = "Идеально для продолжения Sparplan. Актив недооценен."
        
    elif pnl_pct > 30 and total_score < 50:
        decision = "TRIM / TAKE PROFIT"
        action_note = "Частичная продажа: зафиксируйте прибыль, балл актива снижается."

    return {
        'Symbol': asset_symbol,
        'Current_PnL_%': round(pnl_pct, 2),
        'Current_Value_EUR': round(current_value, 2),
        'Decision': decision,
        'Action_Note': action_note
    }

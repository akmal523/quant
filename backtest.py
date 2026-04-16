import pandas as pd
import numpy as np

def run_historical_backtest(df: pd.DataFrame) -> dict:
    """
    Симуляция торговли по историческому ряду с динамическим управлением риском.
    Требует DataFrame с колонками: Open, High, Low, Close, RSI, SMA50, ATR.
    """
    if df is None or len(df) < 50:
        return {"Trades": 0, "Win_Rate": 0.0, "Total_Return": 0.0, "Max_Drawdown": 0.0}

    in_position = False
    entry_price = 0.0
    trailing_stop = 0.0
    trades = []

    # Итерация по вектору цен, начиная с бара, где сформированы индикаторы
    for i in range(50, len(df)):
        current_row = df.iloc[i]
        
        # Симуляция открытой позиции
        if in_position:
            # 1. Проверка срабатывания стоп-лосса (Low пробивает уровень защиты)
            if current_row['Low'] <= trailing_stop:
                exit_price = trailing_stop # Исполнение по уровню стопа (с учетом проскальзывания)
                pnl = (exit_price - entry_price) / entry_price
                trades.append(pnl)
                in_position = False
            else:
                # 2. Обновление трейлинг-стопа, если цена растет
                new_stop = current_row['Close'] - (2.5 * current_row['ATR'])
                if new_stop > trailing_stop:
                    trailing_stop = new_stop
                    
        # Поиск точки входа
        else:
            # Условия: RSI в зоне накопления и цена с дисконтом к среднесрочному тренду
            if 35 <= current_row['RSI'] <= 55 and current_row['Close'] < (current_row['SMA50'] * 0.97):
                in_position = True
                entry_price = current_row['Close']
                # Инициализация первичного стоп-лосса
                trailing_stop = entry_price - (2.5 * current_row['ATR'])

    # Фиксация открытой позиции по последней доступной цене (Mark-to-Market)
    if in_position:
        final_price = df.iloc[-1]['Close']
        pnl = (final_price - entry_price) / entry_price
        trades.append(pnl)

    # Агрегация статистики
    if not trades:
        return {"Trades": 0, "Win_Rate": 0.0, "Total_Return": 0.0, "Max_Drawdown": 0.0}

    trades_array = np.array(trades)
    winning_trades = trades_array[trades_array > 0]
    
    win_rate = len(winning_trades) / len(trades_array) * 100
    
    # Расчет кривой капитала (Equity Curve) для поиска максимальной просадки
    cumulative_returns = np.cumprod(1 + trades_array)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = abs(drawdown.min()) * 100

    total_return = (cumulative_returns[-1] - 1) * 100

    return {
        "Trades": len(trades_array),
        "Win_Rate": round(win_rate, 2),
        "Total_Return": round(total_return, 2),
        "Max_Drawdown": round(max_drawdown, 2)
    }

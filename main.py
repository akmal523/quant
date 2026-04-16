import pandas as pd
import yfinance as yf
import numpy as np
import os

import config
from universe import get_market_universe
from currency import apply_fx_conversion
import sentiment
from scoring import calculate_sector_medians, execute_scoring_pipeline, classify_horizon
from backtest import run_historical_backtest
import news
import indicators
import reporting
import portfolio 

def load_local_portfolio(filepath: str = "portfolio.csv") -> dict:
    portfolio_dict = {}
    if not os.path.exists(filepath): return portfolio_dict
    try:
        df = pd.read_csv(filepath).dropna(how='all')
        for _, row in df.iterrows():
            portfolio_dict[str(row['Symbol']).strip()] = {
                "buy_price": float(row['Buy_Price']),
                "amount": float(row['Amount_EUR'])
            }
    except: pass
    return portfolio_dict

MY_PORTFOLIO = load_local_portfolio()

def generate_reasoning(row: pd.Series) -> str:
    reasons = []
    if row.get('Signal') == 'SELL':
        if row.get('Total_Score', 0) < 40: reasons.append("СЛАБАЯ СТРУКТУРА")
        if row.get('RSI', 50) > 70: reasons.append("ПЕРЕГРЕВ")
    
    risk_drivers = row.get('Risk_Drivers', "")
    if risk_drivers: reasons.append(f"УГРОЗА: {risk_drivers}")
    
    if row.get('Fundamental_Score', 0) >= 15.0: reasons.append("Дисконт к сектору")
    if row.get('Volatility_Penalty', 0) > 0: reasons.append(f"Риск-штраф: -{int(row['Volatility_Penalty'])}")
    
    return " | ".join(reasons) if reasons else "Стабильный фон"

def print_terminal_report(df: pd.DataFrame):
    print("\n" + "="*150)
    print(" СЕКТОРНЫЙ СКАНЕР: ЛИДЕРЫ РЫНКА И ИНВЕСТИЦИОННЫЕ ГОРИЗОНТЫ")
    print("="*150)

    # 1. Глобальный Топ-3 рынка
    print("\n[ АБСОЛЮТНЫЕ ЛИДЕРЫ РЫНКА (ГЛОБАЛЬНЫЙ ТОП-3) ]")
    global_top = df.sort_values('Total_Score', ascending=False).head(3)
    for _, row in global_top.iterrows():
        print(f" {row['Symbol']:<8} | Score: {row['Total_Score']:<5} | Горизонт: {row['Horizon']:<20} | {row['Reasoning']}")

    # 2. Посекторный анализ
    print("\n[ ТОП-3 ПО СТРАТЕГИЧЕСКИМ СЕКТОРАМ ]")
    # Группируем по сектору и берем 3 лучших
    for sector, group in df.groupby('Sector'):
        if sector == 'Unknown': continue
        print(f"\n--- Сектор: {sector.upper()} ---")
        sector_top = group.sort_values('Total_Score', ascending=False).head(3)
        for _, row in sector_top.iterrows():
            print(f"  {row['Symbol']:<8} | Score: {row['Total_Score']:<5} | {row['Horizon']:<22} | {row['Reasoning']}")

def main():
    universe = get_market_universe()
    print(f"Запуск сканирования {len(universe)} активов...")
    
    results = []
    processed_count = 0
    
    for name, symbol in universe.items():
        processed_count += 1
        print(f"[{processed_count}/{len(universe)}] Анализ: {symbol:<8}", end="\r")
        
        try:
            # Ограничиваем время ожидания ответа от Yahoo
            t = yf.Ticker(symbol)
            inf = t.info
            
            # Если info пустое (признак блокировки или таймаута)
            if not inf or 'sector' not in inf:
                continue

            h = t.history(period="5y")
            if h.empty: continue
            
            # Математический конвейер
            h = apply_fx_conversion(h, inf.get("currency", "USD"), "EUR")
            h = indicators.add_all_indicators(h)
            curr = h.iloc[-1]
            
            n_items = news.get_recent_headlines(symbol)
            sent = sentiment.analyze_news_context(n_items)
            
            vol = h['Close'].pct_change().std() * np.sqrt(252)
            div = inf.get('dividendYield', 0.0)
            roe = inf.get('returnOnEquity', np.nan)
            pe = inf.get('trailingPE', np.nan)
            
            data = {
                'Symbol': symbol, 'Name': name, 'Sector': inf.get('sector', 'Unknown'),
                'PE': pe, 'PEG': inf.get('pegRatio', np.nan),
                'ROE': roe, 'Dividend_Yield': div, 'RSI': curr['RSI'],
                'Volatility': vol, 'FinbertSignal': sent['score'],
                'Risk_Drivers': " // ".join(sent['drivers']),
                'Horizon': classify_horizon(div, vol, roe, pe),
                'Close': curr['Close'], 'ATR': curr['ATR'],
                'Price_vs_SMA50': curr['Close'] / curr['SMA50'] if curr['SMA50'] else 1.0
            }
            data.update(run_historical_backtest(h))
            results.append(data)
            
        except Exception as e:
            # Логируем ошибку, но идем дальше
            continue

    print("\nСканирование завершено. Формирование отчета...")
    
    if not results:
        print("!!! КРИТИЧЕСКАЯ ОШИБКА: Ни один актив не был обработан. Проверьте VPN/интернет.")
        return

    df = pd.DataFrame(results)
    
    # Расчет медиан и скоринг
    s_meds = calculate_sector_medians(df)
    for idx, row in df.iterrows():
        # Передаем пустую серию, если сектор неизвестен
        med_val = s_meds.loc[row['Sector']] if row['Sector'] in s_meds.index else pd.Series(dtype=float)
        res = execute_scoring_pipeline(row.to_dict(), med_val)
        for k, v in res.items(): 
            df.at[idx, k] = v

    df['Reasoning'] = df.apply(generate_reasoning, axis=1)
    df = df.sort_values(['Sector', 'Total_Score'], ascending=[True, False])
    
    print_terminal_report(df)
    
    # Создаем папку если нет
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/market_scan.csv", index=False)

if __name__ == "__main__":
    main()

import yfinance as yf
import pandas as pd

def get_fx_ticker(asset_currency: str, base_currency: str) -> str:
    """Определяет корректный тикер для Yahoo Finance."""
    if asset_currency == "GBp":
        return f"GBP{base_currency}=X"
    return f"{asset_currency}{base_currency}=X"

def apply_fx_conversion(asset_df: pd.DataFrame, asset_currency: str, base_currency: str = "EUR") -> pd.DataFrame:
    """
    Приводит исторические цены актива к базовой валюте с учетом исторического кросс-курса.
    """
    if asset_currency == base_currency:
        return asset_df

    fx_ticker = get_fx_ticker(asset_currency, base_currency)
    
    start_date = asset_df.index.min()
    end_date = asset_df.index.max() + pd.Timedelta(days=1)
    
    # Загрузка вектора курса валют, синхронизированного по времени с активом
    fx_data = yf.download(fx_ticker, start=start_date, end=end_date, progress=False)
    
    if fx_data.empty:
        raise ValueError(f"Отсутствуют исторические данные для кросс-курса {fx_ticker}")

    # Изоляция цены закрытия кросс-курса (одномерный вектор)
    fx_close = fx_data['Close'].squeeze()

    # Выравнивание индексов по датам и заполнение пустот (forward fill)
    # Предотвращает NaN в дни несовпадения расписания торгов фондовых и валютных рынков
    aligned_df = asset_df.join(fx_close.rename('FX_Rate'), how='left')
    aligned_df['FX_Rate'] = aligned_df['FX_Rate'].ffill().bfill()

    # Масштабирование британских пенсов
    multiplier = 0.01 if asset_currency == "GBp" else 1.0

    # Точечное перемножение ценовых метрик на вектор исторического курса
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in aligned_df.columns:
            aligned_df[col] = aligned_df[col] * aligned_df['FX_Rate'] * multiplier

    aligned_df.drop(columns=['FX_Rate'], inplace=True)

    return aligned_df

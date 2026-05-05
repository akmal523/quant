import time
import pandas as pd
import yfinance as yf
from universe import SECTOR_UNIVERSE
from database import get_connection

OUTPUT_FILE = "market_data.parquet"
REQUEST_DELAY = 0.5

def main() -> None:
    tickers = [
        (sym, name, sector)
        for sector, instruments in SECTOR_UNIVERSE.items()
        for name, sym in instruments.items()
    ]

    total = len(tickers)
    all_data = []

    for i, (sym, name, sector) in enumerate(tickers):
        print(f"[{i+1}/{total}] Fetching {sym}...")
        
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period="5y", auto_adjust=True)
            
            if not df.empty:
                df['Symbol'] = sym
                df['Sector'] = sector
                
                df = df.reset_index()
                if 'Date' in df.columns:
                    # Устранение конфликтов таймзон при сериализации в Parquet
                    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                    df = df.set_index('Date')
                
                # Исключение неценовых артефактов (Dividends, Stock Splits)
                cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Sector']
                df = df[[c for c in cols_to_keep if c in df.columns]]
                
                all_data.append(df)
            else:
                print(f" [!] Empty history for {sym}")
                
        except Exception as e:
            print(f" [!] Error {sym}: {e}")
            
        time.sleep(REQUEST_DELAY)
        # ... (inside main, after the for loop ends) ...
    
    # 1. Ensure the block is OUTSIDE the for loop
    if all_data:
        final_df = pd.concat(all_data)
        
        # Step A: Move Date from Index to a Column
        final_df = final_df.reset_index()
        
        # Step B: Ensure Date is a string or standard timestamp (DuckDB prefers this)
        if 'Date' in final_df.columns:
            final_df['Date'] = pd.to_datetime(final_df['Date']).dt.strftime('%Y-%m-%d')
        
        conn = get_connection()
        
        # Step C: Load the prepared DataFrame
        conn.execute("CREATE OR REPLACE TABLE market_history AS SELECT * FROM final_df")
        
        print(f"\nWrite complete: {len(final_df)} rows saved to DuckDB.")
    else:
        print("\nFatal: No data acquired.")

if __name__ == "__main__":
    main()

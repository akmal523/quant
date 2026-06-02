import time
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from universe import SECTOR_UNIVERSE
from database import get_connection

OUTPUT_FILE = "market_data.parquet"
REQUEST_DELAY = 0.5
MAX_WORKERS = 10

def fetch_single(sym: str, name: str, sector: str) -> pd.DataFrame | None:
    """Fetch a single ticker's history. Designed for ThreadPoolExecutor."""
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period="5y", auto_adjust=True)

        if not df.empty:
            latest_dt = df.index[-1].strftime('%Y-%m-%d')
            latest_px = df['Close'].iloc[-1]
            print(f" [OK] {sym}: Latest {latest_dt} | Price: {latest_px:.2f}")
            df['Symbol'] = sym
            df['Sector'] = sector

            df = df.reset_index()
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                df = df.set_index('Date')

            cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Sector']
            df = df[[c for c in cols_to_keep if c in df.columns]]
            return df
        else:
            print(f" [!] Empty history for {sym}")
            return None
    except Exception as e:
        print(f" [!] Error {sym}: {e}")
        return None


def main() -> None:
    tickers = [
        (sym, name, sector)
        for sector, instruments in SECTOR_UNIVERSE.items()
        for name, sym in instruments.items()
    ]

    total = len(tickers)
    all_data = []

    print(f"Fetching {total} tickers with {MAX_WORKERS} parallel workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_single, sym, name, sector): sym
            for sym, name, sector in tickers
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                all_data.append(result)
            print(f"  [{i}/{total}] Completed: {futures[future]}")

    if all_data:
        final_df = pd.concat(all_data)
        final_df = final_df.reset_index()

        if 'Date' in final_df.columns:
            final_df['Date'] = pd.to_datetime(final_df['Date']).dt.strftime('%Y-%m-%d')

        conn = get_connection()
        conn.execute("CREATE OR REPLACE TABLE market_history AS SELECT * FROM final_df")

        print(f"\nWrite complete: {len(final_df)} rows saved to DuckDB ({len(all_data)}/{total} tickers fetched).")
    else:
        print("\nFatal: No data acquired.")

if __name__ == "__main__":
    main()

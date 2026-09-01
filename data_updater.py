import time
import datetime as dt
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from universe import SECTOR_UNIVERSE
from database import get_connection, init_db
from taxonomy import resolve_broker, get_instrument_class

OUTPUT_FILE = "market_data.parquet"
REQUEST_DELAY = 0.5
MAX_WORKERS = 10
# Incremental update: only fetch history after this many days back from last_date.
# yfinance needs a small overlap to avoid gaps on non-trading days.
INCREMENTAL_OVERLAP_DAYS = 5


def get_last_dates(conn) -> dict[str, str]:
    """Return {symbol: last_date_str} from market_history. Empty dict if table missing."""
    try:
        rows = conn.execute(
            "SELECT Symbol, MAX(Date) AS last_date FROM market_history GROUP BY Symbol"
        ).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def fetch_single(sym: str, name: str, sector: str, last_date: str | None = None) -> pd.DataFrame | None:
    """Fetch a single ticker's history. Incremental if last_date provided.

    Intent: avoid re-downloading 5y every run. Fetch only data after last_date
    (minus overlap) and append. Drops trailing NaN rows (e.g. future dates).
    Invariants: returns df with Symbol/Sector columns, Date index, no NaN Close.
    Dependencies: yfinance, pandas, taxonomy (broker registry).
    """
    try:
        # Phase 4 (1.2): resolve TR ticker for execution-relevant local prices.
        # For equities/ETFs with a broker mapping, fetch the LS Exchange ticker
        # (e.g. AAPL.DE) so moving averages reflect local trading hours/spreads.
        broker = resolve_broker(sym)
        fetch_ticker = broker["tr_ticker"] if broker["tr_ticker"] else sym
        ticker = yf.Ticker(fetch_ticker)

        if last_date:
            # Incremental: fetch from overlap window before last known date.
            start = (dt.date.fromisoformat(last_date) - dt.timedelta(days=INCREMENTAL_OVERLAP_DAYS))
            df = ticker.history(start=start.isoformat(), auto_adjust=True)
        else:
            # Full fetch: 5 years.
            df = ticker.history(period="5y", auto_adjust=True)

        if df.empty:
            print(f" [!] Empty history for {sym} (via {fetch_ticker})")
            return None

        # Drop rows with NaN Close (future dates, non-trading days, etc.)
        valid = df.dropna(subset=['Close'])
        if valid.empty:
            print(f" [!] No valid Close data for {sym}")
            return None

        latest_px = valid['Close'].iloc[-1]
        latest_dt = valid.index[-1].strftime('%Y-%m-%d')
        print(f" [OK] {sym} (via {fetch_ticker}): Latest {latest_dt} | Price: {latest_px:.2f}")

        df['Symbol'] = sym
        df['Sector'] = sector
        # Phase 4 (3.1): tag instrument_class for bifurcated scoring.
        df['Instrument_Class'] = get_instrument_class(sym, name)

        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = df.set_index('Date')

        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Sector', 'Instrument_Class']
        df = df[[c for c in cols_to_keep if c in df.columns]]
        df = df.dropna(subset=['Close'])
        return df
    except Exception as e:
        print(f" [!] Error {sym}: {e}")
        return None


def build_fetch_list() -> list[tuple[str, str, str]]:
    """Build the ticker fetch list: CORE ETFs + ACTIVE universe + portfolio.

    Intent (Phase 4, 3.2): stop hardcoding all 277 SECTOR_UNIVERSE stocks.
    Fetch only:
      1. CORE — broad ETFs & macro indices (always tracked).
      2. ACTIVE — graduated equities from asset_registry (heavy analysis).
      3. Portfolio — current holdings (must always have fresh prices).
    Phase 5 (v10.2): DELISTED symbols are excluded (stop retrying forever).
    Invariants: returns list of (symbol, name, sector); deduplicated by symbol.
    """
    from database import init_db
    init_db()

    tickers: dict[str, tuple[str, str]] = {}

    # 1. CORE: Broad ETFs sector (always tracked).
    for name, sym in SECTOR_UNIVERSE.get("Broad ETFs", {}).items():
        tickers[sym] = (name, "Broad ETFs")

    # 2. ACTIVE universe from asset_registry (exclude DELISTED).
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT symbol, sector FROM asset_registry "
            "WHERE universe_status = 'ACTIVE' AND universe_status != 'DELISTED'"
        ).fetchall()
        for sym, sector in rows:
            if sym not in tickers:
                tickers[sym] = (sym, sector or "Unknown")
    except Exception:
        pass

    # 3. Portfolio holdings (always fresh prices for PnL).
    try:
        from portfolio import load_portfolio
        port = load_portfolio("portfolio.csv")
        for sym in port["Symbol"].unique():
            if sym not in tickers:
                tickers[sym] = (sym, "Portfolio")
    except Exception:
        pass

    return [(sym, name, sector) for sym, (name, sector) in tickers.items()]


def main() -> None:
    tickers = build_fetch_list()

    total = len(tickers)
    all_data = []

    conn = get_connection()
    init_db()
    last_dates = get_last_dates(conn)
    incremental = bool(last_dates)
    print(f"Fetching {total} tickers (CORE + ACTIVE + Portfolio) with "
          f"{MAX_WORKERS} parallel workers... "
          f"({'INCREMENTAL' if incremental else 'FULL 5y'} mode)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_single, sym, name, sector, last_dates.get(sym)): sym
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

        # Ensure only known columns (matches market_history PK schema).
        cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Symbol', 'Sector', 'Instrument_Class']
        final_df = final_df[[c for c in cols_to_keep if c in final_df.columns]]

        if incremental:
            # INSERT OR REPLACE dedups on PRIMARY KEY (Symbol, Date).
            conn.execute("INSERT OR REPLACE INTO market_history SELECT * FROM final_df")
            print(f"\nIncremental update: {len(final_df)} rows appended/updated "
                  f"({len(all_data)}/{total} tickers fetched).")
        else:
            conn.execute("DELETE FROM market_history")
            conn.execute("INSERT INTO market_history SELECT * FROM final_df")
            print(f"\nWrite complete: {len(final_df)} rows saved to DuckDB "
                  f"({len(all_data)}/{total} tickers fetched).")
    else:
        print("\nFatal: No data acquired.")

if __name__ == "__main__":
    main()

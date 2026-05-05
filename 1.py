# 1.py
import duckdb
import pandas as pd

conn = duckdb.connect("quant_cache.duckdb")

print("--- 1. Checking Table Schema ---")
columns = conn.execute("DESCRIBE market_history").df()
print(columns[['column_name', 'column_type']])

print("\n--- 2. Checking ROG.SW Data ---")
# Check if ROG.SW has rows
count = conn.execute("SELECT COUNT(*) FROM market_history WHERE Symbol = 'ROG.SW'").fetchone()[0]
print(f"ROG.SW row count: {count}")

if count > 0:
    # Check for NaNs in price (Common for Swiss stocks on Yahoo)
    nans = conn.execute("SELECT COUNT(*) FROM market_history WHERE Symbol = 'ROG.SW' AND Close IS NULL").fetchone()[0]
    print(f"ROG.SW rows with NULL prices: {nans}")
    
    # Check Fundamentals
    print("\n--- 3. Checking Fundamentals for ROG.SW ---")
    fund = conn.execute("SELECT * FROM fundamentals WHERE symbol = 'ROG.SW'").df()
    if fund.empty:
        print("[!] FUNDAMENTALS MISSING FOR ROG.SW")
    else:
        print(fund)

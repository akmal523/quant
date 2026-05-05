# test_market_data.py
import pandas as pd
from database import get_connection

def test_market_db():
    conn = get_connection()
    
    # Mock data
    df = pd.DataFrame({"Symbol": ["AAPL", "MSFT"], "Close": [150.0, 300.0]})
    
    # Write
    conn.execute("CREATE OR REPLACE TABLE market_history AS SELECT * FROM df")
    
    # Read
    res_df = conn.execute("SELECT * FROM market_history WHERE Symbol='AAPL'").df()
    
    assert res_df.iloc[0]["Close"] == 150.0, "Market data I/O failed."
    print("Task 3 validation passed. Pandas <-> DuckDB bridge operational.")

if __name__ == "__main__":
    test_market_db()

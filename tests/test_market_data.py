"""test_market_data.py — the pandas <-> DuckDB bridge."""
import pandas as pd

from quant.data.database import get_connection


def test_market_db():
    conn = get_connection()

    # Mock data (used by the DuckDB replacement scan via `SELECT * FROM df`).
    df = pd.DataFrame({"Symbol": ["AAPL", "MSFT"], "Close": [150.0, 300.0]})  # noqa: F841

    # Use a DEDICATED table: replacing the shared market_history would destroy
    # its schema for every later test in the same worker (v10.7.5 fix).
    conn.execute("CREATE OR REPLACE TABLE market_history_bridge_test AS SELECT * FROM df")
    try:
        res_df = conn.execute(
            "SELECT * FROM market_history_bridge_test WHERE Symbol='AAPL'").df()
        assert res_df.iloc[0]["Close"] == 150.0, "Market data I/O failed."
    finally:
        conn.execute("DROP TABLE IF EXISTS market_history_bridge_test")
    print("Task 3 validation passed. Pandas <-> DuckDB bridge operational.")

if __name__ == "__main__":
    test_market_db()

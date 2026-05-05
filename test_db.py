# test_db.py
import os
import duckdb
from db import init_db, get_connection

def test_db_init():
    # Setup
    if os.path.exists("quant_cache.duckdb"):
        os.remove("quant_cache.duckdb")

    # Act
    init_db()
    conn = get_connection()
    
    # Test Fundamentals Schema
    conn.execute("INSERT INTO fundamentals (symbol, pe) VALUES ('AAPL', 25.5)")
    res_fund = conn.execute("SELECT pe FROM fundamentals WHERE symbol = 'AAPL'").fetchone()
    assert res_fund[0] == 25.5, "Fundamentals I/O failed."

    # Test NLP Schema
    conn.execute("INSERT INTO nlp_scores (doc_hash, score) VALUES ('hash123', 85.0)")
    res_nlp = conn.execute("SELECT score FROM nlp_scores WHERE doc_hash = 'hash123'").fetchone()
    assert res_nlp[0] == 85.0, "NLP I/O failed."

    print("db.py validation passed.")

if __name__ == "__main__":
    test_db_init()

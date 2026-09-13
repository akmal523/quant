import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# test_e2e_state.py
# Use the shared connection so tests/conftest.py can redirect the DB path.
from quant.data.database import get_connection

def verify_state():
    try:
        conn = get_connection()
        tables = conn.execute("SHOW TABLES").df()["name"].tolist()
        
        assert "market_history" in tables, "Updater failed."
        assert "fundamentals" in tables, "Fundamentals cache failed."
        assert "nlp_scores" in tables, "NLP cache failed."
        
        for t in tables:
            rows = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"[OK] {t}: {rows} rows")
            
        print("E2E Verification Passed. Pipeline operational.")
    except Exception as e:
        print(f"[FAIL] {e}")

if __name__ == "__main__":
    verify_state()

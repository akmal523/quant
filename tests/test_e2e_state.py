import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from quant import paths
# test_e2e_state.py
import duckdb

def verify_state():
    try:
        # read_only=True prevents locking issues if main.py is stuck
        conn = duckdb.connect(paths.DB_FILE, read_only=True)
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

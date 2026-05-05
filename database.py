# db.py
import duckdb
import threading

DB_PATH = "quant_cache.duckdb"
_local = threading.local()

def get_connection() -> duckdb.DuckDBPyConnection:
    """Provides thread-local DuckDB connection."""
    if not hasattr(_local, "conn"):
        # DuckDB allows concurrent reads, strictly one write process.
        # timeout=15 for WAL-style waiting.
        _local.conn = duckdb.connect(DB_PATH, config={'access_mode': 'READ_WRITE'})
    return _local.conn

def init_db() -> None:
    """Initializes unified OLAP schemas."""
    conn = get_connection()
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals (
            symbol VARCHAR PRIMARY KEY,
            pe DOUBLE,
            peg DOUBLE,
            roe DOUBLE,
            debt_to_equity DOUBLE,
            ebit DOUBLE,
            interest_expense DOUBLE,
            updated_at DOUBLE
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nlp_scores (
            doc_hash VARCHAR PRIMARY KEY,
            score DOUBLE
        )
    """)

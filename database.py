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

    # market_history: PRIMARY KEY (Symbol, Date) enables INSERT OR REPLACE
    # dedup on incremental updates. Without a PK, DuckDB raises BinderException
    # on INSERT OR REPLACE (no UNIQUE constraint to conflict on).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS market_history (
            Date VARCHAR,
            Open DOUBLE,
            High DOUBLE,
            Low DOUBLE,
            Close DOUBLE,
            Volume DOUBLE,
            Symbol VARCHAR,
            Sector VARCHAR,
            Instrument_Class VARCHAR,
            PRIMARY KEY (Symbol, Date)
        )
    """)

    # Migration: legacy market_history was created via CREATE TABLE AS SELECT
    # with NO primary key and NO Instrument_Class column. Rebuild it with the
    # PK schema, preserving existing rows, so INSERT OR REPLACE works.
    try:
        pk_cols = conn.execute(
            "SELECT kcu.column_name FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            "WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = 'market_history'"
        ).fetchall()
        has_pk = len(pk_cols) > 0
        has_ic = conn.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = 'market_history' AND column_name = 'Instrument_Class'"
        ).fetchone()[0] > 0
        if not has_pk or not has_ic:
            conn.execute("CREATE TABLE market_history_new AS SELECT * FROM market_history")
            conn.execute("DROP TABLE market_history")
            conn.execute("""
                CREATE TABLE market_history (
                    Date VARCHAR,
                    Open DOUBLE,
                    High DOUBLE,
                    Low DOUBLE,
                    Close DOUBLE,
                    Volume DOUBLE,
                    Symbol VARCHAR,
                    Sector VARCHAR,
                    Instrument_Class VARCHAR,
                    PRIMARY KEY (Symbol, Date)
                )
            """)
            conn.execute("""
                INSERT INTO market_history (Date, Open, High, Low, Close, Volume, Symbol, Sector)
                SELECT Date, Open, High, Low, Close, Volume, Symbol, Sector FROM market_history_new
            """)
            conn.execute("DROP TABLE market_history_new")
    except Exception:
        # Table may not exist yet on first run; non-fatal.
        pass

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

    # Point-in-time fundamentals history (Pillar 1.3 — no lookahead bias).
    # as_of_date: the date the fundamentals are valid for.
    # published_date: when the market actually knew them.
    # Backtests must filter published_date <= scoring_date to avoid lookahead.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_history (
            symbol VARCHAR,
            as_of_date DATE,
            published_date DATE,
            pe DOUBLE,
            peg DOUBLE,
            roe DOUBLE,
            debt_to_equity DOUBLE,
            ebit DOUBLE,
            interest_expense DOUBLE,
            PRIMARY KEY (symbol, as_of_date)
        )
    """)

    # ── Phase 4: Asset Taxonomy & Universe Management ─────────────────────────
    # instrument_class: EQUITY | ETF | COMMODITY | CASH. Drives bifurcated
    # scoring pipelines (ETFs bypass Fundamentals/NLP; Commodities use macro).
    # isin: Trade Republic routing key (LS Exchange / Tradegate).
    # universe_status: ACTIVE (heavy analysis) | WATCHLIST (light scan) | CORE.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS asset_registry (
            symbol VARCHAR PRIMARY KEY,
            name VARCHAR,
            instrument_class VARCHAR NOT NULL DEFAULT 'EQUITY',
            isin VARCHAR,
            tr_ticker VARCHAR,
            exchange VARCHAR,
            currency VARCHAR,
            universe_status VARCHAR NOT NULL DEFAULT 'ACTIVE',
            sector VARCHAR,
            structure VARCHAR NOT NULL DEFAULT 'PLAIN',
            graduated_at DATE,
            last_signal_date DATE,
            fetch_failures INTEGER NOT NULL DEFAULT 0,
            updated_at DOUBLE
        )
    """)

    # Migration (v10.2): add structure / fetch_failures columns to a legacy
    # asset_registry created before the Phase 5 schema. CREATE TABLE IF NOT
    # EXISTS does not add columns to an existing table, so add them explicitly.
    # DuckDB cannot add columns WITH constraints, so add bare columns and
    # backfill the defaults.
    try:
        cols = {r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'asset_registry'"
        ).fetchall()}
        if "structure" not in cols:
            conn.execute("ALTER TABLE asset_registry ADD COLUMN structure VARCHAR")
            conn.execute("UPDATE asset_registry SET structure = 'PLAIN'")
        if "fetch_failures" not in cols:
            conn.execute("ALTER TABLE asset_registry ADD COLUMN fetch_failures INTEGER")
            conn.execute("UPDATE asset_registry SET fetch_failures = 0")
    except Exception:
        # Table may not exist yet on first run; non-fatal.
        pass

    # ── Phase 5 (v10.2): Universe audit trail ────────────────────────────────
    # Every state change (GRADUATE / DEMOTE / DELIST / PIN / ADD) is logged here
    # so the dashboard can render an auditable event log.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS universe_events (
            ts TIMESTAMP DEFAULT now(),
            symbol VARCHAR,
            event VARCHAR,
            reason VARCHAR
        )
    """)

    # ── Phase 5 (v10.2): Account state ───────────────────────────────────────
    # Persists the user's cash input from the Daily Briefing bucket check.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS account_state (
            key VARCHAR PRIMARY KEY,
            value DOUBLE
        )
    """)

    # ── Phase 5 (v11): Rebalance log ─────────────────────────────────────────
    # Tracks the last rebalance date per symbol for time-gated rebalancing.
    # Intent: CORE assets rebalance quarterly, SATELLITE monthly, SECTOR
    # bi-weekly. Absence of a row = first run -> baseline ease-in (no forced
    # rebalance). portfolio.py reads/writes this table.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rebalance_log (
            symbol VARCHAR PRIMARY KEY,
            last_rebalance_date DATE
        )
    """)

from quant import paths
# db.py
import time
from contextlib import contextmanager

import duckdb
import threading

DB_PATH = paths.DB_FILE
# Max seconds to wait for the DuckDB file lock before giving up. DuckDB allows
# concurrent reads but strictly one writer; if another process (dashboard, a
# previous crashed run) holds the lock, an unguarded connect() can block forever
# and freeze the pipeline.
CONNECT_TIMEOUT = 15.0
_local = threading.local()


def _connect_with_timeout(timeout: float = CONNECT_TIMEOUT) -> duckdb.DuckDBPyConnection:
    """Open the DB read-write, but never block longer than ``timeout`` seconds.

    Intent: Python's duckdb.connect() exposes no lock timeout. Run it in a
    daemon thread and join with a bounded timeout so a locked database raises a
    clear error instead of silently freezing the run.
    """
    result: list = [None]
    error: list = [None]

    def _worker() -> None:
        try:
            result[0] = duckdb.connect(DB_PATH, config={'access_mode': 'READ_WRITE'})
        except Exception as e:  # noqa: BLE001
            error[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        raise TimeoutError(
            f"DuckDB connect timed out after {timeout:.0f}s "
            f"(database locked by another process?). Path: {DB_PATH}"
        )
    if error[0] is not None:
        raise error[0]
    return result[0]


def connect_with_retry(
    attempts: int = 6,
    delay: float = 0.5,
    verbose: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open a read-write connection, retrying on lock contention (spec 4.2).

    Intent: the UI holds short-lived read-only connections; a refresh subprocess
    may briefly collide with one. Retry up to ``attempts`` times with exponential
    backoff (default ~15s total) before giving up. Under ``verbose`` print the
    plain "waiting for database lock" line.
    Invariants: raises the last error if every attempt fails.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return _connect_with_timeout(timeout=2.0)
        except Exception as e:  # noqa: BLE001
            last = e
            if i < attempts - 1:
                if verbose:
                    print("waiting for database lock")
                time.sleep(delay * (2 ** i))
    assert last is not None
    raise last


@contextmanager
def read_only_connection():
    """Short-lived read-only DuckDB connection. Always closed on exit.

    Intent (v10.5.1, spec 4.1): UI queries must NOT hold a persistent
    read-write connection, or the refresh subprocess cannot acquire the file
    lock. Every UI read opens ``read_only=True`` and closes immediately.
    Invariants: the connection is closed even if the body raises.

    Fallback: DuckDB forbids mixing read-only and read-write connections to the
    same file within one process. If a writer already holds the file in this
    process (tests, or an in-process pipeline call), open a short-lived
    read-write connection instead. It is still opened per call and closed on
    exit, so no persistent connection is held.
    """
    try:
        conn = duckdb.connect(DB_PATH, read_only=True)
    except Exception:  # noqa: BLE001
        # Same config as the in-process writer, else DuckDB rejects the connect.
        conn = duckdb.connect(DB_PATH, config={"access_mode": "READ_WRITE"})
    try:
        yield conn
    finally:
        conn.close()


def get_connection() -> duckdb.DuckDBPyConnection:
    """Provides thread-local DuckDB connection (pipeline writers only).

    NOTE: UI code must use ``read_only_connection()`` instead, so it never holds
    the write lock (spec 4.1).
    """
    if not hasattr(_local, "conn") or _local.conn is None:
        # DuckDB allows concurrent reads, strictly one write process.
        _local.conn = _connect_with_timeout()
    return _local.conn


def use_connection(conn: duckdb.DuckDBPyConnection) -> None:
    """Bind ``conn`` as this thread's connection (pipeline writers)."""
    _local.conn = conn

def migrate_registry_display_name(conn) -> None:
    """Idempotently add asset_registry.display_name (v10.5.3, R5).

    Intent: existing databases predate the column. ADD COLUMN IF NOT EXISTS keeps
    the migration safe on both old and fresh schemas.
    """
    conn.execute("ALTER TABLE asset_registry ADD COLUMN IF NOT EXISTS display_name VARCHAR")


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

    # v10.4.0 (Phase 1): ingested_at records WHEN a row entered the DB, distinct
    # from the trading Date. Enables bitemporal audit (data-arrival vs valid-time).
    try:
        cols = {r[0] for r in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'market_history'"
        ).fetchall()}
        if "ingested_at" not in cols:
            conn.execute("ALTER TABLE market_history ADD COLUMN ingested_at TIMESTAMP")
    except Exception:
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

    # ── v10.5.0 (spec 6): first-class NLP evidence table ─────────────────────
    # Documented columns so the Explorer can show WHY a sentiment score exists:
    # source, title, published_at, score, confidence, retrieved_at. A score that
    # used news stores the item count; a score that did not stores confidence
    # 'low'. The UI aggregates the disclaimer into one evidence footnote.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nlp_evidence (
            symbol VARCHAR,
            source VARCHAR,
            title VARCHAR,
            published_at TIMESTAMP,
            score DOUBLE,
            confidence VARCHAR,
            retrieved_at TIMESTAMP DEFAULT now()
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
        # v10.5.3 (R5): friendly display name. Idempotent migration for existing DBs.
        migrate_registry_display_name(conn)
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

    # ── v10.5.0: account_state table removed ─────────────────────────────────
    # Cash, risk profile, and base currency now live in data/account.yaml
    # (single source of truth, spec 2.2). The old DuckDB table is dropped so no
    # widget can compute its own copy of cash (R2).
    try:
        conn.execute("DROP TABLE IF EXISTS account_state")
    except Exception:
        pass

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

    # ── Plan 3 (Phase 1): Broad Universe Master ─────────────────────────────
    # Replaces the hardcoded ~300 SECTOR_UNIVERSE. Holds the full 1000+ ticker
    # pool built from index constituents (S&P 500, Nasdaq 100, Russell 1000)
    # plus broad ETFs. The funnel (funnel.py) filters this pool down to the
    # top survivors for heavy analysis. universe_status mirrors asset_registry
    # semantics (WATCHLIST default) so discovery can graduate from this pool.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS universe_master (
            symbol VARCHAR PRIMARY KEY,
            name VARCHAR,
            source VARCHAR,
            instrument_class VARCHAR NOT NULL DEFAULT 'EQUITY',
            universe_status VARCHAR NOT NULL DEFAULT 'WATCHLIST',
            updated_at DOUBLE
        )
    """)

    # ── Plan 3 (Phase 1): Funnel survivors cache ─────────────────────────────
    # data_updater.py runs the (expensive, 1000+ symbol) funnel once per cycle
    # and persists the top survivors here. main.py then reads this table instead
    # of re-running the funnel, keeping the documented 2-step flow
    # (data_updater -> main) without duplicating the network fetch.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funnel_survivors (
            symbol VARCHAR PRIMARY KEY,
            score DOUBLE,
            updated_at DOUBLE
        )
    """)

    # ── v10.4.0 (Phase 2): Trade log for Transaction Cost Analysis (TCA) ──────
    # Records the signal price/time vs the actual fill price/time so the
    # Implementation Shortfall (slippage in bps) can be measured per trade.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            ts TIMESTAMP DEFAULT now(),
            symbol VARCHAR,
            side VARCHAR,
            signal_price DOUBLE,
            signal_ts TIMESTAMP,
            fill_price DOUBLE,
            fill_ts TIMESTAMP,
            slippage_bps DOUBLE,
            fee_eur DOUBLE
        )
    """)

    # ── v10.4.0 (Phase 2): Daily theoretical portfolio snapshot ───────────────
    # The reconciliation engine diffs this against the Trade Republic CSV export
    # to detect divergence (missing dividend, unexecuted limit order).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshot (
            snapshot_date DATE,
            symbol VARCHAR,
            shares DOUBLE,
            price_eur DOUBLE,
            value_eur DOUBLE,
            PRIMARY KEY (snapshot_date, symbol)
        )
    """)

    # ── v10.5.1 (spec 5.1): portfolio value history ──────────────────────────
    # One row per review, written by the review step. Feeds the Today value
    # chart. Not keyed by date because several reviews can occur in one day.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_history (
            review_ts TIMESTAMP,
            value_eur DOUBLE,
            invested_eur DOUBLE,
            cash_eur DOUBLE,
            pnl_eur DOUBLE
        )
    """)

    # ─ H3-fix part 2: heal a stale registry to the working universe ──────────
    # Membership-only two-way sync against W (see quant.data.registry_repair).
    # No-op on a fresh (empty) registry, so a brand-new DB and the isolated test
    # DB are untouched; a stale local DB is pruned/repopulated once here.
    # Idempotent; never raises (heal must not break startup).
    try:
        from quant.data.registry_repair import sync_registry_to_working_universe

        sync_registry_to_working_universe(conn)
    except Exception:  # noqa: BLE001
        pass

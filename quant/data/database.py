import threading

# db.py
import time
from collections.abc import Iterator
from contextlib import contextmanager

import duckdb

from quant import paths

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

# ── v10.7.2 (Part 1.2): short writer transactions with a bounded retry ────────
# Every writer opens ONE connection through write_connection(), yields, and
# closes it in ``finally``. On a DuckDB lock/IO error it retries up to 5 attempts
# with backoff totaling about 60 seconds. Long computations must NOT wrap the
# whole job in one write transaction; write artifacts in small transactions at
# the end. The Streamlit app holds NO write connection beyond the documented
# startup names backfill (quant.data.names.ensure_display_names), which opens one
# short-lived write connection and skips on lock contention.

def _is_lock_error(exc: Exception) -> bool:
    """True for a DuckDB lock/IO error worth retrying."""
    if isinstance(exc, TimeoutError):
        return True
    text = str(exc).lower()
    return any(token in text for token in
               ("lock", "conflicting", "io error", "busy", "transaction"))

def _write_backoff(attempts: int, total_seconds: float) -> list[float]:
    """Evenly spaced delays so the retry budget totals about ``total_seconds``."""
    sleeps = max(0, attempts - 1)
    if sleeps == 0:
        return []
    return [total_seconds / sleeps] * sleeps

@contextmanager
def write_connection(
    attempts: int | None = None,
    total_seconds: float | None = None,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Short-lived read-write connection with a bounded retry budget.

    Intent (v10.7.2, Part 1.2): one helper for every write path. Opens the
    connection with a bounded retry on DuckDB lock/IO errors (backoff totaling
    about ``total_seconds``), yields it once, and closes it in ``finally``.
    v10.8.10: a generator context manager can yield only once, so the body is
    never retried (the old body retry turned a lock error into "generator didn't
    stop after throw()"); an error in the body propagates as itself.
    """
    from quant.config import WRITE_RETRY_ATTEMPTS, WRITE_RETRY_TOTAL_SECONDS

    attempts = WRITE_RETRY_ATTEMPTS if attempts is None else attempts
    total_seconds = WRITE_RETRY_TOTAL_SECONDS if total_seconds is None else total_seconds
    delays = _write_backoff(attempts, total_seconds)
    conn = None
    for i in range(max(1, attempts)):
        try:
            conn = _connect_with_timeout(timeout=2.0)
            break
        except Exception as e:  # noqa: BLE001
            if _is_lock_error(e) and i < len(delays):
                time.sleep(delays[i])
                continue
            raise
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

@contextmanager
def read_only_connection() -> Iterator[duckdb.DuckDBPyConnection]:
    """Short-lived read-only DuckDB connection. Always closed on exit.

    Intent (v10.5.1, spec 4.1): UI queries must NOT hold a persistent
    read-write connection, or the refresh subprocess cannot acquire the file
    lock. Every UI read opens ``read_only=True`` and closes immediately.
    Invariants: the connection is closed even if the body raises.

    v10.8.0: a foreign writer (a running ``quant update``/``daily``, or a stale
    process) holds the file lock, so the read-only connect fails with a
    "Conflicting lock" error. Retry the read-only connect with a short backoff
    instead of escalating to a write connection (which would fight the writer).

    Fallback: DuckDB forbids mixing read-only and read-write connections to the
    same file within one process. If a writer already holds the file in this
    process (tests, or an in-process pipeline call), the connect fails with a
    non-lock error; open a short-lived read-write connection instead. It is
    still opened per call and closed on exit, so no persistent connection is
    held.
    """
    # v10.8.2: a longer grace so a brief writer collision (a refresh/daily job, or
    # a stale process) does not surface as a failed read. Total ~6.7s.
    delays = [0.2, 0.5, 1.0, 2.0, 3.0]
    last: Exception | None = None
    for i in range(len(delays) + 1):
        try:
            conn = duckdb.connect(DB_PATH, read_only=True)
        except Exception as e:  # noqa: BLE001
            last = e
            if _is_lock_error(e):
                if i < len(delays):
                    time.sleep(delays[i])
                    continue
                raise
            # Same-process writer: DuckDB rejects the read-only connect with a
            # non-lock error. Use the in-process writer's mode instead.
            conn = duckdb.connect(DB_PATH, config={"access_mode": "READ_WRITE"})
        try:
            yield conn
            return
        finally:
            conn.close()
    if last is not None:
        raise last

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

def migrate_registry_display_name(conn: duckdb.DuckDBPyConnection) -> None:
    """Idempotently add asset_registry.display_name (v10.5.3, R5).

    Intent: existing databases predate the column. ADD COLUMN IF NOT EXISTS keeps
    the migration safe on both old and fresh schemas.
    """
    conn.execute("ALTER TABLE asset_registry ADD COLUMN IF NOT EXISTS display_name VARCHAR")
    # v10.8.10: 'yahoo' when ``currency`` is the one Yahoo reports for the symbol
    # (written by refresh); FX conversion trusts only that, else a suffix guess.
    conn.execute("ALTER TABLE asset_registry ADD COLUMN IF NOT EXISTS currency_source VARCHAR")

def init_db() -> None:
    """Initializes unified OLAP schemas.

    F-series: also seeds the writable data dir (dirs + input templates + bundled
    themes) so a fresh install has a valid DB location and input files before the
    first read. Seeding never overwrites user-owned files.
    """
    try:
        from quant.data.bootstrap import seed_user_data

        seed_user_data()
    except Exception:  # noqa: BLE001
        pass
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

    # ── v10.5.0: account_state table removed ─────────────────────────────────
    # Cash, risk profile, and base currency now live in data/account.yaml
    # (single source of truth, spec 2.2). The old DuckDB table is dropped so no
    # widget can compute its own copy of cash (R2).
    try:
        conn.execute("DROP TABLE IF EXISTS account_state")
    except Exception:
        pass

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

    # ── v10.8.7 (T2): the scoring store ───────────────────────────────────────
    # One row per symbol: the numeric composite + confidence tier + the component
    # grades that produced it. Written by the scoring stage (daily + on-demand);
    # read by Research, the decision taxonomy and the invest plan. A missing
    # component is a documented neutral penalty, never a dash (R-C2).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            symbol VARCHAR PRIMARY KEY,
            composite DOUBLE,
            confidence VARCHAR,
            structural_grade DOUBLE,
            tactical_grade DOUBLE,
            sentiment DOUBLE,
            stewardship DOUBLE,
            present_components INTEGER,
            total_components INTEGER,
            updated_at DOUBLE
        )
    """)

    # ── v10.7.6 (Part 2): Tax ledger (Sparerpauschbetrag tracking) ────────────
    # One row per buy, sell, or dividend, with the realized PnL for sells. The
    # tax summary and the valuation backfill read this table.
    conn.execute("CREATE SEQUENCE IF NOT EXISTS trades_id_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id BIGINT DEFAULT nextval('trades_id_seq'),
            date DATE,
            symbol VARCHAR,
            action VARCHAR,
            shares DOUBLE,
            price_eur DOUBLE,
            amount_eur DOUBLE,
            fee_eur DOUBLE DEFAULT 0,
            realized_pnl_eur DOUBLE,
            created_at TIMESTAMP DEFAULT now()
        )
    """)

    # ── v10.7.0: Holdings metadata (shares from the broker CSV sync) ─────────
    # On every CSV sync, shares = Current_Value_EUR / price_eur at sync time.
    # Between syncs, the daily revalue uses shares * latest close and labels the
    # value "estimated". Broker PnL stays the synced truth until the next sync.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS holdings_meta (
            symbol VARCHAR PRIMARY KEY,
            shares DOUBLE,
            sync_date DATE,
            invested_at_sync DOUBLE
        )
    """)

    # ── v10.7.0: Flows (money entering/leaving the INVESTED pool) ────────────
    # Types: buy (money entered invested), sell (money left invested to cash or
    # out), dividend (left invested, landed in cash). Sparplan auto-flows are
    # written when a monthly plan is approved and replaced by actuals later.
    conn.execute("CREATE SEQUENCE IF NOT EXISTS flows_id_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flows (
            id BIGINT DEFAULT nextval('flows_id_seq'),
            date DATE,
            type VARCHAR,
            amount_eur DOUBLE,
            symbol VARCHAR,
            note VARCHAR,
            created_at TIMESTAMP DEFAULT now()
        )
    """)

    # ── v10.8.0 (Phase 1): append-only position snapshots ────────────────────
    # Every broker sync stores a dated snapshot of shares per symbol, so a later
    # sync never rewrites earlier history. The daily value series is built from
    # these snapshots + recorded flows + daily closes.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS position_snapshots (
            snapshot_date DATE,
            symbol VARCHAR,
            shares DOUBLE,
            PRIMARY KEY (snapshot_date, symbol)
        )
    """)

    # ── v10.8.4 (T1): the daily mark-to-market valuation store ───────────────
    # The SINGLE source for every on-screen value: value_eur = shares * price_eur
    # * fx. Written by the daily job, confirm & save, and the on-demand refresh;
    # idempotent per (date, symbol). All charts, deltas, holdings tables and class
    # weights read THIS store (no ad-hoc snapshot joins).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS valuation (
            date DATE,
            symbol VARCHAR,
            shares DOUBLE,
            price_eur DOUBLE,
            fx DOUBLE,
            value_eur DOUBLE,
            source VARCHAR,
            PRIMARY KEY (date, symbol)
        )
    """)
    # One row per date: the portfolio total (sum of valuation.value_eur) + cash.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_total (
            date DATE PRIMARY KEY,
            value_eur DOUBLE,
            cash_eur DOUBLE
        )
    """)
    # Corporate-action split factors, so share reconstruction is honest across a
    # split (factor 2.0 for a 2:1). Written by the ingest path when detected.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS splits (
            symbol VARCHAR,
            effective_date DATE,
            factor DOUBLE,
            PRIMARY KEY (symbol, effective_date)
        )
    """)

    # ── v10.8.4 (T7): the analysis-pipeline validity contract ────────────────
    # Every run (daily or on-demand) writes one row per stage, so a UI empty
    # state reads the manifest instead of guessing why a value is missing.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_manifest (
            run_id VARCHAR,
            stage VARCHAR,
            status VARCHAR,
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            coverage VARCHAR,
            error VARCHAR,
            PRIMARY KEY (run_id, stage)
        )
    """)

    # ── v10.8.4 (T5): the stateful advice log ────────────────────────────────
    # Advice is a persisted object, not a pure re-computation: it carries
    # created_at / open_since / status so a repeated item shows how long it has
    # been open and a done item stops repeating and appears in History.
    conn.execute("CREATE SEQUENCE IF NOT EXISTS advice_log_id_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS advice_log (
            id BIGINT DEFAULT nextval('advice_log_id_seq'),
            verb VARCHAR,
            symbol VARCHAR,
            amount_eur DOUBLE,
            reason VARCHAR,
            horizon VARCHAR,
            class VARCHAR,
            created_at TIMESTAMP DEFAULT now(),
            open_since DATE,
            status VARCHAR DEFAULT 'open'
        )
    """)

    # ── v10.7.0: Alerts (level-triggered, Section 5) ─────────────────────────
    # An alert stays open until the user resolves it. Self-scoring prices a
    # resolved alert 30 days later and records a verdict (the honesty ledger).
    conn.execute("CREATE SEQUENCE IF NOT EXISTS alerts_id_seq START 1")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id BIGINT DEFAULT nextval('alerts_id_seq'),
            created_date DATE,
            symbol VARCHAR,
            kind VARCHAR,
            message VARCHAR,
            action VARCHAR,
            amount_eur DOUBLE,
            status VARCHAR DEFAULT 'new',
            resolve_date DATE,
            resolve_reason VARCHAR,
            price_at_resolve DOUBLE,
            notified BOOLEAN DEFAULT FALSE,
            price_30d DOUBLE,
            verdict VARCHAR,
            condition_cleared BOOLEAN DEFAULT FALSE
        )
    """)
    # v10.7.4 (R8): a dismissed alert must not reopen unless the underlying
    # condition re-arms. condition_cleared records that the condition read false
    # at least once after the dismissal. Added to legacy DBs via a guarded ALTER
    # (a failed ALTER aborts the DuckDB transaction, so check the column first).
    try:
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info('alerts')").fetchall()}
        if "condition_cleared" not in cols:
            conn.execute(
                "ALTER TABLE alerts ADD COLUMN condition_cleared BOOLEAN DEFAULT FALSE")
    except Exception:  # noqa: BLE001
        pass

    # ── v10.7.4 (R7): small key-value meta store (sync-reminder throttle) ────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key VARCHAR PRIMARY KEY,
            value VARCHAR
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

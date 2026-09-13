from quant import paths
import time
import random
import datetime as dt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from quant.data.database import get_connection, init_db
from quant.execution.taxonomy import resolve_broker, get_instrument_class
from quant.data.yf_utils import history_with_timeout, rate_limited
from quant.data.assertions import DataAssertionError
from quant.cli.output import reporter

OUTPUT_FILE = "market_data.parquet"
# Spacing between the (batched-friendly) per-symbol fetches. yfinance 1.x rate
# limits aggressively; a modest delay + jitter avoids tripping it.
REQUEST_DELAY = 0.8
# Lower concurrency: too many simultaneous Yahoo requests trigger YFRateLimitError.
MAX_WORKERS = 5
# Incremental update: only fetch history after this many days back from last_date.
# yfinance needs a small overlap to avoid gaps on non-trading days.
INCREMENTAL_OVERLAP_DAYS = 5


def get_last_dates(conn) -> dict[str, str]:
    """Return {symbol: last_date_str} from market_history. Empty dict if table missing."""
    try:
        rows = conn.execute(
            "SELECT Symbol, MAX(Date) AS last_date FROM market_history GROUP BY Symbol"
        ).fetchall()
        return {r[0]: r[1] for r in rows}
    except Exception:
        return {}


def fetch_single(sym: str, name: str, sector: str, last_date: str | None = None) -> pd.DataFrame | None:
    """Fetch a single ticker's history. Incremental if last_date provided.

    Intent: avoid re-downloading 5y every run. Fetch only data after last_date
    (minus overlap) and append. Drops trailing NaN rows (e.g. future dates).
    Invariants: returns df with Symbol/Sector columns, Date index, no NaN Close.
    Dependencies: yfinance, pandas, taxonomy (broker registry).
    """
    try:
        # Phase 4 (1.2): resolve TR ticker for execution-relevant local prices.
        # For equities/ETFs with a broker mapping, fetch the LS Exchange ticker
        # (e.g. AAPL.DE) so moving averages reflect local trading hours/spreads.
        broker = resolve_broker(sym)
        fetch_ticker = broker["tr_ticker"] if broker["tr_ticker"] else sym

        # Throttle: space out requests with jitter to avoid Yahoo rate-limiting.
        time.sleep(REQUEST_DELAY + random.uniform(0, 0.4))

        if last_date:
            # Incremental: fetch from overlap window before last known date.
            start = (dt.date.fromisoformat(last_date) - dt.timedelta(days=INCREMENTAL_OVERLAP_DAYS))
            df = history_with_timeout(fetch_ticker, start=start.isoformat(), auto_adjust=True)
        else:
            # Full fetch: 5 years.
            df = history_with_timeout(fetch_ticker, period="5y", auto_adjust=True)

        if df is None:
            reason = "rate-limited" if rate_limited() else "timeout/failed"
            reporter.detail(f" [!] {reason.capitalize()} history for {sym} (via {fetch_ticker})")
            return None
        if df.empty:
            reporter.detail(f" [!] Empty history for {sym} (via {fetch_ticker})")
            return None

        # Drop rows with NaN Close (future dates, non-trading days, etc.)
        valid = df.dropna(subset=['Close'])
        if valid.empty:
            reporter.detail(f" [!] No valid Close data for {sym}")
            return None

        latest_px = valid['Close'].iloc[-1]
        latest_dt = valid.index[-1].strftime('%Y-%m-%d')
        reporter.detail(f" [OK] {sym} (via {fetch_ticker}): Latest {latest_dt} | Price: {latest_px:.2f}")

        df['Symbol'] = sym
        df['Sector'] = sector
        # Phase 4 (3.1): tag instrument_class for bifurcated scoring.
        df['Instrument_Class'] = get_instrument_class(sym, name)

        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = df.set_index('Date')

        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'Symbol', 'Sector', 'Instrument_Class']
        df = df[[c for c in cols_to_keep if c in df.columns]]
        df = df.dropna(subset=['Close'])

        # ── Bad-tick repair (MUST run before split detection) ───────────────
        # A single corrupt Yahoo row (spike-and-revert) would otherwise be
        # misread as a split by detect_split, mangling the series and tripping
        # the hard drop assertion (e.g. DFEN 2024-06-03).
        from quant.data.data_quality import repair_isolated_glitches
        df = repair_isolated_glitches(df)

        # ── v10.4.0 (Phase 1): Corporate Actions Engine ─────────────────────
        # Unadjusted data in a backtest guarantees false returns. Detect splits
        # from the price/volume discontinuity and restate pre-split prices so the
        # series is continuous BEFORE features/scoring see it.
        from quant.data.corporate_actions import apply_corporate_actions
        df, split_events = apply_corporate_actions(df)
        split_dates = {pd.Timestamp(e.date) for e in split_events}
        if split_events:
            reporter.detail(f" [CA] {sym}: adjusted {len(split_events)} split(s): "
                            f"{', '.join(f'{e.date.date()} x{e.ratio:g}' for e in split_events)}")

        # ── Part 3 (Gap #1): Data Quality Gate ──────────────────────────────
        # Validate before the data enters DuckDB. Auto-repair common issues;
        # skip the symbol entirely if issues are unfixable.
        from quant.data.data_quality import DataQualityValidator
        validator = DataQualityValidator()
        # Incremental slices are only ~5 days, so the 60-day minimum does not
        # apply when appending to existing data.
        check_min_history = not last_date
        # Extreme daily moves (>25%) are legitimate for volatile names (earnings,
        # biotech, M&A) and must NOT block ingestion. The old code hard-skipped
        # them, permanently excluding BE/SMTC/DELL/ARM/FLEX/TEAM (a symbol that
        # is skipped never gets a last_date, so it is skipped forever). Validate
        # structure only; big moves are legitimate market behaviour.
        is_valid, issues = validator.validate_batch(
            df, sym, check_min_history=check_min_history, check_extreme_moves=False,
        )
        if not is_valid:
            reporter.detail(f" [!] [{sym}] Data quality issues: {issues}")
            repaired = validator.auto_repair(df, sym)
            is_valid, issues = validator.validate_batch(
                repaired, sym, check_min_history=check_min_history,
                check_extreme_moves=False,
            )
            if not is_valid:
                reporter.detail(f" [!] [{sym}] Skipping append — unfixable issues: {issues}")
                return None
            df = repaired

        # ── v10.4.0 (Phase 1): Hard Data Quality Assertions ─────────────────
        # Unlike the soft repair above, a violation here ABORTS the pipeline.
        # Split dates are passed so a legitimate split drop is not flagged.
        from quant.data.assertions import run_assertions
        run_assertions(df, sym, split_dates=split_dates)

        return df
    except DataAssertionError:
        # Hard gate: propagate so main() aborts the whole run.
        raise
    except Exception as e:
        reporter.detail(f" [!] Error {sym}: {e}")
        return None


def build_fetch_list() -> tuple[list[tuple[str, str, str]], dict]:
    """Build the ticker fetch list: CORE ETFs + ACTIVE + portfolio + funnel.

    Intent (Plan 3, Phase 1): the broad 1000+ universe is filtered by funnel.py
    to the top survivors. data_updater fetches FULL 5y history only for those
    survivors plus always-tracked CORE ETFs, ACTIVE registry, and portfolio.
    Phase 5 (v10.2): DELISTED symbols are excluded (stop retrying forever).
    v10.5.0: returns (tickers, meta) where meta carries universe/survivor counts
    for the terse aggregate line.
    Invariants: returns list of (symbol, name, sector); deduplicated by symbol.
    """
    meta = {"universe": 0, "survivors": 0}
    from quant.data.database import init_db
    init_db()

    tickers: dict[str, tuple[str, str]] = {}

    # 1. CORE: broad ETFs (always tracked).
    from quant.data.universe_builder import BROAD_ETFS
    for name, sym in BROAD_ETFS.items():
        tickers[sym] = (name, "Broad ETFs")

    # 2. ACTIVE universe from asset_registry (exclude DELISTED).
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT symbol, sector FROM asset_registry "
            "WHERE universe_status = 'ACTIVE' AND universe_status != 'DELISTED'"
        ).fetchall()
        for sym, sector in rows:
            if sym not in tickers:
                tickers[sym] = (sym, sector or "Unknown")
    except Exception:
        pass

    # 3. Portfolio holdings (always fresh prices for PnL).
    try:
        from quant.portfolio.portfolio import load_portfolio
        port = load_portfolio(paths.DATA_PORTFOLIO)
        for sym in port["Symbol"].unique():
            if sym not in tickers:
                tickers[sym] = (sym, "Portfolio")
    except Exception:
        pass

    # 4. Funnel survivors (top ~24 from the broad universe).
    # The broad 1000+ universe_master must exist before the funnel can filter it.
    # It is normally built by the weekly cron (universe_builder.py), but on a
    # fresh DB it is empty -> the funnel would silently return 0 survivors and
    # the scan universe would collapse to CORE + portfolio only. Build on demand.
    try:
        from quant.data.universe_builder import load_universe_master, build_universe_master
        from quant.data.funnel import run_funnel, save_survivors
        pool = load_universe_master()
        if not pool:
            reporter.detail("  [UNIVERSE] universe_master empty - building broad 1000+ pool...")
            build_universe_master()
            pool = load_universe_master()
        meta["universe"] = len(pool)
        reporter.detail(f"  [FUNNEL] input pool: {len(pool)} symbols")
        if pool:
            result = run_funnel(pool)
            # Persist survivors so main.py reuses them instead of re-running the
            # 1000+ symbol funnel (keeps the 2-step flow: data_updater -> main).
            save_survivors(result["survivors"])
            meta["survivors"] = len(result["survivors"])
            reporter.detail(f"  [FUNNEL] {result['input']} -> {result['stage1']} -> "
                            f"{result['stage2']} survivors")
            for sym in result["survivors"]:
                if sym not in tickers:
                    tickers[sym] = (sym, "Funnel")
    except Exception as e:
        reporter.detail(f"  [!] Funnel failed (non-fatal): {e}")

    return [(sym, name, sector) for sym, (name, sector) in tickers.items()], meta


def main() -> int:
    """Fetch market data + run the funnel. Returns an exit code (spec 3.1)."""
    from quant import __version__
    from quant.reporting.artifacts import new_run_dir

    _t0 = time.time()
    run_dir = new_run_dir()
    reporter.line(f"quant update {__version__}")

    tickers, meta = build_fetch_list()
    total = len(tickers)
    all_data = []

    conn = get_connection()
    init_db()
    last_dates = get_last_dates(conn)
    incremental = bool(last_dates)
    reporter.detail(f"Fetching {total} tickers with {MAX_WORKERS} workers "
                    f"({'INCREMENTAL' if incremental else 'FULL 5y'} mode)")

    # Safety net: each fetch is already bounded by history_with_timeout, but cap
    # the overall wait so a pathological stall cannot hang the run forever.
    overall_timeout = max(120.0, (total / max(MAX_WORKERS, 1)) * 60.0)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fetch_single, sym, name, sector, last_dates.get(sym)): sym
            for sym, name, sector in tickers
        }

        try:
            for i, future in enumerate(as_completed(futures, timeout=overall_timeout), 1):
                try:
                    result = future.result()
                except DataAssertionError as e:
                    # Hard data-quality gate failed: abort the entire run rather
                    # than ingesting corrupt data. Exit code 1 (spec 3.1).
                    reporter.end_progress()
                    reporter.line(f"error: data assertion failed: {e}")
                    reporter.line("remedy: fix the data source, then re-run quant update.")
                    return 1
                if result is not None:
                    all_data.append(result)
                reporter.progress(f"  fetched {i}/{total}")
        except TimeoutError:
            reporter.end_progress()
            reporter.detail(f" [!] Overall fetch timeout ({overall_timeout:.0f}s) reached; "
                            f"proceeding with {len(all_data)} tickers fetched so far.")
    reporter.end_progress()

    if not all_data:
        reporter.line("error: no data acquired.")
        reporter.line("remedy: check network access and the ticker list, then re-run quant update.")
        return 1

    final_df = pd.concat(all_data)
    final_df = final_df.reset_index()

    if 'Date' in final_df.columns:
        final_df['Date'] = pd.to_datetime(final_df['Date']).dt.strftime('%Y-%m-%d')

    # Ensure only known columns (matches market_history PK schema).
    cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Symbol', 'Sector', 'Instrument_Class']
    final_df = final_df[[c for c in cols_to_keep if c in final_df.columns]]

    # Explicit column list: market_history carries an extra ingested_at
    # column (v10.4.0 bitemporal audit) that final_df does not, so a bare
    # `SELECT *` supplies 9 values for 10 columns (BinderException).
    final_df["ingested_at"] = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    insert_cols = ("Date, Open, High, Low, Close, Volume, Symbol, Sector, "
                   "Instrument_Class, ingested_at")
    insert_select = ("SELECT Date, Open, High, Low, Close, Volume, Symbol, "
                     "Sector, Instrument_Class, ingested_at FROM final_df")
    if incremental:
        # INSERT OR REPLACE dedups on PRIMARY KEY (Symbol, Date).
        conn.execute(
            f"INSERT OR REPLACE INTO market_history ({insert_cols}) {insert_select}"
        )
    else:
        conn.execute("DELETE FROM market_history")
        conn.execute(
            f"INSERT INTO market_history ({insert_cols}) {insert_select}"
        )

    # ── v10.4.0 (Phase 4): structured telemetry + EDA event ─────────────
    # Publish market_close_data_ready so subscribers (scoring) can react
    # without a hard call chain. A Yahoo outage cannot cascade.
    try:
        from quant.infra.observability import ObservabilityCollector
        from quant.infra.event_bus import EventBus, EVENTS
        obs = ObservabilityCollector()
        obs.record_metric("fetch_latency_s", time.time() - _t0)
        obs.record_metric("tickers_fetched", len(all_data))
        if rate_limited():
            obs.increment("api_rate_limit_hits")
        EventBus().publish(EVENTS["MARKET_CLOSE_DATA_READY"], {
            "rows": len(final_df),
            "tickers": len(all_data),
            "telemetry": obs.to_json(),
        })
    except Exception:
        pass

    # ── Terse aggregate summary (spec 3.2, max 20 lines) ────────────────────
    latest_bar = final_df["Date"].max() if "Date" in final_df.columns else "unknown"
    reporter.line(f"  universe {meta['universe']} symbols; funnel survivors {meta['survivors']}")
    reporter.line(f"  fetched {len(all_data)}/{total} symbols, +{len(final_df)} rows, "
                  f"latest bar {latest_bar}")
    reporter.line(f"  done in {time.time() - _t0:.0f} s -> {run_dir}/")
    return 0


if __name__ == "__main__":
    main()

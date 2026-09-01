# Changelog

All notable changes to **Quant-AI** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [10.2.0] - 2026-09-01

### Added - Dashboard Clarity, Universe State Machine, Broker Data

1. **Universe State Machine** - [`taxonomy.py`](taxonomy.py), [`discovery.py`](discovery.py)
   - Statuses now `CORE` / `ACTIVE` / `WATCHLIST` / `DELISTED`; new `structure`
     column (`PLAIN` / `INVERSE` / `LEVERAGED`).
   - `CORE_STATUSES` / `DEMOTABLE_STATUSES` sets; CORE is immutable (never
     graduated, never demoted).
   - `GRADUATION_GRACE_MONTHS` grace period: new graduates are not demoted in
     the same run (fixes the graduate-then-demote contradiction).
   - Demotion anchor is the most recent of `graduated_at` / `last_signal_date`.
   - `fetch_failures` tracking: after `MAX_FETCH_FAILURES` consecutive failures,
     a symbol is marked `DELISTED` and excluded from fetching (ZNWD.L).
   - `universe_events` audit table: every GRADUATE / DEMOTE / DELIST / PIN / ADD
     is logged for the dashboard event log.

2. **Registry Repair** - [`scripts/repair_registry.py`](scripts/repair_registry.py) *(new)*
   - Sets `CORE_ETFS` to CORE, marks SDS/SH as INVERSE, clears `graduated_at`
     for WATCHLIST symbols, marks ZNWD.L DELISTED, syncs broker ISINs.

3. **Broker Registry & Routing** - [`taxonomy.py`](taxonomy.py), [`routing.py`](routing.py)
   - `validate_isin()` checksum validator (ISO 6166).
   - `sync_broker_registry()` populates `asset_registry.isin` from the CSV.
   - INVERSE/LEVERAGED structure never routes to SPARPLAN (decay over time).
   - Missing ISIN emits an explicit "ISIN MISSING" instruction.
   - Dynamic fee hurdle: `alpha_bps_from_active_score()` maps active score to
     expected alpha, so `min_trade_size_eur` varies per symbol.

4. **Scoring Differentiation** - [`scoring.py`](scoring.py), [`main.py`](main.py)
   - `etf_quality_score()` cross-sectional ETF structural grade (trend / RS /
     low-vol / momentum) replaces the hardcoded 85.0.
   - `etf_tactical_grade()` continuous tactical grade (regime tilt + momentum z)
     replaces the binary 99.4 / 59.4.
   - `etf_factor_scores()` populates the dashboard Z-score section for ETFs.
   - Per-tier funnel logs: `Tier1 kept X, Tier2 kept Y`.

5. **Dashboard Rebuild** - [`dashboard.py`](dashboard.py), [`.streamlit/config.toml`](.streamlit/config.toml)
   - Zero emoji characters; plain-text headers.
   - `width="stretch"` replaces `use_container_width`.
   - `@st.cache_data(ttl=300)` on all reads.
   - Sidebar: version, last run, market regime, cash APY.
   - Daily Briefing: metric row, SPARPLAN/ACTIVE tables, bucket check, data
     health, backtest expander.
   - Asset Explorer: identity card, price chart with rendered volatility bands,
     factor profile, broker card.
   - Universe Manager: status counts, event log, filterable registry, actions.

6. **Notifier** - [`notifier.py`](notifier.py)
   - Zero emoji message text; run date, regime, action list, bucket violations,
     data-health warnings.
   - `missing_config()` logs the exact missing env variable names.

7. **Tests** - [`test_phase5.py`](test_phase5.py), [`test_no_emoji.py`](test_no_emoji.py) *(new)*
   - Grace period, CORE immunity, delist tracking, ISIN checksum, inverse
     routing, ETF score differentiation.
   - No-emoji lint over `dashboard.py` / `notifier.py` / `reporting.py` / `main.py`.

### Fixed

- [`discovery.py`](discovery.py) graduate-then-demote-in-same-run contradiction.
- [`discovery.py`](discovery.py) CORE ETFs were demoted to WATCHLIST.
- [`discovery.py`](discovery.py) ZNWD.L retried forever; now DELISTED after
  `MAX_FETCH_FAILURES`.
- [`taxonomy.py`](taxonomy.py) WATCHLIST symbols with `graduated_at` populated
  (status/history contradiction) - cleared by the repair script.
- [`routing.py`](routing.py) inverse ETFs (SDS, SH) routed to SPARPLAN.
- [`scoring.py`](scoring.py) degenerate ETF scoring (all 93.6 tie).
- [`dashboard.py`](dashboard.py) empty volatility bands, placeholder Z-scores,
  emoji headers, `use_container_width` deprecation.
- [`database.py`](database.py) legacy `asset_registry` missing `structure` /
  `fetch_failures` columns - added via migration.

### Changed

- [`config.py`](config.py) added `GRADUATION_GRACE_MONTHS`, `STALE_DATA_DAYS`,
  `MAX_FETCH_FAILURES`, `CORE_ETFS`.
- [`artifacts.py`](artifacts.py) added `latest_run_dir()`.
- [`data_updater.py`](data_updater.py) / [`main.py`](main.py) exclude DELISTED
  symbols from fetching and scanning.

---

## [10.1.0] - 2026-09-01

### Added - Phase 4: Broker-Aware Family Office Terminal

1. **Execution Reality (Trade Republic)** - [`optimizer.py`](optimizer.py), [`routing.py`](routing.py)
   - `minimum_trade_size()` / `passes_fee_hurdle()` — 2 EUR round-trip fee hurdle.
   - `route_signal()` — Sparplan (0 EUR buy) vs Active Trade (1 EUR) routing.
   - `broker_registry.csv` — yahoo_ticker → ISIN / tr_ticker / exchange mapping.

2. **Cash & Fee Mathematics** - [`config.py`](config.py), [`risk.py`](risk.py), [`optimizer.py`](optimizer.py)
   - `BROKER_CASH_APY = 0.0225` (TR cash yield as real risk-free rate).
   - `daily_risk_free_rate()` — `(1+APY)^(1/365)-1` used in Sortino/Sharpe.
   - Smart Balance buckets: Safety ≥ 10%, Core ≥ 40%, Alpha ≤ 50% (cvxpy constraints).

3. **Asset Taxonomy & Universe Management** - [`taxonomy.py`](taxonomy.py), [`discovery.py`](discovery.py)
   - `instrument_class` (EQUITY/ETF/COMMODITY/CASH) bifurcated scoring in `main.py`.
   - `asset_registry` table with `universe_status` (ACTIVE/WATCHLIST/CORE).
   - Weekly watchlist scan: 52-week-high / 3x-volume graduation, 6-month demotion.

4. **Local Interface & Automation** - [`dashboard.py`](dashboard.py), [`notifier.py`](notifier.py), [`setup_cron.sh`](setup_cron.sh)
   - 3-page Streamlit dashboard (Daily Briefing / Asset Explorer / Universe Manager).
   - Telegram/Discord daily push notification.
   - Cron installer (daily 18:00 CET + weekly discovery).

### Dependencies

- Added `streamlit`, `plotly`, `requests` to [`requirements.txt`](requirements.txt).

### Fixed

- [`database.py`](database.py) `market_history` had no PRIMARY KEY, so
  `INSERT OR REPLACE` raised `BinderException`. Added `PRIMARY KEY (Symbol, Date)`
  schema + a migration that rebuilds legacy tables with the PK and
  `Instrument_Class` column.
- [`data_updater.py`](data_updater.py) still fetched all 277 `SECTOR_UNIVERSE`
  stocks. New `build_fetch_list()` fetches only **CORE ETFs + ACTIVE universe +
  portfolio holdings** (Core & Satellite model).
- [`taxonomy.py`](taxonomy.py) new symbols defaulted to `ACTIVE`, defeating the
  graduation model. Now default to `WATCHLIST`; only `discovery.graduate()` or
  manual pin promotes to `ACTIVE`.

### Added - Phase 4: Broker-Aware Family Office Terminal

1. **Execution Reality (Trade Republic)** - [`optimizer.py`](optimizer.py), [`routing.py`](routing.py)
   - `minimum_trade_size()` / `passes_fee_hurdle()` — 2 EUR round-trip fee hurdle.
   - `route_signal()` — Sparplan (0 EUR buy) vs Active Trade (1 EUR) routing.
   - `broker_registry.csv` — yahoo_ticker → ISIN / tr_ticker / exchange mapping.

2. **Cash & Fee Mathematics** - [`config.py`](config.py), [`risk.py`](risk.py), [`optimizer.py`](optimizer.py)
   - `BROKER_CASH_APY = 0.0225` (TR cash yield as real risk-free rate).
   - `daily_risk_free_rate()` — `(1+APY)^(1/365)-1` used in Sortino/Sharpe.
   - Smart Balance buckets: Safety ≥ 10%, Core ≥ 40%, Alpha ≤ 50% (cvxpy constraints).

3. **Asset Taxonomy & Universe Management** - [`taxonomy.py`](taxonomy.py), [`discovery.py`](discovery.py)
   - `instrument_class` (EQUITY/ETF/COMMODITY/CASH) bifurcated scoring in `main.py`.
   - `asset_registry` table with `universe_status` (ACTIVE/WATCHLIST/CORE).
   - Weekly watchlist scan: 52-week-high / 3x-volume graduation, 6-month demotion.

4. **Local Interface & Automation** - [`dashboard.py`](dashboard.py), [`notifier.py`](notifier.py), [`setup_cron.sh`](setup_cron.sh)
   - 3-page Streamlit dashboard (Daily Briefing / Asset Explorer / Universe Manager).
   - Telegram/Discord daily push notification.
   - Cron installer (daily 18:00 CET + weekly discovery).

### Dependencies

- Added `streamlit`, `plotly`, `requests` to [`requirements.txt`](requirements.txt).

---

## [10.0.0] - 2026-09-01

### Added - Performance & Architecture

1. **Smart Funnel Architecture (Tiered Execution)** - [`main.py`](main.py)
   - Tier 1 (us): fast fundamental filter (PE, ROE).
   - Tier 2 (ms): fast technical uptrend filter (`Close > 200 SMA`, vectorized).
   - Tier 3 (s): heavy NLP/SEC scraping only on funnel survivors.
   - Cuts total execution time by 50-60%.

2. **EWMA Volatility (primary)** - [`indicators.py`](indicators.py)
   - New `fast_volatility()` - vectorized EWMA, ~1000x faster than GARCH MLE.
   - `add_all_indicators()` now uses EWMA as primary; GARCH retained as fallback.

3. **Market-Regime HMM (fit once)** - [`scoring.py`](scoring.py)
   - New `fit_market_regime()` fits a GaussianHMM once on a broad index (SPY or
     longest-history proxy) and applies the macro bull probability to all assets.
   - Eliminates ~90s of per-asset HMM fitting; statistically sounder.

4. **Batched FinBERT Inference** - [`sentiment.py`](sentiment.py)
   - New `FinBERTBatchScorer.score_texts()` pools chunks across all documents into
     single vectorized forward passes (AVX2).
   - New OOP `NLPScorer` with dependency injection (replaces global `_model`/`_tokenizer`).

5. **Polars Data Engine** - [`main.py`](main.py)
   - Reads DuckDB to Polars natively (`.pl()`); log-returns computed vectorized in Rust.
   - No pandas intermediate, no GIL.

6. **Selectolax SEC Parser** - [`sec_edgar.py`](sec_edgar.py)
   - Replaced BeautifulSoup with the C-based `selectolax` parser (~50x faster).

7. **Incremental Data Updates** - [`data_updater.py`](data_updater.py)
   - `get_last_dates()` reads `MAX(Date)` per symbol; fetches only new data (with
     5-day overlap) instead of re-downloading 5 years every run.

8. **Vectorized Feature Engine** - [`build_features.py`](build_features.py) *(new)*
   - Computes momentum (1d/1m/6m/12m), SMA20/200, vol20/60, ADV20, uptrend,
     trend_strength, max_drawdown_60d across the whole universe in one Polars pass.

9. **Cross-Sectional Factor Scoring** - [`scoring.py`](scoring.py)
   - `factor_scores()` z-scores Value/Quality/Momentum/Low-Risk/Sentiment across the
     universe and combines with weights (0.25/0.25/0.20/0.15/0.15).
   - `sector_neutral_rank()` neutralizes sector bias.

10. **Point-in-Time Fundamentals (No Lookahead)** - [`database.py`](database.py), [`fundamentals.py`](fundamentals.py)
    - New `fundamentals_history` table with `as_of_date`/`published_date`.
    - `save_fundamentals_history()` and `get_fundamentals_as_of()` filter
      `published_date <= as_of_date` to eliminate lookahead bias.

11. **Cost-Aware Backtest** - [`backtest.py`](backtest.py)
    - `run_cost_aware_backtest()` executes at T+1 open, applies 15bps round-trip
      cost, reports net/gross PnL and cost drag.

12. **Portfolio Optimizer** - [`optimizer.py`](optimizer.py) *(new)*
    - `optimize_portfolio()` (cvxpy) with Ledoit-Wolf shrunk covariance and
      max-weight/sector/turnover constraints.
    - `shrunk_covariance()` helper.

13. **Data Validation & Liquidity Filters** - [`validation.py`](validation.py) *(new)*
    - `validate_market_data()`, `sanitize_fundamentals()`, `liquidity_score()`,
      `is_liquid()`, `max_trade_size()`.

14. **Run Artifacts & Structured Logging** - [`artifacts.py`](artifacts.py) *(new)*
    - Timestamped `outputs/run_<ts>/` directories (factor scores, NLP scores, run
      config, metrics as parquet/JSON).
    - JSON `StructuredLogger`.

15. **Unit Tests** - [`test_factors.py`](test_factors.py) *(new)*
    - 9 tests: factor scoring, sector neutralization, no-lookahead fundamentals,
      cost-aware backtest, liquidity, data validation.

### Fixed

- [`scoring.py`](scoring.py) `kelly_position_size()` and `target_volatility_size()`
  referenced `KELLY_FRACTION`/`TARGET_VOLATILITY`/`MAX_POSITION_PCT` without importing
  them - added local imports (unblocked `test_scoring.py`).
- [`main.py`](main.py) DuckDB to Polars conversion: read natively via `.pl()` (no pyarrow
  requirement on pandas frames) and parse the string `Date` column with
  `str.to_datetime()`.
- [`main.py`](main.py) Polars `group_by("Symbol")` yields tuple keys - unpacked to
  scalar strings to avoid `VARCHAR[]` cast errors in `get_fundamentals()`.

### Dependencies

- Added `polars`, `pyarrow`, `selectolax`, `cvxpy` to [`requirements.txt`](requirements.txt).

---

## [9.0.0] - Previous

- Robust NaN data handling in `data_updater.py`.
- FinBERT runs once in the main process (~2.4GB RAM saved).
- Proper logging infrastructure.
- Comprehensive unit test suite (20 scoring + 6 backtest).
- GARCH scale stability with EWMA fallback.
- Parallel data updater (ThreadPoolExecutor, 10 workers).
- Graceful FX degradation.
- Clean configuration (removed 7 redundant constants).
- Fixed critical bugs (NaN close crash, portfolio symbols dropped, duplicate data
  loading, `init_db()` never called, `backtest.py` import error).

---

## Disclaimer

All output is for informational purposes. Probabilistic models and NLP sentiment
analysis involve inherent risk. **Past performance does not guarantee future results.**
The universe contains only currently-listed instruments - historical backtest figures
are systematically overstated due to survivorship bias.
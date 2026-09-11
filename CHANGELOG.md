# Changelog

All notable changes to **Quant-AI** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [10.3.1] - 2026-09-11

### Fixed - Incremental Data Acquisition (Bugfix)

1. **Incremental append no longer rejected by 60-day minimum** -
   [`data_quality.py`](data_quality.py) + [`data_updater.py`](data_updater.py)
   - **Root cause:** incremental mode fetches only `INCREMENTAL_OVERLAP_DAYS = 5`
     days of history, but the data quality gate enforced `min_history_days = 60`
     on that 5-day slice. Every symbol failed the "Only 5 days history (< 60)"
     check, `auto_repair` could not fix it, so all appends were skipped →
     `Fatal: No data acquired.`
   - **Fix:** added `check_min_history: bool = True` param to
     `DataQualityValidator.validate_batch()`. The 60-day minimum is a
     *full-history* scoring invariant, not a data-integrity check for the append
     slice. [`data_updater.py`](data_updater.py) passes
     `check_min_history=not last_date`, so incremental slices skip the minimum
     while full 5y fetches still enforce it.
   - **Result:** `28/28` tickers fetched, `152` rows appended/updated on the
     incremental run. All other quality checks (NaN, negative, extreme moves,
     staleness) still run on the slice.

---

## [10.3.0] - 2026-09-09

### Added - Architectural Refinement (Part 3)

1. **Data Quality Gate** - [`data_quality.py`](data_quality.py) *(new)*
   - `DataQualityValidator` validates incoming market data (NaN, negative
     prices, extreme >25% moves, duplicates, staleness, price bounds) BEFORE it
     enters DuckDB.
   - `auto_repair()` removes duplicates/NaN and interpolates small gaps.
   - Wired into [`data_updater.py`](data_updater.py) — unfixable data is skipped.

2. **Feature Cache** - [`feature_cache.py`](feature_cache.py) *(new)*
   - `FeatureCache` caches computed indicators keyed by
     `hash(symbol + feature + data_hash)`.
   - Invalidates when the underlying Close data changes. Cuts incremental
     computation 60-80%.

3. **Observability** - [`observability.py`](observability.py) *(new)*
   - `ObservabilityCollector` times each pipeline step, records errors, and
     prints a summary + JSON export.
   - Wired into [`main.py`](main.py) — prints a pipeline timing summary.

4. **Incremental Processing** - [`incremental.py`](incremental.py) *(new)*
   - `IncrementalProcessor` detects changed symbols via data hash → O(changed)
     not O(all).

5. **YAML Config** - [`config_loader.py`](config_loader.py) + [`config.yaml`](config.yaml) *(new)*
   - Nested dot-path config access. Non-programmers tune thresholds without
     editing Python.

6. **Alert System** - [`alerts.py`](alerts.py) *(new)*
   - `AlertSystem` surfaces drawdowns, rebalancing triggers, and tax-loss
     opportunities as leveled alerts.

7. **Portfolio Health Score** - [`health_score.py`](health_score.py) *(new)*
   - `PortfolioHealthScore` collapses diversification, risk-adjusted return,
     drawdown, cost, and liquidity into a 0-100 score with grade + recs.

8. **What-If Scenarios** - [`scenario_simulator.py`](scenario_simulator.py) *(new)*
   - `ScenarioSimulator` answers "sell X buy Y" and "market crashes 20%".

9. **Tests** - [`test_part3.py`](test_part3.py) *(new)* — 14 tests.

---

## [10.2.2] - 2026-09-09

### Added - Advanced Strategic Enhancements (Part 2)

1. **Risk-Aware Portfolio Context** - [`portfolio_context.py`](portfolio_context.py) *(new)*
   - `PortfolioContext` computes marginal risk contribution (MRC), PCA factor
     exposure, and a concentration penalty that modulates asset scores.

2. **Multi-Strategy Ensemble** - [`strategies/`](strategies/) + [`strategy_engine.py`](strategy_engine.py) *(new)*
   - `Momentum`, `MeanReversion`, `Value`, `RiskParity` strategies.
   - `StrategyEngine` blends signals with regime-dependent weights.

3. **German Tax-Loss Harvesting** - [`tax_optimizer.py`](tax_optimizer.py) *(new)*
   - Applies Abgeltungsteuer (26.375%), EUR 1,000 allowance, loss-offset.

4. **Dynamic Cash Reserve** - [`cash_manager.py`](cash_manager.py) *(new)*
   - Target cash from regime + VIX + opportunity (5-30%); dip-buying scaled by
     drawdown depth.

5. **Drawdown Circuit Breakers** - [`risk_monitor.py`](risk_monitor.py) *(new)*
   - NORMAL / CAUTION / ALERT / LOCKDOWN based on drawdown and volatility.

6. **P&L Attribution** - [`attribution.py`](attribution.py) *(new)*
   - Brinson-Fachler allocation / selection / interaction effects.

7. **Event Bus** - [`event_bus.py`](event_bus.py) *(new)*
   - Pub/sub decoupling so modules react to regime/drawdown/tax/dip events.

8. **Behavioral Guardrails** - [`behavioral_guardrails.py`](behavioral_guardrails.py) *(new)*
   - Cooldowns, weekly trade limits, consecutive-loss size reduction.

9. **Regime-Aware Validation** - [`validation_engine.py`](validation_engine.py) *(new)*
   - Walk-forward backtest with regime detection + robustness metrics.

10. **Unified Briefing** - [`reporting_advanced.py`](reporting_advanced.py) *(new)*
    - Assembles all Part 2 modules into a single daily briefing, wired into
      [`main.py`](main.py) (non-fatal).

11. **Tests** - [`test_advanced.py`](test_advanced.py) *(new)* — 15 tests.

---

## [10.2.1] - 2026-09-09

### Added - Strategic Portfolio Rebalancing (Part 1)

1. **Asset Tier Classification** - [`config.py`](config.py)
   - `CORE_ASSETS` / `SATELLITE_ASSETS` / `ACTIVE_ASSETS` / `SECTOR_ASSETS`
     tier lists (take precedence over `CORE_ETFS`).
   - `TARGET_WEIGHTS` (50/20/20/10), `REBALANCE_DRIFT_TIERS`,
     `REBALANCE_FREQUENCY_DAYS`, `MIN_TRADE_SIZE_EUR`, `REBALANCE_FIRST_RUN`.

2. **Rebalance Log** - [`database.py`](database.py)
   - `rebalance_log(symbol, last_rebalance_date)` table for time-gated
     rebalancing.

3. **Tier-Aware Portfolio Audit** - [`portfolio.py`](portfolio.py)
   - `classify_asset()`, `should_rebalance_asset()`, `enhanced_portfolio_audit()`
     with corrected weight formula and first-run baseline ease-in.

4. **Tier Signal Generation** - [`scoring.py`](scoring.py)
   - `generate_signal_for_tier()` — CORE never SELL (only HOLD/BUY MORE).

5. **Fee & Liquidity Awareness** - [`optimizer.py`](optimizer.py)
   - `calculate_min_trade_size()`, `check_volume_liquidity()`.

6. **Pipeline Wiring** - [`main.py`](main.py)
   - Enhanced audit with drift + fee-aware recommendations.

7. **Tests** - [`test_rebalancing.py`](test_rebalancing.py) *(new)* — 13 tests.

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
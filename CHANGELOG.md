# Changelog

All notable changes to **Quant-AI** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [10.4.0] - 2026-09-13

### Added - Institutional-Grade Elevation (Phases 1-5)

Shift from *generating signals* to *guaranteeing reliability, execution reality,
and risk survival*. Aligned with CFA Institute / Basel risk-framework practice.

**Phase 1 - Data Trust & Bitemporal Integrity**

1. **Corporate Actions Engine** - [`quant/data/corporate_actions.py`](quant/data/corporate_actions.py) *(new)*
   - `detect_split` finds splits from the price/volume discontinuity (discrete
     ratio set + volume confirmation); `apply_corporate_actions` restates
     pre-split prices and volume; `adjust_cost_basis` fixes the portfolio entry
     price. Wired into [`quant/data/data_updater.py`](quant/data/data_updater.py)
     before features/scoring. Unadjusted data in a backtest guarantees false returns.
2. **Hard Data Quality Assertions** - [`quant/data/assertions.py`](quant/data/assertions.py) *(new)*
   - `run_assertions` is a HARD gate (raises `DataAssertionError`): duplicate
     timestamps, >50% one-day drop without a split flag, and NaN/negative D/E for
     profitable companies. `data_updater.py` aborts the whole run on violation.
3. **Bitemporal PIT enforcement** - [`quant/data/fundamentals.py`](quant/data/fundamentals.py)
   - `get_fundamentals` now persists a PIT snapshot (`as_of_date`/`published_date`);
     new `get_fundamentals_pit` is the only accessor backtests should use (filters
     `published_date <= as_of_date`, fail-closed). `market_history` gains
     `ingested_at` for arrival-vs-valid-time audit.

**Phase 2 - Execution Reality & TCA**

4. **Implementation Shortfall** - [`quant/execution/tca.py`](quant/execution/tca.py) *(new)*
   - `implementation_shortfall` (signal vs fill, bps), `estimate_slippage_bps`
     (half-spread + square-root impact), `record_trade` -> new `trade_log` table,
     `slippage_summary`.
5. **Volatility-aware minimum trade size** - [`quant/portfolio/optimizer.py`](quant/portfolio/optimizer.py)
   - `minimum_trade_size_vol_aware` scales the fee floor by asset volatility; the
     1 EUR fee on a small trade is an instant loss. Wired into
     [`quant/execution/routing.py`](quant/execution/routing.py).
6. **Automated Broker Reconciliation** - [`quant/execution/reconciliation.py`](quant/execution/reconciliation.py) *(new)*,
   [`scripts/reconcile_broker.py`](scripts/reconcile_broker.py) *(new)*
   - Diffs the theoretical portfolio state against the TR CSV export; new
     `portfolio_snapshot` table; flags divergence > 1 EUR.

**Phase 3 - Institutional Risk & Capital Preservation**

7. **Mean-CVaR Optimization** - [`quant/portfolio/optimizer.py`](quant/portfolio/optimizer.py)
   - `optimize_portfolio_cvar` (Rockafellar-Uryasev LP) optimizes the average loss
     in the worst 5% tail; `cvar_of_weights` verifies it. Robust to fat tails.
8. **Hard Kill Switch** - [`quant/portfolio/risk_monitor.py`](quant/portfolio/risk_monitor.py)
   - `check_kill_switch` emits `LIQUIDATE TO CASH` on >3% daily drawdown or
     realized vol > 2x target; publishes `KILL_SWITCH` on the event bus.
9. **Regime-Conditional Constraints** - [`quant/portfolio/regime_constraints.py`](quant/portfolio/regime_constraints.py) *(new)*
   - Bear/Chop hard-caps single-stock weight at 2% and forbids leverage; applied
     inside both optimizers via the `regime` parameter.

**Phase 4 - Software Engineering & CI/CD Rigor**

10. **Golden-File Snapshot Testing** - [`tests/test_golden_snapshot.py`](tests/test_golden_snapshot.py) *(new)*,
    [`tests/golden_util.py`](tests/golden_util.py) *(new)*, [`scripts/make_golden.py`](scripts/make_golden.py) *(new)*
    - Deterministic seeded backtest output; CI fails on >0.01% drift.
11. **Event-Driven Architecture** - [`quant/infra/event_bus.py`](quant/infra/event_bus.py)
    - New canonical events `market_close_data_ready`, `scoring_complete`,
    `kill_switch`; `data_updater` publishes, scoring subscribes.
12. **Structured Telemetry** - [`quant/infra/observability.py`](quant/infra/observability.py)
    - `record_metric`/`increment` track fetch latency, DuckDB query time, and
    rate-limit hits; persisted to `outputs/run_<ts>/telemetry.json`.
13. **GitHub Actions CI** - [`.github/workflows/ci.yml`](.github/workflows/ci.yml) *(new)*
    - Runs the full suite + the golden snapshot gate on every push/PR.

**Phase 5 - Advanced Alpha Metrics & Reporting**

14. **Deflated Sharpe Ratio** - [`quant/analytics/metrics.py`](quant/analytics/metrics.py) *(new)*
    - Bailey & Lopez de Prado DSR penalizes the Sharpe for trials + skew/kurtosis.
15. **Alpha Decay (IC) Curves** - [`quant/analytics/metrics.py`](quant/analytics/metrics.py)
    - `information_coefficient` + `alpha_decay_curve` at T+1/T+5/T+21.
16. **Probabilistic Turnover** - [`quant/analytics/metrics.py`](quant/analytics/metrics.py)
    - `turnover_stats` returns mean AND variance of turnover.
17. **ALPHA QUALITY briefing section** - [`quant/reporting/reporting_advanced.py`](quant/reporting/reporting_advanced.py),
    [`quant/main.py`](quant/main.py) - DSR, IC decay, and turnover variance in the
    daily briefing.

### Tests

- [`tests/test_institutional.py`](tests/test_institutional.py) *(new)* - 11 tests
  covering corporate actions, assertions, Mean-CVaR, kill switch, regime
  constraints, TCA, vol-aware sizing, reconciliation, alpha metrics, events,
  telemetry. Full suite green (18 suites).

---

## [10.3.6] - 2026-09-13

### Fixed - Universe Cleanup & Single Funnel Run

`main.py` was re-running the full 1000+ symbol funnel that `data_updater.py`
already runs (a duplicated network fetch every cycle); the broad universe
contained tickers Yahoo does not carry; and legitimate volatile names were
silently dropped. All cleaned up.

1. **Non-existent tickers pruned** - [`universe_builder.py`](universe_builder.py)
   - `NONEXISTENT_SYMBOLS` (`LBRDK`, `WBS`) are skipped on build and deleted
     from `universe_master`; stale dotted class-share rows are purged on rebuild.
     `universe_master` is now 1082 clean Yahoo symbols (was 1084).

2. **Funnel runs once (2-step flow restored)** - [`database.py`](database.py),
   [`funnel.py`](funnel.py), [`data_updater.py`](data_updater.py), [`main.py`](main.py)
   - New `funnel_survivors` table caches the funnel result. `data_updater.py`
     writes it (`save_survivors`); `main.py` now READS it (`load_survivors`)
     instead of re-running the 1000+ symbol funnel. Removes a duplicated ~76s
     network fetch per daily cycle and honours the documented
     `data_updater.py` -> `main.py` flow.

3. **Legitimate volatility no longer blocked** - [`data_updater.py`](data_updater.py)
   - The `>25% daily move` check hard-skipped volatile names (BE, SMTC, DELL,
     ARM, FLEX, TEAM); a skipped symbol never acquires a `last_date`, so it was
     skipped forever. Extreme moves are now treated as legitimate market
     behaviour (structural checks still run: NaN/negative/duplicate/stale).
     Result: 79/79 tickers fetched, 0 skips.

4. **yfinance delisted noise silenced** - [`yf_utils.py`](yf_utils.py)
   - `download_batch()` wraps `yf.download()` in `_silence_yf_output()`,
     raising the yfinance logger and redirecting stderr, so batch probes of bad
     tickers no longer print `possibly delisted` or surface as `[ERROR]`.

---

## [10.3.5] - 2026-09-13

### Fixed - Pipeline Freeze: Bounded Network I/O & Single-Writer Database

`data_updater.py` froze during runs. The root cause was three compounding
unbounded-blocking issues; all waits are now time-bounded.

1. **yfinance calls had no timeout** - [`yf_utils.py`](yf_utils.py) *(new)*,
   [`data_updater.py`](data_updater.py), [`funnel.py`](funnel.py)
   - New `history_with_timeout()` runs each `Ticker.history()` call in a daemon
     thread with a hard per-attempt timeout (default 15s) plus retries/backoff,
     mirroring the pattern already used in [`fundamentals.py`](fundamentals.py).
     A throttled Yahoo response can no longer hang the run.
   - `fetch_single` (data_updater) and `fetch_snapshot` / `fetch_history`
     (funnel) now use the helper and treat a timeout (`None`) as "no data".
   - The `as_completed` loops now carry an overall timeout safety net so one
     stalled ticker cannot block the whole batch.

2. **Funnel phase was silent** - [`data_updater.py`](data_updater.py)
   - `main()` now prints `Building fetch list...` before `build_fetch_list()`
     and reports the resulting count, so the (potentially slow) funnel is
     visible instead of looking frozen.

3. **DuckDB multi-threaded write contention** - [`taxonomy.py`](taxonomy.py),
   [`database.py`](database.py)
   - Added a reentrant `_DB_WRITE_LOCK` that serializes all `asset_registry` /
     `universe_events` writes (`_upsert_registry`, `log_universe_event`,
     `set_core`, `sync_broker_registry`). Worker threads in the fetch pool no
     longer write to DuckDB concurrently.
   - `get_connection()` now connects through a bounded daemon thread
     (`CONNECT_TIMEOUT = 15s`) and raises a clear error instead of blocking
     forever when the DB file is locked. The old comment claimed a `timeout=15`
     that was never actually passed.

4. **Request throttling** - [`data_updater.py`](data_updater.py)
   - The defined-but-unused `REQUEST_DELAY` is now applied before each fetch
     (with jitter), and worker concurrency was lowered from 10 to 5 to avoid
     tripping Yahoo's rate limiter.

5. **Batched downloads** - [`yf_utils.py`](yf_utils.py), [`funnel.py`](funnel.py)
   - New `download_batch()` fetches many tickers per `yf.download()` request
     (50 per chunk) instead of one request per ticker. The 1084-symbol universe
     now costs ~20 requests, not 1084. Funnel Stage 1 and Stage 2 both use it.

6. **Rate-limit circuit breaker** - [`yf_utils.py`](yf_utils.py)
   - `YFRateLimitError` is detected by name; a global cooldown with exponential
     backoff pauses all workers together, and after 3 consecutive hits the run
     aborts fast (`[!] Yahoo rate limit persists - aborting further fetches`)
     instead of grinding through ~1000 retries.

7. **Funnel Stage 1 soft cap enforced** - [`funnel.py`](funnel.py)
   - `FUNNEL_STAGE1_TARGET` (300) was documented but ignored, so Stage 2
     downloaded 1y history for ~1066 tickers every run. Stage 1 now ranks
     survivors by dollar volume and keeps the top 300. The full funnel
     (1084 -> 300 -> 24) dropped from ~120s to ~76s.

8. **US share-class ticker normalization** - [`universe_builder.py`](universe_builder.py)
   - Wikipedia lists class shares with a dot (`BRK.B`, `BF.A`, `HEI.A`,
     `LEN.B`, `UHAL.B`) while Yahoo uses a dash (`BRK-B`, ...). These were
     reported "possibly delisted" and silently dropped. Dots are now converted
     to dashes for the US-only index tables; European suffixes (`.L`/`.DE`/
     `.AS`) come from the curated `BROAD_ETFS` list and are unaffected.

---

## [10.3.4] - 2026-09-12

### Fixed - Reconciliation, Data Quality Gate & Funnel Noise

1. **Reconciliation formula aligned with Plan 3** - [`portfolio.py`](portfolio.py)
   - `System_Estimated_Value` now uses the **native** `Avg_Entry_Price` (not
     FX-converted): `(Current_Value_EUR / Avg_Entry_Price) * Current_Market_Price_EUR`.
   - **Root cause:** converting `Avg_Entry_Price` to EUR inflated the deviation
     and fired a false `[!]` flag on every position (e.g. AMZN +24.71 instead of
     +0.49). Now only genuinely divergent positions are flagged.

2. **Data Quality Gate no longer drops volatile tickers on incremental** -
   [`data_quality.py`](data_quality.py) + [`data_updater.py`](data_updater.py)
   - Added `check_extreme_moves: bool = True` to `validate_batch()`. The
     `>25% daily move` check is a full-history corruption invariant, not an
     append-slice check. A single big move on a real trading day (earnings/news)
     is legitimate and must not block the update.
   - [`data_updater.py`](data_updater.py) passes `check_extreme_moves=not last_date`
     (mirrors the existing `check_min_history` pattern).
   - **Result:** BE, QRVO, SMTC, OKTA, SANM, DELL, GTLB are no longer skipped.

3. **Funnel silences yfinance delisted-symbol noise** - [`funnel.py`](funnel.py)
   - Added `_silence_yfinance()` context manager that redirects stderr to
     devnull during snapshot/history fetches. The broad universe contains some
     delisted/bad tickers (e.g. `BRK.B`, `BF.B` from Russell 1000); yfinance
     prints `$SYM: possibly delisted` for each. These are non-fatal (the funnel
     skips them) but flooded the console. Now silent.

---

## [10.3.3] - 2026-09-12

### Added - Architecture Restoration & Broker-Sync Overhaul (Plan 3)

1. **Smart 1000+ Universe** - [`universe_builder.py`](universe_builder.py) *(new)*
   - Deleted the hardcoded ~300 `SECTOR_UNIVERSE`. The broad pool now loads
     dynamically from index constituents (S&P 500 = 503, Nasdaq-100 = 102,
     Russell 1000 = 1021) + 49 broad ETFs → **1084 unique symbols** in a new
     `universe_master` table.
   - Uses a browser User-Agent to bypass Wikipedia's HTTP 403 block on
     `pandas.read_html`.

2. **Multi-Stage Funnel** - [`funnel.py`](funnel.py) *(new)*
   - Stage 1 (liquidity/viability): 1-day snapshot, `price > $5` and min daily
     dollar volume → ~300-500 survivors.
   - Stage 2 (trend/momentum): 1y history, SMA/RSI/6m-return composite score →
     top ~24 survivors for heavy FinBERT/GARCH analysis.

3. **Universe Refactor** - [`universe.py`](universe.py)
   - Removed `SECTOR_UNIVERSE`, `get_market_universe`, `symbol_to_sector`,
     `get_sector_symbols`, `universe_stats`. Kept `CURRENCY_SYMBOLS` + GEO
     tables. `is_etf()` now reads `broker_registry.csv` directly (no circular
     import with `taxonomy.classify_instrument`).

4. **Staged Data Pipeline** - [`data_updater.py`](data_updater.py) + [`main.py`](main.py)
   - `build_fetch_list()` fetches full 5y history only for funnel survivors +
     CORE ETFs + ACTIVE + portfolio. `main.py` scan universe = funnel output +
     portfolio + CORE + ACTIVE.

5. **Broker-Sync CSV** - [`portfolio.csv`](portfolio.csv) + [`portfolio.py`](portfolio.py)
   - New schema: `Symbol, Avg_Entry_Price, Current_Value_EUR, Broker_PnL_EUR`.
     `Invested_EUR = Current_Value_EUR - Broker_PnL_EUR`;
     `Real_PnL_EUR = Broker_PnL_EUR` (broker truth, never price-guessed);
     `Real_PnL_Pct = Broker_PnL_EUR / Invested_EUR * 100`. Fixes the phantom
     DCA profit bug.
   - `broker_data.csv` retired; `Broker_PnL_EUR` consolidated into
     `portfolio.csv`.

6. **FX Transparency** - [`portfolio.py`](portfolio.py)
   - Dual-price display: `Current_Price_Native` + `Current_Price_EUR`;
     `FX_Impact_EUR` retained.

7. **Reconciliation Engine** - [`portfolio.py`](portfolio.py)
   - `System_Estimated_Value = shares × current_price_eur`;
     `Recon_Deviation` + `Recon_Flag` `[!]` when deviation > €1.00 (stale CSV
     or high spread).

8. **Discovery Feed** - [`discovery.py`](discovery.py)
   - Graduation engine now scans `universe_master` (watchlist.csv is fallback).

9. **Tests** - [`test_funnel.py`](test_funnel.py) *(new)*, [`test_universe_builder.py`](test_universe_builder.py) *(new)*, [`test_portfolio_fx.py`](test_portfolio_fx.py) *(rewritten)*.

---

## [10.3.2] - 2026-09-11

### Added - Portfolio Audit & FX Reconciliation (Action Plan Items 1-4, 6)

1. **Real PnL in EUR** - [`portfolio.py`](portfolio.py)
   - `enhanced_portfolio_audit` now computes FX-aware EUR PnL:
     `Invested_EUR` = `Original_Amount` (real EUR cost basis),
     `Value_EUR` = `(Current_Price / FX) * Shares` (live, converted to EUR),
     `Real_PnL_EUR` = `Value_EUR - Invested_EUR`,
     `Real_PnL_Pct` = `Real_PnL_EUR / Invested_EUR * 100`.
   - Shares are backed out from the CSV's recorded current EUR value so cost
     basis and FX impact stay independent of today's rate.

2. **FX Impact Tracking** - [`portfolio.py`](portfolio.py)
   - New `FX_Impact_EUR` column = `Real_PnL_EUR - Asset_PnL_EUR`, where
     `Asset_PnL_EUR` is the pure price move converted at today's rate. Positive
     = currency helped; negative = currency hurt.

3. **Broker Reconciliation** - [`portfolio.py`](portfolio.py) + [`broker_data.csv`](broker_data.csv) *(new)*
   - `load_broker_data()` reads `broker_data.csv` (`Symbol,Broker_PnL_EUR`).
   - Audit emits `Broker_Deviation` = `System_PnL_EUR - Broker_PnL_EUR` and a
     `Broker_Flag` `[!]` when `|deviation| > €0.50`.

4. **Restructured Audit Table** - [`portfolio.py`](portfolio.py) + [`main.py`](main.py)
   - Columns: `Symbol | Tier | Avg_Buy_Price | Current_Price_FX | Invested_EUR |
     Value_EUR | Current_Weight | Target_Weight | Drift | Real_PnL_EUR |
     Real_PnL_Pct | FX_Impact_EUR | Broker_Deviation | Broker_Flag | Signal |
     Horizon | Recommendation`.
   - Backward-compat aliases (`PnL_pct`, `PnL_EUR`, `Current_Value`) retained
     for the tax optimizer and effectiveness report.

5. **DIP BUY Gating** - [`main.py`](main.py)
   - DIP BUY alerts only fire for positions underweight vs their target tier
     (or not in the audit), never for overweight positions.

6. **FX Helpers** - [`currency.py`](currency.py)
   - Added `deduce_currency()` and `get_fx_to_eur()` (EUR=1.0, USD=1/EURUSD
     live, GBX via USD proxy, others via `apply_fx_conversion`).

7. **Tests** - [`test_portfolio_fx.py`](test_portfolio_fx.py) *(new)* — 4 tests.

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
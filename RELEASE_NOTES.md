# Release Notes - v10.3.6

**Quant-AI v10.3.6** - Universe Cleanup & Single Funnel Run

## Fixed

1. **Non-existent tickers pruned** - [`universe_builder.py`](universe_builder.py)
   — `LBRDK`/`WBS` and stale dotted class-share rows are removed from
   `universe_master` on rebuild (now 1082 clean Yahoo symbols).
2. **Funnel runs once (2-step flow)** - [`database.py`](database.py),
   [`funnel.py`](funnel.py), [`data_updater.py`](data_updater.py),
   [`main.py`](main.py) — a new `funnel_survivors` cache means `data_updater.py`
   computes survivors and `main.py` reads them, removing a duplicated 1000+
   symbol fetch from every daily cycle.
3. **Volatility no longer blocks ingestion** - [`data_updater.py`](data_updater.py)
   — the `>25%` move check no longer hard-skips BE/SMTC/DELL/ARM/FLEX/TEAM;
   all 79 tickers now update.
4. **Silenced delisted-ticker noise** - [`yf_utils.py`](yf_utils.py) —
   `download_batch()` suppresses yfinance's `possibly delisted` stderr/log
   output during batch probes.

## Test Suite

| Suite | Tests | Status |
|:---|:---:|:---|
| `test_funnel.py` | 5 | Pass |
| `test_universe_builder.py` | 4 | Pass |
| `test_part3.py` | 14 | Pass |

---

# Release Notes - v10.3.5

**Quant-AI v10.3.5** - Pipeline Freeze Fix: Bounded Network I/O & Single-Writer Database

## Fixed

1. **Unbounded yfinance calls** - [`yf_utils.py`](yf_utils.py) *(new)*,
   [`data_updater.py`](data_updater.py), [`funnel.py`](funnel.py) — the new
   `history_with_timeout()` wraps `Ticker.history()` in a daemon thread with a
   hard per-attempt timeout (15s) plus retries/backoff. A throttled Yahoo
   response can no longer hang the run, and the `as_completed` loops now have an
   overall timeout safety net.
2. **Silent funnel phase** - [`data_updater.py`](data_updater.py) — `main()`
   now reports progress before/after `build_fetch_list()`, so the slow funnel no
   longer looks frozen.
3. **DuckDB write contention** - [`taxonomy.py`](taxonomy.py),
   [`database.py`](database.py) — a reentrant write lock serializes registry
   writes from the fetch thread pool; `get_connection()` now uses a bounded
   connect timeout (`CONNECT_TIMEOUT = 15s`) instead of blocking forever on a
   locked database.
4. **Request throttling** - [`data_updater.py`](data_updater.py) — the
   previously-unused `REQUEST_DELAY` is now applied before each fetch (with
   jitter), and worker concurrency was lowered from 10 to 5.
5. **Batched downloads** - [`yf_utils.py`](yf_utils.py), [`funnel.py`](funnel.py)
   — new `download_batch()` fetches 50 tickers per `yf.download()` request; the
   1084-symbol universe now costs ~20 requests instead of 1084.
6. **Rate-limit circuit breaker** - [`yf_utils.py`](yf_utils.py) — detects
   `YFRateLimitError`, backs off globally, and aborts fast after 3 consecutive
   hits rather than grinding through every ticker.
7. **Funnel Stage 1 cap** - [`funnel.py`](funnel.py) — `FUNNEL_STAGE1_TARGET`
   is now enforced (top 300 by dollar volume), cutting Stage 2 from ~1066 to 300
   history downloads; the full funnel dropped from ~120s to ~76s.
8. **US share-class tickers** - [`universe_builder.py`](universe_builder.py) —
   `BRK.B`/`BF.A`/`HEI.A`/`LEN.B`/`UHAL.B` are normalized to Yahoo's dashed
   form (`BRK-B`, ...) so they are no longer dropped as "possibly delisted".
## Test Suite

| Suite | Tests | Status |
|:---|:---:|:---|
| `test_funnel.py` | 5 | Pass |
| `test_universe_builder.py` | 4 | Pass |
| `test_portfolio_fx.py` | 6 | Pass |
| `test_rebalancing.py` | 13 | Pass |
| `test_phase4.py` | 20 | Pass |
| `test_phase5.py` | 13 | Pass |

---

# Release Notes - v10.3.4

**Quant-AI v10.3.4** - Reconciliation, Data Quality & Funnel Noise Fixes

## Fixed

1. **Reconciliation formula aligned with Plan 3** - [`portfolio.py`](portfolio.py)
   — `System_Estimated_Value` uses the native `Avg_Entry_Price`:
   `(Current_Value_EUR / Avg_Entry_Price) * Current_Market_Price_EUR`. Fixes
   false `[!]` flags on every position (AMZN +24.71 → +0.49).
2. **Data Quality Gate** - [`data_quality.py`](data_quality.py) +
   [`data_updater.py`](data_updater.py) — added `check_extreme_moves`; skipped
   on incremental slices so a single legitimate >25% move (earnings/news) no
   longer drops the symbol. BE, QRVO, SMTC, OKTA, SANM, DELL, GTLB now update.
3. **Funnel noise** - [`funnel.py`](funnel.py) — `_silence_yfinance()` redirects
   stderr during fetches, removing the `$SYM: possibly delisted` flood from
   probing bad/delisted Russell 1000 tickers.

## Test Suite

| Suite | Tests | Status |
|:---|:---:|:---|
| `test_funnel.py` | 5 | Pass |
| `test_universe_builder.py` | 4 | Pass |
| `test_portfolio_fx.py` | 6 | Pass |
| `test_rebalancing.py` | 13 | Pass |
| `test_phase4.py` | 20 | Pass |
| `test_phase5.py` | 13 | Pass |

---

# Release Notes - v10.3.3

**Quant-AI v10.3.3** - Architecture Restoration & Broker-Sync Overhaul

## What's New

1. **Smart 1000+ Universe** - [`universe_builder.py`](universe_builder.py) *(new)*
   — the hardcoded ~300 `SECTOR_UNIVERSE` is deleted. The broad pool now loads
   dynamically from index constituents (S&P 500 = 503, Nasdaq-100 = 102,
   Russell 1000 = 1021) + 49 broad ETFs → **1084 unique symbols** in a new
   `universe_master` table.
2. **Multi-Stage Funnel** - [`funnel.py`](funnel.py) *(new)* — Stage 1
   liquidity/viability (price > $5, min daily $ volume) → ~300-500; Stage 2
   trend/momentum (SMA/RSI/6m return) → top ~24 survivors for heavy analysis.
3. **Broker-Sync CSV** - [`portfolio.csv`](portfolio.csv) — new schema
   `Symbol, Avg_Entry_Price, Current_Value_EUR, Broker_PnL_EUR`. PnL is broker
   truth, never price-guessed. Fixes the phantom DCA profit bug.
4. **FX Transparency** - dual-price display (`Current_Price_Native` +
   `Current_Price_EUR`); `FX_Impact_EUR` retained.
5. **Reconciliation Engine** - `System_Estimated_Value = shares ×
   current_price_eur`; `[!]` flag when deviation > €1.00 (stale CSV / spread).
6. **Staged Data Pipeline** - [`data_updater.py`](data_updater.py) +
   [`main.py`](main.py) fetch full history only for funnel survivors + CORE +
   ACTIVE + portfolio.

## Test Suite

| Suite | Tests | Status |
|:---|:---:|:---|
| `test_funnel.py` | 5 | Pass |
| `test_universe_builder.py` | 4 | Pass |
| `test_portfolio_fx.py` | 6 | Pass |
| `test_rebalancing.py` | 13 | Pass |
| `test_phase4.py` | 20 | Pass |
| `test_phase5.py` | 13 | Pass |

---

# Release Notes - v10.3.2

**Quant-AI v10.3.2** - Portfolio Audit & FX Reconciliation

## What's New

1. **Real PnL in EUR** - [`portfolio.py`](portfolio.py) — the audit now shows
   actual money made/lost in EUR (`Real_PnL_EUR`, `Real_PnL_Pct`) instead of
   native-currency price growth. `Invested_EUR` = `Original_Amount`;
   `Value_EUR` = `(Current_Price / FX) * Shares`; `Real_PnL_EUR` =
   `Value_EUR - Invested_EUR`.
2. **FX Impact Tracking** - new `FX_Impact_EUR` column isolates the currency
   contribution to EUR PnL (positive = currency helped, negative = hurt).
3. **Broker Reconciliation** - [`broker_data.csv`](broker_data.csv) lets you
   paste Trade Republic PnL; the audit emits `Broker_Deviation` and flags
   `[!]` when it exceeds €0.50.
4. **Restructured Audit Table** - readable columns: `Symbol | Tier |
   Avg_Buy_Price | Current_Price_FX | Invested_EUR | Value_EUR | Weight |
   Real_PnL_EUR | Real_PnL_Pct | FX_Impact_EUR | Broker_Deviation |
   Rebalance_Action`.
5. **DIP BUY Gating** - [`main.py`](main.py) only suggests DIP BUYs on
   underweight positions, never overweight ones.
6. **FX Helpers** - [`currency.py`](currency.py) adds `deduce_currency()` and
   `get_fx_to_eur()`.

## Test Suite

| Suite | Tests | Status |
|:---|:---:|:---|
| `test_portfolio_fx.py` | 4 | Pass |
| `test_rebalancing.py` | 13 | Pass |
| `test_advanced.py` | 15 | Pass |
| `test_phase5.py` | 13 | Pass |

---

# Release Notes - v10.3.1

**Quant-AI v10.3.1** - Incremental Data Acquisition Bugfix

## Fixed

1. **Incremental append no longer rejected by the 60-day minimum** -
   [`data_quality.py`](data_quality.py) + [`data_updater.py`](data_updater.py)
   - **Root cause:** incremental mode fetches only `INCREMENTAL_OVERLAP_DAYS = 5`
     days of history, but the data quality gate enforced `min_history_days = 60`
     on that 5-day slice. Every symbol failed the "Only 5 days history (< 60)"
     check, `auto_repair` could not fix it, so all appends were skipped →
     `Fatal: No data acquired.`
   - **Fix:** added `check_min_history: bool = True` to
     `DataQualityValidator.validate_batch()`. The 60-day minimum is a
     *full-history* scoring invariant, not a data-integrity check for the append
     slice. [`data_updater.py`](data_updater.py) passes
     `check_min_history=not last_date`, so incremental slices skip the minimum
     while full 5y fetches still enforce it.
   - **Result:** `28/28` tickers fetched, `152` rows appended/updated on the
     incremental run. All other quality checks (NaN, negative, extreme moves,
     staleness) still run on the slice.

---

# Release Notes - v10.3.0

**Quant-AI v10.3.0** - Architectural Refinement Release (Part 3)

This release focuses on architectural hygiene: data quality validation, feature
caching, observability, incremental processing, YAML config, alerts, a portfolio
health score, and what-if scenario analysis. The system is faster, safer,
simpler, and more useful — while staying offline-first and local.

---

## What's New

1. **Data Quality Gate** - [`data_quality.py`](data_quality.py) validates
   incoming market data (NaN, negative prices, extreme moves, duplicates,
   staleness) before it enters DuckDB. `auto_repair()` fixes common issues.
   Wired into [`data_updater.py`](data_updater.py).
2. **Feature Cache** - [`feature_cache.py`](feature_cache.py) caches computed
   indicators keyed by `hash(symbol + feature + data_hash)`, invalidating when
   data changes. Cuts incremental computation 60-80%.
3. **Observability** - [`observability.py`](observability.py) times each
   pipeline step, records errors, and prints a summary + JSON export. Wired
   into [`main.py`](main.py).
4. **Incremental Processing** - [`incremental.py`](incremental.py) detects
   changed symbols via data hash → O(changed) not O(all).
5. **YAML Config** - [`config_loader.py`](config_loader.py) + [`config.yaml`](config.yaml)
   with nested dot-path access.
6. **Alert System** - [`alerts.py`](alerts.py) surfaces drawdowns, rebalancing
   triggers, and tax-loss opportunities.
7. **Portfolio Health Score** - [`health_score.py`](health_score.py) collapses
   diversification, risk-adjusted return, drawdown, cost, and liquidity into a
   0-100 score with grade + recommendations.
8. **What-If Scenarios** - [`scenario_simulator.py`](scenario_simulator.py)
   answers "sell X buy Y" and "market crashes 20%".

---

## Test Suite

| Suite | Tests | Status |
|:---|:---:|:---|
| `test_scoring.py` | 20 | Pass |
| `test_factors.py` | 9 | Pass |
| `test_backtest_validity.py` | 6 | Pass |
| `test_phase4.py` | 20 | Pass |
| `test_phase5.py` | 12 | Pass |
| `test_no_emoji.py` | 1 | Pass |
| `test_rebalancing.py` | 13 | Pass |
| `test_advanced.py` | 15 | Pass |
| `test_part3.py` | 14 | Pass |

---

# Release Notes - v10.2.2

**Quant-AI v10.2.2** - Advanced Strategic Enhancements Release (Part 2)

This release transforms the bot from a single-signal scanner into an adaptive
multi-strategy portfolio manager: portfolio-level risk context, a multi-strategy
ensemble, German tax-loss harvesting, dynamic cash management, drawdown circuit
breakers, P&L attribution, an event bus, overtrading guardrails, and regime-aware
backtest validation.

---

## What's New

1. **Risk-Aware Portfolio Context** - [`portfolio_context.py`](portfolio_context.py)
   computes marginal risk contribution, PCA factor exposure, and a concentration
   penalty that modulates asset scores.
2. **Multi-Strategy Ensemble** - [`strategies/`](strategies/) + [`strategy_engine.py`](strategy_engine.py)
   blend Momentum / MeanReversion / Value / RiskParity with regime-dependent
   weights.
3. **German Tax-Loss Harvesting** - [`tax_optimizer.py`](tax_optimizer.py)
   applies Abgeltungsteuer (26.375%), EUR 1,000 allowance, loss-offset.
4. **Dynamic Cash Reserve** - [`cash_manager.py`](cash_manager.py) targets cash
   from regime + VIX + opportunity (5-30%) and scales dip-buying with drawdown.
5. **Drawdown Circuit Breakers** - [`risk_monitor.py`](risk_monitor.py) returns
   NORMAL / CAUTION / ALERT / LOCKDOWN.
6. **P&L Attribution** - [`attribution.py`](attribution.py) Brinson-Fachler
   allocation / selection / interaction effects.
7. **Event Bus** - [`event_bus.py`](event_bus.py) pub/sub decoupling.
8. **Behavioral Guardrails** - [`behavioral_guardrails.py`](behavioral_guardrails.py)
   cooldowns, weekly limits, consecutive-loss size reduction.
9. **Regime-Aware Validation** - [`validation_engine.py`](validation_engine.py)
   walk-forward backtest with robustness metrics.
10. **Unified Briefing** - [`reporting_advanced.py`](reporting_advanced.py)
    assembles all modules into a single daily briefing, wired into [`main.py`](main.py).

---

## Test Suite

| Suite | Tests | Status |
|:---|:---:|:---|
| `test_advanced.py` | 15 | Pass |

---

# Release Notes - v10.2.1

**Quant-AI v10.2.1** - Strategic Portfolio Rebalancing Release (Part 1)

This release differentiates CORE (buy-and-hold) from ACTIVE (tactical) assets,
rebalances only on meaningful drift, respects the 2 EUR round-trip fee, and
never emits SELL on core buy-and-hold ETFs.

---

## What's New

1. **Asset Tier Classification** - [`config.py`](config.py) defines
   `CORE_ASSETS` / `SATELLITE_ASSETS` / `ACTIVE_ASSETS` / `SECTOR_ASSETS` tier
   lists (precedence over `CORE_ETFS`), `TARGET_WEIGHTS` (50/20/20/10), drift
   thresholds, and rebalance frequencies.
2. **Rebalance Log** - [`database.py`](database.py) adds the `rebalance_log`
   table for time-gated rebalancing.
3. **Tier-Aware Portfolio Audit** - [`portfolio.py`](portfolio.py) adds
   `classify_asset()`, `should_rebalance_asset()`, `enhanced_portfolio_audit()`
   with corrected weight formula and first-run baseline ease-in.
4. **Tier Signal Generation** - [`scoring.py`](scoring.py) adds
   `generate_signal_for_tier()` — CORE never SELL.
5. **Fee & Liquidity Awareness** - [`optimizer.py`](optimizer.py) adds
   `calculate_min_trade_size()`, `check_volume_liquidity()`.
6. **Pipeline Wiring** - [`main.py`](main.py) uses the enhanced audit.

---

## Test Suite

| Suite | Tests | Status |
|:---|:---:|:---|
| `test_rebalancing.py` | 13 | Pass |

---

# Release Notes - v10.2.0

**Quant-AI v10.2.0** - Dashboard Clarity Release

This release fixes the universe state machine contradictions, completes the
broker ISIN registry, differentiates ETF scoring, and rebuilds the dashboard
with zero emoji characters.

---

## Defects Fixed (from the v10.1 run)

1. **Graduate-then-demote contradiction** - 18 symbols graduated, then 15
   demoted seconds later. New graduates now get a `GRADUATION_GRACE_MONTHS`
   grace period and are never demoted in the same run.
2. **CORE ETFs demoted to WATCHLIST** - URTH, IWDA.AS, EUNL.DE, VOO, XEON.DE
   and the core set are now immutable (`CORE` status, never demoted).
3. **Contradictory registry state** - WATCHLIST symbols with `graduated_at`
   populated are cleared by the repair script.
4. **Delisted symbol retried forever** - ZNWD.L is now marked `DELISTED` after
   `MAX_FETCH_FAILURES` consecutive failures and excluded from fetching.
5. **Broker registry mostly empty** - ISINs synced into `asset_registry`; a
   missing ISIN now emits an explicit "ISIN MISSING" instruction.
6. **Inverse ETFs routed to Sparplan** - SDS and SH are marked `INVERSE` and
   never route to SPARPLAN (they decay over time).
7. **Degenerate ETF scoring** - `etf_quality_score()` and `etf_tactical_grade()`
   rank ETFs cross-sectionally, killing the 93.6 tie.
8. **Half-empty dashboard** - volatility bands now render, Z-scores are
   populated for ETFs, headers are plain text, `use_container_width` replaced.
9. **Stale data not surfaced** - the Data Health section flags prices older than
   `STALE_DATA_DAYS`.
10. **Static fee hurdle** - `min_trade_size_eur` now varies with expected alpha.

---

## Highlights

- **Universe state machine** - `CORE`/`ACTIVE`/`WATCHLIST`/`DELISTED` statuses,
  `structure` flags, grace period, delist tracking, `universe_events` audit log.
- **Registry repair** - [`scripts/repair_registry.py`](scripts/repair_registry.py)
  fixes the registry in one run.
- **ISIN validation** - `validate_isin()` checksum validator.
- **ETF scoring** - cross-sectional structural grade + continuous tactical grade.
- **Dashboard rebuild** - 3 pages, zero emojis, cached reads, rendered bands.
- **No-emoji lint** - [`test_no_emoji.py`](test_no_emoji.py) enforces the rule.

---

## What's New

1. **Universe State Machine** - `taxonomy.py`, `discovery.py`, `database.py`.
2. **Registry Repair** - `scripts/repair_registry.py`.
3. **Broker Registry & Routing** - `validate_isin`, `sync_broker_registry`,
   inverse exclusion, dynamic fee hurdle.
4. **Scoring Differentiation** - `etf_quality_score`, `etf_tactical_grade`,
   `etf_factor_scores`, per-tier funnel logs.
5. **Dashboard Rebuild** - 3 pages, `.streamlit/config.toml`, cached reads.
6. **Notifier** - zero-emoji message, explicit missing-config logging.
7. **Tests** - `test_phase5.py`, `test_no_emoji.py`.

---

## Test Suite

| Suite | Tests | Status |
|:---|:---:|:---|
| `test_scoring.py` | 20 | Pass |
| `test_factors.py` | 9 | Pass |
| `test_backtest_validity.py` | 6 | Pass |
| `test_phase4.py` | 20 | Pass |
| `test_phase5.py` | 12 | Pass |
| `test_no_emoji.py` | 1 | Pass |

---

# Release Notes - v10.1.0

**Quant-AI v10.1.0** - Broker-Aware Family Office Terminal

This release transitions the system from a theoretical research script into a
practical, broker-aware, solo-family-office wealth management terminal, aligned
with Trade Republic's asymmetric fee structure.

---

## Highlights

- **Core & Satellite universe** — [`data_updater.py`](data_updater.py) fetches
  only CORE ETFs + ACTIVE (graduated) universe + portfolio holdings (23 symbols),
  not the old 277-stock hardcoded set.
- **1-EUR fee asymmetry** — [`optimizer.py`](optimizer.py) `minimum_trade_size()`
  rejects signals below the 2 EUR round-trip hurdle.
- **Cash as risk-free baseline** — [`risk.py`](risk.py) uses the TR 2.25% APY
  daily yield in Sortino/Sharpe.
- **Smart Balance buckets** — Safety ≥ 10%, Core ≥ 40%, Alpha ≤ 50% as hard
  cvxpy constraints.
- **Bifurcated scoring** — [`taxonomy.py`](taxonomy.py) tags EQUITY/ETF/COMMODITY/
  CASH; ETFs bypass Fundamentals/NLP, scored on macro regime + trend.
- **Universe graduation** — [`discovery.py`](discovery.py) promotes watchlist
  anomalies to ACTIVE, demotes stale assets after 6 months.
- **Local UI & automation** — [`dashboard.py`](dashboard.py) (Streamlit),
  [`notifier.py`](notifier.py) (Telegram/Discord), [`setup_cron.sh`](setup_cron.sh).

---

## What's New

1. **Execution Reality** - `broker_registry.csv`, `minimum_trade_size()`, `routing.py`.
2. **Cash & Fee Mathematics** - `BROKER_CASH_APY`, `daily_risk_free_rate()`, bucket constraints.
3. **Asset Taxonomy** - `instrument_class`, `asset_registry` table, bifurcated scoring.
4. **Universe Management** - `discovery.py`, `watchlist.csv`, ACTIVE/WATCHLIST status.
5. **Local Interface** - 3-page Streamlit dashboard, Telegram/Discord notifier, cron.

## Bug Fixes

- `market_history` missing PRIMARY KEY caused `BinderException` on `INSERT OR REPLACE`.
- `data_updater.py` still fetched all 277 stocks — now Core & Satellite only.
- New symbols defaulted to ACTIVE — now default to WATCHLIST.
- `data_updater.py` missing `init_db` import (`NameError`).

---

# Release Notes - v10.0.0

**Quant-AI v10.0.0** - Production-Grade Quant Research Platform

This release transforms the project from a script-based scanner into a
production-grade quant research platform. Python remains the orchestrator; the
slow parts are pushed into Rust (Polars), SQL (DuckDB), and C++-backed tools
(cvxpy, selectolax).

---

## Highlights

- ~50-60% faster execution via the Smart Funnel (tiered execution).
- ~1000x faster volatility via EWMA (replaces per-asset GARCH MLE).
- ~90s saved by fitting the market-regime HMM once instead of per-asset.
- Faster NLP via batched FinBERT inference across all documents.
- No-lookahead backtests via point-in-time fundamentals.
- Realistic backtests with transaction costs and T+1 execution.

---

## What's New

### Performance
1. **Smart Funnel Architecture** - fundamental, technical, NLP tiered execution.
2. **EWMA Volatility** - `fast_volatility()` as primary volatility source.
3. **Market-Regime HMM** - fit once on a broad index, applied to all assets.
4. **Batched FinBERT** - `FinBERTBatchScorer` pools chunks into single forward passes.
5. **Polars Data Engine** - native DuckDB to Polars reads, vectorized Rust computation.
6. **Selectolax SEC Parser** - C-based parser, ~50x faster than BeautifulSoup.
7. **Incremental Data Updates** - fetch only new data after last stored date.

### Strategy Quality
8. **Vectorized Feature Engine** - `build_features.py` (momentum, SMA, vol, ADV, drawdown).
9. **Cross-Sectional Factor Scoring** - z-scores across the universe, sector-neutralized.
10. **Point-in-Time Fundamentals** - `fundamentals_history` table, no lookahead bias.
11. **Cost-Aware Backtest** - T+1 execution with 15bps round-trip costs.
12. **Portfolio Optimizer** - cvxpy mean-variance with Ledoit-Wolf covariance.

### Architecture
13. **Data Validation & Liquidity Filters** - `validation.py`.
14. **Run Artifacts & Structured Logging** - `artifacts.py`.
15. **Unit Tests** - `test_factors.py` (9 tests).

---

## Bug Fixes

- `kelly_position_size()` / `target_volatility_size()` missing config imports.
- DuckDB to Polars conversion (native `.pl()`, string `Date` parsing).
- Polars `group_by` tuple-key unpacking (VARCHAR[] cast error).

---

## Dependencies Added

`polars`, `pyarrow`, `selectolax`, `cvxpy`

---

## Test Suite

| Suite | Tests | Status |
|:---|:---:|:---|
| `test_scoring.py` | 20 | Pass |
| `test_backtest_validity.py` | 6 | Pass |
| `test_factors.py` | 9 | Pass |
| `test_db.py` | - | Pass |
| `test_market_data.py` | - | Pass |
| `test_async.py` | - | Pass |

---

## Known Pre-Existing Issues (not introduced in v10.0)

- `test_cache.py` asserts an `ICR` key that `_get_from_cache()` does not return.
- `test_e2e_state.py` requires `market_history` to be populated by `data_updater.py`.

---

## Documentation

- [`README.md`](README.md) - updated architecture, features, scoring model, setup, tests.
- [`CHANGELOG.md`](CHANGELOG.md) - full versioned changelog.
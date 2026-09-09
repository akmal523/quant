# Quant-AI v10.3.0 - Broker-Aware Family Office Terminal (EUR-Native)

A professional-grade Python pipeline for systematic multi-sector equity analysis. Uses a **Core & Satellite universe** (CORE ETFs + ACTIVE graduated equities + portfolio holdings) instead of a hardcoded 277-stock set, normalises global currencies to EUR, and scores assets using a cross-sectional factor model, market-regime HMM, EWMA volatility, batched FinBERT NLP sentiment, and sector-aware fundamental stewardship. Fully aligned with Trade Republic's asymmetric 1-EUR fee structure and 2.25% cash APY.

**Trade Republic Ready.** The engine detects native currency (USD, CHF, GBP, DKK, NOK, SEK, CAD, AUD, KRW, GBX) and converts to EUR using live FX rates from Yahoo Finance and the ECB (Frankfurter API).

**Architecture principle:** Python is the orchestrator; the slow parts are pushed into Rust (Polars), SQL (DuckDB), and C++-backed tools (cvxpy, selectolax).

---

## Architecture

```
quant/
├── main.py                 # Orchestration engine (Smart Funnel -> Batch FinBERT -> Multiprocessing -> Audit)
├── data_updater.py          # Incremental market data fetcher (ThreadPoolExecutor, 10x faster)
├── build_features.py        # Vectorized cross-sectional feature engine (Polars)
├── optimizer.py             # Portfolio optimizer (cvxpy, Ledoit-Wolf covariance + risk buckets)
├── validation.py            # Data validation + liquidity filters
├── artifacts.py             # Run artifacts + structured JSON logging
├── currency.py              # Real-time FX normalisation with graceful degradation
├── async_fetcher.py         # Concurrent SEC 8-K + News RSS network I/O
├── scoring.py               # Factor model, market-regime HMM, stewardship, capital allocation
├── sentiment.py             # Batched FinBERT NLP (NLPScorer + FinBERTBatchScorer)
├── indicators.py            # EWMA volatility (primary) + RSI + ATR
├── risk.py                  # Empirical VaR, Sortino/Sharpe (broker cash risk-free rate)
├── backtest.py              # WFO + cost-aware backtest (T+1 execution)
├── portfolio.py             # Portfolio audit with PnL tracking
├── fundamentals.py          # Hierarchical fundamentals + point-in-time history
├── database.py              # DuckDB thread-local connection management
├── universe.py              # 20-sector asset universe + ETF detection + geo risk tables
├── taxonomy.py              # Asset taxonomy (EQUITY/ETF/COMMODITY/CASH) + broker registry  [NEW]
├── routing.py               # Signal routing: Sparplan vs Active Trade + fee hurdle       [NEW]
├── discovery.py             # Universe graduation: watchlist -> ACTIVE scan               [NEW]
├── notifier.py              # Telegram/Discord daily push notification                    [NEW]
├── dashboard.py             # 3-page Streamlit local UI (Daily Briefing/Explorer/Universe)[NEW]
├── config.py                # Centralised runtime settings
├── reporting.py             # Terminal output + Excel/CSV export
├── sec_edgar.py             # SEC 8-K downloader (selectolax C-parser)
├── news.py                  # Yahoo Finance RSS headline fetcher
├── mailer.py                # Optional email reporting
│
├── broker_registry.csv      # yahoo_ticker -> ISIN / tr_ticker / exchange mapping        [NEW]
├── watchlist.csv            # Satellite universe (potential graduates)                   [NEW]
├── setup_cron.sh            # Cron installer (daily 18:00 CET + weekly discovery)        [NEW]
│
├── test_scoring.py          # 20 unit tests for scoring engine
├── test_backtest_validity.py # 6 validation tests for backtesting
├── test_factors.py          # 9 tests: factor model, no-lookahead, costs, liquidity
├── test_cache.py            # Fundamentals cache + ICR validation
├── test_db.py               # DuckDB schema I/O validation
├── test_async.py            # Concurrent text fetch speed validation
├── test_e2e_state.py        # End-to-end pipeline state verification
├── test_market_data.py      # DuckDB <-> Pandas bridge test
│
├── portfolio.csv            # Your Trade Republic holdings
├── quant_cache.duckdb       # Unified OLAP storage (market history + fundamentals + NLP)
├── outputs/                 # Market scan reports + run artifacts (run_<timestamp>/)
└── plans/                   # Engineering change proposals
```

---
## New in v10.2

> **Dashboard clarity release.** The universe state machine no longer fights
> itself (no graduate-then-demote contradictions), the broker ISIN registry is
> complete, ETF scoring is differentiated, and the dashboard is rebuilt with
> zero emoji characters.

### 1. Universe State Machine
- Statuses `CORE` / `ACTIVE` / `WATCHLIST` / `DELISTED`; `structure` flags
  (`PLAIN` / `INVERSE` / `LEVERAGED`).
- CORE is immutable; new graduates get a `GRADUATION_GRACE_MONTHS` grace period.
- `fetch_failures` tracking marks a symbol `DELISTED` after `MAX_FETCH_FAILURES`.
- `universe_events` audit table logs every state change.

### 2. Broker Registry & Routing
- `validate_isin()` checksum validator; `sync_broker_registry()` populates ISINs.
- INVERSE/LEVERAGED products never route to SPARPLAN.
- Missing ISIN emits an explicit "ISIN MISSING" instruction.
- Dynamic fee hurdle: `min_trade_size_eur` varies with expected alpha.

### 3. Scoring Differentiation
- `etf_quality_score()` cross-sectional ETF structural grade.
- `etf_tactical_grade()` continuous tactical grade (kills the 93.6 tie).
- `etf_factor_scores()` populates the dashboard Z-score section for ETFs.

### 4. Dashboard Rebuild (3 pages, zero emojis)
- Daily Briefing, Asset Explorer, Universe Manager.
- `.streamlit/config.toml` disables telemetry and runs headless.
- All reads cached via `@st.cache_data(ttl=300)`; `width="stretch"` replaces
  `use_container_width`.

### 5. Notifier & Tests
- Zero-emoji message text; explicit missing-config logging.
- `test_phase5.py` + `test_no_emoji.py` enforce the fixes permanently.

---

## New in v10.2.1 — Strategic Portfolio Rebalancing

> **Strategy release.** The bot stops treating all assets equally. It now
> classifies each holding into a management tier (CORE / SATELLITE / ACTIVE /
> SECTOR), rebalances only on meaningful drift, respects the 2 EUR round-trip
> fee, and never emits SELL on core buy-and-hold ETFs.

### 1. Asset Tier Classification
- [`config.py`](config.py) defines `CORE_ASSETS` / `SATELLITE_ASSETS` /
  `ACTIVE_ASSETS` / `SECTOR_ASSETS`. Tier lists take **precedence** over
  `CORE_ETFS` (e.g. `SXRV.DE` is in `CORE_ETFS` but classified SATELLITE).
- [`portfolio.py`](portfolio.py) `classify_asset()` maps a symbol to one tier;
  unknown symbols default to ACTIVE.

### 2. Drift-Based Rebalancing
- `TARGET_WEIGHTS` (CORE 50% / SATELLITE 20% / ACTIVE 20% / SECTOR 10%).
- `should_rebalance_asset()` rebalances only when drift exceeds the per-tier
  threshold (`REBALANCE_DRIFT_TIERS`) AND the frequency window
  (`REBALANCE_FREQUENCY_DAYS`) has elapsed.
- **First-run ease-in**: no `rebalance_log` row = baseline recorded, no forced
  trade. Set `REBALANCE_FIRST_RUN = True` to force rebalancing to targets.

### 3. CORE Never Sells
- [`scoring.py`](scoring.py) `generate_signal_for_tier()` overrides the generic
  signal. CORE assets emit only `HOLD` or `BUY MORE (DCA OK)` — never SELL.
- SATELLITE trims only on >10% overweight; ACTIVE keeps the full tactical
  spectrum; SECTOR rotates cyclically.

### 4. Fee & Liquidity Awareness
- [`optimizer.py`](optimizer.py) `calculate_min_trade_size()` returns the max of
  (drift value, fee-hurdle capital, `MIN_TRADE_SIZE_EUR`).
- `check_volume_liquidity()` rejects trades exceeding 1% of 20-day ADV.

### 5. Rebalance Log
- [`database.py`](database.py) adds the `rebalance_log` table
  (`symbol`, `last_rebalance_date`) for time-gated rebalancing.

### 6. Tests
- `test_rebalancing.py` (13 tests): tier classification, fee hurdle, per-tier
  rebalance logic, CORE never SELL, drift thresholds, first-run baseline.

---

## New in v10.2.2 — Advanced Strategic Enhancements

> **Adaptive portfolio manager release.** Part 1 differentiated CORE vs ACTIVE
> assets. Part 2 adds portfolio-level context, a multi-strategy ensemble,
> tax-loss harvesting, dynamic cash management, drawdown circuit breakers,
> P&L attribution, an event bus, overtrading guardrails, and regime-aware
> backtest validation.

### 1. Risk-Aware Portfolio Context
- [`portfolio_context.py`](portfolio_context.py) `PortfolioContext` computes
  marginal risk contribution (MRC), PCA factor exposure, and a concentration
  penalty that modulates asset scores. Catches hidden concentration risk
  (e.g. SXRV.DE + AMZN ~80% correlated).

### 2. Multi-Strategy Ensemble
- [`strategies/`](strategies/) — `Momentum`, `MeanReversion`, `Value`,
  `RiskParity` strategies.
- [`strategy_engine.py`](strategy_engine.py) `StrategyEngine` blends signals
  with regime-dependent weights. In high-vol regimes it shifts from momentum
  to mean-reversion; in bear markets value + risk-parity dominate.

### 3. German Tax-Loss Harvesting
- [`tax_optimizer.py`](tax_optimizer.py) `TaxOptimizer` applies Abgeltungsteuer
  (26.375%), the EUR 1,000 Sparerpauschbetrag, and loss-offset rules. Proposes
  harvesting non-CORE losers with a correlated replacement when tax savings
  clear the 2 EUR fee by a 3x buffer.

### 4. Dynamic Cash Reserve
- [`cash_manager.py`](cash_manager.py) `CashManager` targets cash allocation
  from regime + VIX + opportunity set (5-30%), and scales dip-buying with
  drawdown depth.

### 5. Drawdown Circuit Breakers
- [`risk_monitor.py`](risk_monitor.py) `RiskMonitor` returns NORMAL / CAUTION /
  ALERT / LOCKDOWN based on drawdown and volatility, with recommended actions.

### 6. P&L Attribution
- [`attribution.py`](attribution.py) `BrinsonFachlerAttribution` decomposes
  P&L into allocation / selection / interaction effects vs a benchmark.

### 7. Event-Driven Architecture
- [`event_bus.py`](event_bus.py) `EventBus` pub/sub decouples modules so they
  react to regime change, drawdown, tax opportunity, and dip events.

### 8. Behavioral Guardrails
- [`behavioral_guardrails.py`](behavioral_guardrails.py) enforces per-symbol
  cooldowns, a weekly trade limit, and size reduction after consecutive losses.

### 9. Regime-Aware Validation
- [`validation_engine.py`](validation_engine.py) `ValidationEngine` runs
  walk-forward backtests with regime detection and reports robustness metrics.

### 10. Unified Briefing
- [`reporting_advanced.py`](reporting_advanced.py) assembles all modules into a
  single daily briefing, wired into [`main.py`](main.py) (non-fatal).

### 11. Tests
- `test_advanced.py` (15 tests): risk contribution, concentration penalty,
  strategy ensemble, tax harvesting, cash bounds, circuit breakers,
  attribution, event bus, guardrails, validation robustness.

---

## New in v10.3.0 — Architectural Refinement

> **Hygiene release.** Removes complexity, modernizes patterns, and closes
> gaps: data quality validation, feature caching, observability, incremental
> processing, YAML config, alerts, health score, and what-if scenarios.

### 1. Data Quality Gate (P0)
- [`data_quality.py`](data_quality.py) `DataQualityValidator` validates
  incoming market data (NaN, negative prices, extreme moves, duplicates,
  staleness) BEFORE it enters DuckDB. `auto_repair()` fixes common issues.
- Wired into [`data_updater.py`](data_updater.py) — unfixable data is skipped.

### 2. Feature Cache (P0)
- [`feature_cache.py`](feature_cache.py) `FeatureCache` caches computed
  indicators keyed by `hash(symbol + feature + data_hash)`. Invalidates when
  the underlying Close data changes. Cuts incremental computation 60-80%.

### 3. Observability (P0)
- [`observability.py`](observability.py) `ObservabilityCollector` times each
  pipeline step, records errors, and prints a summary + JSON export.
- Wired into [`main.py`](main.py) — prints a pipeline timing summary.

### 4. Incremental Processing (P1)
- [`incremental.py`](incremental.py) `IncrementalProcessor` detects which
  symbols changed (via data hash) so the pipeline is O(changed) not O(all).

### 5. YAML Config (P1)
- [`config_loader.py`](config_loader.py) `Config` loads [`config.yaml`](config.yaml)
  with nested dot-path access. Non-programmers can tune thresholds without
  editing Python.

### 6. Alert System (P2)
- [`alerts.py`](alerts.py) `AlertSystem` surfaces drawdowns, rebalancing
  triggers, and tax-loss opportunities as leveled alerts.

### 7. Portfolio Health Score (P2)
- [`health_score.py`](health_score.py) `PortfolioHealthScore` collapses
  diversification, risk-adjusted return, drawdown, cost, and liquidity into a
  single 0-100 score with a grade and recommendations.

### 8. What-If Scenarios (P2)
- [`scenario_simulator.py`](scenario_simulator.py) `ScenarioSimulator` answers
  "what if I sell X buy Y" and "what if the market crashes 20%" before
  committing capital.

### 9. Tests
- `test_part3.py` (14 tests): data quality, feature cache, observability,
  incremental, YAML config, alerts, health score, scenario simulator.

---

## New in v10.1

> **Universe model change:** [`data_updater.py`](data_updater.py) no longer fetches
> all 277 `SECTOR_UNIVERSE` stocks. It fetches only **CORE ETFs + ACTIVE
> (graduated) universe + portfolio holdings** via `build_fetch_list()`. New
> symbols default to `WATCHLIST`; only `discovery.py` graduation or manual pin
> promotes them to `ACTIVE`. This keeps the heavy-analysis universe lean.

### 1. Execution Reality (Trade Republic Integration)
- **1-EUR fee asymmetry** — [`optimizer.py`](optimizer.py) adds `minimum_trade_size()` and `passes_fee_hurdle()`. Formula: `Min Capital = (Round_Trip_Fee / Alpha_BPS) * 10000`. A 200 bps alpha needs ≥ 100 EUR to clear the 2 EUR round-trip fee.
- **ISIN mapping** — [`broker_registry.csv`](broker_registry.csv) maps `yahoo_ticker` → `isin` / `tr_ticker` / `exchange` / `currency`. [`data_updater.py`](data_updater.py) fetches the LS Exchange ticker for execution-relevant local prices.
- **Signal routing** — [`routing.py`](routing.py) routes high-structural/low-tactical to **Sparplan** (0 EUR buy) and high-tactical to **Active Trade** (1 EUR).

### 2. Portfolio Architecture (Buckets & Cash)
- **Cash as risk-free baseline** — [`config.py`](config.py) `BROKER_CASH_APY = 0.0225`. [`risk.py`](risk.py) converts to daily yield `(1+APY)^(1/365)-1` and uses it in Sortino/Sharpe.
- **Smart Balance buckets** — [`optimizer.py`](optimizer.py) enforces `Safety ≥ 10%`, `Core ≥ 40%`, `Alpha ≤ 50%` as hard cvxpy inequality constraints.

### 3. Asset Taxonomy & Universe Management
- **Bifurcated scoring** — [`taxonomy.py`](taxonomy.py) tags `EQUITY`/`ETF`/`COMMODITY`/`CASH`. [`main.py`](main.py) bypasses Fundamentals/NLP for ETFs/commodities, scoring them on macro regime + trend + relative strength.
- **Graduation universe** — [`discovery.py`](discovery.py) scans [`watchlist.csv`](watchlist.csv) weekly (5-day data only). 52-week-high or 3x-volume anomalies graduate to ACTIVE; stale ACTIVE assets demote after 6 months.

### 4. Local Interface & Automation
- **Streamlit dashboard** — [`dashboard.py`](dashboard.py): Daily Briefing, Asset Explorer (Plotly + GARCH bands), Universe Manager.
- **Push notifications** — [`notifier.py`](notifier.py) sends daily Telegram/Discord summaries (trades, cash allocation, risk warnings).
- **Cron** — [`setup_cron.sh`](setup_cron.sh) installs daily 18:00 CET + weekly discovery jobs.

---

## New in v10.0

### 1. Smart Funnel Architecture (Tiered Execution)
[`main.py`](main.py) now runs a 3-tier funnel instead of scoring everything:
- **Tier 1 (us):** Fast fundamental filter (PE, ROE).
- **Tier 2 (ms):** Fast technical filter - `Close > 200 SMA` uptrend check (vectorized).
- **Tier 3 (s):** Heavy NLP/SEC scraping only on funnel survivors.

This cuts total execution time by 50-60% by skipping expensive NLP on assets with poor technicals.

### 2. EWMA Volatility (replaces per-asset GARCH MLE)
[`indicators.py`](indicators.py) adds `fast_volatility()` - vectorized EWMA (~1000x faster than GARCH MLE) and makes it the primary volatility source in `add_all_indicators()`. GARCH retained as an optional fallback path.

### 3. Market-Regime HMM (fit once, not per-asset)
[`scoring.py`](scoring.py) adds `fit_market_regime()` - fits a GaussianHMM **once** on a broad index (SPY or longest-history proxy) to derive the macro bull probability, then applies it to all assets. Eliminates ~90s of per-asset HMM fitting and is statistically sounder.

### 4. Batched FinBERT Inference
[`sentiment.py`](sentiment.py) adds `FinBERTBatchScorer.score_texts()` - pools chunks from all documents, pads them into a single tensor, and runs one forward pass per batch (AVX2 vectorization). Also introduces the OOP `NLPScorer` with dependency injection.

### 5. Polars Data Engine
[`main.py`](main.py) reads DuckDB to Polars natively (`.pl()`), computing log-returns vectorized across all symbols in Rust. No pandas intermediate, no GIL.

### 6. Selectolax SEC Parser
[`sec_edgar.py`](sec_edgar.py) replaces BeautifulSoup with the C-based `selectolax` parser (~50x faster on multi-MB SEC filings).

### 7. Incremental Data Updates
[`data_updater.py`](data_updater.py) now fetches only data after each symbol's `MAX(Date)` (with overlap) instead of re-downloading 5 years every run. Reduces update time from minutes to seconds.

### 8. Vectorized Feature Engine
[`build_features.py`](build_features.py) computes momentum (1d/1m/6m/12m), SMA20/200, vol20/60, ADV20, uptrend, trend_strength, and max-drawdown across the whole universe in one Polars pass.

### 9. Cross-Sectional Factor Scoring
[`scoring.py`](scoring.py) adds `factor_scores()` - z-scores value/quality/momentum/low-risk/sentiment across the universe and combines with weights (0.25/0.25/0.20/0.15/0.15), replacing hardcoded absolute thresholds. `sector_neutral_rank()` neutralizes sector bias.

### 10. Point-in-Time Fundamentals (No Lookahead)
[`database.py`](database.py) adds the `fundamentals_history` table; [`fundamentals.py`](fundamentals.py) adds `save_fundamentals_history()` and `get_fundamentals_as_of()` which filter `published_date <= as_of_date` - eliminating lookahead bias in backtests.

### 11. Cost-Aware Backtest
[`backtest.py`](backtest.py) adds `run_cost_aware_backtest()` - executes at T+1 open, applies 15bps round-trip cost, reports net/gross PnL and cost drag.

### 12. Portfolio Optimizer
[`optimizer.py`](optimizer.py) adds `optimize_portfolio()` (cvxpy) with Ledoit-Wolf shrunk covariance and max-weight/sector/turnover constraints - replacing independent BUY/SELL labels with risk-aware weights.

### 13. Data Validation & Liquidity Filters
[`validation.py`](validation.py) adds `validate_market_data()`, `sanitize_fundamentals()`, `liquidity_score()`, `is_liquid()`, and `max_trade_size()` to prevent bad data from corrupting signals.

### 14. Run Artifacts & Structured Logging
[`artifacts.py`](artifacts.py) writes timestamped `outputs/run_<ts>/` directories (factor scores, NLP scores, run config, metrics as parquet/JSON) and provides a JSON `StructuredLogger`.

### 15. Fixed Pre-Existing Bugs
- [`scoring.py`](scoring.py) `kelly_position_size()` and `target_volatility_size()` referenced `KELLY_FRACTION`/`TARGET_VOLATILITY`/`MAX_POSITION_PCT` without importing them - added local imports (unblocked `test_scoring.py`).

---

## New in v9.0

### Robust NaN Data Handling
Yahoo Finance often appends future trading dates with NaN prices as the last row of `history()` output. `data_updater.py` now uses `df.dropna(subset=['Close'])` before reading the latest close price, ensuring all fetched tickers have valid price data regardless of trailing NaN rows.

Assets with no valid price data are no longer silently dropped. `main.py` returns skeleton scan results for such symbols (rather than `None`), and `portfolio.py` shows `"NO DATA"` in the portfolio audit instead of `"NOT SCANNED"` with NaN PnL.

### FinBERT Runs Once in Main Process
Previous versions loaded the FinBERT NLP model in **every multiprocessing worker** (4 workers x ~800MB = 3.2GB RAM). Now FinBERT is initialised once in the main process before the worker pool starts. Workers receive pre-computed `nlp_data` dicts - no model loading, no `init_worker()` needed. **~2.4GB RAM saved.**

### Proper Logging Infrastructure
All `print()` debugging replaced with structured `logging.info/warning/exception` calls throughout the pipeline. Timestamped output (`HH:MM:SS [LEVEL] message`), with worker crashes captured via `logger.exception()` for full tracebacks.

### Comprehensive Unit Test Suite
- **20 tests** for the scoring engine: HMM regime detection, stewardship scoring (general & financials), structural/tactical grade composition, capital allocation (CORE/SPECULATIVE/HOLD), Kelly/volatility position sizing, fast filter boundaries.
- **6 tests** for the backtesting engine: survivorship bias warning emission, WFO window alignment, insufficient-data handling, macro/historical backtest edge cases.

### GARCH Scale Stability
Auto-scales log-returns to unit variance before GARCH(1,1) fitting, suppressing `DataScaleWarning` for low-price assets (e.g. ETFs trading at 0.67 EUR). Falls back to EWMA when GARCH cannot converge or data < 252 observations.

### Parallel Data Updater
`data_updater.py` rewritten with `ThreadPoolExecutor` (10 workers). The Core & Satellite universe (CORE ETFs + ACTIVE + portfolio) is fetched in seconds instead of minutes.

### Graceful FX Degradation
`currency.py` no longer crashes the entire scan when EUR/USD rate cannot be fetched (ECB API + Yahoo both down). Falls back to 1.0 with a warning instead of `RuntimeError`.

### Clean Configuration
Removed 7 redundant threshold constants from `config.py` that were duplicated under different names (e.g. `MIN_ROE`/`FILTER_MIN_ROE`, `MAX_PE`/`FILTER_MAX_PE`, `GEN_MAX_DE`/`STW_GEN_MAX_DE`). One source of truth per parameter.

### Better NLP Reasoning
"No text data found" changed to `"No SEC/News data available - neutral score applied"` with debug-level logging to distinguish genuine data gaps from scoring issues.

### Fixed Critical Bugs
- **NaN close price crash** in `data_updater.py` - Yahoo Finance appends future trading dates with NaN prices as the last row. `df['Close'].iloc[-1]` picked up NaN for ALL US-listed tickers, blocking 174/277 instruments. Fixed with `df.dropna(subset=['Close'])` before accessing the latest close.
- **Portfolio symbols silently dropped** from scan results in `main.py` - `process_asset()` returned `None` on NaN close, causing portfolio holdings to show as `"NOT SCANNED"` in the audit. Fixed by returning a skeleton result with `Active_Score: 0.0` and `Signal: N/A`.
- **Missing portfolio symbols fallback** - symbols with no market data at all (not in DuckDB) now get a placeholder entry in scan results instead of being silently absent.
- **Duplicate data loading** in `main.py` - market data was loaded twice, the second load discarding chronological sorting. HMM/GARCH received scrambled data.
- **`init_db()` never called** - NLP cache `INSERT` could crash with `CatalogException`.
- **`backtest.py` import error** - `from indicators import rsi, atr` referenced non-existent functions. Both implemented with proper EWMA computation.

---

## Features

- **Core & Satellite Universe** - Fetches only CORE ETFs + ACTIVE (graduated) equities + portfolio holdings, not a hardcoded 277-stock set. New symbols default to `WATCHLIST`; `discovery.py` graduates anomalies to `ACTIVE`.
- **Bifurcated Scoring** - [`taxonomy.py`](taxonomy.py) tags `EQUITY`/`ETF`/`COMMODITY`/`CASH`. ETFs/commodities bypass Fundamentals/NLP and score on macro regime + trend + relative strength.
- **Trade Republic Execution** - 1-EUR fee asymmetry via `minimum_trade_size()`; Sparplan vs Active Trade routing; ISIN/`tr_ticker` resolution via `broker_registry.csv`.
- **Smart Balance Buckets** - Safety ≥ 10%, Core ≥ 40%, Alpha ≤ 50% hard constraints in the cvxpy optimizer; cash earns the 2.25% broker APY as the risk-free rate.
- **20-Sector Universe** - Uranium, Energy, Oil & Gas, Defense, Cybersecurity, Gold, Silver, Copper, Lithium, Quantum, Semiconductors, AI/Cloud, Logistics, Banking, Insurance, Healthcare, Water, Agriculture, Real Estate, Broad ETFs.
- **Universal ETF Detection** - Automatically identifies ETFs by sector, display name keywords, and hardcoded fallbacks. ETFs bypass corporate fundamental filters and score purely on market-regime momentum.
- **Global Sentiment Fallback** - US equities scored via SEC 8-K filings; international assets fall back to News RSS.
- **Batched Local FinBERT NLP** - Air-gapped sentiment analysis ([ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert)). `FinBERTBatchScorer` pools chunks across documents into single vectorized forward passes. No API calls.
- **Market-Regime HMM** - Gaussian HMM fit once on a broad index to derive the macro Bull/Bear probability, applied to all assets.
- **Cross-Sectional Factor Model** - Value/Quality/Momentum/Low-Risk/Sentiment z-scores combined with weights, replacing hardcoded thresholds. Sector-neutralized ranking.
- **Smart Funnel** - Tiered execution: fundamental filter, technical uptrend filter, heavy NLP only on survivors.
- **Stewardship Override** - The "Quality Floor": even stocks with strong momentum are downgraded to SPECULATIVE if Debt-to-Equity exceeds limits or Interest Coverage Ratio is inadequate.
- **Dynamic Currency Normalisation** - OHLCV data auto-converted to EUR using live FX rates. Supports USD, CHF, GBP, GBX, DKK, NOK, SEK, CAD, AUD, KRW.
- **Portfolio Audit** - Cross-references `portfolio.csv` against live data. Issues BUY MORE (DCA OK), HOLD, URGENT SELL directives with accurate EUR-native PnL.
- **Cost-Aware Backtest** - T+1 execution with 15bps round-trip costs; reports net/gross PnL and cost drag.
- **Portfolio Optimizer** - cvxpy mean-variance with Ledoit-Wolf covariance, max-weight/sector/turnover constraints.
- **Point-in-Time Fundamentals** - `fundamentals_history` table with `published_date` filtering to eliminate lookahead bias.
- **Walk-Forward Optimisation** - Rolling in-sample / out-of-sample windows prove strategy generalisation. Survivorship bias warning emitted on every call.
- **DuckDB + Polars Analytical Core** - High-performance OLAP storage (340K+ rows) with vectorized Rust feature computation.
- **Run Artifacts** - Each run writes `outputs/run_<timestamp>/` with factor scores, NLP scores, config, and metrics.

---

## Scoring Model (v10.1)

| Category | Weight | Components |
|:---|:---:|:---|
| **Fundamentals** | 30% | PE, PEG, ROE - reward capital efficiency |
| **Stewardship** | 30% | D/E, P/B, ICR - sector-aware balance sheet stress test |
| **Technical (HMM)** | 15% | EWMA volatility -> market-regime HMM -> Bull/Bear probability |
| **Sentiment (FinBERT)** | 25% | SEC 8-K or News RSS -> batched attention-masked inference |

> **v10.0 factor model:** The pipeline also exposes a cross-sectional factor score
> (`factor_scores()` in `scoring.py`) that z-scores Value/Quality/Momentum/Low-Risk/
> Sentiment across the universe and combines them with weights
> (0.25/0.25/0.20/0.15/0.15), with optional sector-neutralized ranking.

### Horizon & Signal Logic

| Horizon | Requirements | Signal Logic |
|:---|:---|:---|
| **CORE (12-Month)** | Structural >= 75, Stewardship >= 15, Tactical >= 60 | BUY if all met; score = 0.4xstruct + 0.6xtact |
| **HOLD** | Stewardship >= 15 but structural < 75 | HOLD; score = structural grade |
| **SPECULATIVE** | Stewardship < 15 OR structural < 50 | BUY if Tactical >= 70, else SELL |

---

## Setup & Execution

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/quant.git
cd quant
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Update Market Data

```bash
python3 data_updater.py
```

Fetches the **Core & Satellite universe** (CORE ETFs + ACTIVE graduated equities +
portfolio holdings) with 10 parallel workers. **Incremental mode:** on subsequent
runs it fetches only data after each symbol's last stored date (with a 5-day
overlap), reducing update time from minutes to seconds. New symbols default to
`WATCHLIST`; run `discovery.py` weekly to graduate anomalies to `ACTIVE`.

### 3. Configure Portfolio

Edit `portfolio.csv` with your Trade Republic holdings (EUR cost basis):

```csv
Symbol,Buy_Price,Amount_EUR
IWDA.AS,120.50,115.00
HEI.DE,185.20,200.00
LMT,509.88,75.00
RHO.DE,356.40,48.26
```

Comments after `#` are automatically stripped.

### 4. Run the Scan

```bash
python3 main.py
```

### 5. Run the Streamlit Dashboard

```bash
streamlit run dashboard.py
```

The dashboard has three pages: **Daily Briefing**, **Asset Explorer**, and
**Universe Manager**. It reads directly from DuckDB with cached reads
(`@st.cache_data(ttl=300)`). The `.streamlit/config.toml` in the repo root
disables telemetry and runs headless, so no browser opens on the server and no
usage-stats email prompt appears. Run it from the repo root so Streamlit picks
up the config.

### 6. Weekly Universe Discovery

```bash
python3 discovery.py
```

### 7. Daily Automation (optional)

```bash
bash setup_cron.sh   # installs daily 18:00 CET + weekly discovery cron jobs
```

---

## Output Signals

| Decision | Condition |
|:---|:---|
| **BUY MORE (DCA OK)** | High quality (Active Score > 80) and position at or below entry, or within high-conviction window |
| **HOLD** | Fundamentals remain strong but tactical timing or profit level suggests waiting |
| **URGENT SELL** | Significant fundamental decay (low stewardship) or extreme negative sentiment |
| **NO DATA** | Symbol found in portfolio but has no valid price data (e.g. Yahoo Finance unavailable) |
| **NOT SCANNED** | Asset not found in current market universe |

---

## Configuration

All tunable parameters in [`config.py`](config.py):

| Parameter | Default | Description |
|:---|---:|:---|
| `WEIGHT_FUNDAMENTALS` | 30 | PE + PEG + ROE contribution |
| `WEIGHT_STEWARDSHIP` | 30 | D/E + ICR + Payout policy |
| `WEIGHT_TECHNICAL` | 15 | HMM regime score |
| `WEIGHT_SENTIMENT` | 25 | FinBERT NLP sentiment |
| `FILTER_MAX_PE` | 25.0 | Screener PE ceiling |
| `FILTER_MIN_ROE` | 0.20 | Screener ROE floor |
| `STW_GEN_MAX_DE` | 0.5 | Max D/E for non-financials |
| `STW_GEN_MIN_ICR` | 5.0 | Min interest coverage ratio |
| `MIN_STRUCT_GRADE_FOR_BUY` | 75 | Minimum structural grade for CORE classification |
| `MIN_TACT_GRADE_FOR_BUY` | 70 | Minimum tactical grade for SPECULATIVE BUY |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly position sizing |
| `TARGET_VOLATILITY` | 0.15 | 15% annualised target vol |
| `MAX_POSITION_PCT` | 0.10 | 10% hard cap per position |
| `BROKER_CASH_APY` | 0.0225 | TR cash yield (2.25%) as risk-free rate |
| `ROUND_TRIP_FEE_EUR` | 2.0 | Active trade round-trip fee (1 EUR buy + 1 EUR sell) |
| `SAFETY_BUCKET_MIN` | 0.10 | Safety bucket floor (cash & short-term bonds) |
| `CORE_BUCKET_MIN` | 0.40 | Core bucket floor (broad ETFs via Sparplan) |
| `ALPHA_BUCKET_MAX` | 0.50 | Alpha bucket ceiling (active equities) |
| `SPARPLAN_STRUCT_MIN` | 75.0 | Structural grade threshold for Sparplan routing |
| `ACTIVE_TACT_MIN` | 70.0 | Tactical grade threshold for Active Trade routing |
| `WATCHLIST_VOLUME_MULT` | 3.0 | Volume > 3x 20-day avg graduates to ACTIVE |
| `ACTIVE_DEMOTE_MONTHS` | 6 | No signals for 6 months demotes to WATCHLIST |
| `TARGET_WEIGHTS` | 50/20/20/10 | Per-tier target weights (CORE/SATELLITE/ACTIVE/SECTOR) |
| `REBALANCE_DRIFT_TIERS` | 10/7.5/5/6% | Per-tier drift threshold that triggers rebalance |
| `REBALANCE_FREQUENCY_DAYS` | 90/30/1/14 | Per-tier min days between rebalances |
| `MIN_TRADE_SIZE_EUR` | 50.0 | Minimum trade to clear the 2 EUR round-trip fee |
| `REBALANCE_FIRST_RUN` | False | True = force rebalance to targets on first run |

---

## Test Suite

```bash
# Scoring engine (20 tests)
python3 test_scoring.py

# Backtesting validity (6 tests)
python3 test_backtest_validity.py

# Factor model, no-lookahead, costs, liquidity (9 tests)
python3 test_factors.py

# Phase 4: fee hurdle, cash rf, routing, taxonomy, graduation (20 tests)
python3 test_phase4.py

# Phase 5: state machine, ISIN, inverse routing, ETF scoring (12 tests)
python3 test_phase5.py

# v10.2.1: tier classification, fee hurdle, rebalance logic, CORE never SELL (13 tests)
python3 test_rebalancing.py

# v10.2.2: portfolio context, strategy ensemble, tax, cash, risk, attribution (15 tests)
python3 test_advanced.py

# v10.3.0: data quality, feature cache, observability, alerts, health, scenarios (14 tests)
python3 test_part3.py

# No-emoji lint (dashboard/notifier/reporting/main)
python3 test_no_emoji.py

# Database I/O
python3 test_db.py

# Fundamentals cache + ICR
python3 test_cache.py

# End-to-end pipeline state
python3 test_e2e_state.py
```

---

## Disclaimer

All output is for informational purposes. Probabilistic models and NLP sentiment analysis involve inherent risk. **Past performance does not guarantee future results.** The universe contains only currently-listed instruments - historical backtest figures are systematically overstated due to survivorship bias.

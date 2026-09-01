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
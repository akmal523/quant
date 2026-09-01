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
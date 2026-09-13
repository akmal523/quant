# Quant-AI

**Broker-aware, EUR-native systematic equity analysis for a solo family office on Trade Republic.**

[![Version](https://img.shields.io/badge/version-10.4.0-blue)](CHANGELOG.md)
[![CI](https://github.com/akmal523/quant/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

Quant-AI scans a smart 1000+ ticker universe, filters it to a handful of
survivors, and produces EUR-native buy/hold/sell guidance plus a broker-synced
portfolio audit. Python orchestrates; the heavy lifting runs in Rust (Polars),
SQL (DuckDB), and C++-backed libraries (cvxpy, selectolax).

> **New here?** Read [`CONTEXT.md`](CONTEXT.md) for the domain vocabulary, data
> contracts, and architecture map. It is the canonical reference for every term
> used below.

---

## Overview

| Capability | Summary |
|:--|:--|
| **Smart universe** | S&P 500 + Nasdaq-100 + Russell 1000 + broad ETFs (~1080 symbols), two-stage funnel to the top ~24 survivors. |
| **Bifurcated scoring** | Equities run fundamentals + NLP; ETFs/commodities score on macro regime + trend. |
| **Trade Republic aware** | ISIN/`tr_ticker` routing, Sparplan vs Active Trade, 1 EUR asymmetric fee, 2.25% cash APY as the risk-free rate. |
| **Broker-synced** | Portfolio PnL is copied from the broker, never price-guessed. |
| **EUR-native** | Global currencies normalised with live FX (Yahoo + ECB). |
| **Institutional-grade** | Bitemporal PIT data, hard assertions, TCA, Mean-CVaR, kill switch, golden-file CI. |

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional NER

# 2. Configure
#    data/portfolio.csv   -> your Trade Republic holdings
#    .env.example         -> optional notifier / API keys
#    quant/config.py      -> all tunable thresholds

# 3. Run (two steps)
python3 data_updater.py   # step 1: fetch data + cache funnel survivors
python3 main.py           # step 2: score, audit, report

# 4. Dashboard
streamlit run quant/dashboard.py

# 5. Automate (optional)
bash scripts/setup_cron.sh   # daily 18:00 CET + weekly discovery
```

### Portfolio file (`data/portfolio.csv`)

```csv
Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR
EUNL.DE,125.03,281.25,12.00
AMZN,220.55,150.41,5.20
```

`Invested_EUR = Current_Value_EUR − Broker_PnL_EUR` is derived; `Broker_PnL_EUR`
is broker truth, never price-guessed. Comments after `#` are stripped.

---

## Architecture

### Pipeline (two steps)

```
quant.data.universe_builder ─► universe_master (1000+ symbols, DuckDB)
        │
        ▼
quant.data.data_updater ─► corporate actions ─► quality gate ─► assertions
        │                          └─► market_history (DuckDB)
        ▼
quant.main ─► scoring ─► portfolio audit ─► reports + notifications
        ▲
quant.dashboard (Streamlit) ── reads DuckDB
```

Step 1 (`data_updater.py`) fetches market data, adjusts corporate actions, runs
the hard data-quality gate, and caches the funnel survivors. Step 2 (`main.py`)
reads that cache, scores, audits, and reports. `main.py` does **not** re-fetch
the universe.

### Key modules

| Module | Responsibility |
|:--|:--|
| [`quant/main.py`](quant/main.py) | Orchestration: load → filter → NLP → score → audit → report |
| [`quant/data/data_updater.py`](quant/data/data_updater.py) | Incremental DuckDB market data + funnel; caches survivors |
| [`quant/data/universe_builder.py`](quant/data/universe_builder.py) | Builds `universe_master` from index constituents + ETFs |
| [`quant/data/funnel.py`](quant/data/funnel.py) | Two-stage filter (liquidity → momentum) + survivor cache |
| [`quant/analytics/scoring.py`](quant/analytics/scoring.py) | Factor model, regime HMM, stewardship, capital allocation |
| [`quant/portfolio/portfolio.py`](quant/portfolio/portfolio.py) | Broker-synced audit, EUR PnL, reconciliation |
| [`quant/portfolio/optimizer.py`](quant/portfolio/optimizer.py) | cvxpy Mean-Variance + Mean-CVaR with bucket constraints |
| [`quant/strategy/backtest.py`](quant/strategy/backtest.py) | Walk-forward, cost-aware (T+1) backtest |
| [`quant/execution/taxonomy.py`](quant/execution/taxonomy.py) | Instrument class, broker registry, universe state machine |
| [`quant/execution/routing.py`](quant/execution/routing.py) | Sparplan vs Active Trade + dynamic fee hurdle |
| [`quant/data/database.py`](quant/data/database.py) | DuckDB schema + connection management |
| [`quant/dashboard.py`](quant/dashboard.py) | 3-page Streamlit UI (Briefing / Explorer / Universe) |
| [`quant/paths.py`](quant/paths.py) | CWD-independent path resolver |

---

## Institutional Capabilities (v10.4.0)

The system verifies data, measures execution reality, and survives tail risk.

| Domain | Capability | Module |
|:--|:--|:--|
| **Data** | Bitemporal PIT fundamentals (`as_of_date` / `published_date`) | [`quant/data/fundamentals.py`](quant/data/fundamentals.py) |
| **Data** | Corporate actions engine (splits → prices + cost basis) | [`quant/data/corporate_actions.py`](quant/data/corporate_actions.py) |
| **Data** | Hard assertions that abort the pipeline | [`quant/data/assertions.py`](quant/data/assertions.py) |
| **Execution** | Implementation Shortfall (TCA) + `trade_log` | [`quant/execution/tca.py`](quant/execution/tca.py) |
| **Execution** | Volatility-aware minimum trade size | [`quant/portfolio/optimizer.py`](quant/portfolio/optimizer.py) |
| **Execution** | Automated broker reconciliation | [`quant/execution/reconciliation.py`](quant/execution/reconciliation.py) |
| **Risk** | Mean-CVaR optimizer (Rockafellar-Uryasev) | [`quant/portfolio/optimizer.py`](quant/portfolio/optimizer.py) |
| **Risk** | Hard kill switch → `LIQUIDATE TO CASH` | [`quant/portfolio/risk_monitor.py`](quant/portfolio/risk_monitor.py) |
| **Risk** | Regime-conditional limits (Bear/Chop = 2%) | [`quant/portfolio/regime_constraints.py`](quant/portfolio/regime_constraints.py) |
| **Metrics** | Deflated Sharpe Ratio, Alpha Decay (IC), turnover variance | [`quant/analytics/metrics.py`](quant/analytics/metrics.py) |
| **Ops** | Structured JSON telemetry | [`quant/infra/observability.py`](quant/infra/observability.py) |
| **CI** | Golden-file snapshot gate | [`tests/test_golden_snapshot.py`](tests/test_golden_snapshot.py) |

Reconcile the portfolio against the broker export:

```bash
python3 scripts/reconcile_broker.py
```

---

## Project Structure

```
trade/
├── main.py · data_updater.py        # thin entry points (run these)
├── quant/                           # main package
│   ├── config.py · config_loader.py · paths.py
│   ├── data/        database, data_updater, data_quality, assertions,
│   │                corporate_actions, universe, universe_builder, funnel,
│   │                yf_utils, currency, async_fetcher, sec_edgar, news,
│   │                fundamentals, incremental
│   ├── features/    build_features, indicators, feature_cache
│   ├── analytics/   scoring, sentiment, validation, validation_engine,
│   │                health_score, metrics
│   ├── portfolio/   portfolio, portfolio_context, optimizer, risk,
│   │                risk_monitor, regime_constraints, cash_manager,
│   │                tax_optimizer, attribution, behavioral_guardrails
│   ├── strategy/    strategy_engine, backtest, strategies/
│   ├── execution/   taxonomy, routing, discovery, tca, reconciliation
│   ├── reporting/   reporting, reporting_advanced, artifacts, notifier,
│   │                mailer, alerts
│   ├── infra/       event_bus, observability
│   └── dashboard.py · scenario_simulator.py
├── data/            portfolio.csv, broker_registry.csv, watchlist.csv, config.yaml
├── tests/           test_*.py, golden/ (snapshot fixtures)
├── scripts/         setup_cron.sh, repair_registry.py, reconcile_broker.py,
│                    make_golden.py
├── plans/           engineering proposals
├── .github/         workflows/ci.yml
└── outputs/         generated run artifacts
```

---

## Configuration

All tunables live in [`quant/config.py`](quant/config.py). Key settings:

| Parameter | Default | Description |
|:---|---:|:---|
| `WEIGHT_FUNDAMENTALS` / `_STEWARDSHIP` / `_TECHNICAL` / `_SENTIMENT` | 30 / 30 / 15 / 25 | Scoring weights |
| `MIN_STRUCT_GRADE_FOR_BUY` | 75 | Structural floor for CORE classification |
| `BROKER_CASH_APY` | 0.0225 | TR cash yield (risk-free rate) |
| `ROUND_TRIP_FEE_EUR` | 2.0 | Active-trade round-trip fee |
| `SAFETY_BUCKET_MIN` / `CORE_BUCKET_MIN` / `ALPHA_BUCKET_MAX` | 0.10 / 0.40 / 0.50 | Bucket constraints |
| `TARGET_WEIGHTS` | 50/20/20/10 | Per-tier targets (CORE/SATELLITE/ACTIVE/SECTOR) |
| `REBALANCE_DRIFT_TIERS` | 10/7.5/5/6% | Per-tier drift trigger |
| `FUNNEL_STAGE1_TARGET` / `FUNNEL_TOP_N` | 300 / 24 | Funnel caps |
| `CVAR_ALPHA` | 0.05 | Mean-CVaR tail (Expected Shortfall) |
| `MAX_DAILY_DRAWDOWN` / `VOL_KILL_MULTIPLIER` | 0.03 / 2.0 | Kill-switch thresholds |
| `REPORTING_LAG_DAYS` | 45 | PIT publication lag (no lookahead) |

---

## Testing & CI

Tests are standalone scripts (they add the repo root to `sys.path`):

```bash
for t in tests/test_*.py; do echo "== $t =="; python3 "$t" || exit 1; done
# or, with pytest:
python3 -m pytest tests/ -q
```

| Suite | Coverage |
|:--|:--|
| [`tests/test_institutional.py`](tests/test_institutional.py) | Corporate actions, assertions, Mean-CVaR, kill switch, regime, TCA, metrics |
| [`tests/test_golden_snapshot.py`](tests/test_golden_snapshot.py) | Golden-file regression (>0.01% drift fails) |
| [`tests/test_scoring.py`](tests/test_scoring.py) | Factor model, regime, stewardship |
| [`tests/test_backtest_validity.py`](tests/test_backtest_validity.py) | WFO validity, survivorship warning |
| [`tests/test_factors.py`](tests/test_factors.py) | Factor scoring, no-lookahead, costs, liquidity |
| [`tests/test_phase4.py`](tests/test_phase4.py) | Fee hurdle, cash rf, routing, taxonomy, graduation |
| [`tests/test_phase5.py`](tests/test_phase5.py) | Universe state machine, ISIN, inverse routing, ETF scoring |
| [`tests/test_rebalancing.py`](tests/test_rebalancing.py) | Tier classification, rebalance logic, CORE never SELL |
| [`tests/test_advanced.py`](tests/test_advanced.py) | Portfolio context, ensemble, tax, cash, risk, attribution |
| [`tests/test_part3.py`](tests/test_part3.py) | Data quality, feature cache, observability, alerts, scenarios |
| [`tests/test_funnel.py`](tests/test_funnel.py) · [`tests/test_universe_builder.py`](tests/test_universe_builder.py) | Funnel rules + universe parsing |
| [`tests/test_portfolio_fx.py`](tests/test_portfolio_fx.py) | Portfolio FX + reconciliation |
| [`tests/test_no_emoji.py`](tests/test_no_emoji.py) | No-emoji lint |
| [`tests/test_db.py`](tests/test_db.py) · [`tests/test_market_data.py`](tests/test_market_data.py) · [`tests/test_cache.py`](tests/test_cache.py) · [`tests/test_e2e_state.py`](tests/test_e2e_state.py) · [`tests/test_async.py`](tests/test_async.py) | DB I/O, DuckDB↔Pandas, cache, end-to-end, concurrency |

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the full suite
plus the golden snapshot gate on every push and pull request.

Regenerate the golden file only for an intentional backtest change:

```bash
python3 scripts/make_golden.py
```

---

## Documentation

| Document | Purpose |
|:--|:--|
| [`CONTEXT.md`](CONTEXT.md) | Domain vocabulary, data contracts, module map, ADR notes |
| [`CHANGELOG.md`](CHANGELOG.md) | Full version history |
| [`plans/`](plans/) | Engineering change proposals |
| [`ANALYSIS_REPORT.md`](ANALYSIS_REPORT.md) | Historical v8.5 code review (superseded) |

---

## Disclaimer

All output is for informational purposes. Probabilistic models and NLP sentiment
analysis involve inherent risk. **Past performance does not guarantee future
results.** The universe contains only currently-listed instruments — historical
backtest figures are systematically overstated due to survivorship bias.

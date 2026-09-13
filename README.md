# Quant-AI v10.3.6

**Broker-aware, EUR-native systematic equity analysis for a solo family office on Trade Republic.**

Quant-AI scans a smart 1000+ ticker universe, filters it down to a handful of
survivors, and produces EUR-native buy/hold/sell guidance plus a broker-synced
portfolio audit. Python orchestrates; the slow parts run in Rust (Polars), SQL
(DuckDB), and C++-backed libraries (cvxpy, selectolax).

Current version: **v10.3.6** — see [`CHANGELOG.md`](CHANGELOG.md) for the history.

---

## Overview

- **Smart universe** — S&P 500 + Nasdaq-100 + Russell 1000 + broad ETFs (~1080 symbols), filtered by a two-stage funnel to the top ~24 survivors.
- **Bifurcated scoring** — equities run fundamentals + NLP; ETFs/commodities score on macro regime + trend.
- **Trade Republic aware** — ISIN/`tr_ticker` routing, Sparplan vs Active Trade, 1 EUR asymmetric fee, 2.25% cash APY as the risk-free rate.
- **Broker-synced** — portfolio PnL is copied from the broker, never price-guessed.
- **EUR-native** — global currencies normalised with live FX (Yahoo + ECB).

---

## Project Layout

```
trade/
├── main.py · data_updater.py     # thin entry points (run these)
├── quant/                        # main package
│   ├── config.py · config_loader.py · paths.py
│   ├── data/       database, data_updater, data_quality, universe,
│   │               universe_builder, funnel, yf_utils, currency,
│   │               async_fetcher, sec_edgar, news, fundamentals, incremental
│   ├── features/   build_features, indicators, feature_cache
│   ├── analytics/  scoring, sentiment, validation, validation_engine, health_score
│   ├── portfolio/  portfolio, portfolio_context, optimizer, risk, risk_monitor,
│   │               cash_manager, tax_optimizer, attribution, behavioral_guardrails
│   ├── strategy/   strategy_engine, backtest, strategies/
│   ├── execution/  taxonomy, routing, discovery
│   ├── reporting/  reporting, reporting_advanced, artifacts, notifier, mailer, alerts
│   ├── infra/      event_bus, observability
│   └── dashboard.py · scenario_simulator.py
├── data/           portfolio.csv, broker_registry.csv, watchlist.csv, config.yaml
├── tests/          test_*.py
├── scripts/        setup_cron.sh, repair_registry.py
├── plans/          engineering proposals
└── outputs/        generated run artifacts
```

---

## Architecture

### Data flow (two steps)

```
quant.data.universe_builder ─► universe_master (1000+ symbols, DuckDB)
        │
        ▼
quant.data.data_updater ─► quant.data.funnel (top ~24 survivors, cached)
        │                          └─► market_history (DuckDB)
        ▼
quant.main ─► analytics scoring ─► portfolio audit ─► reports + notifications
        ▲
quant.dashboard (Streamlit) ── reads DuckDB
```

Step 1 (`data_updater.py`) fetches market data and caches the funnel survivors.
Step 2 (`main.py`) reads that cache, scores, audits, and reports. `main.py` does
**not** re-fetch the universe.

### Key modules

| Module | Responsibility |
|:--|:--|
| [`quant/main.py`](quant/main.py) | Orchestration: load → filter → NLP → score → audit → report |
| [`quant/data/data_updater.py`](quant/data/data_updater.py) | Incremental DuckDB market data + funnel; caches survivors |
| [`quant/data/universe_builder.py`](quant/data/universe_builder.py) | Builds `universe_master` from index constituents + ETFs |
| [`quant/data/funnel.py`](quant/data/funnel.py) | Two-stage filter (liquidity → momentum) + survivor cache |
| [`quant/analytics/scoring.py`](quant/analytics/scoring.py) | Factor model, regime HMM, stewardship, capital allocation |
| [`quant/portfolio/portfolio.py`](quant/portfolio/portfolio.py) | Broker-synced audit, EUR PnL, reconciliation |
| [`quant/portfolio/optimizer.py`](quant/portfolio/optimizer.py) | cvxpy mean-variance with bucket constraints |
| [`quant/strategy/backtest.py`](quant/strategy/backtest.py) | Walk-forward, cost-aware (T+1) backtest |
| [`quant/execution/taxonomy.py`](quant/execution/taxonomy.py) | Instrument class, broker registry, universe state machine |
| [`quant/execution/routing.py`](quant/execution/routing.py) | Sparplan vs Active Trade + dynamic fee hurdle |
| [`quant/data/database.py`](quant/data/database.py) | DuckDB schema + connection management |
| [`quant/dashboard.py`](quant/dashboard.py) | 3-page Streamlit UI (Briefing / Explorer / Universe) |
| [`quant/paths.py`](quant/paths.py) | CWD-independent path resolver |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional NER
```

### 2. Configure

- [`data/portfolio.csv`](data/portfolio.csv) — your Trade Republic holdings.
- [`.env.example`](.env.example) — optional notifier and API keys.
- [`quant/config.py`](quant/config.py) — all tunable thresholds.

### 3. Run (two steps)

```bash
python3 data_updater.py   # step 1: fetch data + cache funnel survivors
python3 main.py           # step 2: score, audit, report
```

### 4. Dashboard

```bash
streamlit run quant/dashboard.py
```

### 5. Automate (optional)

```bash
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

## Features

- **Smart 1000+ universe** with a batched two-stage funnel (~20 requests, not 1084).
- **Bifurcated scoring** — EQUITY (fundamentals + FinBERT) vs ETF/COMMODITY (macro regime + trend).
- **Market-regime HMM** fit once on a broad index, applied to all assets.
- **Batched local FinBERT** sentiment — air-gapped, no API calls.
- **Cross-sectional factor model** — Value / Quality / Momentum / Low-Risk / Sentiment.
- **Trade Republic execution** — ISIN, Sparplan vs Active Trade, dynamic fee hurdle.
- **Broker-synced portfolio audit** with EUR PnL and `[!]` reconciliation flags.
- **Cost-aware walk-forward backtest** (T+1, 15 bps round-trip).
- **cvxpy optimizer** with Safety / Core / Alpha bucket constraints.
- **Adaptive layer** — portfolio context, strategy ensemble, cash manager, risk monitor, tax-loss harvesting, behavioral guardrails.
- **3-page Streamlit dashboard** and Telegram/Discord notifications.

---

## Testing

Unit tests are standalone scripts (they add the repo root to `sys.path`):

```bash
for t in tests/test_*.py; do echo "== $t =="; python3 "$t" || exit 1; done
# or, with pytest:
python3 -m pytest tests/ -q
```

| Suite | Coverage |
|:--|:--|
| [`tests/test_scoring.py`](tests/test_scoring.py) | Factor model, regime, stewardship |
| [`tests/test_backtest_validity.py`](tests/test_backtest_validity.py) | WFO validity, survivorship warning |
| [`tests/test_factors.py`](tests/test_factors.py) | Factor scoring, no-lookahead, costs, liquidity |
| [`tests/test_phase4.py`](tests/test_phase4.py) | Fee hurdle, cash rf, routing, taxonomy, graduation |
| [`tests/test_phase5.py`](tests/test_phase5.py) | Universe state machine, ISIN, inverse routing, ETF scoring |
| [`tests/test_rebalancing.py`](tests/test_rebalancing.py) | Tier classification, rebalance logic, CORE never SELL |
| [`tests/test_advanced.py`](tests/test_advanced.py) | Portfolio context, ensemble, tax, cash, risk, attribution |
| [`tests/test_part3.py`](tests/test_part3.py) | Data quality, feature cache, observability, alerts, scenarios |
| [`tests/test_funnel.py`](tests/test_funnel.py), [`tests/test_universe_builder.py`](tests/test_universe_builder.py) | Funnel rules + universe parsing |
| [`tests/test_portfolio_fx.py`](tests/test_portfolio_fx.py) | Portfolio FX + reconciliation |
| [`tests/test_no_emoji.py`](tests/test_no_emoji.py) | No-emoji lint |
| [`tests/test_db.py`](tests/test_db.py), [`tests/test_market_data.py`](tests/test_market_data.py), [`tests/test_cache.py`](tests/test_cache.py), [`tests/test_e2e_state.py`](tests/test_e2e_state.py), [`tests/test_async.py`](tests/test_async.py) | DB I/O, DuckDB↔Pandas, cache, end-to-end, concurrency |

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

---

## Documentation

- [`CHANGELOG.md`](CHANGELOG.md) — full version history.
- [`CONTEXT.md`](CONTEXT.md) — domain vocabulary, data contracts, ADR notes.
- [`plans/`](plans/) — engineering change proposals.

---

## Disclaimer

All output is for informational purposes. Probabilistic models and NLP sentiment
analysis involve inherent risk. **Past performance does not guarantee future
results.** The universe contains only currently-listed instruments — historical
backtest figures are systematically overstated due to survivorship bias.

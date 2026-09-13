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

## Architecture

### Data flow (two steps)

```
universe_builder.py ─► universe_master (1000+ symbols)
        │
        ▼
data_updater.py ──► funnel.py (top ~24 survivors, cached) ──► market_history (DuckDB)
        │                                                          ▲
        ▼                                                          │
main.py ──► scoring ──► portfolio audit ──► reports + notifications ─┘
        ▲
dashboard.py (Streamlit) ── reads DuckDB
```

Step 1 (`data_updater.py`) fetches market data and caches the funnel survivors.
Step 2 (`main.py`) reads that cache, scores, audits, and reports. `main.py` does
**not** re-fetch the universe.

### Modules

| Module | Responsibility |
|:--|:--|
| [`main.py`](main.py) | Orchestration: load → filter → NLP → score → audit → report |
| [`data_updater.py`](data_updater.py) | Incremental DuckDB market data + funnel; caches survivors |
| [`universe_builder.py`](universe_builder.py) | Builds `universe_master` from index constituents + ETFs |
| [`funnel.py`](funnel.py) | Two-stage filter (liquidity → momentum) + survivor cache |
| [`scoring.py`](scoring.py) | Factor model, regime HMM, stewardship, capital allocation |
| [`portfolio.py`](portfolio.py) | Broker-synced audit, EUR PnL, reconciliation |
| [`optimizer.py`](optimizer.py) | cvxpy mean-variance with bucket constraints |
| [`backtest.py`](backtest.py) | Walk-forward, cost-aware (T+1) backtest |
| [`taxonomy.py`](taxonomy.py) | Instrument class, broker registry, universe state machine |
| [`routing.py`](routing.py) | Sparplan vs Active Trade + dynamic fee hurdle |
| [`discovery.py`](discovery.py) | Watchlist → ACTIVE graduation engine |
| [`build_features.py`](build_features.py) | Vectorized cross-sectional features (Polars) |
| [`sentiment.py`](sentiment.py) | Batched local FinBERT NLP |
| [`indicators.py`](indicators.py) | EWMA volatility, RSI, ATR |
| [`risk.py`](risk.py) | Empirical VaR, Sortino/Sharpe, risk penalty |
| [`fundamentals.py`](fundamentals.py) | Fundamentals fetch + point-in-time cache |
| [`sec_edgar.py`](sec_edgar.py), [`news.py`](news.py) | SEC 8-K and News RSS text sources |
| [`currency.py`](currency.py) | Live FX normalisation to EUR |
| [`database.py`](database.py) | DuckDB schema + connection management |
| [`dashboard.py`](dashboard.py) | 3-page Streamlit UI (Briefing / Explorer / Universe) |
| [`notifier.py`](notifier.py) | Telegram / Discord daily push |
| [`config.py`](config.py) | All tunable settings |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional NER
```

### 2. Configure

- [`portfolio.csv`](portfolio.csv) — your Trade Republic holdings.
- [`.env.example`](.env.example) — optional notifier and API keys.
- [`config.py`](config.py) — all tunable thresholds.

### 3. Run (two steps)

```bash
python3 data_updater.py   # step 1: fetch data + cache funnel survivors
python3 main.py           # step 2: score, audit, report
```

### 4. Dashboard

```bash
streamlit run dashboard.py
```

### 5. Automate (optional)

```bash
bash setup_cron.sh        # daily 18:00 CET + weekly discovery
```

### Portfolio file

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

Unit tests are standalone scripts (no pytest required):

```bash
for t in test_*.py; do echo "== $t =="; python3 "$t" || exit 1; done
```

| Suite | Coverage |
|:--|:--|
| [`test_scoring.py`](test_scoring.py) | Factor model, regime, stewardship |
| [`test_backtest_validity.py`](test_backtest_validity.py) | WFO validity, survivorship warning |
| [`test_factors.py`](test_factors.py) | Factor scoring, no-lookahead, costs, liquidity |
| [`test_phase4.py`](test_phase4.py) | Fee hurdle, cash rf, routing, taxonomy, graduation |
| [`test_phase5.py`](test_phase5.py) | Universe state machine, ISIN, inverse routing, ETF scoring |
| [`test_rebalancing.py`](test_rebalancing.py) | Tier classification, rebalance logic, CORE never SELL |
| [`test_advanced.py`](test_advanced.py) | Portfolio context, ensemble, tax, cash, risk, attribution |
| [`test_part3.py`](test_part3.py) | Data quality, feature cache, observability, alerts, scenarios |
| [`test_funnel.py`](test_funnel.py), [`test_universe_builder.py`](test_universe_builder.py) | Funnel rules + universe parsing |
| [`test_portfolio_fx.py`](test_portfolio_fx.py) | Portfolio FX + reconciliation |
| [`test_no_emoji.py`](test_no_emoji.py) | No-emoji lint |
| [`test_db.py`](test_db.py), [`test_market_data.py`](test_market_data.py), [`test_cache.py`](test_cache.py), [`test_e2e_state.py`](test_e2e_state.py), [`test_async.py`](test_async.py) | DB I/O, DuckDB↔Pandas, cache, end-to-end, concurrency |

---

## Configuration

All tunables live in [`config.py`](config.py). Key settings:

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

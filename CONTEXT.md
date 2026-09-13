# CONTEXT.md — Domain Vocabulary & Architecture Map

Canonical terms for the Quant-AI family office terminal. Keep terminology
consistent across code, docs, and AI agents.

## Domain Vocabulary

| Term | Canonical Meaning |
|------|-------------------|
| **instrument_class** | Asset taxonomy: `EQUITY` \| `ETF` \| `COMMODITY` \| `CASH`. Drives bifurcated scoring. |
| **universe_status** | `ACTIVE` (heavy analysis) \| `WATCHLIST` (light scan) \| `CORE` (always tracked). |
| **Structural Grade** | Long-term fundamental quality (0-100). Stewardship + PE/PEG/ROE. |
| **Tactical Grade** | Short-term timing (0-100). HMM regime + sentiment - risk. |
| **Sparplan** | TR savings plan. 0 EUR buy, 1 EUR sell. Long-term accumulation. |
| **Active Trade** | TR tactical order. 1 EUR buy + 1 EUR sell = 2 EUR round-trip. |
| **Round-trip fee** | 2 EUR (active). The asymmetric hurdle rate for position sizing. |
| **Risk-free rate** | TR cash APY (2.25%), not US Treasury. Daily = `(1+APY)^(1/365)-1`. |
| **Safety Bucket** | Cash & short-term bonds. Constraint: ≥ 10%. |
| **Core Bucket** | Broad ETFs via Sparplan. Constraint: ≥ 40%. |
| **Alpha Bucket** | Active equities. Constraint: ≤ 50%. |
| **Graduation** | Watchlist → ACTIVE on 52-week-high or 3x-volume anomaly. |
| **Demotion** | ACTIVE → Watchlist after 6 months with no signals. |
| **universe_master** | Broad 1000+ ticker pool (S&P 500 + Nasdaq 100 + Russell 1000 + ETFs). Built by `universe_builder.py`. |
| **Funnel** | Two-stage filter: Stage 1 liquidity/viability (price>$5, min $ volume) → ~300-500; Stage 2 trend/momentum → top ~24 survivors. |
| **Broker-synced CSV** | `portfolio.csv` schema `Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR`. Invested = Value − Broker_PnL; PnL is broker truth, never price-guessed. |
| **Reconciliation** | `System_Estimated_Value = shares × current_price_eur`; `[!]` flag when deviation > €1.00 (stale CSV / high spread). |
| **Bitemporal PIT** | `as_of_date` (valid-for) + `published_date` (market saw it). Backtests filter `published_date <= as_of_date`; no lookahead. |
| **Corporate Action** | Split/merger/dividend adjustment. `detect_split` -> `apply_corporate_actions` restates prices + cost basis. |
| **Data Assertion** | Hard gate (`DataAssertionError`) that aborts the pipeline: duplicate timestamps, unexplained >50% drop, bad D/E. |
| **Implementation Shortfall** | Signal price vs actual fill price, in bps. Positive = worse execution. Logged to `trade_log`. |
| **Mean-CVaR** | Optimizes average loss in the worst `CVAR_ALPHA` (5%) tail (Expected Shortfall), not variance. |
| **Kill Switch** | Hard stop: >3% daily drawdown or vol > 2x target -> `LIQUIDATE TO CASH` + pause scanner. |
| **Regime Constraint** | HMM regime -> `(max_single_weight, max_leverage)`. Bear/Chop = 2% / 0x. |
| **Deflated Sharpe Ratio** | Bailey & Lopez de Prado DSR: Sharpe penalized for trials + skew/kurtosis. |
| **Alpha Decay (IC)** | Information Coefficient of a signal at T+1/T+5/T+21; smooth decay validates exits. |
| **Golden File** | Saved deterministic backtest output; CI fails on >0.01% drift. |
| **trade_log** | TCA table: `symbol, side, signal_price, fill_price, slippage_bps, fee_eur`. |
| **portfolio_snapshot** | Daily theoretical portfolio state, diffed against the broker export. |

## Module Map (Callers)

```
quant.data.universe_builder ─> quant.data.database (universe_master), quant.execution.taxonomy
quant.data.funnel ───────────> quant.config (thresholds), yfinance (snapshots/history)
quant.data.data_updater ─────> quant.data.universe_builder, quant.data.funnel, quant.execution.taxonomy, quant.data.database
quant.main ──────────────────> quant.data.funnel, quant.data.universe_builder, quant.analytics.scoring, quant.execution.taxonomy, quant.execution.routing, quant.reporting.notifier
quant.portfolio.optimizer ───> quant.portfolio.risk (daily rf), quant.config (buckets, fees)
quant.execution.discovery ───> quant.data.universe_builder, quant.execution.taxonomy, quant.data.database (asset_registry)
quant.dashboard ─────────────> quant.data.database, quant.execution.taxonomy, quant.execution.routing, quant.portfolio.risk
quant.reporting.notifier ────> quant.config, quant.portfolio.risk

> Package layout: modules live under `quant/<subpackage>/`. Entry points are the
> thin root wrappers `main.py` and `data_updater.py`. Filesystem paths resolve via
> `quant/paths.py` (CWD-independent).
```

## Data Contract (asset_registry)

| Column | Type | Purpose |
|--------|------|---------|
| symbol | VARCHAR PK | yahoo ticker |
| instrument_class | VARCHAR | EQUITY/ETF/COMMODITY/CASH |
| isin | VARCHAR | TR routing key |
| tr_ticker | VARCHAR | LS Exchange ticker |
| exchange | VARCHAR | LS Exchange / Tradegate |
| currency | VARCHAR | native currency |
| universe_status | VARCHAR | ACTIVE/WATCHLIST/CORE |
| graduated_at | DATE | promotion date |
| last_signal_date | DATE | last signal for demotion |

## v10.4.0 Module Map (Institutional Layer)

```
quant.data.corporate_actions ─> quant.data.data_updater (adjust before ingest)
quant.data.assertions ────────> quant.data.data_updater (hard gate, aborts run)
quant.data.fundamentals ──────> fundamentals_history (PIT persist) + get_fundamentals_pit
quant.execution.tca ──────────> trade_log (implementation shortfall)
quant.execution.reconciliation > portfolio_snapshot + scripts/reconcile_broker.py
quant.portfolio.optimizer ────> optimize_portfolio_cvar, minimum_trade_size_vol_aware
quant.portfolio.regime_constraints ─> optimizer (regime caps)
quant.portfolio.risk_monitor ─> check_kill_switch -> event_bus KILL_SWITCH
quant.analytics.metrics ──────> reporting_advanced (DSR / IC / turnover)
quant.infra.observability ────> telemetry.json (fetch latency, DuckDB ms, rate limits)
```

## ADR Notes

- **Cash as risk-free rate**: TR pays 2.25% APY on uninvested cash. Using this
  as R_f (instead of US Treasuries) is the correct opportunity cost for a
  EUR-based solo family office. Hard to reverse, real trade-off → documented.

- **Mean-CVaR over Sharpe (v10.4.2)**: optimization minimizes the average loss
  in the worst `CVAR_ALPHA` (5%) tail (Expected Shortfall) instead of variance.
  Sharpe penalizes upside volatility and assumes elliptical returns; equity
  returns are fat-tailed. Mean-CVaR (Rockafellar-Uryasev LP) is coherent and
  tail-faithful. Sharpe is retained for reporting, not for sizing. Hard to
  reverse, real trade-off → documented.

---

## v10.4.2 Domain Additions

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **Coverage ratchet** | A `fail_under` floor in `.coveragerc` that may only be raised. Baseline 42%, target 80%. |
| **Golden drift** | Relative change (> 0.01%) in the deterministic backtest snapshot vs `tests/golden/backtest_2024.json`. |
| **CLI entry point** | The `quant` console command (`update` / `run` / `reconcile` / `all`). Root wrappers are shims. |

### Data contracts

- `market_history` columns: `Date, Open, High, Low, Close, Volume, Symbol,
  Sector, Instrument_Class` + `ingested_at TIMESTAMP` (bitemporal arrival time).
- `trade_log`: `ts, symbol, side, signal_price, signal_ts, fill_price, fill_ts,
  slippage_bps, fee_eur`.
- `portfolio_snapshot`: `snapshot_date, symbol, shares, price_eur, value_eur`
  (PK `(snapshot_date, symbol)`).

### Module dependencies (v10.4.2)

```
quant.cli ────────────────────> quant.data.data_updater, quant.main,
                                quant.execution.reconciliation
quant.main ───────────────────> quant.analytics.scoring, quant.portfolio.*
quant.data.data_updater ──────> quant.data.universe_builder, quant.data.funnel
quant.portfolio.optimizer ────> quant.portfolio.risk, quant.portfolio.regime_constraints
tests.conftest ──────────────> quant.paths, quant.data.database (isolated DB)
```

### Test isolation contract

- All tests run against an ephemeral DuckDB injected by [`tests/conftest.py`](tests/conftest.py).
- `quant.paths.DB_FILE` and `quant.data.database.DB_PATH` are patched for the session.
- Tests must never read/write live `data/` or `outputs/`; use `tmp_path` / `tempfile`.
- All randomness seeded (`np.random.default_rng`).

---

## v10.5.0 Domain Additions

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **account.yaml** | `data/account.yaml`: single source of truth for `base_currency`, `cash_eur`, `risk_profile`. Replaces the `account_state` DuckDB table. |
| **Risk profile** | One of `conservative` / `balanced` / `aggressive`. Maps to `(safety_min, core_min, alpha_max, max_position, cash_floor)` in `quant/config.py` `RISK_PROFILES`. |
| **Published Briefing** | Static read-only HTML (`outputs/run_<ts>/web/index.html` + `data.json`) generated by `quant publish`, deployed to GitHub Pages. |
| **Terse output** | Default CLI stdout: aggregate lines only, at most 20 lines. Per-symbol detail goes to `outputs/run_<ts>/pipeline.log` and to stdout only under `--verbose`. |
| **Canonical action** | An action derived from the portfolio audit + broker registry by `quant.reporting.actions.build_actions`. One source for CLI, dashboard, and web. |
| **NLP evidence** | `nlp_evidence` table: `symbol, source, title, published_at, score, confidence, retrieved_at`. Explains WHY a sentiment score exists. |

### Risk profiles (one sentence each)

- `conservative` keeps at least 20 percent in safety assets and at least 15 percent in cash.
- `balanced` keeps at least 10 percent in safety assets and at least 10 percent in cash.
- `aggressive` keeps at least 5 percent in safety assets and at least 5 percent in cash.

### Evidence transparency (NLP)

- News comes from SEC 8-K filings and news feeds, scored with FinBERT.
- Sentiment maps to a score in `[-1, 1]`; the tactical grade consumes it.
- `confidence low` means no news evidence was found: the sentiment component is
  neutral and the tactical grade is penalized (`SENTIMENT_NO_DATA_PENALTY`).
- The UI and report never repeat the per-row disclaimer; they aggregate it into
  one evidence footnote and expose per-symbol detail in Explorer.

### Module dependencies (v10.5.0)

```
quant.cli ────────────────────> quant.cli.output, quant.data.data_updater,
                                quant.main, quant.reporting.web
quant.cli.output ─────────────> quant.reporting.artifacts (run dir + log)
quant.main ───────────────────> quant.reporting.actions, quant.reporting.briefing,
                                quant.portfolio.account
quant.reporting.web ──────────> quant.reporting.actions, quant.portfolio.account
quant.dashboard ──────────────> quant.portfolio.account, quant.portfolio.editor,
                                quant.reporting.actions, quant.reporting.artifacts
quant.portfolio.account ──────> quant.config (RISK_PROFILES), quant.paths
```

### Source-of-truth contract (v10.5.0)

| Fact | Writer | Reader(s) |
|------|--------|-----------|
| Positions, broker PnL | user via Portfolio editor / `data/portfolio.csv` | pipeline, workspace, report |
| Cash, risk profile, base currency | user via Account form / `data/account.yaml` | pipeline, workspace, report |
| ISIN, routing, min size | `data/broker_registry.csv` | pipeline, workspace, report |
| Prices, history | pipeline (`quant update`) | everything |
| Scores, grades, actions, regime | pipeline (`quant run`) | everything |
| Version | `quant/__init__.py` | CLI, sidebar, report footer |

---

## v10.5.1 Domain Additions

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **Copy module** | `quant/ui/copy.py`: the single source of every user-facing string and number formatter. Pages import from it so vocabulary is consistent and the banned-token test scans one file. |
| **Banned token** | An internal identifier (run ids, paths, column names, enum values, `n/a`) that must never render in the UI. Enforced by `tests/test_ui_copy.py`. |
| **Portfolio history** | `portfolio_history` table: one row per review (`review_ts, value_eur, invested_eur, cash_eur, pnl_eur`). Feeds the Today value chart. |
| **Orchestrator mutex** | `quant/ui/runner.py`: one in-process lock serializing UI-triggered refresh/review runs. A second tab gets "A review is already running in another tab." |
| **Read-only connection** | `quant.data.database.read_only_connection()`: a short-lived, always-closed connection the UI uses so a refresh subprocess can take the write lock. |
| **Cash rate schedule** | `quant/portfolio/cash_rate.py`: dated `(effective_date, apy, source_url)` rows. 2.5 percent from 16 Sep 2026; 2.25 percent before. |

### Cash rate

- Trade Republic raised its cash APY to 2.5 percent on 16 Sep 2026.
- `current_cash_apy(as_of, live=False)` returns the scheduled rate; `live=True`
  attempts a documented public fetch and falls back to the schedule, never raising.
- `update_cash_rate(apy, effective_date, source_url)` records a new dated fact.
- Consumers (`routing`, `risk`, `cash_manager`, `notifier`, dashboard) call
  `current_cash_apy()` instead of a constant; `config.BROKER_CASH_APY` is the fallback.

### Concurrency contract (spec 4)

1. UI reads use `read_only_connection()` and close immediately; the UI holds no
   persistent read-write connection.
2. `connect_with_retry(attempts=6, delay=0.5)` retries the write lock over ~15s.
3. `quant.ui.runner` serializes UI-triggered runs with a non-blocking mutex.
4. UI runs spawn `sys.executable -m quant.cli` and inherit the environment;
   output streams to the run log, not the UI.

### Module dependencies (v10.5.1)

```
quant.ui.copy ────────────────> (pure; datetime only)
quant.ui.runner ──────────────> quant.cli (subprocess), quant.ui.copy
quant.ui.search ──────────────> quant.data.database (read path), universe_builder
quant.dashboard ──────────────> quant.ui.copy, quant.ui.runner, quant.ui.search,
                                quant.data.database (read_only_connection),
                                quant.portfolio.{account,editor,history,cash_rate},
                                quant.reporting.actions
quant.portfolio.history ──────> quant.data.database
quant.portfolio.cash_rate ────> (pure; urllib for the optional live fetch)
```

---

## v10.5.2 Domain Additions

### Doctrine

- **F1 — No synthesized identifiers.** The codebase never invents, guesses, or
  hardcodes financial identifiers (ISIN, CUSIP, SEDOL). Identifiers enter only
  through (a) files the user owns, (b) validated live metadata from a named
  source, or (c) a curated file with an audit trail. Every identifier carries
  provenance. `quant.data.identifiers.is_valid_isin` is the mechanical gate.
- **P4 — Empty states describe missing data only.** A computation that ran and
  failed is a Health item, never an empty state. Today shows "unavailable (see
  Health)"; Settings Health names the failure. Never mask a failure as missing
  history.

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **isin_source** | Provenance of a registry ISIN: `user` (pre-existing), `curated` (`data/isin_curated.csv`), `yahoo` (live metadata). Internal; drives the confirm-in-broker caveat. |
| **Curated ISIN map** | `data/isin_curated.csv` (`symbol,isin,verified_by,verified_at`): user-verified values that override live metadata forever. Ships with a header and zero rows. |
| **regime_error** | `metrics.json` flag: the regime HMM fit ran and failed. Today shows "unavailable"; Health shows the failure sentence. Distinct from missing history. |
| **Canonical status** | One of `On track` / `Waiting until {date}` / `Add` / `Trim` / `Blocked`, produced by `quant.ui.copy.status_for` and persisted with the audit. Table and action cards read the same object. |

### Source-of-truth additions (v10.5.2)

| Fact | Writer | Reader(s) |
|------|--------|-----------|
| ISIN provenance | `scripts/repair_registry.py` (via `quant.data.registry_repair`) | Explore caveat, tests |
| Regime failure flag | `quant run` (`metrics.json`) | Today, Settings Health |
| Canonical status | `quant run` (portfolio audit) | Today table + action cards |

### Module dependencies (v10.5.2)

```
quant.data.identifiers ───────> (pure)
quant.data.registry_repair ───> quant.paths, quant.data.identifiers, yfinance (optional)
scripts.repair_registry ──────> quant.data.registry_repair, quant.execution.taxonomy
quant.ui.runner.run_repair ───> quant.data.registry_repair (in-process, under the mutex)
quant.reporting.actions ──────> quant.ui.copy (status_for)
quant.dashboard ──────────────> quant.portfolio.cash_rate (current_rate), quant.ui.runner (run_repair)
```

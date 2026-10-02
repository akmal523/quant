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

## v10.5.3 Domain Additions

### Regime block + confidence

- `metrics.json` carries `regime = {state, label, prob, confidence, as_of, error}`.
- `state` is `estimated | insufficient_history | failed`; `label` is `rising | falling | mixed`.
- `confidence` = `regime_confidence(prob)`: `|prob - 0.5| >= 0.30 -> high`,
  `>= 0.15 -> medium`, else `low` ([`quant/analytics/scoring.py`](quant/analytics/scoring.py)).
- UI mapping: estimated -> `Market trend: {label} ({confidence} confidence).`;
  insufficient_history -> `Market trend: not enough history yet.`;
  failed -> `Market trend: unavailable (see Health).` (Health names the failure).

### State vocabulary (canonical status)

`On track`, `Add`, `Trim`, `Waiting until {date}`, `Below minimum order`,
`Blocked`, `Not reviewed yet` (single mapper `quant.ui.copy.status_for`).

### Artifact accessors (five)

All UI artifact reads go through `quant.reporting.artifacts`:
`latest_review()`, `read_regime()`, `read_actions()`, `read_scores(symbol)`,
`read_history()`. No page touches run directories or parquet paths directly.

### News stores (v10.5.3, R7)

Two stores, two roles. The **news cache** (`outputs/news_cache.json`, keyed by
symbol, 24 h TTL, atomic temp+rename writes) is the single fetch-and-cache path
for Explore's on-demand news AND the review's holdings coverage, so both render
the same headlines for a symbol on the same day; a source-outage counter
(`outputs/news_state.json`) drives the Health line after three consecutive review
failures. The **DuckDB `nlp_evidence` / `nlp_scores`** tables keep their distinct
role: scored sentiment inputs consumed by the scoring pipeline, not display.

### Registry write paths and the working universe (H3-fix)

**Working universe (canonical).** `W` is the single source of the term:

```
W = funnel survivors (cached)
  ∪ CORE ∪ ACTIVE (asset_registry.universe_status)
  ∪ portfolio.csv symbols
  ∪ broker_registry.csv symbols
  ∪ curated ISIN/name symbols
  ∪ BROAD_ETFS (the always-tracked constant)
```

`asset_registry` holds **exactly** `W`, nothing else. `universe_master` (1000+
symbols) is excluded everywhere.
`quant.data.registry_repair.working_universe()` recomputes `W` from the live
inputs at run time (never hardcoded); the doctor prints
`registry rows: {n} (working universe)`.

**Two stores, two roles.** `data/broker_registry.csv` is user-owned routing input
(ISIN / venue / class per routable symbol); its ceiling is
`portfolio + CORE_ETFS + 20`. `asset_registry` (DuckDB) is the working universe.

**Writers (audit).** Every writer of `asset_registry` and its source set:

| Writer | Source set |
|--------|-----------|
| `registry_repair.sync_registry_to_working_universe` | `W` (insert missing, prune outside; blank cells healed) |
| `registry_repair.ensure_registry_rows` | portfolio + CORE/ACTIVE (broker_registry.csv rows) |
| `registry_repair.repair_isins` | missing ISIN cells only (curated > existing > live) |
| `taxonomy.sync_broker_registry` | reads broker_registry.csv (never writes it) |
| `taxonomy.set_core` / `set_structure` / `add_to_watchlist` / `mark_delisted` | individual, explicit calls |
| `taxonomy._upsert_registry` (via `get_instrument_class`) | the fetch list, which is a subset of `W` |
| `discovery.graduate` | one symbol, with a universe event |
| `discovery.run_discovery` (fetch-failure counter) | **updates tracked rows only** — never inserts a `universe_master` symbol |

No path bulk-inserts `universe_master` into `asset_registry` — that caused the
1090-row explosion. `ensure_registry_rows` refuses a bulk source with a logged
warning. `tests/test_registry_bounded.py` enforces: `asset_registry == W`,
idempotent membership, the CSV ceiling, and the bulk-source refusal.

---

## v10.6.0 Domain Additions (H3.4)

### Themes (Explore search)

- `data/themes.csv` (`theme,symbols`) maps **prose theme tags** to symbols and/or
  other themes. Themes are human labels, **NOT financial identifiers** — F1 does
  not apply; the file is user-editable.
- Rows are unquoted; the `symbols` column is a comma list parsed on the first
  comma. A token that names another theme links to it (theme-to-theme), so
  `space -> aerospace -> 5J50.DE` and `space -> defence -> DFEN`.
- Search corpus = display_name + name + symbol + ISIN + themes, all
  case-insensitive substring ([`quant/ui/search.py`](quant/ui/search.py)). An
  empty query renders the helper line; a non-empty query with no match renders
  the exact zero-match sentence.
- News rows render weekday dates ([`fmt_weekday_date`](quant/ui/copy.py)), cap at
  5 with an `Earlier items ({n} more)` expander, and drop the sentiment word with
  one Diagnostics line when the FinBERT stack is absent.

### Explore card and the discovery loop (H3.5)

- [`quant.ui.cards.explore_card_fields`](quant/ui/cards.py) is the ONE source for
  the Explore card title / class / subtitle (registry values win; the CSV routing
  metadata is the fallback). The doctor probes it
  (`explore card {symbol}: title=… subtitle=…`), so a regression is diagnosable.
- `label_for` never duplicates the base ticker: if the base (symbol before the
  first dot) already appears in a name paren group, the name stands alone
  ("Global Aero & Def (5J50)", not "... (5J50) (5J50.DE)").
- Discovery loop: Explore's zero-match names a `universe_master` match
  ("{label} is in the discovery universe but not tracked. Add it in Portfolio to
  track it."); the Portfolio add-input also offers `universe_master` candidates
  ("{label} - not tracked yet"). Selecting one adds a held row → enters `W` on
  save, fetched at the next update. No `universe_master` row enters
  `asset_registry` without being held or CORE/ACTIVE.

### Failed-review shadowing, news dates, sentiment provenance (H3.6)

- Data reads (`read_scores`, `read_regime`) use the most recent **successful**
  review: `latest_review(ok_only=True)` / `latest_ok_run_dir()`. The Today S4
  card and Settings Health read the latest attempt of any status; when the two
  differ, Explore shows one line `Scores as of {weekday d mon, hh:mm}.`
- `fmt_weekday_date` / `fmt_weekday_ts` parse ISO-8601 **and** RFC-2822, so the
  live news cache (RFC-2822) never renders a raw timestamp.
- Each news cache entry carries `scorer: model | default`. The sentiment word is
  rendered only for `scorer: model`; entries lacking the field migrate to
  `default` on read. The doctor prints the distribution
  (`sentiment cache: … scorer model a / default b, pos p neg q neu r`).

### Legacy artifacts, update freshness, chart legibility (H3.7)

- A review lives in a TIMESTAMPED run dir carrying `metrics.json`; `run_latest`
  and update-only dirs never shadow a review. A review without `review_status`
  is legacy and counts as ok; only `"failed"` excludes it. Data reads use
  `latest_review(ok_only=True)`; the S4 card / Health read the latest attempt.
- `quant update` writes `outputs/update_state.json`; the Settings status line
  composes live values (instrument count, prices-through) with the update
  timestamp.
- The Reviews list drops rows newer than the last ok review (legacy phantom
  rows). `fmt_review_ts` omits a bogus `00:00` for a date-only source.
- The value-chart annotation sits in the top margin on a white box; sub-3-day
  ranges label the x-axis by `hh:mm`.

### Advice engine, legacy runs, discovery names (H3.8)

- **One drift calculation.** The advice is driven by drift vs the tier threshold
  — shared by the holdings table, the engine, statuses, and the S3 footnote. An
  over-threshold drift is never "On track". The rebalance time gate only sets a
  cooldown → `Waiting until {date}` + footnote, never silence; `Cooldown_Until`
  rides on the audit row.
- A review lives in a timestamped dir with metrics.json; legacy reviews (no
  `review_status`) count as ok. The Today header reads ONE artifact (never a
  borrowed close date). The Reviews list keeps legacy rows and hides phantoms.
- `read_history` / Explore charts read DuckDB (`portfolio_history` /
  `market_history`), never run-dir snapshots.
- `universe_master` names are backfilled (bounded, cached in
  `outputs/universe_names.json`) so the discovery index resolves a company query.
- `ERROR_RUNNING` copy is operation-agnostic; the only as-of string is
  `Scores as of {date}.` (no "From the review of" preamble).

### Calendar lines and savings plan (R8)

- `data/account.yaml` gains optional `savings_plan_day` (1-31). `AccountState`
  reads it, clamps out-of-range/non-numeric to None, and persists it only when
  set. README documents the key.
- [`quant.ui.copy.savings_plan_line(today, day)`](quant/ui/copy.py) is the ONE
  source of the Today sentence: same day -> `Savings plan executes today.`;
  otherwise `Savings plan executes in {days} days ({date}); ...`. Pure datetime
  (no `calendar` import); a day beyond the month clamps to the month's last day;
  the date carries its weekday (`fmt_weekday_date`).
- [`quant.execution.routing.holding_routes_to_savings_plan`](quant/execution/routing.py)
  reuses `route_signal` (ETF/CASH, plain structure -> SPARPLAN) so the calendar
  line and execution routing can never disagree.
- Today renders the countdown under the actions block only when at least one
  holding routes to a savings plan. The header gains the markets-closed
  freshness line (`MARKETS_CLOSED`) when today is non-trading and the latest bar
  is the previous session (Friday).

## v10.6.1 Domain Additions (F-series — fresh install)

### Path resolution: development vs production (F2)

- [`quant/paths.py`](quant/paths.py) resolves the writable root by mode:
  - **source checkout** (a `pyproject.toml` or `.git` sits next to the package)
    -> the repo root (tests and the repo `data/` are used);
  - **installed wheel** -> `platformdirs.user_data_dir("quant-ai")` (a wheel must
    never write into site-packages);
  - `QUANT_DATA_DIR` (env) overrides both, for tests, CI, and portable installs.
- `PACKAGE_DIR` is always the installed package dir (shipped assets + the
  dashboard script); `PROJECT_ROOT` is the writable root. Importing `paths`
  performs **no** filesystem writes; `ensure_dirs()` creates the runtime dirs.

### First-run seeding (F5)

- `themes.csv` ships as package data at [`quant/_data/themes.csv`](quant/_data/themes.csv)
  (included in the wheel via `packages = ["quant"]`).
- [`quant/data/bootstrap.py`](quant/data/bootstrap.py) `seed_user_data()` creates
  the runtime dirs and seeds the **empty input templates** (headers only, shipped
  in code so no user portfolio/registry leaks into the wheel) plus the bundled
  `themes.csv` into the writable data dir. It NEVER overwrites a user-owned file
  and never raises. Called from `init_db()`.

### Doctor initializes the DB (F1/F3)

- `quant doctor` calls `init_db()` before reading, so on a fresh install registry
  counts read `0` (not `-1`) and `market_history` resolves instead of raising a
  `CatalogException`. The card probe prints `details={has_details}` (F4).

### Explore empty state (F4)

- [`quant.ui.cards.explore_card_fields`](quant/ui/cards.py) returns `has_details`
  (a real registry row exists). [`quant/ui/render.py`](quant/ui/render.py) renders
  `EXPLORE_NO_DETAILS` for an instrument with no registry row, never the bare
  ticker as a title.

---

## v10.6.0 Cycle Ledger (V1-V8, R1-R9, H2-H3.8)

Purpose: everything built, decided, broken, and fixed since 10.5.2, so a
continuation can resume from this file alone. All unreleased work since tag
10.5.2 ships as **10.6.0**.

### Version timeline

| Version | Content |
|---|---|
| 10.4.0 | Starting point: 3-page Streamlit dashboard, requirements.txt, failing CI test |
| 10.4.2 | Professional transformation: pyproject/hatchling, `quant` CLI, conftest isolation, mkdocs, ruff/pre-commit, release workflows, community files, ADRs |
| 10.5.0 | Polish v2/v3 first pass: four-page workspace, copy module, concurrency fix, cash-rate schedule |
| 10.5.1 | P1-P10 polish: read-only connections, retry, runner mutex, copy catalog + banned-token test, portfolio_history, names/search, workspace rebuild |
| 10.5.2 | Punchlist v3: empty states, regime masking guard, status single mapper, visuals, autocomplete, ISIN repair infrastructure |
| 10.5.2 + unreleased (→10.6.0) | R1-R9 program and H2/H3 hotfix cycles (below). Last code commit: 5844631; docs commits: 738f3d4, 9e88de5. R8 (`2022302`) + H4 (`441271a`) + ruff gate (`df314a8`) followed. Suite **327 passed**, released as **10.6.0** (tag `v10.6.0`) |
| 10.6.1 | F-series fresh-install hotfix: dev/production path split (`user_data_dir("quant-ai")`), first-run seeding + bundled themes, doctor DB init, Explore empty state. Suite **339 passed** |

Unreleased commit chain: `00ecd8e` → `570f875` → `981c65f`, `1074e0d` → `1d527ee`
→ `707a1b6` → `3292226` → `caff829`, `2ea724d` → `3aff21d`, `79c2821` → `2e6e1b3`
→ `70d8c44` → `b929ae9` → `6affffc` → `352065f` → `d6e95f4` → `5c15c9e`
→ `b26a864` → `5844631` (last code) → `738f3d4` → `9e88de5` (docs).

### Product identity

- **Daily portfolio manager, not a trading terminal**: one snapshot after market
  close, plain-language add/trim/leave advice, orders in the broker app.
- **Browser-first**: `quant dash` is the product; the terminal is an optional
  power tool (three commands: `update`, `run`, `publish`).
- **No paid server**: hosted surface = static Published Briefing on GitHub Pages
  via free Actions cron; the interactive workspace runs locally.
- **Read-only UI for derived data**: the app writes only input files; the single
  documented exception is the startup names backfill (`ensure_display_names`),
  which opens one short-lived write connection and skips on lock contention.
- Once-daily cadence, long-horizon resource management; no urgency language.

### Doctrine (binding)

P1-P14 (P1 action/trust test; P2 no internal identifiers; P3 no per-line
provenance; P4 no silent defaults + explicit empty states; P5 verb-phrase
buttons; P6 plain error + remedy, raw text only in log/View log; P7 technical
data collapsed in Diagnostics; P8 one accent, semantic colors, no emoji/exclaims;
P9 mobile-first single column, tables ≤5 columns, full-width primary buttons;
P10 units inline, human dates, scores "67 / 100"; P11 friendly names primary;
P12 one job per page; P13 empty states guide the next step; P14 all strings in
[`quant/ui/copy.py`](quant/ui/copy.py), guarded by `tests/test_ui_copy.py`).
F1 (no synthesized identifiers; sources user files / validated live metadata /
curated files; `isin_source` provenance; `is_valid_isin` ISO 6166 gate). D1
(recorded-source tests: production default path against a checked-in fixture,
source stubbed at the boundary). Single-home rule (one message home per surface).

**Status vocabulary** from [`copy.status_for`](quant/ui/copy.py): On track /
Add / Trim / Waiting until {date} / Below minimum order / Blocked / Not reviewed
yet. Drift is computed once and shared by table, engine, statuses, and the S3
footnote; a cooldown surfaces (never silences); over-threshold drift is never
"On track".

**State machines**: Today S0-S4 and Explore states with exact catalog copy. Data
reads use `latest_review(ok_only=True)` (legacy artifacts without `review_status`
count as ok; only explicit "failed" excludes); S4/Health read the latest attempt
of any status; review dirs are timestamped dirs containing `metrics.json`.

**Feedback contract**: disabled verb-ing buttons; dedicated progress container
cleared on completion; one numbered outcome sentence; plain failure + collapsed
View log; `Open Today` only after success via `st.switch_page`.

**Mutex**: one acquisition per update+run sequence; `outputs/.runner.lock`
heartbeat refreshed every 30 s; stale after 600 s with auto-release; foreign live
heartbeat → other-tab sentence; own session → generic `An operation is already
running.`

### Governance

- **Commit discipline**: one commit per step, conventional messages; checkpoint
  per step with hash + `git diff --stat`; git-based evidence only.
- **Ruling protocol**: stop and report spec contradictions with a proposed ruling
  instead of improvising (recorded rulings: build on `b929ae9`; BROAD_ETFS in W;
  curated ISIN ingest; ruff scope).
- **Coverage ratchet**: floor 42.39% baseline, raise-only, target 80%
  ([`.coveragerc`](.coveragerc)).
- **Ruff gate (R9)**: zero new findings on changed files plus repo-wide F-codes
  fixed; legacy E/I/N deferred to a tracked issue.
- **Manual passes gate the next step**; user-reported deviations become hotfixes.
- **Golden-file policy**: regenerate only for intentional backtest changes;
  schema additions default so the golden payload never moves.
- **No fabrication**: screenshots, ISINs, names, test vectors only from real or
  recorded sources.

### Architecture and stores

Modules added this cycle: [`quant/ui/{copy,render,cards,search,runner}.py`](quant/ui/),
[`quant/pages/{today,portfolio,explore,settings}.py`](quant/pages/),
[`quant/data/{names,identifiers,registry_repair,news}.py`](quant/data/),
[`quant/reporting/{artifacts,actions,web}.py`](quant/reporting/),
[`quant/portfolio/{history,cash_rate}.py`](quant/portfolio/),
[`quant/cli/output.py`](quant/cli/output.py); [`quant/dashboard.py`](quant/dashboard.py)
is the navigation entry only.

- **Five artifact accessors** (the only UI read path): `latest_review(ok_only)`,
  `read_regime`, `read_actions`, `read_scores`, `read_history`.
- Stores: DuckDB (`market_history`, `asset_registry`, `portfolio_history`, nlp
  tables); `outputs/run_<ts>/` review dirs gated by `metrics.json`;
  `outputs/{update_state,names_state,news_state}.json`; news cache JSON (atomic
  temp + `os.replace`); `.runner.lock`.
- **Working universe W** (canonical; see "Registry write paths"): funnel
  survivors ∪ CORE ∪ ACTIVE ∪ portfolio ∪ broker_registry ∪ curated ∪ BROAD_ETFS;
  `universe_master` excluded everywhere; `sync_registry_to_working_universe`
  inserts missing W and prunes outside W (idempotent); bulk universe_master
  inserts refused; `discovery.run_discovery` no longer inserts the pool.
- Registry class precedence: CSV `instrument_class` wins over taxonomy; W inserts
  are classified with known names so keyword rules fire.
- Cells: NULL never empty string; symbol-valued display_name/name = missing and
  repaired.
- `quant doctor`: db/registry/missing counts, names state, metadata probes, news
  cache age + sentiment distribution, search probes (apple/amazon/gold/space/
  samsung), history bar probes, explore-card probes, actions oracle, runner lock
  with stale detection.

### Input file contracts

- [`data/portfolio.csv`](data/portfolio.csv): broker truth (value, PnL; invested
  derived).
- [`data/account.yaml`](data/account.yaml): `base_currency`, `cash_eur`,
  `risk_profile` (conservative/balanced/aggressive), optional `savings_plan_day`
  (R8, pending).
- [`data/broker_registry.csv`](data/broker_registry.csv): routing (bounded;
  ceiling guard; `isin_source` provenance).
- [`data/isin_curated.csv`](data/isin_curated.csv),
  [`data/names_curated.csv`](data/names_curated.csv),
  [`data/themes.csv`](data/themes.csv): curated inputs with audit trail; all
  un-ignored in [`.gitignore`](.gitignore).
- Cash rate: dated schedule in [`quant/portfolio/cash_rate.py`](quant/portfolio/cash_rate.py)
  (2.25% from 2024-01-01, 2.50% from 2026-09-16), injectable live fetch with
  schedule fallback.

### Bug ledger (symptom → root cause → fix → guard)

| ID | Symptom | Root cause | Fix | Guard |
|---|---|---|---|---|
| CI-1 | test_cache CatalogException | schema never initialized in tests | conftest init_db fixture + ephemeral DB patching both path bindings | conftest |
| LK-1 | update fails: DuckDB lock held by UI | UI held persistent write connection | read-only short-lived connections, writer retry, runner mutex | test_ui_connections |
| RG-1 | regime 0.50/unknown silent | fallback prior shown as measurement | unconditional regime block with states | test_regime_masking |
| CP-1 | "(source: csv)", "Regime Regime:", daily PnL 0.00, bucket banner | spec-first widgets, dual truths | copy catalog, single sources, deletions | test_ui_copy |
| VS-1 | red primary, donut semantic colors, clipped legends | palette misuse | primaryColor #1F3B73, categorical palette, ≥5% labels, legend rules | chart tests |
| DP-1 | "Gold Miners (GDX) (GDX)" | formatter appended ticker already present | `label_for` dedup (incl. dotted base) ; banned tokens | test_ui_copy |
| SR-1 | "amazon"/"apple" no matches live | names empty in live registry; backfill only at update | startup `ensure_display_names`, clean at render | test_search_real_registry |
| PS-1 | names populated but tickers shown | 25 rows poisoned display_name==symbol | symbol-valued = missing + repair; D1 recorded test | test_names_recorded_source |
| RG-2 | registry 1090 rows, 1000 blank | bulk universe_master insert | W sync + prune + bounded test; discovery root cause | test_registry_bounded |
| RG-3 | registry under-inclusive (32) | rollback pruned to CSV set only | sync both directions to W; ISIN/currency heal over W | test_registry_bounded |
| CL-1 | progress-line leaks, header not first | unguided prints | reporter.detail, TTY-gated progress, header-first | test_cli_hygiene |
| PB-1 | "published X" then X missing | run_latest link not verified | publish verifies + prints real path, exit 1 on failure | test_publish |
| RV-1 | phantom history row; Today S4 vs Settings silence | status not persisted; history written on failure | `review_status` artifact; no history on failure | test_review_status |
| NW-1 | news dead; review/Explore divergence; corruption risk | no coverage; two paths; non-atomic writes | shared `load_news`, atomic writes, outage counter + Health line | test_explore_news |
| NT-1 | raw tz timestamps in news rows | formatter missed RFC-2822/ISO | `fmt_weekday_ts` both formats | test_news_format |
| SN-1 | all rows "neutral" | no scorer provenance | per-entry `scorer: model|default`; word only for model; doctor distribution | test_h3_6 |
| SH-1 | scores vanished after a failed review | latest-dir shadowing | `latest_review(ok_only=True)` + as-of line | test_h3_6 |
| LG-1 | Today S0 with six reviews | ok_only excluded legacy artifacts | missing status = legacy ok; unified reads | test_h3_7 |
| ST-1 | Settings line stale after refresh | read review-era state | `update_state.json` + live compose | test_h3_7 |
| PH-1 | failed row still listed (amended, Ruling A) | false S4 card pre-H3.3; NO phantom row existed (the 23:24 review was genuine ok) | H3.3 no-history-on-failure prevents future phantoms; L3 keeps legacy ok rows; read-side filter | test_h3_7 |
| AO-1 | "From the review of …, 00:00" | bogus time, non-catalog string | `fmt_review_ts`, single catalog as-of string, banned token | test_h3_7/8 |
| CH-1 | illegible annotation; "13 Sep" x5 | in-plot gray text; day-only ticks | top-margin white box; hh:mm under 3 days | test_h3_7 |
| SF-1 | Settings "No market data" mid-refresh | transient locked read erased state | fallback to update_state | test_h3_8 |
| RV-2 | Reviews list empty | legacy ts parsing in L3 filter | parsing fix | test_h3_8 |
| OC-1 | "A review is already running" during refresh | operation-specific copy | generic operation copy | test_h3_8 |
| AC-1 | zero actions on a drifting portfolio (core value) | time gate silenced advice; drift/base inconsistency | drift-driven advice; cooldown → Waiting + footnote; shared drift; doctor actions oracle | test_h3_8 recorded oracle |
| HD-1 | "Review of 14 Sep close, prepared 13 Sep" | header from two runs | single-artifact header; legacy renders prepared line | test_h3_8 |
| GR-1 | Growth annotation "(0.00 EUR)" | mode-confused | percent-only in Growth | test_h3_8 |
| DS-1 | discovery sentence dead live | index lacked names | cached universe_master name backfill feeding index | test_h3_8 |
| MH-1 | "No price history" suspicion | hypothesis | disproven (charts read market_history); regression test | test_h3_8 |
| CU-1 | SGLN.L "Stock", missing currency/ISIN | taxonomy over CSV; empty strings | CSV class precedence; NULL not ''; named classification | test_h3_5 |
| CU-2 | cannot track Apple (closed universe) | W prune correct; no UI path | discovery sentence + Portfolio add-input candidates "- not tracked yet"; RESOLVED LIVE 14 Sep: AAPL tracked via discovery loop, fetched, charted (registry 89) | test_h3_5 |
| IS-1 | 5J50.DE held but unroutable; false remedy | no registry row; remedy lied | `ensure_registry_rows`; curated ingest; repair single-home Settings | test_registry_repair |
| NW-2 | model available yet every news entry `scorer: default`; no word ever shown | the fetch write path never invoked the scorer | `load_news` gains an injectable scorer/scorer_factory, batch-scores headlines per symbol: model -> `scorer: model`, absent/failure/timeout -> `default` + one log line | test_h4 |
| EX-1 | external URL printed; deprecation warnings | bind default; old API | localhost default, `--lan`, README proxy warning; `width="stretch"` | manual + lint |
| OB-1 | Open Today dead; leftover progress; "advice below" empty | session-state advice | artifact advice + `st.switch_page`; container cleared | test_feedback_contract |

### Feature contracts shipped

- **Charts**: 1M/3M/1Y/Max (default Max), dotted baseline, mode-aware annotation
  (percent-only in Growth), 3-point rule, Value|Growth rebase-to-100, holdings
  multi-select ≤3, benchmark IWDA toggle default off, hh:mm axis under 3 days.
- **News**: review covers holdings via the shared path; Explore on-demand 24 h
  cache (spinner, 10 s timeout); cap 5 + `Earlier items ({n} more)`; weekday
  dates; scorer provenance; outage Health line after three consecutive failures.
- **Search**: corpus display_name + name + symbol + ISIN + themes; `data/themes.csv`
  prose tags with theme-to-theme links; exact zero-match sentence; discovery
  sentence for universe_master matches.
- **Evidence**: per-symbol news list; review evidence summary; glossary expander
  only with scores.

### Test suite evolution

145 → … → 262 → 273 → 278 → 286 → 294 → **304 passed**. Golden
[`tests/golden/backtest_2024.json`](tests/golden/backtest_2024.json) unmoved
throughout. Key guards: `test_run_to_ui`, `test_ui_copy`, `test_registry_bounded`,
`test_names_recorded_source` (D1), `test_feedback_contract`, `test_review_status`,
`test_explore_news`, `test_chart_rules`, `test_themes`, `test_news_format`,
`test_h3_5/6/7/8`, `test_doctor`, `test_golden_snapshot`.

### Current state and open items

Released **10.6.1** (F-series fresh-install hotfix; tag `v10.6.0` marks 10.6.0);
suite 339; ruff zero new on changed files plus repo-wide F-codes fixed (legacy
E/I/N/UP deferred);
live doctor healthy: W=90, names clean, 18 non-routable ISINs explained, search
probes correct, actions oracle 2, sentiment cache 30 default-scorer (legacy),
cards correct.

1. **R8 — DONE** (`2022302`, feat(today)). Calendar lines: optional
   `savings_plan_day` (1-31) in `data/account.yaml`; `copy.savings_plan_line`
   (same day -> `SAVINGS_TODAY`; month wrap + month-end clamp; the date carries
   its weekday per the audit rule); `routing.holding_routes_to_savings_plan`
   reuses `route_signal`; Today shows the countdown under the actions block only
   when a holding routes to a savings plan; the header gains the markets-closed
   freshness line on a non-trading day after the previous session. README key
   documented. Guard `tests/test_r8_calendar.py`.
   Ruling R8-DATE: the explicit audit rule ("every rendered date carries its
   weekday via `fmt_weekday_date`") as it applies to the savings date.
2. **R9** release: CHANGELOG narrative, bump 10.6.0, `mkdocs --strict`, ruff gate
   per settled scope, checklist (D1 audit, names clean on fresh install, publish
   path assertion, CLI ≤20 lines per command, `git ls-files data/` audit, bounded
   test, codecov badge either uploading or removed), tag `v10.6.0`.
3. **H3.8 manual-pass matrix** — REPORTED 14 Sep 2026: 7 PASS (1 cooldown footnotes
   + Waiting statuses; 3 5J50.DE chart; 4 Settings stable; 5 six genuine ok rows;
   6 apple tracked via discovery; 7 news weekday/cap/word). Item 2 (Growth IWDA)
   evidence accepted via `test_chart_rules` (Ruling B). Item 8 FAILED -> H4.
   Live: `Scores as of Monday 14 Sep 2026.` with AMZN/GDX bars after the 14 Sep
   review (N1 verified).
4. **H4 — DONE** (`441271a`, fix(news)). Scoring write path was unwired:
   `fetch_news_items` hardcoded `default` and `load_news` (the only cache writer)
   never called the scorer. Fixed: injectable scorer boundary in `load_news`,
   batch per symbol, model available -> `scorer: model`, absent/failure/timeout
   -> `default` + one log line. Guard `tests/test_h4.py` (model/absent/timeout/
   cache-hit). Remains: live manual verify (Explore AMZN words after cache
   expiry; doctor `scorer model > 0`).
5. Follow-up issues (non-gating): capture `docs/assets/today.png`; curated ISINs
   beyond 5J50.DE; legacy ruff E/I/N; PyPI name + Trusted Publishing; GitHub
   topics; coverage ratchet toward 80.
6. **Step 3 — launch**: exact topics + description + Pages link, repo-metadata
   and follow-up-issue `gh` commands, the 10.6.0 launch post + demo script. Live
   briefing link verified HTTP 200 (https://akmal523.github.io/quant/). Repo
   metadata and issue creation need an authenticated `gh` (unauthenticated in the
   agent session).

### Continuation protocol

Checkpoint format (changed/verified/remains with hashes + diffstat); one commit
per step; ruling protocol for contradictions; manual passes gate steps; doctor
first for any live anomaly; never fabricate identifiers, names, screenshots, or
test vectors; doctrine P1-P14, F1, D1 binding; the W definition is immutable
without a recorded ruling.

---

## v10.6.2 Domain Additions (Three-Tier Architecture)

### Rulings

- **R-TIER-1 (tier model).** The legacy 4-tier system (CORE / SATELLITE /
  ACTIVE / SECTOR) is replaced by the 3-tier system (FORTRESS / ALPHA /
  SPECULATIVE). Tier assignments live in a new user-editable
  [`data/tiers.csv`](data/tiers.csv) (`symbol,tier,last_updated,notes`);
  [`data/portfolio.csv`](data/portfolio.csv) stays broker-synced and untouched.
  Legacy mapping: CORE -> FORTRESS; SATELLITE / ACTIVE / SECTOR -> ALPHA;
  default ALPHA. The legacy `classify_asset` and `enhanced_portfolio_audit` are
  retained unchanged for backward compatibility.
- **R-PDF-1 (report export).** The Weekly Friday Report ships as Markdown plus a
  self-contained HTML file with print CSS. The user prints to PDF from the
  browser. No PDF library is added; new lightweight deps are `markdown` and
  `jinja2`.
- **R-DSR-1 (deflated Sharpe).** The v10.6.2 spec's
  `Phi^-1(1 - p_adj/2) * se` form returns +inf for a highly significant result
  (p_adj underflows to 0). The standard expected-max-Sharpe deflation is used
  instead; it is finite and returns the raw Sharpe for a single trial.

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **Fortress** | Tier 1. Eternal holdings, never sold to avoid capital gains tax. Broad ETFs + 3-5 core stocks via Sparplan. Structural grade only; tactical and NLP ignored. Quarterly review of Sparplan amounts only. |
| **Alpha** | Tier 2. Active accumulation, the liquid reserve. Full scoring pipeline; weekly rebalancing on Fridays; sell when cash is needed. |
| **Speculative** | Tier 3. High-risk bets, hard 2 percent cap. Momentum and volume only. Stop-loss -50 percent, take-profit +100 percent. |
| **tiers.csv** | `data/tiers.csv`: user-editable tier assignments. The single source of the 3-tier classification. |
| **Conviction** | Signal-quality word for Alpha: `0.3*structural + 0.4*tactical + 0.3*nlp`; HIGH > 75, MEDIUM > 60, else LOW. |
| **Liquidity score** | 0-100 measure of how quickly/cheaply an asset can be sold. `0.4*volume + 0.4*spread - time_penalty`. |
| **Weekly VaR** | 5-day VaR: `VaR_daily * sqrt(5)`. The Alpha risk horizon. |
| **Signal cache** | `outputs/signal_cache.json`: Friday signals served Monday through Thursday. |
| **Weekly report** | `outputs/reports/weekly_<date>.md` + `.html` (print-to-PDF). |

### Module dependencies (v10.6.2)

```
quant.portfolio.tier_manager ─> quant.paths, quant.config
quant.portfolio.fortress ─────> quant.analytics.scoring, quant.config
quant.portfolio.alpha ────────> quant.analytics.scoring, quant.portfolio.risk
quant.portfolio.speculative ──> quant.config
quant.portfolio.signal_cache ─> quant.paths, quant.portfolio.alpha
quant.portfolio.portfolio ────> quant.portfolio.tier_manager, quant.portfolio.{fortress,speculative}
quant.reporting.weekly_report > quant.portfolio.{portfolio,risk}, markdown, jinja2
quant.reporting.actions ──────> quant.config (tier caps)
quant.execution.reconciliation > quant.execution.tca
quant.ui.render ──────────────> quant.portfolio.{portfolio,risk}, quant.ui.copy
quant.cli ────────────────────> quant.reporting.weekly_report
```

### Source-of-truth additions (v10.6.2)

| Fact | Writer | Reader(s) |
|------|--------|-----------|
| Tier assignment | user via `data/tiers.csv` / migration script | tier_audit, actions, UI, report |
| Friday signals | `quant run` (Friday) | signal cache, UI, report |
| Weekly report | `quant weekly-report` | user (browser print-to-PDF) |

### Test suite (v10.6.2)

21 tests in [`tests/test_three_tier.py`](tests/test_three_tier.py) across 7
categories: look-ahead bias, survivorship bias, mathematical soundness, data
integrity, behavioral biases, economic realism, system robustness. Coverage
ratchet raised 42 -> 55 (measured 57.64 percent).

---

## v10.6.3 Domain Additions (Three-Tier Completion and Polish)

### Corrections

- The three-tier release is **10.6.2** (not 10.6.22). Every reference was
  renamed. This release is **10.6.3**.
- **No emoji anywhere.** [`tests/test_no_emoji.py`](tests/test_no_emoji.py)
  scans the whole project (quant, scripts, tests, docs, root markdown) and the
  full emoji ranges (pictographs, misc symbols, dingbats, variation selectors).

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **Unclassified asset** | A symbol in `portfolio.csv` with no row in `tiers.csv`. Detected and given a recommended tier (ETF/CASH to FORTRESS, EQUITY to ALPHA, else SPECULATIVE). |
| **Tier validation** | `validate_tiers_csv`: missing columns, invalid tiers, duplicates, orphans, and allocation-cap violations. |
| **Tier repair** | `repair_tiers_csv`: dedupe, drop orphans, default invalid tiers to ALPHA. Does not change allocations. |
| **Emergency sell plan** | `emergency_sell_plan`: tier-aware order (ALPHA, SPECULATIVE, then FORTRESS as a last resort with a tax warning). |
| **Tax-aware prioritization** | `prioritize_sells_with_tax`: losers first, then winners within the 1000 EUR Freistellungsauftrag. |
| **Stale signal cache** | A cache older than `max_age_days` (default 7) is invalidated. |
| **Weekly trade cap** | `check_alpha_weekly_limit` returns `{allowed, trades_remaining, warning, override_required}`; `track_weekly_trades` counts the week's trades from `trade_log`. |
| **Batch scoring** | `batch_score_assets` scores many assets in parallel, routed by tier, with a cached structural grade. |

### Module dependencies (v10.6.3)

```
quant.portfolio.tier_manager ─> quant.execution.taxonomy (get_instrument_class)
quant.portfolio.risk ─────────> quant.portfolio.tier_manager (tier_map)
quant.portfolio.behavioral_guardrails ─> quant.data.database (trade_log)
quant.analytics.scoring ──────> quant.portfolio.{alpha,speculative}
quant.reporting.weekly_report > quant.portfolio.risk (emergency_sell_plan)
quant.ui.render ──────────────> quant.portfolio.{tier_manager,behavioral_guardrails}
quant.cli ────────────────────> quant.portfolio.tier_manager (validate/repair)
```

### Test suite (v10.6.3)

13 edge-case tests in [`tests/test_v10_6_3.py`](tests/test_v10_6_3.py):
unclassified detection, auto-assign, tier validation/repair, allocation limits,
empty-portfolio emergency plan, all-FORTRESS plan, tax-aware prioritization,
empty-portfolio report, stale cache, the dict guardrail, `track_weekly_trades`,
and batch scoring.

---

## v10.6.4 Domain Additions (Auto-Balance and Final Polish)

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **Tier allocation** | Per-tier value, percent, limit, and violation flag from `analyze_tier_allocations`. |
| **Auto-balance suggestion** | A proposed tier reassignment that reduces a violated tier. Advisory only; the user approves or rejects each one. |
| **Apply rebalance** | `apply_rebalance_suggestions`: change the tier of approved symbols in `data/tiers.csv` only. Never executes a trade. |
| **Safe tier load** | `load_tiers_safe`: never crashes on a missing, empty, or corrupted tiers file; returns `(tiers_df, warnings)`. |
| **Corruption repair** | `_repair_corrupted_csv`: recover valid rows from a malformed tiers file. |
| **Health check** | `run_health_check`: data freshness, tiers validation, tier allocations, database integrity, signal cache, data source. Status in {HEALTHY, WARNING, CRITICAL}. |
| **Parallel batch scoring** | `batch_score_assets_parallel`: process pool for 50+ assets; thread path below 20. |

### Module dependencies (v10.6.4)

```
quant.portfolio.autobalance ──> quant.config, quant.execution.taxonomy
quant.cli.health ─────────────> quant.data.database, quant.portfolio.{portfolio,tier_manager,autobalance,signal_cache}
quant.analytics.scoring ──────> quant.portfolio.{alpha,speculative} (parallel workers)
quant.ui.render ──────────────> quant.portfolio.autobalance
quant.cli ────────────────────> quant.portfolio.autobalance, quant.cli.health
```

### Source-of-truth additions (v10.6.4)

| Fact | Writer | Reader(s) |
|------|--------|-----------|
| Tier reassignment | user via `quant apply-rebalance` / UI / wizard | tier_audit, actions, UI, report |
| Health status | `quant health-check` (read-only) | user |

### Test suite (v10.6.4)

8 tests in [`tests/test_autobalance.py`](tests/test_autobalance.py) and 4 tests
in [`tests/test_v10_6_4.py`](tests/test_v10_6_4.py) (batch-scoring performance,
safe load, corruption repair, health check).

---

## v10.6.5 Domain Additions (Final Optimization and Polish)

No new features; behavior is preserved.

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **Batch query** | `batch_query_portfolio_data`: one `IN (...)` query for many symbols (prices from `market_history`). |
| **Lazy load** | `load_prices_lazy`: stream `market_history` rows in chunks to bound memory. |
| **Optimized batch scoring** | `batch_score_assets_optimized`: one query, then parallel scoring. |
| **Graceful degradation** | `score_asset_with_fallbacks`: neutral fallback values with a `data_quality` marker (FULL / PARTIAL / FALLBACK). |
| **Disk cache** | `disk_cache`: JSON result cache keyed by function name and arguments, with a max age. |
| **Error hierarchy** | `quant.errors`: `QuantError` and subclasses. Existing functions keep their contracts. |
| **Retry** | `quant.utils.retry.retry_with_backoff`: exponential backoff for transient external failures. |

### Module dependencies (v10.6.5)

```
quant.errors ─────────────────> (pure)
quant.utils.retry ────────────> (pure)
quant.analytics.cache ────────> quant.paths
quant.data.database ──────────> (batch query, lazy load)
quant.analytics.scoring ──────> quant.data.database (batch query)
quant.cli ────────────────────> quant.analytics.cache
```

### Test suite (v10.6.5)

3 integration tests in [`tests/test_integration.py`](tests/test_integration.py),
4 stress tests in [`tests/test_stress.py`](tests/test_stress.py), and 3
performance tests in [`tests/test_performance.py`](tests/test_performance.py).

---

## v10.7.0 Domain Additions ("The System That Talks Sense")

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **Invested pool** | Everything held at the broker. The ONLY pool that is scored, weighted, charted, and advised. All targets and limits apply to this pool only. |
| **Savings-plan budget** | A monthly amount the user decides from salary. A FLOW into Invested, not a balance. Sparplan buys cost 0 EUR on the buy side. |
| **Operational cash** | Daily-life money at the broker. Earns the cash yield. NEVER an investment buffer: never in weights, never in the donut, never a source of forced trades. |
| **Four channels** | Daily silent monitor, Friday weekly summary, Monthly decision, and rare urgent alerts. Nothing else may interrupt the user. |
| **Level-triggered alert** | A condition that inspects the current state (not intraday events), so a monitoring gap delays an alert but never loses it. |
| **Modified Dietz** | Return over a period that excludes deposits: `R = (V_end - V_start - F) / (V_start + sum(w_i * F_i))`. Deposits do not count as profit. |
| **Sizing laws** | Section 6 rules: FORTRESS never sells; sell = min(drift, value - 1 EUR) rounded down to 5 EUR, suppressed below 25 EUR; positions below 100 EUR are untouchable; cooldown blocks sells only; buys round to 10 EUR; active buys need HIGH conviction or the money goes to cash. |
| **Advice record** | The self-scoring honesty ledger: resolved alerts are priced 30 days later and scored correct/wrong. |
| **Alert kinds** | `structural_break`, `tactical_collapse`, `position_crash`, `regime_flip`, `speculative_stop`. Each is level-triggered and stays open until resolved. |
| **Sizing laws** | FORTRESS never sells; sell = min(drift, value - 1 EUR) rounded down to 5 EUR, suppressed below 25 EUR; positions below 100 EUR untouchable; cooldown blocks sells only; buys round to 10 EUR; active buys need HIGH conviction. |

### v10.7.0 rulings

Ambiguities were resolved conservatively (simpler, more honest, more
conservative with the user's money):

- **Trading calendar**: weekend-only rule (Saturday/Sunday skip). No holiday
  calendars yet.
- **Cash in weights**: operational cash is excluded from all weights, targets,
  and the distribution donut. It competes only inside the Monthly allocator.
- **Cash floor**: the old `cash floor 10%` constraint is removed from
  [`quant/config.py`](quant/config.py) and all risk-profile displays. Risk
  profiles now describe the INVESTED pool only as a 3-tuple
  `(long_term_min, active_max, max_position)`.
- **Alert conditions**: level-triggered, evaluated on the latest snapshot.
- **Telegram**: sent with the standard library (`urllib`); no new dependency.
- **Scheduler**: systemd user timer is the default; a cron fallback is
  documented for non-systemd machines.
- **Golden backtest**: the optimizer bucket constraints were renamed to the
  invested-only meaning; the golden path does not use them, so
  [`tests/golden/backtest_2024.json`](tests/golden/backtest_2024.json) is
  unmoved.

---

## v10.7.1 Domain Additions ("Finish the Redesign")

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **Advice** | One record from `quant/engine/advice.build_advice`: `kind` (buy/top_up/sell_part/keep/change_savings_plan/to_cash), symbol, company name, EUR, from_where, why, fee, tier word, source. |
| **RejectedNote** | A considered-but-suppressed advice (symbol, considered action, plain reason). The Overview "Not this week" block renders exactly these. |
| **One advice pipeline** | Every surface that tells the user to do something renders from `build_advice`; no surface computes its own TRIM/BUY labels. |

### v10.7.1 rulings

- **Legacy adapter**: `quant/reporting/actions.build_actions` is kept as a thin
  adapter over `build_advice` so older callers (the web briefing, the run status
  map, the doctor probe) keep working. It preserves the legacy status semantics
  (WAITING/ADD/TRIM) while the action word comes from the dictionary.
- **Legacy audits**: an audit row without `Value_EUR` uses a nominal value, and
  without `Current_Weight` derives the current weight from `Target_Weight +
  Drift`, so the sizing laws still apply to legacy fixtures.
- **Candidate grouping**: Find investments groups funnel survivors by structure
  (>= 75) and tactics (>= 70); the same list feeds the Monthly decision.
- **Golden backtest**: unmoved.

---

## v10.7.2 Domain Additions ("Real-World Hardening")

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **Runner lock** | `outputs/.runner.lock`: one shared lock owned by `quant.engine.lock`. Carries `pid`, `started_at` (ISO), `command` (plus the legacy `owner`/`ts`). Every writer respects it. |
| **Takeover** | Taking the lock from a dead pid (immediate) or a live pid above `HARD_LOCK_MINUTES` (with a warning). Recorded so the doctor renders the exact phrase. |
| **write_connection** | `quant.data.database.write_connection()`: a short-lived read-write connection that closes in `finally` and retries on DuckDB lock/IO errors. |
| **last_failed_run** | `outputs/.daily_last_failed`: the plain reason a daily run could not acquire the database. The morning slot retries. |
| **news_pillar** | Persisted status `active` | `absent`, recomputed on the Friday run. Absent when zero model-scored items in the last 30 days. |
| **Backup archive** | `data/backups/quant-backup-YYYYMMDD-HHMM.tar.gz`: the user-owned state, pruned to the 5 most recent. |

### v10.7.2 rulings

- **Lock thresholds.** `STALE_LOCK_MINUTES = 90`, `HARD_LOCK_MINUTES = 240`. A
  dead pid is always taken over immediately. A live pid at or below the stale
  threshold is respected (retry 5 attempts over about 60 seconds, then abort
  gracefully). A live pid above the hard cap is taken over with a warning. The
  middle band (above stale, at or below hard) behaves like the below-stale case
  (retry then abort): the system never steals a live process's lock unless it is
  clearly runaway. `PermissionError` from the liveness probe counts as alive.
- **One lock, one owner.** The UI's old 10-minute heartbeat window is replaced by
  the shared lock. The UI holds the lock and tells its spawned subprocess it
  inherited it via `QUANT_LOCK_HELD=1`, so the subprocess never deadlocks against
  its own parent. `quant all` re-enters the lock for the same pid.
- **News-pillar wiring finding (Part 2.2).** The wiring EXISTS: H4 wired
  `load_news` to score headlines, and `quant run` scores holdings through
  `NLPScorer`. The zero model-scored count is data/legacy-caused: the daily timer
  never fetches news (only the review and Explore do), and the existing cache
  entries are legacy `default`. No pipeline change was made. The news cache
  `scorer` field (`model` | `default`) is the `scored_by` provenance the
  diagnostic reads; `nlp_scores` keeps its `doc_hash, score` shape.
- **Backup member list.** `data/portfolio.csv`, `data/tiers.csv`,
  `data/account.yaml`, `quant_cache.duckdb` (plus `quant_cache.duckdb.wal` when
  present). `data/notify.toml` is included ONLY with `--include-secrets`. Members
  are stored at their project-relative paths; restore unpacks the archive over
  the project folder (the one containing `data/`), then runs `quant doctor`.
- **Tier source of truth (Part 5.1).** The audit's `Tier` column comes from the
  legacy `classify_asset` (CORE/SATELLITE/ACTIVE/SECTOR), so a FORTRESS symbol
  could be misread as ALPHA and sold. `build_advice` now resolves the tier from
  the `tiers` argument (tiers.csv) first; the holding's tier is only a fallback.
  `build_actions` loads the `tiers.csv` map and passes it. A FORTRESS sell can no
  longer be expressed by any consumer.
- **Holdings meta sync (Part 5.2).** The run pipeline seeds `holdings_meta` from
  the broker CSV with `first_time_only=True`: a symbol already present keeps its
  recorded `sync_date`, so the 35-day reminder still reflects the user's last CSV
  export. `quant doctor` shows the last broker sync date.
- **Golden backtest**: unmoved.

---

## v10.7.3 Domain Additions ("UI Truth Pass")

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **display_name** | `quant/data/names.py`: the ONE resolver (registry `display_name` -> cached yfinance `longName` probe -> symbol). Every user-facing surface calls it. |
| **TARGET_WEIGHTS_INVESTED** | `quant/config.py`: the ONE per-symbol invested-pool target map. `build_advice`, the allocator, the split reason lines, and the drift computations all read it. |
| **target_pct** | An optional column in `data/tiers.csv` that overrides a symbol's `TARGET_WEIGHTS_INVESTED` value (a value > 1 is read as a percent). |
| **Estimated value** | `shares * latest close`, shown when `holdings_meta` is newer than the broker CSV. Labeled "estimated, as of <date>". |
| **Broker statement** | The four fields the user typed, exactly as the broker reported them at the last sync. The truth for profit and loss. |
| **Ad-hoc buy** | A purchase recorded before the monthly plan is approved. Flow type `buy`, source `ad-hoc`; no reconciliation deviation line. |

### v10.7.3 rulings

- **One target map (Part 5).** `TARGET_WEIGHTS_INVESTED` is the single source of
  per-symbol invested-pool targets, seeded `EUNL.DE 0.50, SXRV.DE 0.20, AMZN
  0.20, 5J50.DE 0.10` and overridable by an optional `target_pct` column in
  `data/tiers.csv`. The equal-split-within-Fortress logic is removed. The
  allocator and `build_advice` therefore name the same top-up symbol; the
  EUNL-vs-5J50 contradiction is resolved.
- **Estimated-value rule (Part 3).** When `holdings_meta` has a sync or actuals
  newer than the broker CSV (or the pending-sync marker exists), the Overview
  Invested plaque, the holdings Value column, the per-asset expanders, and the
  average entry price read the REVALUED estimate, labeled "estimated, as of
  <date>". The broker statement table keeps the CSV numbers, labeled "broker
  statement, as of <sync date>". `enter_actuals` ADDS to the existing estimated
  position, so a recorded buy moves the visible balance.
- **Verdict law (Part 4).** The holdings Verdict column renders ONLY from
  `build_advice` kinds; the legacy status mapping is removed from the render
  path. A FORTRESS row can only show "Keep, do nothing" or the savings-plan
  top-up sentence.
- **One top-up sentence (Part 4.2).** `copy.STEP_TOP_UP` is rendered by both
  `quant run` and the Overview steps, so the two surfaces can never disagree.
- **Pre-approval actuals (Part 6.1).** Actuals saved while the month is not
  approved are ad-hoc buys; the reconciliation deviation line never appears for
  them.
- **Golden backtest**: unmoved.

---

## v10.7.4 Domain Additions ("Classification and Decision Test Grid")

### Glossary

| Term | Canonical Meaning |
|------|-------------------|
| **Decision grid** | The end-to-end test suite that proves a ticker enters, is classified, flows through scoring and advice, and yields (or suppresses) actionable signals under every realistic portfolio state. |
| **Advice set** | The set of advice kinds a tier may produce; grid rows assert the SET, not a single action. |
| **Fee hurdle** | A buy is allowed only when `expected_alpha_bps / 10000 * amount_eur >= fee_eur`. |
| **Re-arm** | A dismissed alert may reopen only after its underlying condition reads false at least once, then true again. |
| **Budget conservation** | The allocator's legs sum to the budget exactly; the rounding residual goes to cash. |

### v10.7.4 rulings

- **R1 (numeric rows).** The sizing laws are the contract; the spec's table
  numbers were illustrative and are corrected. 500 EUR at +6.9 percent drift
  yields a 30 EUR sell (min of drift and value minus 1, floored to 5-EUR steps,
  suppressed below 25); a 45 EUR buy rounds to 40. The fee hurdle boundary is
  inclusive (`>=`).
- **R2 (FORTRESS far over target).** A FORTRESS holding more than 10 points over
  target gets `change_savings_plan` with the wording "Consider lowering or
  pausing the savings-plan leg for <name>; it is X percent of invested vs Y
  percent target." It never implies selling. Under-target gaps keep the top-up
  wording.
- **R3 (ALPHA MEDIUM under target).** Emit an explicit `keep` plus a rejected
  note ("conviction MEDIUM, needs HIGH for a buy"). Silence is forbidden for any
  action that was considered. LOW conviction generates no advice.
- **R4 (buy below 100 EUR).** The untouchable law applies to ACTIVE buys: no buy,
  rejected note "position below 100 EUR; the 1 EUR fee makes small buys
  inefficient." Savings-plan legs (fee 0) are exempt.
- **R5 (SPECULATIVE advisories).** A take-profit advisory at +100 percent and a
  cap-violation advisory when the bets weight exceeds 2 percent. Both are notes,
  never forced sells.
- **R6 (emergency ordering).** At equal liquidity, losers (negative broker PnL)
  sort before winners. The order stays stable and deterministic.
- **R7 (sync reminder).** The 35-day reminder appears at most once per 7 days,
  never on the same day as a successful sync, and lists pending estimated
  positions by name. The throttle stamp lives in the `meta` table.
- **R8 (dismissed alerts).** A dismissed alert must not reopen unless re-armed:
  the underlying condition must read false for at least one full run and then
  become true again. The `alerts.condition_cleared` column records the re-arm.
- **R9 (regime override).** In a bear regime, HIGH-conviction ALPHA buys are
  suppressed; the pipeline emits a single `to_cash` advice plus one rejected note
  per suppressed buy. Sells and FORTRESS savings-plan advice are unaffected. The
  regime input exists to override micro signals in adverse macro conditions;
  allowing HIGH-conviction buys in bear would make the regime decorative and
  would let the advice pipeline contradict the monthly allocator, which already
  routes the active pool to cash in bear.
- **R10 (allocator).** The long base goes to the SINGLE largest-gap FORTRESS
  holding (ties break alphabetically by symbol); with no FORTRESS holding it goes
  to one broad ETF. The split sums to the budget exactly (the residual goes to
  cash). Every leg's reason states the actual gap in percentage points. The bets
  pool is carved out of the active pool.
- **Golden backtest**: unmoved.

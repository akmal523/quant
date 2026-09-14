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
| 10.5.2 + unreleased (→10.6.0) | R1-R9 program and H2/H3 hotfix cycles (below). HEAD `5844631`, suite **304 passed**, version string still 10.5.2 |

Unreleased commit chain: `00ecd8e` → `570f875` → `981c65f`, `1074e0d` → `1d527ee`
→ `707a1b6` → `3292226` → `caff829`, `2ea724d` → `3aff21d`, `79c2821` → `2e6e1b3`
→ `70d8c44` → `b929ae9` → `6affffc` → `352065f` → `d6e95f4` → `5c15c9e`
→ `b26a864` → `5844631`.

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
| PH-1 | failed row still listed | pre-H3.3 DB row | read-side filter to ok rows | test_h3_7 |
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
| CU-2 | cannot track Apple (closed universe) | W prune correct; no UI path | discovery sentence + Portfolio add-input candidates "- not tracked yet" | test_h3_5 |
| IS-1 | 5J50.DE held but unroutable; false remedy | no registry row; remedy lied | `ensure_registry_rows`; curated ingest; repair single-home Settings | test_registry_repair |
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

HEAD `5844631`; version string 10.5.2; suite 304; ruff zero new on changed files;
live doctor healthy: W=90, names clean, 18 non-routable ISINs explained, search
probes correct, actions oracle 2, sentiment cache 30 default-scorer (legacy),
cards correct.

1. **R8** calendar lines (one commit): `savings_plan_day` countdown + same-day
   variant, markets-closed freshness line, README key.
2. **R9** release: CHANGELOG narrative, bump 10.6.0, `mkdocs --strict`, ruff gate
   per settled scope, checklist (D1 audit, names clean on fresh install, publish
   path assertion, CLI ≤20 lines per command, `git ls-files data/` audit, bounded
   test, codecov badge either uploading or removed), tag `v10.6.0`.
3. **H3.8 manual-pass matrix** (gates R8): Today cards or cooldown footnote on the
   live portfolio; Growth IWDA line + percent-only annotation; 5J50.DE history;
   Settings line stable during refresh; Reviews six ok rows; apple discovery
   sentence.
4. **Verify post-H3.6 fetches write `scorer: model`** (live cache still all
   default; if new fetches stay default, the scoring write path is unwired).
5. Follow-up issues (non-gating): capture `docs/assets/today.png`; curated ISINs
   beyond 5J50.DE; legacy ruff E/I/N; PyPI name + Trusted Publishing; GitHub
   topics; `.rooignore` manual entries; coverage ratchet toward 80.
6. Launch sequence post-tag: topics, launch post, demo, the two issues above.

### Continuation protocol

Checkpoint format (changed/verified/remains with hashes + diffstat); one commit
per step; ruling protocol for contradictions; manual passes gate steps; doctor
first for any live anomaly; never fabricate identifiers, names, screenshots, or
test vectors; doctrine P1-P14, F1, D1 binding; the W definition is immutable
without a recorded ruling.

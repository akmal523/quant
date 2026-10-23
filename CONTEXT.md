# CONTEXT.md — Vocabulary, Architecture & Binding Rules

The current state of Quant-AI (v10.8.10). Keep terminology consistent across
code, docs and AI agents. This file states what is true **now**; superseded
rulings are not repeated here. The version-by-version rulings up to v10.8.9 are
archived verbatim in
[`docs/history/CONTEXT-to-v10.8.9.md`](docs/history/CONTEXT-to-v10.8.9.md), and
[`CHANGELOG.md`](CHANGELOG.md) records every release. Code comments cite rules by
their id (R-C1, R-F1, ...); the ids below are the ones still in force.

## Product

A daily portfolio manager for one EUR-based investor on Trade Republic. The user
types what they own; the app compares it with the targets and says in plain
language what to buy, sell or leave alone. Orders are placed by the user in the
broker app. The browser app (five pages) is the product; the CLI is optional.

The workflow: **Update holdings → Review changes → Confirm and save → act on the
ONE list in the broker → repeat.**

## Vocabulary

| Term | Meaning |
|------|---------|
| **Holdings table** | `portfolio.csv`: `Symbol, Avg_Entry_Price, Current_Value_EUR, Broker_PnL_EUR` (+ optional `Plan_EUR_month`). The ONLY input. Derived, never typed: invested = Value − Profit; shares = invested / average cost per share (fee-inclusive, Trade Republic "Buy In"). |
| **Class** (tier) | `FORTRESS` / `ALPHA` / `SPECULATIVE`, shown as **Long-term / Active / Speculative** (descriptive "Speculative (high risk)") via the one map in [`quant/ui/terminology.py`](quant/ui/terminology.py). "Small bets" and "FORTRESS" are banned in UI strings. |
| **Instrument class** | Stock / Fund (ETF) / Commodity (`EQUITY` / `ETF` / `COMMODITY`); automatic with one override; never "cash". |
| **Composite** | The 0–100 score from technical (0.40), fundamental (0.30), sentiment (0.20) and stewardship (0.10) components, plus a confidence tier. |
| **Idea** | A watched (not held) symbol with composite ≥ `IDEA_FLOOR` (55). |
| **The ONE list** | Every action the user should take, grouped **Today** / **This month**; each item is one verb, one name + ticker, one amount, one reason, one evidence pointer. |
| **Regime** | The market trend from an HMM fit on a broad index: engine keys `bull` / `bear` / `chop`, shown as rising / falling / mixed. |
| **Valuation store** | `valuation` + `portfolio_total` (+ `splits`): one mark-to-market point per trading day. |
| **Run manifest** | `run_manifest`: one row per analysis stage per run, with status, coverage and error. |
| **Data folder** | The per-user writable root (`PROJECT_ROOT`), e.g. `~/.local/share/quant-ai`; `QUANT_DATA_DIR` overrides it. Never inside the code checkout. |

## Architecture

```
            quant/ui (pages.py, render.py, columns.py, copy.py)   READ-ONLY
                 │ reads
   ┌─────────────┼───────────────────────────────────────────────┐
   │ holdings_view   today_list.build_items   artifacts.read_*   │  quant/engine,
   │ (positions +    (advice + taxonomy +     (scores, regime,   │  quant/reporting
   │  tiers+targets)  pairing; weights.py)     value series)     │
   └─────────────┬───────────────────────────────────────────────┘
                 │ DuckDB (quant_cache.duckdb in the data folder)
   writers ──────┘  confirm.confirm_save · daily.run_daily · cli refresh/run
                    · once-per-process startup backfills (names, valuation)
```

- **Pages** ([`quant/ui/pages.py`](quant/ui/pages.py)): Portfolio (home),
  Update holdings, History, Full analysis (Holdings analysis / Research / Market /
  Pipeline health tabs), Settings.
- **CLI** ([`quant/cli/__init__.py`](quant/cli/__init__.py)): `dash`, `refresh`,
  `daily`, `upgrade`, `backup`, `doctor` (plus hidden maintenance commands, e.g.
  `run`, `schedule`). Without an activated venv: `.venv/bin/python -m quant.cli`.
- **Paths** ([`quant/paths.py`](quant/paths.py)): `CODE_ROOT` is the checkout,
  `PROJECT_ROOT` the data folder. A one-time, copy-only migration
  ([`quant/data/user_data.py`](quant/data/user_data.py)) brought pre-10.8.1 repo
  state forward; the doctor lists stale copies left in the checkout.

### Analysis chain

[`quant/engine/analysis.py`](quant/engine/analysis.py): `prices → fundamentals →
nlp → regime → scoring → optimization`, run by the daily job and by **Run
analysis** (which runs `refresh` then `run`). Every stage writes one
`run_manifest` row:

| Status | Meaning |
|--------|---------|
| `ok-fresh` | a fetch stage fetched in this run (only when a fetcher is injected) |
| `ok-reused` | a fetch stage found stored data fresher than its TTL |
| `ok` | a compute stage (regime, scoring, optimization) ran |
| `stale` | a startup backfill (`names`, `valuation`) failed after retries |
| `failed` / `missing` | the stage errored / had no input; the UI shows the reason |

Fetching itself happens in `refresh` (and the daily job's update step) before the
chain, so production runs normally show `ok-reused`. TTLs: prices 1 **trading**
day (Friday's close is fresh over the weekend), fundamentals 7 days, nlp 1 day.
The `optimization` stage only checks that scores exist (coverage). The mean-CVaR
optimizer ([`quant/portfolio/optimizer.py`](quant/portfolio/optimizer.py)) is
not called by the product today (only its `minimum_trade_size` fee hurdle is);
it is library code kept under test and mutation testing.

The regime stage fits on `URTH` (`fit_market_regime(..., max_points=1.0)` is a
probability), maps it with `regime_from_bull_prob` (≥ 0.60 bull, ≤ 0.40 bear) and
stores it in `meta`; `artifacts.read_regime` returns the newer of that estimate
and the last review's regime block. Consumers compare with
`regime_constraints.is_bear` / `regime_key`, which accept both vocabularies.

## Binding rules

### Doctrine

- **P1–P14 (UI).** P1 every screen passes the action/trust test; P2 no internal
  identifiers in UI text (run ids, paths, column names, enum values); P3 no
  per-line provenance; P4 no silent defaults, explicit empty states; P5
  verb-phrase buttons; P6 plain error + remedy (raw text only in logs); P7
  technical detail collapsed; P8 one accent, semantic colors, no emoji or
  exclamation marks; P9 single-column layout, full-width primary buttons (table
  widths: R-F1); P10 units inline, human dates, scores "67 / 100"; P11 friendly
  names first; P12 one job per page; P13 empty states guide the next step; P14
  every user-facing string lives in [`quant/ui/copy.py`](quant/ui/copy.py),
  guarded by `tests/test_ui_copy.py` and `tests/test_copy_lints.py`.
- **F1 — no synthesized identifiers.** ISINs and names come from user files,
  validated live metadata or the curated files (`isin_source` provenance; ISO 6166
  check). The writers of `asset_registry` and their sources are listed in
  [`quant/data/registry_repair.py`](quant/data/registry_repair.py).
- **D1 — recorded-source tests.** A feature fed by a live source has a test that
  runs the production path against a checked-in fixture, with the source stubbed
  at the boundary, never our own functions (`tests/test_mock_boundary.py`).

### Data and writes

- **R-C4 — single writer.** Writes happen only in Confirm and save, the daily
  job, `refresh` / `run`, and the once-per-process startup backfills (bounded
  retry + backoff; on final failure one log line and a `stale` stage). The render
  path is read-only: the ONE list is computed by
  [`today_list.build_items`](quant/engine/today_list.py) and annotated with
  [`advice_state.annotate`](quant/engine/advice_state.py); the writers persist it
  with `today_list.persist` (→ `advice_state.sync`). The only page-initiated
  writes are explicit user actions (Confirm and save, Mark done, Back up now,
  Settings fields).
- **R-C1 — cash is not an entity.** Uninvested cash is never a row, target,
  line, weight or verb. Free money exists only as the amount typed into the
  invest block and whatever sits at the broker, untracked. The only trace: a
  sell with no buy to pair says "hold proceeds at the broker, untracked". The
  allocator's `kind="cash"` leg is the unallocated part of a budget, reported as
  "not allocated". `portfolio_total.cash_eur` is frozen audit data: written NULL,
  never read (a grep guard enforces it).
- **One value per datum.** The chart, the header delta and the totals read only
  the valuation store through the named helpers in
  [`quant/reporting/artifacts.py`](quant/reporting/artifacts.py) and
  [`quant/engine/mtm.py`](quant/engine/mtm.py): one point per trading day, no
  forward fill, tail = latest close, `day_delta(D) = total(D) − total(D−1)`.
- **Honest return.** "since first purchase" is Modified Dietz over the stored
  series and the ledger flows after the first date
  ([`flows.since_first_return`](quant/engine/flows.py)); deposits are not gains.

### Decisions

- **R-C3 — one weight math.** [`weights.instrument_weights` /
  `class_weights`](quant/engine/weights.py) (invested-only basis) are the single
  source for the Holdings table, class table, drift, advice and the allocator. The
  legacy review audit (`portfolio.enhanced_portfolio_audit`, `quant run`) keeps its
  own per-row weight for the review artifact only.
- **R-C2 — scoring degrades, never a dash.** [`composite_score`](quant/analytics/scoring.py)
  scores whatever components exist; each missing component contributes
  `SCORE_MISSING_PENALTY` (12/100). This is deliberately **below** neutral: a
  symbol with prices only tops out at 47.2 and cannot become an Idea. Confidence:
  present/total ≥ 0.8 high, ≥ 0.5 medium, else low. The composite is stored in
  `scores` for every symbol with a price, and every table (Research, Holdings
  analysis) and every decision (threat floor, swap) reads that composite.
- **The four verb classes** ([`action_taxonomy.py`](quant/engine/action_taxonomy.py)),
  thresholds in [`quant/config.py`](quant/config.py):
  - **Rebalance** — drift beyond the class threshold (from `build_advice`, which
    also yields the paired drift buy/sell).
  - **Threat** — Speculative stop-loss / take-profit; drawdown from cost beyond
    `THREAT_DRAWDOWN`; composite below `SCORE_FLOOR`; a bear regime (de-risk the
    Active part).
  - **Swap** — a scored **Active or Speculative** holding whose composite trails
    the best Idea by more than `SWAP_HURDLE`. Rendered as one paired item; a
    swapped symbol carries no standalone buy/sell.
  - **Plan change** — a Long-term holding more than `PLAN_CHANGE_GAP` under target
    for `PLAN_CHANGE_MONTHS` consecutive month-ends, counted from the valuation
    store (`today_list.months_under_target`).
- **Pairing** ([`pairing.route_sell`](quant/engine/pairing.py)): a sell's proceeds
  go to the largest underweight holding, else the best Idea, else "hold proceeds
  at the broker, untracked".
- **Stateful advice** (`advice_log`): an item keeps `open_since` while it is
  produced; "Mark done" closes it (History keeps it); an item that is produced
  again later opens a new row.
- **Notifications**: Telegram only, at most two messages per local day, only when
  the list changed ([`quant/engine/notify.py`](quant/engine/notify.py)).

### UI and copy

- **R-F1 — column contract.** [`quant/ui/columns.py`](quant/ui/columns.py)
  `TABLE_SPECS` is the one registry of table columns: non-empty headers, the
  currency in money headers, a column budget ≤ 8, low-priority detail in the
  drawer. Specs: editor (`Name | Instrument | Avg cost / share (paid ccy) | Value
  (EUR) | Profit (EUR) | Plan (EUR/m)`), entry_check (`Instrument | Shares
  (derived) | Typed value (EUR) | Live value (EUR) | Delta %`, amber when
  \|Delta\| > 1%), research (`Symbol | Class | Stage | Score | Confidence | News |
  Now / Target`), holdings_analysis (`Symbol | Class | Now / Target | Drift (pp) |
  Score | Alerts | Reason`), invest_class, invest_instrument, home_holdings. The
  AppTest table audit in [`tests/test_v10_8_9_phase1.py`](tests/test_v10_8_9_phase1.py)
  guards overflow.
- **R-P1 — editor grid.** Only typed inputs plus the two identity columns; live
  and derived values live in the read-only Entry-check table below it. The `Name`
  cell never repeats the ticker.
- **R-P2 — derived status cells.** `Reason` / `Alerts` / the coverage line
  ([`holdings_analysis`](quant/engine/holdings_analysis.py),
  [`manifest.coverage_line`](quant/engine/manifest.py)) are computed, never
  placeholders; unknown means a reasoned empty state, never a bare "-".
- **Inspect drawer.** Each Full-analysis tab has one lazy "Inspect" select; the
  drawer renders five sections in a fixed order (price chart, score + components,
  news, SEC, open advice), each with an explicit empty state. Widget keys are
  scoped per tab (`ha_*` / `rs_*`); each tab body has an error boundary.
- **R-P3 — one delta badge.** [`copy.delta_badge`](quant/ui/copy.py) owns every
  delta: icon/color from the sign of EUR (neutral when \|EUR\| < 0.5), text
  `"<eur:+.0f> EUR (<pct:+.1f>%)"`; a non-zero value that rounds to zero is a
  signed "less than" (`-<1 EUR`, `+<0.1%`). `-0 EUR` and `-0.0%` are
  unrepresentable.
- **R-F2 — view-bound overlays.** The benchmark toggle exists only in Growth view
  (state kept in `val_bench_on`); the chart carries no caption that repeats a
  header metric.
- **R-F3 — one copy lint.** [`tests/test_copy_lints.py`](tests/test_copy_lints.py)
  scans the copy module, the rendered pages and the Telegram builders for every
  known bad pattern. All user-facing strings live in
  [`quant/ui/copy.py`](quant/ui/copy.py).

### Backups

`backup.create_backup` (Settings → Back up now, `quant backup`, and before every
`update`) writes `data/backups/quant-backup-*.tar.gz` (table, tiers, account,
database; `notify.toml` only with `--include-secrets`), keeps 5, and alone sets
`last_backup_at`. Confirm and save writes a separate CSV/YAML-only snapshot to
`data/backups/auto/` (keeps 10) that never prunes or counts as a backup. See
[`docs/backup.md`](docs/backup.md).

## Guards (tests that enforce the rules)

| Rule | Guard |
|------|-------|
| Read-only render, list logic, regime, badge | `tests/test_v10_8_10.py` |
| Declared fallbacks / no stubs | `scripts/audit_stubs.py`, `tests/test_fallback_registry.py`, `tests/test_audit_stubs.py` ([policy](docs/stub_policy.md)) |
| Decision core is never mocked | `tests/test_mock_boundary.py` |
| Valuation invariants | `tests/test_v10_8_6_invariants.py` |
| Column contract | `tests/test_v10_8_9_phase1.py` |
| Copy lint | `tests/test_copy_lints.py` |
| No `cash_eur` reads | `tests/test_v10_8_2_no_cash.py` |
| Deterministic scoring | `tests/test_golden_snapshot.py` |

## ADR notes

- **Cash APY as the risk-free rate.** Trade Republic pays interest on uninvested
  cash; that rate (not US Treasuries) is the opportunity cost used in the risk
  math. It is a rate, not a tracked position (R-C1).
- **Mean-CVaR over Sharpe.** The (currently unwired) optimizer minimizes
  expected shortfall in the worst 5% tail; Sharpe is reported, not used for
  sizing.

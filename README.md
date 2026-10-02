# Quant-AI
**Daily portfolio manager for a solo family office on Trade Republic**

[![CI](https://github.com/akmal523/quant/actions/workflows/ci.yml/badge.svg)](https://github.com/akmal523/quant/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/akmal523/quant/branch/main/graph/badge.svg)](https://codecov.io/gh/akmal523/quant)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Quant-AI is a self-hosted portfolio manager for one investor on Trade Republic.
Once a day, after the close, it reviews your holdings against their targets and
says in plain language what to add, trim, or leave alone. It advises and
explains; you place the orders in the broker app. The browser app is the
product; the terminal is an optional power tool.

Live briefing: https://akmal523.github.io/quant/

> **New here?** Read [`CONTEXT.md`](CONTEXT.md) for the domain vocabulary, data
> contracts, architecture map, and the full **v10.6.0 cycle ledger** (every
> decision, bug, and guard since 10.5.2).

---

## What it does

- **Three-tier portfolio** — Fortress (never sell), Alpha (weekly trading),
  Speculative (2 percent cap). Assign tiers in the user-editable
  [`data/tiers.csv`](data/tiers.csv); [`data/portfolio.csv`](data/portfolio.csv)
  holds the four fields you enter from the broker app (Trade Republic does not
  export a portfolio CSV).
- **Daily advice** — drift versus target drives add/trim; a rebalance cooldown
  surfaces as "Waiting until {date}", never silence.
- **Weekly cadence** — signals generate on Friday and are cached Monday through
  Thursday. `quant weekly-report` writes a Markdown plus self-contained HTML
  report (print to PDF from the browser).
- **Emergency liquidity** — enter a cash amount and get a tax-aware sell order
  (most liquid first, losers first for tax-loss harvesting).
- **Auto-balance** — `quant suggest-rebalance` proposes tier reassignments to
  fix allocation violations; you approve each one. It edits `data/tiers.csv`
  only and never executes a trade.
- **Risk controls** — Mean-CVaR tail optimization, kill-switch drawdown
  breakers, weekly VaR, and hard data-quality gates that fail closed.
- **Honest data** — bitemporal point-in-time fundamentals (no lookahead),
  broker-synced PnL, and FinBERT news sentiment with no dictionary fallback.

## How to use it

Run `quant setup` once, then live your life:

```bash
quant setup           # guided first-week setup (data, tiers, schedule, alerts, backup)
quant dash            # open the app
```

Once a month, open the **Monthly decision** page, enter your savings-plan
budget, and approve the split. Everything else runs by itself. The full
first-week checklist is in [`docs/first_week.md`](docs/first_week.md).

---

## What's new in 10.7.2

- **Real-world hardening** — one shared runner lock with ownership and takeover
  (a dead process is taken over, a runaway one is taken over with a warning, a
  live one is respected), short writer transactions with a bounded retry, and a
  plain, retryable failure when the database is busy.
- **News pillar, demoted honestly** — `quant news-doctor` diagnoses why the
  heavy FinBERT stack contributes nothing; when it does, the pillar is marked
  absent, torch is never imported, and one honest line explains the tactical
  score.
- **Backup** — `quant backup` writes a restorable tar.gz of your state and keeps
  the five most recent; the system reminds you at most once a week.
- **First-week setup** — `quant setup` walks six idempotent steps and prints the
  first-week checklist; `quant setup --check` prints the statuses.

Older releases are in [`CHANGELOG.md`](CHANGELOG.md).

---

## Quick start

Run `quant setup` once, then live your life. The command walks six idempotent
steps (market data, tiers, schedule, notifications, backup, and the first-week
checklist), showing the current status of each and offering a skip. The full
checklist is in [`docs/first_week.md`](docs/first_week.md).

```bash
quant setup           # guided setup
quant setup --check   # print the statuses without prompting
quant dash            # open the app
```

Open the address Streamlit prints (usually http://localhost:8501). Display names
and ISINs are backfilled automatically the first time the app runs. On a phone
on the same Wi-Fi, run `quant dash --lan` and open the printed LAN address.

Everything else happens in the browser:

1. Open My holdings and add your holdings (type a name, symbol, or ISIN).
2. Set cash and your risk profile.
3. Press Save and review, then read the advice on Overview.

## Environment

The project runs in a Python 3.11+ virtual environment. This machine already has
one at `~/Downloads/vacancy_results/myenv` with an editable install of this
project:

```bash
alias myenv='source ~/Downloads/vacancy_results/myenv/bin/activate'
myenv
quant --version
```

The `quant` command exists only inside the environment. Without an active
environment, call the interpreter directly:

```bash
~/Downloads/vacancy_results/myenv/bin/python -m quant.cli --version
```

To create a fresh environment (a new machine, or a clean rebuild):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dashboard,test]"
quant --version
```

## Notifications

```bash
quant notify-setup
```

Choose Telegram, email, or none. For Telegram, create a bot via BotFather, send
it any message, then read your chat id from
`https://api.telegram.org/bot<TOKEN>/getUpdates`. The configuration is written to
`data/notify.toml` (kept local, gitignored). The command sends a test message and
records whether it worked; `quant doctor` shows the notification status.

## Backup

```bash
quant backup
```

Writes a restorable `data/backups/quant-backup-YYYYMMDD-HHMM.tar.gz` and keeps
the five most recent. See [`docs/backup.md`](docs/backup.md) for the contents and
the restore procedure.

## Advanced: command line

The terminal is optional. Three commands cover the daily cycle:

| Command | What it does |
|:--|:--|
| `quant update` | Refresh market data (step 1) |
| `quant run` | Review the portfolio and write the briefing (step 2) |
| `quant publish` | Render the static Published Briefing for the web |
| `quant doctor` | Read-only diagnosis (registry, names, news, search, advice) |
| `quant news-doctor` | Read-only news-pillar diagnostic (`--enable` forces it active) |
| `quant backup` | Archive the user-owned state (`--include-secrets` adds notify.toml) |
| `quant setup` | Guided first-week setup (`--check` prints statuses) |
| `quant health-check` | System health (data, tiers, allocations, database, cache) |
| `quant suggest-rebalance` | Suggest tier reassignments to fix allocation violations |
| `quant apply-rebalance` | Apply approved tier reassignments (`--symbols`, `--dry-run`) |
| `quant autobalance-wizard` | Interactively review and apply rebalancing suggestions |
| `quant cache-stats` | Show cache statistics (entries, size, location) |
| `quant clear-cache` | Clear all cached calculations |

Add `--verbose` for per-symbol detail. Logs live under `outputs/run_*/`.

### Performance and verification

```bash
python scripts/benchmark.py          # measure batch query, scoring, lazy load
python scripts/audit_type_hints.py   # type-hint coverage
python scripts/audit_docstrings.py   # docstring coverage
python scripts/final_verification.py # tests + ruff + audits + docs + benchmarks
```

See [`docs/performance.md`](docs/performance.md) for the tuning guide and the
measured benchmarks.

The External URL Streamlit prints is your public IP only if your router forwards
the port; by default it does not. Never expose Quant-AI to the internet without
putting it behind an authenticating reverse proxy. `quant dash` binds localhost
only; to serve on your local network (phone on the same Wi-Fi), run
`quant dash --lan`.

### Portfolio file (`data/portfolio.csv`)

Trade Republic does not export a portfolio CSV. You enter four fields per
position by hand from the app screens (Symbol, average entry price per share,
current value in EUR, profit in EUR). Four positions take about 5 minutes once
per month. The app's **My holdings** page has an editable **Broker statement**
table for exactly this.

```csv
Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR
EUNL.DE,125.03,281.25,12.00
AMZN,220.55,150.41,5.20
```

`Invested_EUR = Current_Value_EUR - Broker_PnL_EUR` is derived; `Broker_PnL_EUR`
is broker truth, never price-guessed. Comments after `#` are stripped. Between
syncs the app shows estimates from known shares and the latest price, labeled
estimated.

### Account file (`data/account.yaml`)

```yaml
base_currency: EUR
cash_eur: 10400.0
risk_profile: balanced
savings_plan_day: 1   # optional, 1-31
```

`risk_profile` is one of `conservative` / `balanced` / `aggressive`. The selected
profile in `data/account.yaml` overrides the bucket defaults in
[`quant/config.py`](quant/config.py).

`savings_plan_day` is optional (1-31). When set and at least one holding routes to
a savings plan, Today shows the countdown under the actions block
(`Savings plan executes in {n} days ({date}); additions before that date apply
this month.`), or `Savings plan executes today.` on the day itself.

### Tier file (`data/tiers.csv`)

```csv
symbol,tier,last_updated,notes
EUNL.DE,FORTRESS,2026-10-01,Migrated from legacy CORE
AMZN,ALPHA,2026-10-01,Migrated from legacy ACTIVE
GME,SPECULATIVE,2026-10-01,Meme stock, max 2 percent
```

`tier` is one of `FORTRESS` / `ALPHA` / `SPECULATIVE`. An unlisted symbol
defaults to `ALPHA`. Run `python scripts/migrate_tiers.py` once to seed the file
from the existing portfolio; edit it freely afterwards. `portfolio.csv` is never
modified.

### Weekly report

```bash
quant weekly-report --as-of 2026-10-02 --emergency 300
```

Writes `outputs/reports/weekly_<date>.md` and `.html`. Open the HTML file in a
browser and print to PDF (Ctrl+P / Cmd+P). The `--emergency` amount adds a
tax-aware sell order to the report.

### Tier validation

```bash
quant validate-tiers   # report issues in data/tiers.csv
quant repair-tiers     # fix duplicates, orphans, and invalid tiers
```

### Automated Friday report

To generate the report automatically every Friday at 06:00, add a cron entry:

```cron
0 6 * * 5 cd /path/to/quant && python -m quant.cli weekly-report
```

### Migration

Upgrading from the legacy four-tier system? See
[`docs/migration_v10.6.2.md`](docs/migration_v10.6.2.md). A sample tier file is
in [`examples/tiers_example.csv`](examples/tiers_example.csv).

---

## Architecture

```mermaid
flowchart TD
    U[Universe builder 1000+ symbols] --> F[Funnel stage1 liquidity stage2 momentum]
    F --> S[Scoring fundamentals + NLP + regime]
    S --> A[Audit broker-synced PnL]
    A --> R[Reports + notifications]
    D[(DuckDB bitemporal store)] --- U
    D --- F
    D --- S
    D --- A
    DASH[Streamlit dashboard] --- D
```

Step 1 (`quant update`) fetches market data, adjusts corporate actions, runs the
hard data-quality gate, and caches the funnel survivors. Step 2 (`quant run`)
reads that cache, scores, audits, and reports. Step 2 does **not** re-fetch the
universe.

### Key modules

| Module | Responsibility |
|:--|:--|
| [`quant/main.py`](quant/main.py) | Orchestration: load -> filter -> NLP -> score -> audit -> report |
| [`quant/data/data_updater.py`](quant/data/data_updater.py) | Incremental DuckDB market data + funnel; caches survivors |
| [`quant/data/universe_builder.py`](quant/data/universe_builder.py) | Builds `universe_master` from index constituents + ETFs |
| [`quant/data/funnel.py`](quant/data/funnel.py) | Two-stage filter (liquidity -> momentum) + survivor cache |
| [`quant/analytics/scoring.py`](quant/analytics/scoring.py) | Factor model, regime HMM, stewardship, capital allocation |
| [`quant/portfolio/portfolio.py`](quant/portfolio/portfolio.py) | Broker-synced audit, EUR PnL, reconciliation |
| [`quant/portfolio/optimizer.py`](quant/portfolio/optimizer.py) | cvxpy Mean-Variance + Mean-CVaR with bucket constraints |
| [`quant/execution/tca.py`](quant/execution/tca.py) | Implementation Shortfall (TCA) + `trade_log` |
| [`quant/cli/__init__.py`](quant/cli/__init__.py) | `quant` console command + `doctor` |
| [`quant/ui/copy.py`](quant/ui/copy.py) | Every user-facing string + formatter (P14) |
| [`quant/ui/render.py`](quant/ui/render.py) | The four page renderers |
| [`quant/ui/cards.py`](quant/ui/cards.py) | Explore card fields (one source) |
| [`quant/ui/search.py`](quant/ui/search.py) | Instrument + theme search index |
| [`quant/reporting/artifacts.py`](quant/reporting/artifacts.py) | The five UI artifact accessors |
| [`quant/data/names.py`](quant/data/names.py) | Display-name cleaning + backfill |
| [`quant/data/registry_repair.py`](quant/data/registry_repair.py) | Working-universe sync + ISIN heal |
| [`quant/data/news.py`](quant/data/news.py) | News fetch + 24 h cache + outage counter |
| [`quant/portfolio/history.py`](quant/portfolio/history.py) | Portfolio value history |
| [`quant/portfolio/cash_rate.py`](quant/portfolio/cash_rate.py) | Dated cash-rate schedule |
| [`quant/paths.py`](quant/paths.py) | CWD-independent path resolver |

---

## Configuration

All tunables live in [`quant/config.py`](quant/config.py). Key settings:

| Parameter | Default | Description |
|:---|---:|:---|
| `WEIGHT_FUNDAMENTALS` / `_STEWARDSHIP` / `_TECHNICAL` / `_SENTIMENT` | 30 / 30 / 15 / 25 | Scoring weights |
| `MIN_STRUCT_GRADE_FOR_BUY` | 75 | Structural floor for CORE classification |
| Cash rate | dated schedule in `quant/portfolio/cash_rate.py` | Trade Republic yield used as the risk-free rate (2.5 percent from 16 Sep 2026) |
| `ROUND_TRIP_FEE_EUR` | 2.0 | Active-trade round-trip fee |
| `SAFETY_BUCKET_MIN` / `CORE_BUCKET_MIN` / `ALPHA_BUCKET_MAX` | 0.10 / 0.40 / 0.50 | Bucket constraints |
| `FUNNEL_STAGE1_TARGET` / `FUNNEL_TOP_N` | 300 / 24 | Funnel caps |
| `CVAR_ALPHA` | 0.05 | Mean-CVaR tail (Expected Shortfall) |
| `MAX_DAILY_DRAWDOWN` / `VOL_KILL_MULTIPLIER` | 0.03 / 2.0 | Kill-switch thresholds |
| `REPORTING_LAG_DAYS` | 45 | PIT publication lag (no lookahead) |

Environment variables (optional) are documented in [`.env.example`](.env.example).

---

## Testing

Tests are hermetic: an ephemeral DuckDB is injected by
[`tests/conftest.py`](tests/conftest.py); production data is never touched.

```bash
myenv                     # activate the environment (see Environment above)
pytest -n auto            # parallel; coverage floor enforced by .coveragerc
pytest tests/test_v10_7_2_phase_a.py   # a single file
```

The coverage floor is a ratchet in [`.coveragerc`](.coveragerc) (v10.4.2
baseline 42%, current floor 55%, target 80%). The `dashboard` extra (streamlit)
is required so the UI tests run and the floor is met. CI
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the full suite plus
the golden snapshot gate on every push and PR. See
[`tests/README.md`](tests/README.md) for the isolation model, determinism rules,
and golden-file regeneration process.

---

## Contributing

All user-facing language lives in [`quant/ui/copy.py`](quant/ui/copy.py), guarded
by [`tests/test_ui_copy.py`](tests/test_ui_copy.py); change copy there, never
inline. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, style (ruff, line
length 100), type checking (pyright), and the conventional-commit convention. By
participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Documentation

| Document | Purpose |
|:--|:--|
| [`CONTEXT.md`](CONTEXT.md) | Domain vocabulary, data contracts, module map, ADRs |
| [`CHANGELOG.md`](CHANGELOG.md) | Full version history |
| [`tests/README.md`](tests/README.md) | Test isolation, coverage, golden files |
| API docs | `mkdocs serve` (deployed to GitHub Pages) |

---

## Disclaimer

All output is for informational purposes. Probabilistic models and NLP sentiment
analysis involve inherent risk. **Past performance does not guarantee future
results.** The universe contains only currently-listed instruments - historical
backtest figures are systematically overstated due to survivorship bias.

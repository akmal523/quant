# Quant-AI

**A daily portfolio manager for one investor on Trade Republic.**

Quant-AI is a self-hosted tool for a single investor. Once a day, after the
market close, it reviews your holdings against their targets and says in plain
language what to add, trim, or leave alone. It advises and explains; you place
the orders in the broker app. The browser app is the product; the terminal is an
optional power tool.

[![CI](https://github.com/akmal523/quant/actions/workflows/ci.yml/badge.svg)](https://github.com/akmal523/quant/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/akmal523/quant.git
cd quant
./install.sh          # Windows: install.bat
```

## Use

```bash
./start.sh            # Windows: start.bat
```

The first time, run the guided setup once: `./start.sh` opens the app, and
`quant setup` (inside the app's environment) walks the first-week checklist.

Open the address Streamlit prints (usually http://localhost:8501). Everything
else happens in the browser:

1. Open **My holdings** and add your positions (type a name, symbol, or ISIN).
2. Set your cash and risk profile in **Settings**.
3. Open **Monthly decision**, enter your budget, and approve the split.

The app then runs a daily check after the market close and shows what to do on
**Overview**. The full first-week checklist is in
[`docs/first_week.md`](docs/first_week.md).

## Update

```bash
quant upgrade
```

## Where your data lives

Your positions, cash, tiers, and history live in `data/` and in the local DuckDB
file `quant_cache.duckdb`. They are never committed to git. Back them up with:

```bash
quant backup
```

See [`docs/backup.md`](docs/backup.md) for the archive contents and the restore
steps.

## Documentation

| Document | Purpose |
|:--|:--|
| [`CONTEXT.md`](CONTEXT.md) | Domain vocabulary, data contracts, architecture, decisions |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |
| [`docs/`](docs/) | First week, backup, performance, stub policy, API reference |

## Disclaimer

All output is for informational purposes. Probabilistic models and NLP sentiment
analysis involve inherent risk. **Past performance does not guarantee future
results.** The universe contains only currently-listed instruments - historical
backtest figures are systematically overstated due to survivorship bias.

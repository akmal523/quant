# Quant-AI

A daily portfolio manager for one EUR-based investor on Trade Republic.

You enter what you own; the app reviews it against your targets and says in plain
language what to buy, sell, or leave alone. You place the orders in the broker app.

## Install

Requires Python 3.11 or newer.

    ./install.sh        # Windows: install.bat

## Start

    ./start.sh          # Windows: start.bat

## Update

    ./update.sh         # Windows: update.bat

`update` backs up your data first, pulls the new version, reinstalls and migrates
the database.

## How it works

1. **Update holdings**: type each position from the broker app (value, profit,
   average cost per share) and confirm. Saving is the only way data enters.
2. **Portfolio**: total value, the day's change, and the ONE list of what to do,
   grouped into *Today* and *This month*. Each item is one action with one reason.
   Set each holding's class and target under **Update holdings → Class and
   target**; a holding without a target gets no buy/sell advice.
3. **Full analysis**: per-holding drill-down, the research funnel (ideas), the
   market trend and the pipeline health. **Run analysis** fetches fresh prices,
   fundamentals and news headlines and rescores everything.
4. **History** and **Settings**: the trade ledger, the advice log, the Telegram
   bot, Refresh prices, Update software, Back up now.

Scores are simple and explained: price trend (momentum and the 200-day
average), five fundamentals for stocks, and the tone of recent news headlines
(FinBERT, optional `news` extra). A part without enough data is left out and the
confidence drops; it is never guessed.

An optional daily timer (`schedule`, see below; Linux with systemd) refreshes
prices and re-evaluates the list on trading days, and each morning sends one
Telegram digest: value, day change, the market trend, what to do, holdings vs
target, notable news and any problems. Mail is not supported.

## Where your data lives

Your positions, settings, database and backups live in your per-user data folder
(shown in Settings; on Linux `~/.local/share/quant-ai`), never in the repository.
Set `QUANT_DATA_DIR` to use a different folder.

Back up with **Settings → Back up now**. See [docs/backup.md](docs/backup.md).

## Command line (optional)

The app needs no terminal. For scripting, the same commands are available as

    .venv/bin/python -m quant.cli <command>      # Windows: .venv\Scripts\python -m quant.cli <command>

(or plain `quant <command>` inside an activated environment). Commands: `dash`,
`refresh`, `daily`, `upgrade`, `backup`, `doctor`; `schedule` installs the daily
timer (`schedule --off` removes it, `schedule --status` reports it).

## Documentation

- [`CONTEXT.md`](CONTEXT.md) — vocabulary, architecture and the binding rules
- [`CHANGELOG.md`](CHANGELOG.md) — version history
- [`docs/`](docs/) — backup, performance, the stub/fallback policy, API reference

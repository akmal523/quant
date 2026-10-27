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

## Telegram bot: update holdings and log actions from your phone

Set up Telegram first (Settings → Set up Telegram, or `notify-setup`). Then
start the bot on the computer that holds your data, either way:

- **Settings → Telegram bot → Run the bot in the background** (or
  `bot --install`): a systemd user service that starts with the computer and
  restarts itself after an error. **Stop the bot** (`bot --off`) removes it.
- `bot` in a terminal, for a try-out (Ctrl+C stops it).

`bot --status` says whether it runs; `bot --restart` restarts it; `upgrade`
restarts it for you, so after an update it runs the new code. Its log:
`journalctl --user -u quant-bot`.

Send `/commands` (or tap "/" in the chat for the menu). Only the chat id saved
in Settings is answered.

| Command | What it does |
|---------|--------------|
| `/portfolio` | holdings, value, profit, now vs target |
| `/set SYMBOL VALUE PROFIT [COST]` | the broker numbers for one holding (COST is needed for a new one) |
| `/buy SYMBOL EUR [PRICE]` | add a buy; the price per share defaults to the latest close |
| `/sell SYMBOL EUR\|all [PRICE]` | record a sell |
| `/confirm`, `/cancel` | save or drop the pending change |
| `/todo`, `/done N` | the ONE list, numbered; mark item N done |
| `/history` | the latest trades and done actions |
| `/commands`, `/help` | the list with examples |

A change is shown as the same review as **Update holdings** and is written only
on `/confirm`, through **Confirm and save** (table, ledger, value chart, the ONE
list), into the data folder on this computer. `/done` is **Mark done**.
Decimals may use a comma (`12,5`).

**While the computer is off or asleep**, Telegram keeps your messages for up to
24 hours; when the bot starts again it answers them in order, and each such
answer says when you sent it. A pending change (sent `/buy`, not yet
`/confirm`) survives a restart. Messages older than 24 hours are dropped by
Telegram; for an always-on bot, run the app on a machine that stays on.

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

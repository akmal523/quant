# Migration to v10.7.0

v10.7.0 is a root redesign of how the system converses with you. This note
explains the two changes that affect existing data.

## The cash-floor constraint is removed

The old model mixed three kinds of money into one "portfolio", which produced
the 92 percent cash donut and nonsense "invest from cash" advice. From v10.7.0
the system keeps three separate pools:

- **Invested** — everything held at the broker. The only pool that is scored,
  weighted, charted, and advised.
- **Savings-plan budget** — a monthly amount you decide from salary. A flow into
  Invested, not a balance.
- **Operational cash** — daily-life money at the broker. Never an investment
  buffer.

The `cash floor 10%` constraint is gone from `quant/config.py` and all
risk-profile displays. `RISK_PROFILES` is now a 3-tuple
`(long_term_min, active_max, max_position)` of the invested pool. If you had a
custom risk profile, re-check it against the new plain descriptions.

## Report history is deduplicated

The Settings report history now shows one entry per trading day, keeping the
latest per day. Existing duplicate rows are collapsed at render time; no data is
deleted.

## New tables

The daily job creates `holdings_meta`, `flows`, `portfolio_value_history`,
`alerts`, and `monthly_plans` on first run. No manual migration is needed.

## New commands

- `quant schedule` — install/remove the local daily timer.
- `quant notify-setup` — configure Telegram/email.
- `quant daily` — the job body (used by the timer).
- `quant ack <id> --status done|declined --reason "..."` — resolve an alert.

## What did not change

- The broker CSV schema (`Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR`).
- The existing CLI commands and the golden backtest.
- Local-first privacy: portfolio data never leaves the machine.

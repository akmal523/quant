# Backup and restore

One command produces a restorable archive of everything you own in Quant-AI.
The system reminds you about backups at most once a week, never nagging.

## Create a backup

```bash
quant backup
```

This writes `data/backups/quant-backup-YYYYMMDD-HHMM.tar.gz` and keeps the five
most recent archives. It prints the path, the size in MB, the member list, and
the one-line restore instruction.

## What is inside the archive

| Member | What it is |
|:--|:--|
| `data/portfolio.csv` | Your broker-synced holdings (value and profit). |
| `data/tiers.csv` | Your tier assignments (long-term / active / small bets). |
| `data/account.yaml` | Cash, risk profile, base currency, savings-plan day. |
| `quant_cache.duckdb` | The DuckDB store (prices, scores, history, alerts). |
| `quant_cache.duckdb.wal` | The write-ahead log, when present. |

`data/notify.toml` (your Telegram token or email password) is **not** included by
default. Add it only when you want it in the archive:

```bash
quant backup --include-secrets
```

The command prints a warning line when it does. Keep that archive private.

## Restore

1. Stop every Quant-AI process (close the app, stop the timer with
   `quant schedule --off`).
2. Unpack the archive over your project folder (the folder that contains
   `data/`):

   ```bash
   tar -xzf data/backups/quant-backup-YYYYMMDD-HHMM.tar.gz -C /path/to/quant
   ```

3. Run `quant doctor` to confirm the database, registry, and files are healthy.

## Recommendation

Copy the archive to a cloud drive once a month. The archive is small and
self-contained; a monthly copy is enough to survive a disk failure.

## Where the status appears

- `quant doctor` prints `Last backup: <date>`, or the gentle line
  `Last backup N days ago. Run quant backup.` when the last backup is older than
  30 days (or `No backup yet. Run quant backup.` when there is none).
- The Settings page shows the same line in the Automation block, at most once a
  week.

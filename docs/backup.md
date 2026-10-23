# Backup and restore

One action produces a restorable archive of everything you own in Quant-AI.

## Create a backup

In the app: **Settings → Back up now**. From a terminal:

```bash
.venv/bin/python -m quant.cli backup        # Windows: .venv\Scripts\python -m quant.cli backup
```

This writes `data/backups/quant-backup-YYYYMMDD-HHMM.tar.gz` inside your data
folder (the path shown in Settings) and keeps the five most recent archives. The
command prints the path, the size in MB, the member list, and the one-line
restore instruction. `./update.sh` makes one before every update.

## What is inside the archive

Paths inside the archive are relative to the data folder.

| Member | What it is |
|:--|:--|
| `data/portfolio.csv` | Your holdings table (value, profit, average cost). |
| `data/tiers.csv` | Your class assignments (Long-term / Active / Speculative). |
| `data/account.yaml` | Risk profile, base currency, savings-plan day. |
| `quant_cache.duckdb` | The database (prices, scores, ledger, valuation, advice log). |
| `quant_cache.duckdb.wal` | The write-ahead log, when present. |

`data/notify.toml` (your Telegram token) is **not** included by default. Add it
only when you want it in the archive:

```bash
.venv/bin/python -m quant.cli backup --include-secrets
```

The command prints a warning line when it does. Keep that archive private.

## Automatic snapshots

Every **Confirm and save** first writes a small snapshot of the three input files
(no database) to `data/backups/auto/quant-snapshot-*.tar.gz`, keeping the ten most
recent. Snapshots are a safety net for the last edits only: they never replace,
prune or count as a full backup.

## Restore

1. Close the app (stop every Quant-AI process).
2. Unpack the archive into your data folder (the path shown in Settings):

   ```bash
   tar -xzf <data folder>/data/backups/quant-backup-YYYYMMDD-HHMM.tar.gz -C <data folder>
   ```

3. Run the doctor (`.venv/bin/python -m quant.cli doctor`) to confirm the
   database, registry, and files are healthy.

## Recommendation

Copy the archive to a cloud drive once a month. It is small and self-contained;
a monthly copy is enough to survive a disk failure.

## Where the status appears

- The doctor prints `Last backup: <date>`, or `Last backup N days ago. Use Back up
  now in Settings.` when the last full backup is older than 30 days (`No backup
  yet. ...` when there is none).
- Settings shows the backup folder and the **Back up now** button.

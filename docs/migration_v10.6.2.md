# Migration Guide: v10.6.2 Three-Tier Architecture

## Overview

Version 10.6.2 introduces a three-tier portfolio architecture:

- **FORTRESS**: eternal holdings (never sell, tax-free accumulation).
- **ALPHA**: active trading (weekly rebalancing, sell when money is needed).
- **SPECULATIVE**: high-risk bets (max 2 percent allocation).

This replaces the legacy four-tier system (CORE / SATELLITE / ACTIVE / SECTOR).

## Automatic migration

Run the migration script:

```bash
python scripts/migrate_tiers.py
```

This will:

1. Read your existing `data/portfolio.csv`.
2. Classify each asset with the legacy four-tier system.
3. Map to the new three-tier system:
   - CORE to FORTRESS
   - SATELLITE to ALPHA
   - ACTIVE to ALPHA
   - SECTOR to ALPHA
4. Create `data/tiers.csv` with the results.

`data/portfolio.csv` is never modified.

## Post-migration: auto-balance

After migration, your tier allocations may violate a limit. For example, ALPHA
at 66.1 percent (limit: 50 percent). This is expected. The system provides an
auto-balance tool to fix it:

```bash
quant suggest-rebalance                              # view suggestions
quant autobalance-wizard                             # interactive review
quant apply-rebalance --symbols NVDA,TSM             # apply specific symbols
quant apply-rebalance --symbols NVDA,TSM --dry-run   # preview only
```

Auto-balance only changes tier assignments in `data/tiers.csv`. It does not
execute trades. You must manually buy or sell in Trade Republic.

## Manual adjustments

After migration, review `data/tiers.csv` and adjust as needed:

```csv
symbol,tier,last_updated,notes
URTH,FORTRESS,2026-10-01,"Core ETF, never sell"
SPY,FORTRESS,2026-10-01,"Core ETF, never sell"
AAPL,FORTRESS,2026-10-01,"Core stock, tax-free accumulation"
NVDA,ALPHA,2026-10-01,"Tactical trading, sell when needed"
GME,SPECULATIVE,2026-10-01,"Meme stock, max 2 percent"
```

## Validation

After migration, validate your tiers:

```bash
quant validate-tiers
```

If issues are found, attempt automatic repair:

```bash
quant repair-tiers
```

Repair fixes duplicate symbols, orphan symbols (in tiers but not in the
portfolio), and invalid tier values. It does not change tier allocations; an
allocation that exceeds a cap is reported for manual review.

Run a comprehensive health check:

```bash
quant health-check
```

## Rollback

Migration is one-way. There is no automatic rollback from three-tier to
four-tier.

If you need to roll back:

1. Restore `data/portfolio.csv` from backup.
2. Delete `data/tiers.csv`.
3. Revert to the previous version.

## FAQ

**What if I add a new asset to portfolio.csv but forget tiers.csv?**

The system detects unclassified assets and recommends a tier based on asset
type (ETFs to FORTRESS, stocks to ALPHA). The Portfolio page shows a warning
with an auto-assign action.

**Can I change an asset's tier later?**

Yes. Edit `data/tiers.csv` manually or use the Portfolio page.

**What happens to my existing signals?**

Signals are regenerated based on the new tier assignments. FORTRESS assets show
the structural grade only; ALPHA assets show the full scoring.

**Is my portfolio.csv still broker-synced?**

Yes. `portfolio.csv` is unchanged (Symbol, Avg_Entry_Price, Current_Value_EUR,
Broker_PnL_EUR). Tier assignments are stored separately in `tiers.csv`.

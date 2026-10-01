# Performance Tuning Guide

## Overview

This guide explains how to keep the system fast for large portfolios (50+
assets) and slow environments.

## Measured benchmarks

Measured with `python scripts/benchmark.py` on the live portfolio (4 assets,
v10.6.5):

| Benchmark | Symbols | Elapsed | Peak memory |
|:--|--:|--:|--:|
| Batch query | 4 | 0.091 s | 1.45 MB |
| Batch scoring | 4 | 0.116 s | 1.43 MB |
| Lazy loading (5J50.DE, 345 rows) | 1 | 0.042 s | 0.39 MB |

The synthetic 100-asset tests in `tests/test_performance.py` assert batch
scoring completes in under 60 seconds and peak memory stays under 500 MB.

## Batch scoring

For portfolios with 50+ assets, use parallel batch scoring:

```python
from quant.analytics.scoring import batch_score_assets_parallel

results = batch_score_assets_parallel(
    symbols=symbols,
    price_data=prices,
    fundamentals=fundamentals,
    news=news,
    tiers=tiers,
    max_workers=4,
)
```

Recommendations:

- Under 20 assets: the thread path is used automatically (lower overhead).
- 20 to 50 assets: `max_workers=4`.
- 50+ assets: `max_workers=8` or the CPU core count.

`batch_score_assets_optimized(symbols, tiers)` loads all market history in a
single query, then scores in parallel.

## Database optimization

### Query batching

Instead of N per-symbol queries, use one batch query:

```python
from quant.data.database import batch_query_portfolio_data

data = batch_query_portfolio_data(symbols)
prices = data["prices"]
```

### Lazy loading

For very long histories, stream rows in chunks:

```python
from quant.data.database import load_prices_lazy

for chunk in load_prices_lazy("AAPL", chunk_size=5000):
    process(chunk)
```

## Caching

Cache expensive, JSON-serializable calculations on disk:

```python
from quant.analytics.cache import disk_cache

@disk_cache(max_age_days=1)
def expensive_calculation(x: int) -> int:
    return x * x
```

Cache management:

```bash
quant cache-stats   # entry count, total size, location
quant clear-cache   # delete all entries
```

## Reliability

- `quant.utils.retry.retry_with_backoff` retries transient external failures
  with exponential backoff.
- `quant.analytics.scoring.score_asset_with_fallbacks` degrades gracefully when
  data is missing, marking `data_quality` as PARTIAL or FALLBACK.
- `quant.portfolio.tier_manager.load_tiers_safe` never crashes on a missing,
  empty, or corrupted tiers file.

## Monitoring

Use the health check to spot performance issues:

```bash
quant health-check
```

Look for WARNING on "Data freshness" (slow updates) or "Signal cache" (slow
scoring).

## Benchmarking

Run the performance suite:

```bash
pytest tests/test_performance.py -v
```

It reports batch scoring time for 50 and 100 assets, batch query time, and
memory usage.

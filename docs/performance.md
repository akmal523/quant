# Performance

What costs time in Quant-AI, and how the code keeps it bounded on an old laptop.

## Where the time goes

| Step | Cost | How it is bounded |
|:--|:--|:--|
| `refresh` (market data + funnel) | network-bound, minutes | The 1000+ symbol `universe_master` is filtered once per cycle (Stage 1 price/liquidity, Stage 2 trend) to about 24 survivors cached in `funnel_survivors` and reused for 7 days. Price history is fetched as ranges, so a missed day is backfilled in one call; only held symbols and symbols with a new split are refetched in full. |
| FinBERT news sentiment | heavy (torch) | Optional `news` extra. Only new headlines are scored, once, by one model instance during refresh; the scores are stored in `nlp_evidence` and pruned after 60 days. Without torch the news step is skipped and sentiment is left out of the score. |
| Analysis chain (`daily`, `quant run`, Run analysis) | seconds | Pure scoring from stored prices, fundamentals and headlines; fetch stages only check freshness against their TTL (prices 1 trading day, fundamentals 7 days, nlp 1 day). |
| Pages | fast | Every page read opens a short-lived read-only DuckDB connection; the render path never writes (R-C4). |

## Database concurrency

DuckDB allows one writer per file. Writers use
`quant.data.database.write_connection` (bounded retry with backoff on lock
errors); UI reads use `read_only_connection`, which retries briefly while a
`refresh`/`daily` job holds the lock instead of escalating to a write lock. The
runner lock (`quant.engine.lock`) keeps two jobs from running at once; a failed
daily run writes a marker the doctor shows, and the next timer slot retries.

## Reliability

- The startup backfills (display names, valuation catch-up) retry with
  exponential backoff and record a `stale` stage on final failure.
- `quant.portfolio.tier_manager.load_tiers_safe` never crashes on a missing,
  empty, or corrupted tiers file.
- Every swallowed failure that can reach a number on screen is declared in
  `quant/engine/fallback_registry.py` and tested (see the
  [stub and fallback policy](stub_policy.md)).

## Monitoring

```bash
.venv/bin/python -m quant.cli doctor    # read-only, safe to paste: data freshness, tiers, database, news, Telegram
```

The Full analysis page's **Pipeline health** tab shows every analysis stage with
its status, coverage and last run.

## Honest performance math

A naive "value now / value then" return lies whenever money enters. The
Portfolio header's **since first purchase** line uses **Modified Dietz** over the
valuation store, with the ledger flows (`flows` table, written by Confirm and
save) after the first stored date
(`quant.engine.flows.since_first_return`):

```
R = (V_end - V_start - F) / (V_start + sum(w_i * F_i))
w_i = (days_in_period - days_from_start_to_flow) / days_in_period
```

Deposits do not count as profit. Flow signs: `buy` = +amount (money entered
invested), `sell` = -amount (money left invested), `dividend` = -amount (left
invested). Flows on the first date are already inside `V_start`.

The value series is the mark-to-market store (`portfolio_total`, one point per
trading day, invested positions only; `quant.engine.mtm`). The day delta is
`total(D) - total(D-1)` from the same store.

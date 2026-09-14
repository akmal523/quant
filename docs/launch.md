# Launch & Community Plan

Operational checklist for discoverability, engagement, and trust. Nothing here
is code; it is the go-to-market for the repository.

## 1. Repository metadata

Set these in GitHub repo settings (Settings -> General -> Topics):

```
quantitative-finance  systematic-trading  polars  duckdb
portfolio-optimization  cvar  backtesting  algorithmic-trading  python
```

Description:

> Daily portfolio manager for a solo family office: one snapshot after the
> close, plain-language add/trim advice, broker-synced PnL, DuckDB + Polars.

## 2. Launch post (draft)

Title: **Quant-AI - an institutional-grade systematic equity pipeline in Python**

Body outline:

1. The problem: retail quant tools look impressive but leak lookahead bias and
   ignore execution reality.
2. What this is: a two-step pipeline (fetch -> score/audit) over a 1000+ ticker
   universe, stored in DuckDB, EUR-native, broker-synced to a real broker CSV.
3. What makes it trustworthy:
   - Bitemporal PIT fundamentals (`as_of_date` + `published_date`).
   - Hard data-quality assertions that abort on bad data.
   - Mean-CVaR optimization + a hard kill switch.
   - Golden-file CI that fails on > 0.01% backtest drift.
4. Honest limitations: survivorship bias in the current universe; not
   investment advice.
5. Call to action: star, open an issue, join Discussions.

Channels: r/algotrading, r/quant, Hacker News (Show HN), X/Twitter.

## 3. Two-minute demo script

| Time | On screen | Narration |
|:--|:--|:--|
| 0:00-0:15 | README above the fold | "Institutional-grade equity pipeline, one screen." |
| 0:15-0:35 | `quant update` running | "Fetches 1000+ tickers, adjusts splits, hard data gate." |
| 0:35-1:00 | `quant run` output | "Scores, audits the broker-synced portfolio, reports." |
| 1:00-1:20 | `quant reconcile` | "Diffs the book against the broker export." |
| 1:20-1:40 | Four-page workspace | "Today decides, Portfolio edits, Explore explains, Settings maintains." |
| 1:40-1:55 | `quant doctor` | "One read-only diagnosis: registry, names, news, search, advice." |
| 1:55-2:10 | `quant publish` + CI | "Static briefing for the web; every change verified, backtest drift fails CI." |

## 4. Engagement

- GitHub Discussions enabled for Q&A (see `.github/ISSUE_TEMPLATE/config.yml`).
- Respond to issues within 48 hours (target).
- Label triage: `bug`, `enhancement`, `good first issue`.

## 5. Trust

- [SECURITY.md](https://github.com/akmal523/quant/blob/main/SECURITY.md) for
  private vulnerability reports.
- [CODE_OF_CONDUCT.md](https://github.com/akmal523/quant/blob/main/CODE_OF_CONDUCT.md).
- Conventional commits + generated CHANGELOG for an auditable history.

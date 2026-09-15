# Launch & Community Plan

Operational checklist for discoverability, engagement, and trust. Nothing here
is code; it is the go-to-market for the repository.

## 1. Repository metadata

Set these in GitHub repo settings (Settings -> General -> Topics) — exact set:

```
quantitative-finance  systematic-trading  portfolio-optimization  polars
duckdb  algorithmic-trading  mean-cvar  python
```

Description:

> Daily portfolio manager for a solo family office: one snapshot after the
> close, plain-language add/trim advice, broker-synced PnL, DuckDB + Polars.

Pages link (live briefing, verified HTTP 200):

> https://akmal523.github.io/quant/

Set metadata with the authenticated CLI (one-time):

```
gh repo edit akmal523/quant \
  --description "Daily portfolio manager for a solo family office: one snapshot after the close, plain-language add/trim advice, broker-synced PnL, DuckDB + Polars." \
  --add-topic quantitative-finance --add-topic systematic-trading \
  --add-topic portfolio-optimization --add-topic polars --add-topic duckdb \
  --add-topic algorithmic-trading --add-topic mean-cvar --add-topic python \
  --homepage "https://akmal523.github.io/quant/"
```

## 2. Launch post (draft, 10.6.0)

Title: **Quant-AI - a daily portfolio manager that says what to add, trim, or leave alone**

Body outline:

1. The problem: retail quant tools look impressive but leak lookahead bias,
   ignore execution reality, and never tell you what to actually do today.
2. What this is: a browser-first daily portfolio manager for a solo family
   office — one snapshot after the close, plain-language add/trim advice, orders
   placed in the broker app. Four pages: Today decides, Portfolio edits, Explore
   explains, Settings maintains.
3. What makes it trustworthy:
   - Bitemporal PIT fundamentals (`as_of_date` + `published_date`).
   - A working universe `W` (registry holds exactly the tracked set).
   - Honest advice: drift vs target drives add/trim; a cooldown surfaces as
     "Waiting until {date}", never silence.
   - `quant doctor`: one read-only diagnosis (registry, names, news, search,
     advice oracle).
   - Hard data-quality assertions; Mean-CVaR optimization; a hard kill switch.
   - Golden-file CI that fails on > 0.01% backtest drift; 327 tests.
4. Honest limitations: survivorship bias in the current universe; not
   investment advice.
5. Call to action: star, open an issue, join Discussions.

Channels: r/algotrading, r/quant, Hacker News (Show HN), X/Twitter.

## 3. Two-minute demo script

| Time | On screen | Narration |
|:--|:--|:--|
| 0:00-0:15 | README above the fold | "A daily portfolio manager for a real broker account." |
| 0:15-0:35 | `quant dash` -> Today | "One snapshot after the close; add/trim advice in plain language." |
| 0:35-1:00 | Portfolio page | "Edit holdings; type a name, symbol, or ISIN. Values come from the broker." |
| 1:00-1:25 | Explore page | "Search by name, symbol, ISIN, or theme; news with sentiment provenance." |
| 1:25-1:45 | `quant update` then `quant run` | "Refresh prices under a data-quality gate, then review." |
| 1:45-2:00 | `quant doctor` | "One read-only diagnosis: registry, names, news, search, advice oracle." |
| 2:00-2:15 | `quant publish` + CI | "Static briefing for the web; every change verified, backtest drift fails CI." |

## 4. Engagement

- GitHub Discussions enabled for Q&A (see `.github/ISSUE_TEMPLATE/config.yml`).
- Respond to issues within 48 hours (target).
- Label triage: `bug`, `enhancement`, `good first issue`.

## 5. Trust

- [SECURITY.md](https://github.com/akmal523/quant/blob/main/SECURITY.md) for
  private vulnerability reports.
- [CODE_OF_CONDUCT.md](https://github.com/akmal523/quant/blob/main/CODE_OF_CONDUCT.md).
- Conventional commits + generated CHANGELOG for an auditable history.

## 6. Follow-up issues (keep open, non-gating)

Create with the authenticated CLI:

```
gh issue create --title "docs: capture Today screenshot (docs/assets/today.png)" \
  --body "Add a real Today-page screenshot (no fabrication) per the CONTEXT ledger checklist."
gh issue create --title "data: curated ISINs beyond 5J50.DE" \
  --body "Populate data/isin_curated.csv with user-verified ISINs from the broker app."
gh issue create --title "release: PyPI name availability + Trusted Publishing" \
  --body "Confirm the PyPI name; configure Trusted Publishing for the tag workflow."
gh issue create --title "chore: .rooignore manual entries" \
  --body "Add the manual entries the ledger lists for .rooignore."
gh issue create --title "test: raise coverage ratchet toward 80" \
  --body "Raise the .coveragerc fail_under floor incrementally from the current baseline."
gh issue create --title "chore(lint): defer legacy E/I/N/UP ruff findings" \
  --body "R9 fixed repo-wide F-codes; legacy E501/E402/I001/N812/UP037 remain and are tracked here."
```

# Migration to v10.7.6

v10.7.6 is the "Buffett Filter, Tax Accounting, and Remove the Noise" release.
It adds two features and removes unused infrastructure. No user data migration
is required.

## What changed for you

- **Buffett filter.** Stocks are scored on Buffett-style fundamental quality
  (P/E, ROE, ROIC, debt-to-equity, stable earnings) with an economic moat
  estimate. Passing stocks appear in a "Buffett candidates" section on Find
  investments, and each holding shows a Buffett score in My holdings. The filter
  is additive: it never changes the structural or tactical grade, and it is not
  applied to ETFs or commodities.
- **Tax summary page.** A new page tracks realized gains and dividends, applies
  the German Sparerpauschbetrag (1000 EUR single, 2000 EUR married), estimates
  the Abgeltungssteuer at 26.375 percent, and lists losses you can use to lower
  tax. Record a trade once, on My holdings (the quick-events form); the Tax
  summary page reads the same ledger.
- **No PyPI.** The project is a local-first personal tool. The PyPI publishing
  and git-cliff release workflows are removed. Install with `pip install -e .`.

## What to do

Nothing. Your `data/portfolio.csv`, `data/tiers.csv`, and `data/account.yaml`
are unchanged. The new `trades` table is created on first run. Open the Tax
summary page to record trades and see your yearly position.

## For developers

- New modules: `quant/analytics/buffett.py`, `quant/portfolio/tax_accounting.py`.
- New page: `quant/pages/tax.py` (renderer `page_tax` in `quant/ui/render.py`).
- New table: `trades` (separate from `trade_log`, which is TCA).
- New tests: `tests/test_buffett.py`, `tests/test_tax_accounting.py`.
- Removed: `.github/workflows/publish.yml`, `.github/workflows/release.yml`,
  `cliff.toml`.
- The rulings R12 through R15 are recorded in `CONTEXT.md`.

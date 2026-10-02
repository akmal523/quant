# Migration to v10.7.4

v10.7.4 is the "Classification and Decision Test Grid" release. It adds an
end-to-end test grid that proves the core decision flow, and it closes the
behavior gaps the grid exposed. No user data migration is required.

## What changed for you

- **FORTRESS far over target.** A long-term holding more than 10 points over its
  target now gets a savings-plan note ("Consider lowering or pausing the
  savings-plan leg ..."). It is never sold.
- **Bear regime.** In a falling market, high-conviction active buys are
  suppressed; new active money goes to cash. Sells and long-term savings-plan
  advice are unaffected.
- **Small active buys.** A buy below 100 EUR is not recommended (the 1 EUR fee
  makes it inefficient). Savings-plan legs are exempt.
- **Monthly split.** The long-term base now goes to the single largest-gap
  long-term holding, and the split sums to your budget exactly. Every leg states
  its actual gap in percentage points.
- **Sync reminder.** The reminder appears at most once per 7 days, never on the
  same day as a successful sync, and lists the positions you recorded.
- **Dismissed alerts.** A dismissed alert stays closed until the underlying
  condition clears and then breaks again.

## What to do

Nothing. Your `data/portfolio.csv`, `data/tiers.csv`, and `data/account.yaml`
are unchanged. Run `quant doctor` to see the new portfolio-data and unpriceable
diagnostics.

## For developers

- New grid files: `tests/test_classification_grid.py`, `test_advice_grid.py`,
  `test_freedom_grid.py`, `test_consistency_grid.py`, `test_adversarial_grid.py`,
  `test_properties.py`, `test_regression_grid.py`.
- Shared fixtures: `tests/fixtures/live_portfolio.py`.
- `hypothesis` is a test-only dependency.
- The rulings R1 through R10 are recorded in `CONTEXT.md`.

# Tests

## Running

```bash
pip install -e ".[test,dashboard]"   # dashboard: the Streamlit AppTest UI tests
pytest -n auto                 # parallel, coverage via .coveragerc
pytest tests/test_decision_rules.py   # single file
```

Tests are hermetic and offline. No network, no production data. CI runs the same
command on every push and pull request.

## Isolation model

[`tests/conftest.py`](conftest.py) redirects the DuckDB store to an ephemeral
file for the whole session:

- `quant.paths.DB_FILE` and `quant.data.database.DB_PATH` are patched to a
  `tmp_path_factory` file.
- The thread-local connection is dropped before `init_db()` so the patched path
  is honoured (`database.py` binds `DB_PATH` at import time).
- Production `quant_cache.duckdb` is never opened or deleted.

**Rule:** a test must never read or write live files under `data/` or
`outputs/`. Use `tempfile` / `tmp_path` to build deterministic fixtures (see
[`tests/test_portfolio_fx.py`](test_portfolio_fx.py) for the pattern).

**Determinism:** seed all randomness (`np.random.default_rng(SEED)`). Unseeded
`np.random.normal` makes statistical assertions flaky.

## Recorded-source tests (D1)

Any feature whose data comes from a live external source (Yahoo metadata, news,
FX) must have at least one test that runs the **production default source path**
against a checked-in recorded fixture, with the source stubbed at the boundary
(e.g. `yfinance.Ticker`), never our own functions. Injected-fetcher tests prove
logic; recorded-source tests prove wiring. See
[`tests/test_names_recorded_source.py`](test_names_recorded_source.py) and
[`tests/fixtures/`](fixtures/).

## Coverage

Coverage floor lives in [`.coveragerc`](../.coveragerc) as `fail_under` (55%;
measured about 63% at v10.8.10). The target is **80%**.

Ratchet policy:

1. Never lower `fail_under`.
2. When coverage improves, raise the floor in the same PR.
3. New modules must ship with tests.

## Where a feature is tested

Test files from v10.6 on are named after the release that introduced them, and
one release usually touched several features. This index is the feature view
(current rules: [`CONTEXT.md`](../CONTEXT.md)):

| Feature | Files |
|---------|-------|
| Holdings table, Review diff, Confirm and save | `test_v10_8_2_diff.py`, `test_v10_8_2_confirm.py`, `test_v10_8_2_plan_column.py`, `test_v10_8_2_workflow.py`, `test_v10_8_1_persistence.py` |
| Positions, ledger, editable history | `test_v10_8_0_positions.py`, `test_v10_8_0_ledger.py`, `test_v10_8_3.py` |
| The ONE list: format, advice state, pairing, verbs | `test_v10_8_2_decisions.py`, `test_v10_8_4_phase2.py`, `test_v10_8_5.py` (phase C), `test_v10_8_10.py`, `test_decision_rules.py`, `test_advice_grid.py` |
| Read-only render (R-C4), regime vocabulary | `test_v10_8_10.py` |
| Weights, invest plan | `test_v10_8_7_phase3.py`, `test_v10_8_4_phase3.py` |
| Scoring, news, analysis chain, run manifest | `test_v10_8_10.py`, `test_v10_8_7_phase2.py`, `test_v10_8_4_phase1.py` |
| Valuation store, day delta, chart, FX | `test_v10_8_4_phase1.py`, `test_v10_8_5.py` (phase B), `test_v10_8_6_invariants.py`, `test_v10_8_2_value_series.py`, `test_v10_8_9_phase2.py`, `test_v10_8_10.py` (FX); `store_readers.py` is the independent SQL oracle |
| Cash is not an entity (R-C1) | `test_v10_8_2_no_cash.py` (despite the name, the v10.8.7 rule), `test_v10_8_7_phase1.py` |
| Tables and column contract (R-F1, R-P1, R-P2) | `test_v10_8_8_phase1.py`, `test_v10_8_8_phase2.py`, `test_v10_8_9_phase1.py`, `test_v10_8_4_phase4.py` |
| Copy, delta badge, palette | `test_copy_lints.py`, `test_ui_copy.py`, `test_v10_8_8_phase3.py`, `test_v10_8_0_palette.py`, `test_v10_8_2_verify.py` |
| Backup, user data folder, migration | `test_v10_7_2_phase_c.py`, `test_v10_8_1_user_data.py`, `test_v10_8_1_persistence.py` |
| Locks, CLI, install, doctor | `test_v10_7_2_phase_a.py`, `test_v10_8_2_cli.py`, `test_v10_8_0_install.py`, `test_doctor.py` |
| Telegram digest, daily job, scheduler | `test_v10_8_2_notify.py`, `test_v10_7_0_phase3.py` |
| FX, rates, splits, tiers and targets | `test_v10_8_0_fx.py`, `test_v10_8_0_rates.py`, `test_v10_8_10.py`, `test_three_tier.py` |
| Stubs, fallbacks, mock boundary | `test_audit_stubs.py`, `test_fallback_registry.py`, `test_mock_boundary.py` |
| Math core, exact values (mutation suite, 100%) | `test_core_score_model.py`, `test_core_action_taxonomy.py`, `test_core_mtm.py`, `test_core_sizing_flows.py` |

New tests go into the file of the feature they guard when one fits; a release
that adds a new rule may add `test_v<version>.py` and a row here.

## Contract guards from earlier cycles

- [`test_ui_copy.py`](test_ui_copy.py) — banned tokens on every page (P2/P14).
- [`test_registry_bounded.py`](test_registry_bounded.py) — `asset_registry == W`,
  idempotent membership, CSV ceiling, bulk-source refusal.
- [`test_names_recorded_source.py`](test_names_recorded_source.py) — D1.
- [`test_themes.py`](test_themes.py), [`test_h3_5.py`](test_h3_5.py),
  [`test_doctor.py`](test_doctor.py).

# Tests

## Running

```bash
pip install -e ".[test]"
pytest -n auto                 # parallel, coverage via .coveragerc
pytest tests/test_scoring.py   # single file
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

## Coverage

Coverage floor lives in [`.coveragerc`](../.coveragerc) as `fail_under`. The
v10.4.2 baseline is **42%**; the target is **80%**.

Ratchet policy:

1. Never lower `fail_under`.
2. When coverage improves, raise the floor in the same PR.
3. New modules must ship with tests.

## Golden snapshot

[`tests/test_golden_snapshot.py`](test_golden_snapshot.py) compares a
deterministic seeded backtest against [`tests/golden/backtest_2024.json`](golden/backtest_2024.json)
and fails on **> 0.01%** relative drift.

- **PR:** the golden gate runs and fails on drift. Never regenerate to make a
  failing test pass.
- **`main`:** [`.github/workflows/golden.yml`](../.github/workflows/golden.yml)
  regenerates the golden file and commits it when an intentional backtest change
  lands.

Regenerate locally only for an intentional change:

```bash
python scripts/make_golden.py
```

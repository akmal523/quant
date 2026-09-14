# Contributing to Quant-AI

Thanks for your interest. This guide covers setup, style, testing, and the PR
process.

## Setup

```bash
git clone https://github.com/akmal523/quant
cd quant
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
pre-commit install
```

## Style

- **Linter/formatter:** [ruff](https://docs.astral.sh/ruff/) (replaces flake8,
  isort, black). Line length 100. Rules `E, F, I, N, W, UP`, `quant` is
  first-party. Config lives in [`pyproject.toml`](pyproject.toml).
- **Types:** [pyright](https://github.com/microsoft/pyright) in `basic` mode
  ([`pyrightconfig.json`](pyrightconfig.json)). Fix type errors before pushing.
- **Formatting is mandatory:** run `pre-commit run --all-files` before committing.
- **Ruff scope (release gate):** zero new findings on changed files, plus all
  repo-wide `F` (correctness) codes fixed. Legacy `E`/`I`/`N` findings are
  deferred to a tracked issue with a per-module ratchet.

## Doctrine & copy

The product doctrine (P1-P14, F1, D1) is binding and recorded in
[`CONTEXT.md`](CONTEXT.md). Two rules bite most often:

- **P14 — one copy home.** Every user-facing string lives in
  [`quant/ui/copy.py`](quant/ui/copy.py) and is guarded by
  [`tests/test_ui_copy.py`](tests/test_ui_copy.py). Change copy there, never
  inline; no internal identifiers (run ids, paths, column names, enum values) in
  UI text.
- **D1 — recorded-source tests.** A feature whose data comes from a live source
  must have a test that runs the production default path against a checked-in
  fixture, with the source stubbed at the boundary (never our own functions).

## Comments & docstrings

Document the **why** before the **what**. Use structural tags:

- `Intent:` why the module/function exists
- `Invariants:` conditions that must always hold
- `State Transition:` initial -> trigger -> new state
- `Dependencies:` external logic relied upon

Keep them terse.

## Testing

```bash
pytest -n auto            # full suite + coverage floor
```

- Tests are hermetic: an ephemeral DuckDB is provided by
  [`tests/conftest.py`](tests/conftest.py). Never read/write live `data/` or
  `outputs/`; build fixtures with `tmp_path` / `tempfile`.
- Seed all randomness (`np.random.default_rng`).
- Use the **tracer bullet** approach: one failing test -> minimal fix -> repeat.
- Coverage floor is a ratchet in [`.coveragerc`](.coveragerc). Never lower it;
  raise it when coverage improves.
- See [`tests/README.md`](tests/README.md) for the golden-file process.

## Commits

Conventional Commits are required (drives changelog + versioning):

```
feat: add regime-conditional leverage cap
fix: cast volume to float before split adjustment
docs: expand CONTEXT glossary
test: isolate portfolio CSV fixtures
chore: bump ruff to 0.6
```

## Pull requests

1. Branch from `main`.
2. Keep the diff focused; atomic changes only.
3. CI must be green (tests, coverage gate, golden drift gate).
4. Update [`CHANGELOG.md`](CHANGELOG.md) under `[Unreleased]` when behavior changes.
5. Do not regenerate golden files in a PR; intentional changes regenerate on
   `main`.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

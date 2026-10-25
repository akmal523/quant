# Stub and fallback policy

This project runs on a real laptop with real money. A function that returns a
canned value, or an `except` path that swallows a failure and returns a default
without saying so, destroys trust the day real data disagrees with the canned
value. This page is the policy for contributors.

## The distinction

- A **legitimate fallback** is declared, marked, tested, and surfaced to the user.
  Example: a scoring component that cannot be read contributes
  `SCORE_MISSING_PENALTY` and lowers the confidence tier shown next to the score;
  an offline FX lookup uses the last live EUR/USD rate. Legitimate fallbacks stay.
- A **stub** is undeclared, silent, untested, or returns success while computing
  nothing. Stubs must fail CI.

## The static auditor

`python scripts/audit_stubs.py` scans `quant/` (excluding tests) with an AST and
reports ten finding kinds:

| Kind | Meaning |
|------|---------|
| `pass_body` | a public function whose body is only `pass` |
| `not_implemented` | `raise NotImplementedError` in non-abstract public code |
| `todo_marker` | TODO / FIXME / XXX / STUB / PLACEHOLDER in `quant/` |
| `bare_except` | an `except` that passes without logging or a marker |
| `constant_return` | a long docstring but a single literal return |
| `trivial_complexity` | a docstring verb but complexity 1 and no calls |
| `test_branch` | an `os.environ` read of a TEST / MOCK / FAKE / STUB key |
| `vacuous_constraint` | a weight cap >= 1.0, a score threshold <= 0, a fee < 0 |
| `canary_number` | a float literal equal to a known live value |
| `undeclared_fallback` | an `except` returning a default not in the registry |

The auditor exits nonzero on any blocking finding. It runs in CI.

## The three tiers (R11)

- **Tier S — blocking.** The narrow user-facing computation set: score_model,
  flows, sizing, positions, portfolio, account, the ONE list (today_list, action_taxonomy, pairing, weights,
  advice_state), invest_plan, the valuation store (mtm), the analysis chain, and
  the listed loaders (`scripts/audit_allowlist.toml`). Every `except` path
  whose returned value can flow into a user-visible number or decision must be in
  `quant/engine/fallback_registry.py` with its marker and a triggering test.
- **Tier C — pattern allowlist.** Trivial cleanup and IO shapes (pass-only,
  log-only, cleanup calls). Non-blocking; the auditor verifies the shape.
- **Tier W — warn plus ratchet.** Everything else. Reported; the per-kind counts
  live in `tests/warn_baseline.txt` and may only shrink.

## Adding a fallback

1. Add an entry to `FALLBACKS` in `quant/engine/fallback_registry.py` with
   `failure`, `default`, `marker`, and `test`.
2. Add a trigger to `tests/test_fallback_registry.py` that reproduces the failure
   and returns the default. The registry test iterates the map, so a missing
   trigger fails CI.
3. If the fallback is user-facing, surface the marker (a briefing line, a doctor
   line, or a UI caption).

## Mutation testing

`mutmut` mutates the math core (`pyproject.toml` `[tool.mutmut]`: score_model,
flows, sizing, action_taxonomy, mtm) and runs only the tests listed there. The
weekly job (`.github/workflows/mutation.yml`, `scripts/mutation_report.py`)
requires **100 percent**: every mutant is killed by a test.

A mutant that no test can tell apart from the original (an *equivalent* mutant)
is removed at the source, in this order:

1. Delete the code that cannot matter (an unreachable guard, a default no caller
   relies on, a clamp a formula never exceeds).
2. Move constant text out of function bodies (SQL lives in module-level
   constants; keyword case cannot change a query).
3. Only then mark the one line `# pragma: no mutate`, with a comment that says
   why the mutant is equivalent (e.g. `>` vs `>=` on a continuous measure).

`tests/mutation_survivors_allowlist.txt` stays empty; the file may only shrink.

## Mock boundary

No test may patch the decision core (score_model, flows, sizing,
action_taxonomy). Fakes are allowed only at true I/O boundaries (network, clock,
filesystem, subprocess, the database, the UI render layer) and heavy external
solvers. `tests/test_mock_boundary.py` enforces this.

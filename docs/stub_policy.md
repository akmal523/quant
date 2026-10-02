# Stub and fallback policy

This project runs on a real laptop with real money. A function that returns a
canned value, or an `except` path that swallows a failure and returns a default
without saying so, destroys trust the day real data disagrees with the canned
value. This page is the policy for contributors.

## The distinction

- A **legitimate fallback** is declared, marked, tested, and surfaced to the user.
  Example: `score_asset_with_fallbacks` sets `data_quality = FALLBACK`; the
  optimizer equal-weight fallback sets a marker the weekly report renders.
  Legitimate fallbacks stay.
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
| `canary_number` | a float literal equal to a known live or golden value |
| `undeclared_fallback` | an `except` returning a default not in the registry |

The auditor exits nonzero on any blocking finding. It runs in CI.

## The three tiers (R11)

- **Tier S — blocking.** The narrow user-facing computation set (scoring, flows,
  allocator, sizing, advice, alerts, valuation, news_pillar, optimizer, risk,
  portfolio, account, cash_rate, and the listed loaders). Every `except` path
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

`mutmut` mutates only the math core. The kill threshold is 90 percent. Survivors
are listed in `tests/mutation_survivors_allowlist.txt` with a reason; the file may
only shrink. A survivor that touches a user-facing number is blocking and must be
killed by a test, not allowlisted.

## Mock boundary

No test may patch the decision core (advice, allocator, flows, sizing, scoring,
optimizer, risk). Fakes are allowed only at true I/O boundaries (network, clock,
filesystem, subprocess, the database, the UI render layer) and heavy external
solvers. `tests/test_mock_boundary.py` enforces this.

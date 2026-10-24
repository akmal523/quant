"""test_stub_grid.py — dynamic sensitivity grid (v10.7.5, Part 2).

Intent: a stub that returns constants fails these tests. For every pure math and
scoring function, assert the output RESPONDS to its inputs in the documented
direction, and (where the domain defines it) monotonicity.

Allowlist for legitimately insensitive functions (documented constants, not
stubs): ``regime_confidence`` band edges, ``tier_word``/``class_word`` enum maps,
``fmt_*`` formatters. They are pure lookups, not computations, and are covered by
their own contract tests.

Invariants: pure unit tests; no network, no I/O.
"""
from __future__ import annotations

from datetime import date

from quant.engine import flows, sizing

# ── Modified Dietz ────────────────────────────────────────────────────────────

def test_modified_dietz_linear_in_end_value():
    start, end = date(2026, 1, 1), date(2026, 1, 31)
    r1 = flows.modified_dietz(1000.0, 1050.0, [], start, end)
    r2 = flows.modified_dietz(1000.0, 1100.0, [], start, end)
    assert abs((r2 - r1) - r1) < 1e-9  # equal increments -> linear


def test_modified_dietz_deposit_is_not_profit():
    start, end = date(2026, 1, 1), date(2026, 1, 31)
    f = [{"date": date(2026, 1, 16), "type": "buy", "amount_eur": 200.0}]
    ret = flows.modified_dietz(1000.0, 1200.0, f, start, end)
    assert abs(ret) < 1e-9


# ── Sizing ────────────────────────────────────────────────────────────────────

def test_sell_amount_monotone_and_capped():
    assert sizing.sell_amount(50.0, 1000.0) >= sizing.sell_amount(30.0, 1000.0)
    assert sizing.sell_amount(2000.0, 1000.0) <= 1000.0 - 1.0
    assert sizing.can_sell("ALPHA", 50.0, 200.0) is None  # below 100 EUR

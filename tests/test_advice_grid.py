"""test_advice_grid.py — advice correctness (v10.7.4, Part 2).

Intent: the user trusts the system not to recommend buys that are mathematically
irrational, and to split the monthly budget honestly. This grid asserts the math
under realistic portfolio states.

Invariants:
  - A buy clears the fee hurdle (R1: expected alpha >= fee).
  - Conviction maps to action deterministically (R9: bear overrides).
  - The allocator conserves the budget and states truthful gaps (R10).
  - Modified Dietz excludes deposits and is honest about negative returns.
"""
from __future__ import annotations

from datetime import date

from quant.engine import flows

# ── Part 2.4: Modified Dietz correctness ──────────────────────────────────────

def test_modified_dietz_zero_flows():
    ret = flows.modified_dietz(1000.0, 1100.0, [], date(2026, 1, 1), date(2026, 1, 31))
    assert abs(ret - 0.10) < 1e-9


def test_modified_dietz_single_mid_period_buy_excluded():
    f = [{"date": date(2026, 1, 16), "type": "buy", "amount_eur": 200.0}]
    ret = flows.modified_dietz(1000.0, 1200.0, f, date(2026, 1, 1), date(2026, 1, 31))
    expected = (1200.0 - 1000.0 - 200.0) / (1000.0 + 0.5 * 200.0)
    assert abs(ret - expected) < 1e-9
    assert abs(ret) < 1e-9


def test_modified_dietz_dividend_is_outflow():
    f = [{"date": date(2026, 1, 16), "type": "dividend", "amount_eur": 20.0}]
    ret = flows.modified_dietz(1000.0, 1000.0, f, date(2026, 1, 1), date(2026, 1, 31))
    expected = (1000.0 - 1000.0 + 20.0) / (1000.0 - 10.0)
    assert abs(ret - expected) < 1e-9


def test_modified_dietz_multiple_flows():
    f = [
        {"date": date(2026, 1, 10), "type": "buy", "amount_eur": 200.0},
        {"date": date(2026, 1, 20), "type": "sell", "amount_eur": 50.0},
        {"date": date(2026, 1, 25), "type": "dividend", "amount_eur": 20.0},
    ]
    ret = flows.modified_dietz(1000.0, 1200.0, f, date(2026, 1, 1), date(2026, 1, 31))
    net = 200.0 - 50.0 - 20.0
    weighted = (21 / 30) * 200.0 + (11 / 30) * (-50.0) + (6 / 30) * (-20.0)
    expected = (1200.0 - 1000.0 - net) / (1000.0 + weighted)
    assert abs(ret - expected) < 1e-9


def test_modified_dietz_negative_return_with_deposit():
    f = [{"date": date(2026, 1, 16), "type": "buy", "amount_eur": 500.0}]
    ret = flows.modified_dietz(1000.0, 1400.0, f, date(2026, 1, 1), date(2026, 1, 31))
    assert ret is not None and ret < 0

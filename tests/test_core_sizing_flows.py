"""test_core_sizing_flows.py — exact behavior of the sizing laws and the return math.

quant/engine/sizing.py and quant/engine/flows.py are in the mutation scope
(pyproject [tool.mutmut]); every boundary here is pinned by a value on each side
so a changed comparison or constant fails a test.
"""
from __future__ import annotations

from datetime import date

import duckdb
import pytest

from quant.engine import flows, sizing

D = date


# ── sizing ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value, step, expected", [
    (7.5, 0, 7.5),            # no step: unchanged
    (7.5, -5, 7.5),           # negative step: unchanged
    (7.5, 1, 7.0),            # a step of 1 still rounds
    (24.0, 5, 20.0),
    (200.0 - 1e-12, 5, 200.0),          # float noise below a step boundary
    (19.9999999996, 1, 20.0),           # quotient rounded to 9 places first
    (12.0, 2.5, 10.0),
])
def test_round_down(value, step, expected):
    assert sizing.round_down(value, step) == expected


def test_round_down_returns_a_float_for_a_zero_step():
    assert isinstance(sizing.round_down(7, 0), float)


@pytest.mark.parametrize("value, expected", [
    (99.99, True), (100.0, False), (100.01, False), ("bad", True), (None, True)])
def test_is_untouchable(value, expected):
    assert sizing.is_untouchable(value) is expected


@pytest.mark.parametrize("drift, value, expected", [
    (200.0, 1000.0, 200.0),
    (203.0, 1000.0, 200.0),     # 5 EUR steps, down
    (500.0, 101.0, 100.0),      # capped at value - 1 EUR
    (25.0, 1000.0, 25.0),       # exactly the minimum is allowed
    (24.9, 1000.0, None),       # below the minimum
    (0.0, 1000.0, None),
    (-50.0, 1000.0, None),
    (50.0, 0.0, None),
    (50.0, -10.0, None),
    ("x", 1000.0, None),
    (50.0, None, None),
])
def test_sell_amount(drift, value, expected):
    assert sizing.sell_amount(drift, value) == expected


@pytest.mark.parametrize("amount, expected", [(129.0, 120.0), (10.0, 10.0), (9.99, 0.0)])
def test_round_buy(amount, expected):
    assert sizing.round_buy(amount) == expected


# ── flows: signs and dates ────────────────────────────────────────────────────

@pytest.mark.parametrize("kind, sign", [
    ("buy", 1.0), ("BUY", 1.0), ("sell", -1.0), ("dividend", -1.0), ("gift", 0.0)])
def test_flow_sign(kind, sign):
    result = flows.flow_sign(kind)
    assert result == sign and isinstance(result, float)


def test_as_date_accepts_dates_and_iso_timestamps():
    assert flows._as_date(D(2024, 1, 5)) == D(2024, 1, 5)
    assert flows._as_date("2024-01-05T10:00:00") == D(2024, 1, 5)


# ── flows: the ledger reader ──────────────────────────────────────────────────

@pytest.fixture
def ledger():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE flows (id INTEGER, date DATE, type VARCHAR, "
                 "amount_eur DOUBLE, symbol VARCHAR, note VARCHAR)")
    for row in [(3, D(2024, 3, 1), "sell", 30.0, "C", "n3"),
                (1, D(2024, 1, 1), "buy", 10.0, "A", "n1"),
                (2, D(2024, 2, 1), "dividend", 20.0, "B", "n2")]:
        conn.execute("INSERT INTO flows VALUES (?, ?, ?, ?, ?, ?)", list(row))
    yield conn
    conn.close()


def _flow(i, d, t, a, s, n):
    return {"id": i, "date": d, "type": t, "amount_eur": a, "symbol": s, "note": n}


def test_load_flows_returns_every_column_in_date_order(ledger):
    assert flows.load_flows(ledger) == [
        _flow(1, D(2024, 1, 1), "buy", 10.0, "A", "n1"),
        _flow(2, D(2024, 2, 1), "dividend", 20.0, "B", "n2"),
        _flow(3, D(2024, 3, 1), "sell", 30.0, "C", "n3"),
    ]


def test_load_flows_bounds_are_inclusive(ledger):
    ids = [f["id"] for f in flows.load_flows(ledger, D(2024, 2, 1), D(2024, 3, 1))]
    assert ids == [2, 3]
    assert [f["id"] for f in flows.load_flows(ledger, start=D(2024, 2, 1))] == [2, 3]
    assert [f["id"] for f in flows.load_flows(ledger, end=D(2024, 2, 1))] == [1, 2]


# ── flows: Modified Dietz ─────────────────────────────────────────────────────

def test_modified_dietz_weights_a_flow_by_the_time_it_was_invested():
    # 1000 -> 1600 over 10 days with a 500 deposit on day 4: weight 0.6.
    flow = {"date": D(2024, 1, 5), "type": "buy", "amount_eur": 500.0}
    r = flows.modified_dietz(1000.0, 1600.0, [flow], D(2024, 1, 1), D(2024, 1, 11))
    assert r == pytest.approx(100.0 / 1300.0)


def test_modified_dietz_signs_and_defaults():
    start, end = D(2024, 1, 1), D(2024, 1, 11)
    sell = {"date": D(2024, 1, 1), "type": "sell", "amount_eur": 100.0}
    assert flows.modified_dietz(1000.0, 1000.0, [sell], start, end) == pytest.approx(100 / 900)
    untyped = {"date": D(2024, 1, 1), "amount_eur": 100.0}        # treated as a buy
    assert flows.modified_dietz(1000.0, 1100.0, [untyped], start, end) == 0.0
    no_amount = {"date": D(2024, 1, 1), "type": "buy"}           # counts as 0
    assert flows.modified_dietz(1000.0, 1100.0, [no_amount], start, end) == pytest.approx(0.1)
    unknown = {"date": D(2024, 1, 1), "type": "gift", "amount_eur": 100.0}
    assert flows.modified_dietz(1000.0, 1100.0, [unknown], start, end) == pytest.approx(0.1)


def test_modified_dietz_period_and_denominator_guards():
    assert flows.modified_dietz(100.0, 110.0, [], D(2024, 1, 1), D(2024, 1, 1)) is None
    assert flows.modified_dietz(100.0, 110.0, [], D(2024, 1, 2), D(2024, 1, 1)) is None
    one_day = flows.modified_dietz(100.0, 110.0, [], D(2024, 1, 1), D(2024, 1, 2))
    assert one_day == pytest.approx(0.1)
    assert flows.modified_dietz(0.0, 10.0, [], D(2024, 1, 1), D(2024, 1, 2)) is None
    assert flows.modified_dietz(-1.0, 10.0, [], D(2024, 1, 1), D(2024, 1, 2)) is None
    assert flows.modified_dietz(0.5, 1.0, [], D(2024, 1, 1), D(2024, 1, 2)) == pytest.approx(1.0)


# ── flows: since first purchase ───────────────────────────────────────────────

def test_since_first_return_needs_two_points():
    assert flows.since_first_return([], []) is None
    assert flows.since_first_return(None, []) is None
    assert flows.since_first_return([(D(2024, 1, 1), 100.0)], []) is None
    two = [(D(2024, 1, 1), 100.0), (D(2024, 1, 2), 110.0)]
    assert flows.since_first_return(two, None) == pytest.approx(0.1)


def test_since_first_return_uses_the_last_point_and_later_flows_only():
    series = [(D(2024, 1, 1), 1000.0), (D(2024, 1, 3), 5000.0), (D(2024, 1, 11), 1600.0)]
    on_first = {"date": D(2024, 1, 1), "type": "buy", "amount_eur": 999.0}   # in V_start
    deposit = {"date": D(2024, 1, 5), "type": "buy", "amount_eur": 500.0}
    on_last = {"date": D(2024, 1, 11), "type": "buy", "amount_eur": 100.0}   # counted
    after = {"date": D(2024, 1, 12), "type": "buy", "amount_eur": 777.0}     # outside
    r = flows.since_first_return(series, [on_first, deposit, on_last, after])
    assert r == pytest.approx(flows.modified_dietz(
        1000.0, 1600.0, [deposit, on_last], D(2024, 1, 1), D(2024, 1, 11)))
    assert r == pytest.approx(0.0 / 1300.0)

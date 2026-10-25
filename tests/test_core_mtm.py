"""test_core_mtm.py — the valuation store (quant/engine/mtm.py), exactly.

Runs the real module against an in-memory DuckDB with the store's tables; only
the true boundaries are faked (the connection factory, the FX source, the clock).
Every row the rebuild writes is compared in full.
"""
from __future__ import annotations

import contextlib
import logging
from datetime import date

import duckdb
import pytest

from quant.engine import mtm

D1, D2, D3, D4, D5 = (date(2024, 1, d) for d in (1, 2, 3, 4, 5))


@pytest.fixture
def db():
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE position_snapshots (snapshot_date DATE, symbol VARCHAR, "
                 "shares DOUBLE)")
    conn.execute("CREATE TABLE splits (effective_date DATE, symbol VARCHAR, factor DOUBLE)")
    conn.execute("CREATE TABLE market_history (Symbol VARCHAR, Date VARCHAR, Close DOUBLE)")
    conn.execute("CREATE TABLE valuation (date DATE, symbol VARCHAR, shares DOUBLE, "
                 "price_eur DOUBLE, fx DOUBLE, value_eur DOUBLE, source VARCHAR, "
                 "PRIMARY KEY (date, symbol))")
    conn.execute("CREATE TABLE portfolio_total (date DATE PRIMARY KEY, value_eur DOUBLE, "
                 "cash_eur DOUBLE)")
    yield conn
    conn.close()


class Tracked:
    """A connection wrapper that counts close() calls."""

    def __init__(self, inner):
        self.inner, self.closed = inner, 0

    def execute(self, *args):
        return self.inner.execute(*args)

    def close(self):
        self.closed += 1


def _snap(conn, day, **shares):
    for sym, sh in shares.items():
        conn.execute("INSERT INTO position_snapshots VALUES (?, ?, ?)", [day, sym, sh])


def _px(conn, sym, *points):
    for day, close in points:
        conn.execute("INSERT INTO market_history VALUES (?, ?, ?)", [sym, str(day), close])


def _valuation(conn):
    return conn.execute("SELECT date, symbol, shares, price_eur, fx, value_eur, source "
                        "FROM valuation ORDER BY date, symbol").fetchall()


def _totals(conn):
    return conn.execute("SELECT date, value_eur, cash_eur FROM portfolio_total "
                        "ORDER BY date").fetchall()


# ── small helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    (None, None), (D1, D1), ("2024-01-01T10:00:00", D1), ("2024-01-01", D1),
    ("bad", None), (12345, None)])
def test_as_date(raw, expected):
    assert mtm._as_date(raw) == expected


def test_num():
    assert mtm._num(None) == 0.0 and mtm._num("x") == 0.0
    assert mtm._num("x", 1.0) == 1.0 and mtm._num("2") == 2.0


# ── restate / shares_on (pure) ────────────────────────────────────────────────

@pytest.mark.parametrize("splits, since, expected", [
    ([(D2, "A", 2.0)], D1, 20.0),                 # a later split multiplies
    ([(D2, "A", 2.0)], D2, 10.0),                 # same day: already counted
    ([(D2, "A", 2.0)], D3, 10.0),                 # earlier split: already counted
    ([(D2, "A", 2.0)], None, 20.0),               # unknown date: every split
    ([(D2, "B", 2.0)], D1, 10.0),                 # another symbol
    ([(D2, "A", 0.0)], D1, 10.0),                 # nonsense factor
    ([(D2, "A", -2.0)], D1, 10.0),
    ([("bad", "A", 2.0)], D1, 10.0),              # unparseable date
    ([(D2, "A", None)], D1, 10.0),                # unknown factor counts as 1
    ([(D2, "A")], D1, 10.0),                      # short row
    ([(D2, "A", 2.0), (D3, "A", 3.0)], D1, 60.0),
    ([(D2, "A", 0.5)], D1, 5.0),                  # reverse split
    (None, D1, 10.0),
])
def test_restate(splits, since, expected):
    assert mtm.restate(10.0, "A", since, splits) == expected


def test_shares_on_uses_the_latest_saved_table_on_or_before_the_day():
    snaps = [(D1, {"A": 10.0, "Z": 0.0}), (D3, {"A": 1 / 3})]
    splits = [(D2, "A", 2.0)]
    assert mtm.shares_on(snaps, splits, date(2023, 12, 31)) == {}
    assert mtm.shares_on(snaps, splits, D1) == {"A": 20.0}
    assert mtm.shares_on(snaps, splits, D2) == {"A": 20.0}
    assert mtm.shares_on(snaps, splits, D3) == {"A": 0.333333}
    assert mtm.shares_on(snaps, splits, D5) == {"A": 0.333333}
    assert mtm.shares_on([], splits, D5) == {}


# ── readers of the inputs ─────────────────────────────────────────────────────

def test_snapshots_group_by_day_in_date_order(db):
    _snap(db, D3, A=5.0)
    _snap(db, D1, A=10.0, B=2.0)
    assert mtm._snapshots(db) == [(D1, {"A": 10.0, "B": 2.0}), (D3, {"A": 5.0})]


def test_snapshots_fall_back_to_the_current_table(db, monkeypatch):
    from quant.engine import positions

    monkeypatch.setattr(positions, "positions_now", lambda _c: [
        {"symbol": "A", "shares": 2.0}, {"symbol": "H", "shares": 0.5},
        {"symbol": "Z", "shares": 0.0}])
    assert mtm._snapshots(db) == [(date.today(), {"A": 2.0, "H": 0.5})]
    monkeypatch.setattr(positions, "positions_now", lambda _c: [])
    assert mtm._snapshots(db) == []


def test_split_rows(db):
    db.execute("INSERT INTO splits VALUES (?, 'A', 2.0), (?, 'B', 3.0)", [D3, D1])
    assert mtm._split_rows(db) == [(D1, "B", 3.0), (D3, "A", 2.0)]


def test_fx_rate(monkeypatch):
    from quant.data import currency

    monkeypatch.setattr(currency, "get_fx_to_eur", lambda s: "0.9" if s == "AMZN" else 1)
    assert mtm._fx_rate("AMZN") == 0.9


def test_price_history_per_symbol_sorted(db):
    _px(db, "A", ("bad-date", 5.0), (D2, None))
    _px(db, "A", (D3, 30.0), (D1, 10.0))
    _px(db, "B", (D2, 2.0))
    _px(db, "C", (D2, 99.0))
    assert mtm._price_history(db, {"A", "B"}) == {
        "A": [(D1, 10.0), (D3, 30.0)], "B": [(D2, 2.0)]}
    assert mtm._price_history(db, set()) == {}


@pytest.mark.parametrize("day, expected", [
    (date(2023, 12, 31), None), (D1, 10.0), (D2, 10.0), (D3, 30.0), (D5, 30.0)])
def test_close_on_or_before(day, expected):
    assert mtm._close_on_or_before([(D1, 10.0), (D3, 30.0)], day) == expected


def test_close_on_or_before_empty():
    assert mtm._close_on_or_before([], D1) is None


# ── the rebuild ───────────────────────────────────────────────────────────────

def test_backfill_rebuilds_every_row(db):
    db.execute("INSERT INTO valuation VALUES (?, 'OLD', 1, 1, 1, 1, 'x')", [D1])
    db.execute("INSERT INTO portfolio_total VALUES (?, 1, NULL)", [date(2023, 1, 1)])
    _snap(db, D2, A=10.0, B=1 / 3)
    _snap(db, D4, A=4.0)
    db.execute("INSERT INTO splits VALUES (?, 'A', 2.0)", [D3])
    _px(db, "A", (D1, 1.0), (D2, 10.0), (D3, 5.0), (D5, 6.0))
    _px(db, "B", (D3, 10.0))
    calls: list = []

    def fx(sym):
        calls.append(sym)
        return 0.0 if sym == "B" else 2.0          # a zero rate counts as 1.0

    res = mtm.backfill(db, fx_lookup=fx, source="test")
    assert res == {"days": 3, "rows": 4}
    assert _valuation(db) == [
        (D2, "A", 20.0, 10.0, 2.0, 400.0, "test"),     # 10 shares, 2:1 split on D3
        (D3, "A", 20.0, 5.0, 2.0, 200.0, "test"),
        (D3, "B", 0.333333, 10.0, 1.0, 3.33, "test"),
        (D5, "A", 4.0, 6.0, 2.0, 48.0, "test"),        # the D4 table, after the split
    ]
    assert _totals(db) == [(D2, 400.0, None), (D3, 203.33, None), (D5, 48.0, None)]
    assert sorted(calls) == ["A", "B"]                  # one FX lookup per symbol


def test_backfill_writes_a_total_on_a_day_without_any_price(db):
    _snap(db, D1, A=1.0, B=1.0, N=5.0)               # N never has a price
    _px(db, "B", (D1, 0.0))
    _px(db, "A", (D2, 3.0))
    res = mtm.backfill(db, fx_lookup=lambda _s: 1.0)
    assert res == {"days": 2, "rows": 3}
    assert _totals(db) == [(D1, 0.0, None), (D2, 3.0, None)]
    assert {r[6] for r in _valuation(db)} == {"backfill"}


def test_backfill_uses_the_live_rate_by_default(db, monkeypatch):
    from quant.data import currency

    monkeypatch.setattr(currency, "get_fx_to_eur", lambda _s: 0.5)
    _snap(db, D1, A=2.0)
    _px(db, "A", (D1, 10.0))
    mtm.backfill(db)
    assert _valuation(db) == [(D1, "A", 2.0, 10.0, 0.5, 10.0, "backfill")]


def test_backfill_without_any_saved_table_leaves_the_store_alone(db, monkeypatch):
    from quant.engine import positions

    monkeypatch.setattr(positions, "positions_now", lambda _c: [])
    db.execute("INSERT INTO portfolio_total VALUES (?, 7, NULL)", [D1])
    assert mtm.backfill(db) == {"days": 0, "rows": 0}
    assert _totals(db) == [(D1, 7.0, None)]


# ── readers of the store ──────────────────────────────────────────────────────

def _store(db):
    for d, v in ((D1, 100.0), (D3, 110.0), (D2, 90.0)):
        db.execute("INSERT INTO portfolio_total VALUES (?, ?, NULL)", [d, v])
    for d, s, v in ((D2, "B", 5.0), (D1, "A", 1.0), (D2, "A", 2.0)):
        db.execute("INSERT INTO valuation VALUES (?, ?, 1, 1, 1, ?, 'x')", [d, s, v])


def test_store_readers(db):
    assert mtm.read_total_series(db) == []
    assert mtm.latest_total(db) == {}
    assert mtm.latest_price_date(db) is None
    assert mtm.read_symbol_series(db) == {}
    assert mtm.day_delta(db) == (0.0, 0.0, None)
    _store(db)
    assert mtm.read_total_series(db) == [{"date": D1, "value_eur": 100.0},
                                         {"date": D2, "value_eur": 90.0},
                                         {"date": D3, "value_eur": 110.0}]
    assert mtm.latest_total(db) == {"date": D3, "value_eur": 110.0}
    assert mtm.latest_price_date(db) == D2
    assert mtm.read_symbol_series(db) == {
        "A": [{"date": D1, "value_eur": 1.0}, {"date": D2, "value_eur": 2.0}],
        "B": [{"date": D2, "value_eur": 5.0}]}
    eur, pct, when = mtm.day_delta(db)
    assert (eur, when) == (20.0, D3) and pct == pytest.approx(20.0 / 90.0 * 100.0)


def test_store_readers_on_an_unreadable_store():
    class Broken:
        def execute(self, *_a):
            raise RuntimeError("io")

    assert mtm.latest_price_date(Broken()) is None
    assert mtm.latest_total(Broken()) == {} and mtm.read_symbol_series(Broken()) == {}


def test_query_opens_a_read_only_connection_when_none_is_given(db, monkeypatch):
    from quant.data import database

    _store(db)

    @contextlib.contextmanager
    def ro():
        yield db

    monkeypatch.setattr(database, "read_only_connection", ro)
    assert mtm.latest_total() == {"date": D3, "value_eur": 110.0}


# ── startup catch-up ──────────────────────────────────────────────────────────

@pytest.fixture
def boundary(db, monkeypatch):
    """Fake connection factory + clock; records connects, sleeps and stale marks."""
    import time

    from quant.data import database

    state = {"conns": [], "sleeps": [], "fail": 0, "stale": [], "kwargs": []}

    def connect(**kwargs):
        state["kwargs"].append(kwargs)
        if state["fail"]:
            state["fail"] -= 1
            raise RuntimeError("locked")
        conn = Tracked(db)
        state["conns"].append(conn)
        return conn

    monkeypatch.setattr(database, "connect_with_retry", connect)
    monkeypatch.setattr(time, "sleep", state["sleeps"].append)
    monkeypatch.setattr(mtm, "_mark_valuation_stale", state["stale"].append)
    return state


def test_catch_up_without_prices_skips(boundary):
    assert mtm.catch_up() == {"ok": True, "skipped": True, "reason": "no prices"}
    assert [c.closed for c in boundary["conns"]] == [1]
    assert boundary["kwargs"] == [{"attempts": 1}]


def test_catch_up_when_the_store_is_current_skips(boundary, db):
    _px(db, "A", (D2, 1.0))
    db.execute("INSERT INTO valuation VALUES (?, 'A', 1, 1, 1, 1, 'x')", [D2])
    assert mtm.catch_up() == {"ok": True, "skipped": True, "latest": "2024-01-02"}
    db.execute("INSERT INTO valuation VALUES (?, 'A', 1, 1, 1, 1, 'x')", [D3])
    assert mtm.catch_up() == {"ok": True, "skipped": True, "latest": "2024-01-03"}


def test_catch_up_rebuilds_when_behind(boundary, db):
    _snap(db, D1, A=1.0)
    _px(db, "A", (D1, 5.0), (D2, 6.0))
    db.execute("INSERT INTO valuation VALUES (?, 'A', 1, 1, 1, 1, 'x')", [D1])
    got = mtm.catch_up()
    assert got == {"ok": True, "rows": 2, "days": 2, "latest": "2024-01-02"}
    assert {r[6] for r in _valuation(db)} == {"refresh"}
    assert [c.closed for c in boundary["conns"]] == [1]


def test_catch_up_rebuilds_an_empty_store(boundary, db):
    _snap(db, D1, A=1.0)
    _px(db, "A", (D1, 5.0))
    assert mtm.catch_up() == {"ok": True, "rows": 1, "days": 1, "latest": "2024-01-01"}


def test_catch_up_retries_with_backoff_then_marks_stale(boundary):
    boundary["fail"] = 99
    got = mtm.catch_up(retries=3, base_delay=0.5)
    assert got == {"ok": False, "error": "locked"}
    assert boundary["sleeps"] == [0.5, 1.0]
    assert [str(e) for e in boundary["stale"]] == ["locked"]
    assert len(boundary["kwargs"]) == 3


@pytest.mark.parametrize("retries, tries", [(1, 1), (0, 1), (-2, 1), ("2", 2)])
def test_catch_up_always_tries_at_least_once(boundary, retries, tries):
    boundary["fail"] = 99
    mtm.catch_up(retries=retries, base_delay=1.0)
    assert len(boundary["kwargs"]) == tries
    assert boundary["sleeps"] == [1.0 * 2 ** a for a in range(tries - 1)]


def test_catch_up_defaults(boundary):
    boundary["fail"] = 99
    mtm.catch_up()
    assert boundary["sleeps"] == [0.5, 1.0] and len(boundary["kwargs"]) == 3


def test_catch_up_recovers_after_a_failure(boundary):
    boundary["fail"] = 1
    assert mtm.catch_up(retries=3, base_delay=2.0)["ok"] is True
    assert boundary["sleeps"] == [2.0] and boundary["stale"] == []


def test_catch_up_closes_the_connection_even_when_a_query_fails(boundary, monkeypatch):
    class Broken(Tracked):
        def execute(self, *args):
            raise RuntimeError("io")

    from quant.data import database

    made: list = []

    def connect(**_k):
        made.append(Broken(None))
        return made[-1]

    monkeypatch.setattr(database, "connect_with_retry", connect)
    assert mtm.catch_up(retries=2, base_delay=0.0) == {"ok": False, "error": "io"}
    assert [c.closed for c in made] == [1, 1]


# ── marking the stage stale ───────────────────────────────────────────────────

@pytest.mark.parametrize("latest, run", [("r1", "r1"), (None, "new")])
def test_mark_valuation_stale(monkeypatch, caplog, latest, run):
    from quant.data import database
    from quant.engine import manifest

    conn = Tracked(None)
    seen: dict = {}
    monkeypatch.setattr(database, "connect_with_retry",
                        lambda **k: seen.setdefault("kwargs", k) and conn)
    monkeypatch.setattr(manifest, "latest_run_id", lambda c: latest if c is conn else "?")
    monkeypatch.setattr(manifest, "new_run_id", lambda: "new")
    monkeypatch.setattr(manifest, "record_stage",
                        lambda *a, **k: seen.update(args=a, kw=k))
    with caplog.at_level(logging.WARNING, logger="quant.engine.mtm"):
        mtm._mark_valuation_stale(RuntimeError("boom"))
    assert caplog.messages == ["valuation catch-up failed: boom"]
    assert seen["kwargs"] == {"attempts": 1}
    assert seen["args"] == (conn, run, "valuation", manifest.STATUS_STALE)
    assert seen["kw"] == {"error": "boom"}
    assert conn.closed == 1


def test_mark_valuation_stale_never_raises(monkeypatch):
    from quant.data import database
    from quant.engine import manifest

    class BadClose(Tracked):
        def close(self):
            raise RuntimeError("close")

    monkeypatch.setattr(database, "connect_with_retry", lambda **_k: BadClose(None))
    monkeypatch.setattr(manifest, "latest_run_id", lambda c: "r")
    monkeypatch.setattr(manifest, "record_stage", lambda *a, **k: None)
    mtm._mark_valuation_stale("x")
    monkeypatch.setattr(database, "connect_with_retry",
                        lambda **_k: (_ for _ in ()).throw(RuntimeError("down")))
    mtm._mark_valuation_stale("x")

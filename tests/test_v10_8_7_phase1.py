"""test_v10_8_7_phase1.py — v10.8.7 Phase 1: T1 (cash removal) + T5 (write hygiene).

T1 (R-C1): uninvested cash is not an entity. With a 10 000 EUR money-in flow in the
ledger and 1 082 EUR invested, the Total value is 1 082; the class table Now sums to
100%; no advice mentions a cash percentage; the invest remainder line is neutral and
grammatical.

T5 (R-C4): every startup backfill runs once per process, before first render, with
bounded retry + exponential backoff; on final failure ONE quiet log line and a
``names`` stage ``stale`` flag; zero writes during reruns.

Synthetic fixtures only; the user's real DB and data dir are never touched.
"""
from __future__ import annotations

import logging
from datetime import date

import pytest

from tests import store_readers

# ── T1: cash is not an entity ─────────────────────────────────────────────────

_INVESTED = {
    "EUNL.DE": (5.0, 100.0),   # 500
    "SXRV.DE": (2.0, 100.0),   # 200
    "AMZN": (3.0, 100.0),      # 300
    "5J50.DE": (0.82, 100.0),  # 82
}
_DAY = date(2026, 10, 1)


@pytest.fixture
def invested_ledger():
    """A ledger with a 10 000 EUR money-in flow and 1 082 EUR invested."""
    from quant.data.database import write_connection
    from quant.engine import mtm

    with write_connection() as conn:
        conn.execute("DELETE FROM valuation")
        conn.execute("DELETE FROM portfolio_total")
        conn.execute("DELETE FROM position_snapshots")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM splits")
        conn.execute("DELETE FROM flows")
        for sym in _INVESTED:
            conn.execute("DELETE FROM market_history WHERE Symbol = ?", [sym])
        for sym, (shares, price) in _INVESTED.items():
            conn.execute(
                "INSERT INTO position_snapshots (snapshot_date, symbol, shares) "
                "VALUES (?, ?, ?)", [_DAY, sym, shares])
            conn.execute(
                "INSERT OR REPLACE INTO market_history (Symbol, Date, Close) "
                "VALUES (?, ?, ?)", [sym, str(_DAY), price])
        # A 10 000 EUR money-in flow: audit only, never a cash balance.
        conn.execute("INSERT INTO flows (date, type, amount_eur) VALUES (?, ?, ?)",
                     [_DAY, "buy", 10000.0])
        mtm.backfill(conn, fx_lookup=lambda _s: 1.0, source="test")


def test_total_value_is_invested_only(invested_ledger):
    """T1 acceptance: Total = 1 082 (the money-in flow is not a cash balance)."""
    from quant.engine import mtm

    assert abs(store_readers.portfolio_total(_DAY) - 1082.0) < 1e-9
    latest = mtm.latest_total()
    assert abs(float(latest["value_eur"]) - 1082.0) < 1e-9
    assert "cash_eur" not in latest


def test_class_now_sums_to_100_percent():
    """T1 acceptance: the class table Now column sums to 100%."""
    from quant.engine.invest_plan import propose_by_class

    holdings = [
        {"symbol": "EUNL.DE", "tier": "FORTRESS", "value_eur": 500.0},
        {"symbol": "SXRV.DE", "tier": "ALPHA", "value_eur": 200.0},
        {"symbol": "AMZN", "tier": "ALPHA", "value_eur": 300.0},
        {"symbol": "5J50.DE", "tier": "SPECULATIVE", "value_eur": 82.0},
    ]
    plan = propose_by_class(100.0, holdings, "balanced")
    now_sum = sum(r["now_pct"] for r in plan["classes"])
    assert abs(now_sum - 100.0) < 1e-6, f"class Now sums to {now_sum}"


def test_invest_remainder_line_is_neutral_and_grammatical():
    """T1: the remainder line has correct singular/plural and no doubled currency."""
    from quant.ui import copy as C

    one = C.invest_remainder_line(1.0)
    many = C.invest_remainder_line(50.0)
    assert one == "1.00 EUR remains at the broker (not tracked)."
    assert many == "50.00 EUR remain at the broker (not tracked)."
    assert "EUR EUR" not in one and "EUR EUR" not in many


def test_route_sell_never_targets_tracked_cash():
    """T1: a sell with no eligible buy holds proceeds at the broker, untracked."""
    from quant.engine.pairing import ROUTE_HOLD, route_sell

    route = route_sell({"symbol": "SXRV.DE", "amount_eur": 65.0},
                       [{"symbol": "SXRV.DE", "current_weight": 0.3,
                         "target_weight": 0.2, "value_eur": 300.0}])
    assert route["kind"] == ROUTE_HOLD
    assert "untracked" in route["text"]
    assert "cash target" not in route["text"]


# ── T5: write hygiene ─────────────────────────────────────────────────────────

def test_startup_backfill_runs_once_per_process(monkeypatch):
    """T5: both backfills run exactly once per process; a rerun writes nothing."""
    from quant import dashboard

    calls = {"names": 0, "catch_up": 0}

    import quant.data.names as names_mod
    import quant.engine.mtm as mtm_mod

    monkeypatch.setattr(names_mod, "ensure_display_names",
                        lambda: calls.__setitem__("names", calls["names"] + 1))
    monkeypatch.setattr(mtm_mod, "catch_up",
                        lambda *a, **k: calls.__setitem__("catch_up",
                                                          calls["catch_up"] + 1))
    monkeypatch.setattr(dashboard, "_STARTUP_DONE", False)
    dashboard._startup_backfill()
    dashboard._startup_backfill()  # a rerun
    assert calls["names"] == 1, "names backfill must run once per process"
    assert calls["catch_up"] == 1, "valuation catch-up must run once per process"


def test_locked_db_logs_one_line_and_marks_names_stale(monkeypatch, caplog, capsys):
    """T5: a held lock yields ONE quiet log line, a stale flag, no print, no raise."""
    import quant.data.database as db
    import quant.data.names as names_mod

    def _locked(*_a, **_k):
        raise RuntimeError("database is locked")

    stale = {"n": 0}
    monkeypatch.setattr(db, "connect_with_retry", _locked)
    monkeypatch.setattr(names_mod, "_mark_names_stale",
                        lambda _e: stale.__setitem__("n", stale["n"] + 1))
    with caplog.at_level(logging.WARNING, logger="quant.ui"):
        res = names_mod.ensure_display_names()
    assert res == {"skipped_reason": "lock"}
    assert stale["n"] == 1, "must mark the names stage stale exactly once"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, f"expected one quiet log line, got {len(warnings)}"
    assert capsys.readouterr().out == "", "no stdout spam on a held lock"


def test_mark_names_stale_writes_a_manifest_row(monkeypatch):
    """T5: the stale flag is a run_manifest row for the ``names`` stage."""
    import quant.data.database as db
    import quant.data.names as names_mod
    from quant.engine import manifest

    recorded = {}

    class _Conn:
        def close(self):
            pass

    monkeypatch.setattr(db, "connect_with_retry", lambda *a, **k: _Conn())
    monkeypatch.setattr(manifest, "latest_run_id", lambda _c: "run-1")
    monkeypatch.setattr(manifest, "record_stage",
                        lambda _c, run, stage, status, **k: recorded.update(
                            {"run": run, "stage": stage, "status": status}))
    names_mod._mark_names_stale(RuntimeError("database is locked"))
    assert recorded["stage"] == "names"
    assert recorded["status"] == manifest.STATUS_STALE

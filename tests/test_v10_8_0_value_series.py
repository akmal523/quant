"""test_v10_8_0_value_series.py — the daily invested-value series (v10.8.0, Phase 1).

The Overview chart must work on a database with zero reviews. The series is
built from append-only position snapshots, recorded flows, and daily closes.
These tests pin the invariants:

  - two holdings produce the sum of their EUR values;
  - a mid-period buy adds shares at the flow-day close (no phantom gain);
  - a later re-sync with a different share count is a neutral correction from
    its date forward and never rewrites past points.
"""
from __future__ import annotations

from datetime import date

import pytest

from quant.engine.value_series import build_value_series


def _closes(rows: dict[tuple[str, str], float]) -> dict:
    return rows


def test_two_holdings_sum_of_eur_values():
    snapshots = [
        {"date": "2026-01-01", "symbol": "AAA", "shares": 10.0},
        {"date": "2026-01-01", "symbol": "BBB", "shares": 5.0},
    ]
    closes = _closes({
        ("AAA", "2026-01-01"): 100.0,
        ("BBB", "2026-01-01"): 200.0,
    })
    series = build_value_series(snapshots, [], closes, {})
    assert len(series) == 1
    assert series[0]["date"] == date(2026, 1, 1)
    # 10 * 100 + 5 * 200 = 2000
    assert series[0]["value_eur"] == 2000.0


def test_mid_period_buy_adds_shares_without_phantom_gain():
    snapshots = [
        {"date": "2026-01-01", "symbol": "AAA", "shares": 10.0},
    ]
    flows = [
        {"date": "2026-01-03", "type": "buy", "amount_eur": 100.0,
         "symbol": "AAA"},
    ]
    closes = _closes({
        ("AAA", "2026-01-01"): 100.0,
        ("AAA", "2026-01-02"): 100.0,
        ("AAA", "2026-01-03"): 100.0,
        ("AAA", "2026-01-04"): 100.0,
    })
    series = build_value_series(snapshots, flows, closes, {})
    by_day = {s["date"].isoformat(): s["value_eur"] for s in series}

    # Before the buy: 10 shares * 100 = 1000.
    assert by_day["2026-01-02"] == 1000.0
    # On the buy day the 100 EUR buys 1 share at the 100 close: 11 * 100 = 1100.
    assert by_day["2026-01-03"] == 1100.0
    # The buy itself creates no gain: the value rose by exactly the cash added.
    assert by_day["2026-01-03"] - by_day["2026-01-02"] == 100.0


def test_resync_is_neutral_correction_and_does_not_alter_past_points():
    snapshots = [
        {"date": "2026-01-01", "symbol": "AAA", "shares": 10.0},
        # A later broker re-sync reports a different share count.
        {"date": "2026-01-04", "symbol": "AAA", "shares": 12.0},
    ]
    closes = _closes({
        ("AAA", "2026-01-01"): 100.0,
        ("AAA", "2026-01-02"): 100.0,
        ("AAA", "2026-01-03"): 100.0,
        ("AAA", "2026-01-04"): 100.0,
    })
    series = build_value_series(snapshots, [], closes, {})
    by_day = {s["date"].isoformat(): s["value_eur"] for s in series}

    # Past points use the old snapshot: 10 * 100 = 1000.
    assert by_day["2026-01-01"] == 1000.0
    assert by_day["2026-01-02"] == 1000.0
    assert by_day["2026-01-03"] == 1000.0
    # From the re-sync date forward the new count is truth: 12 * 100 = 1200.
    assert by_day["2026-01-04"] == 1200.0


def test_missing_close_carries_previous_close():
    snapshots = [
        {"date": "2026-01-01", "symbol": "AAA", "shares": 10.0},
    ]
    # A dividend on 2026-01-03 extends the range without changing shares.
    flows = [
        {"date": "2026-01-03", "type": "dividend", "amount_eur": 5.0,
         "symbol": "AAA"},
    ]
    closes = _closes({
        ("AAA", "2026-01-01"): 100.0,
        # 2026-01-02 has no close (weekend/holiday).
        ("AAA", "2026-01-03"): 110.0,
    })
    series = build_value_series(snapshots, flows, closes, {})
    by_day = {s["date"].isoformat(): s["value_eur"] for s in series}
    assert by_day["2026-01-02"] == 1000.0  # carried 100
    assert by_day["2026-01-03"] == 1100.0  # 10 * 110


def test_fx_converts_native_close_to_eur():
    snapshots = [
        {"date": "2026-01-01", "symbol": "USDX", "shares": 10.0},
    ]
    closes = _closes({("USDX", "2026-01-01"): 100.0})
    series = build_value_series(snapshots, [], closes, {"USDX": 0.9})
    # 10 * 100 * 0.9 = 900
    assert series[0]["value_eur"] == 900.0


def test_empty_inputs_return_empty_series():
    assert build_value_series([], [], {}, {}) == []


# ── Integration: the chart works on a database with zero reviews ──────────────

def test_chart_renders_with_zero_reviews(tmp_path, monkeypatch):
    """The Overview chart reads the value series, not the review history."""
    pytest.importorskip("streamlit")
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from quant import paths
    from quant.data import database
    from quant.ui import copy

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    conn = database.get_connection()
    conn.execute("DELETE FROM position_snapshots")
    conn.execute("DELETE FROM market_history WHERE Symbol = 'AAA'")
    for day, close in (("2026-01-01", 100.0), ("2026-01-02", 101.0),
                       ("2026-01-03", 102.0)):
        conn.execute(
            "INSERT INTO market_history (Date, Close, Symbol) VALUES (?, ?, 'AAA')",
            [day, close])
    for day in ("2026-01-01", "2026-01-02", "2026-01-03"):
        conn.execute(
            "INSERT INTO position_snapshots (snapshot_date, symbol, shares) "
            "VALUES (?, 'AAA', 10.0)", [day])

    dashboard = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
    try:
        at = AppTest.from_file(dashboard, default_timeout=60)
        at.run()
        at.switch_page("pages/today.py").run()
        assert not at.exception
        text = " ".join(str(getattr(el, "value", ""))
                        for attr in ("markdown", "info", "warning", "error",
                                     "caption", "text", "title", "header",
                                     "subheader")
                        for el in getattr(at, attr, []))
        assert copy.CHART_BUILDING not in text
    finally:
        # Keep the shared session DB clean so other chart tests stay
        # order-independent.
        conn.execute("DELETE FROM position_snapshots")
        conn.execute("DELETE FROM market_history WHERE Symbol = 'AAA'")

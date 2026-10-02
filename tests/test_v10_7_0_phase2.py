"""
test_v10_7_0_phase2.py — Valuation, flows, performance math, retention (v10.7.0).

Intent: lock the honest-money layer. Shares come from the broker CSV sync;
between syncs values are estimated. Modified Dietz excludes deposits. Retention
downsamples old daily bars and prunes the news cache. The daily-run marker
drives the doctor and the app-open fallback.

Invariants:
  - Pure helpers do no I/O.
  - I/O helpers use the isolated test DB from conftest.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from quant.data.database import get_connection
from quant.engine import daily, flows, retention, valuation

# ── Valuation ─────────────────────────────────────────────────────────────────

def test_compute_shares_from_value_and_price():
    assert valuation.compute_shares(1000.0, 100.0) == 10.0
    assert valuation.compute_shares(1000.0, 0.0) is None
    assert valuation.compute_shares(None, 100.0) is None


def test_estimated_label_is_visible():
    label = valuation.estimated_label(date(2026, 10, 1))
    assert "estimated" in label
    assert "1 Oct 2026" in label


def test_sync_holdings_meta_writes_shares():
    conn = get_connection()
    conn.execute("DELETE FROM holdings_meta")
    df = pd.DataFrame([
        {"Symbol": "EUNL.DE", "Current_Value_EUR": 1000.0, "Invested_EUR": 900.0},
        {"Symbol": "AMZN", "Current_Value_EUR": 500.0, "Invested_EUR": 400.0},
    ])
    written = valuation.sync_holdings_meta(
        conn, df, {"EUNL.DE": 100.0, "AMZN": 50.0}, date(2026, 10, 1))
    assert written == 2
    row = conn.execute(
        "SELECT shares, invested_at_sync FROM holdings_meta WHERE symbol = 'EUNL.DE'"
    ).fetchone()
    assert row[0] == 10.0
    assert row[1] == 900.0


def test_holdings_meta_sync_on_run():
    """Real-world bug (v10.7.2, Part 5.2): quant run reads portfolio.csv but did
    not populate holdings_meta. After the sync, holdings_meta must contain all
    symbols with computed shares and today's sync_date."""
    conn = get_connection()
    conn.execute("DELETE FROM holdings_meta")
    portfolio_df = pd.DataFrame([
        {"Symbol": "EUNL.DE", "Current_Value_EUR": 287.19, "Invested_EUR": 278.00},
        {"Symbol": "SXRV.DE", "Current_Value_EUR": 270.49, "Invested_EUR": 253.90},
        {"Symbol": "AMZN", "Current_Value_EUR": 150.03, "Invested_EUR": 150.00},
        {"Symbol": "5J50.DE", "Current_Value_EUR": 143.53, "Invested_EUR": 151.00},
    ])
    prices = {"EUNL.DE": 95.50, "SXRV.DE": 85.20, "AMZN": 150.00, "5J50.DE": 151.00}
    written = valuation.sync_holdings_meta(conn, portfolio_df, prices, date.today())
    assert written == 4
    rows = conn.execute(
        "SELECT symbol, shares, sync_date FROM holdings_meta ORDER BY symbol").fetchall()
    assert len(rows) == 4
    assert {r[0] for r in rows} == {"AMZN", "EUNL.DE", "SXRV.DE", "5J50.DE"}
    assert all(r[1] > 0 for r in rows)
    assert all(str(r[2])[:10] == date.today().isoformat() for r in rows)


def test_sync_first_time_only_preserves_sync_date():
    """The review path seeds new symbols but never overwrites an existing row, so
    the recorded sync_date keeps reflecting the user's last CSV export."""
    conn = get_connection()
    conn.execute("DELETE FROM holdings_meta")
    df = pd.DataFrame([{"Symbol": "AMZN", "Current_Value_EUR": 100.0,
                        "Invested_EUR": 90.0}])
    valuation.sync_holdings_meta(conn, df, {"AMZN": 50.0}, date(2026, 1, 1))
    written = valuation.sync_holdings_meta(
        conn, df, {"AMZN": 50.0}, date(2026, 10, 2), first_time_only=True)
    assert written == 0
    row = conn.execute(
        "SELECT sync_date FROM holdings_meta WHERE symbol = 'AMZN'").fetchone()
    assert str(row[0])[:10] == "2026-01-01"


def test_revalue_holdings_uses_latest_close():
    conn = get_connection()
    conn.execute("DELETE FROM holdings_meta")
    df = pd.DataFrame([{"Symbol": "EUNL.DE", "Current_Value_EUR": 1000.0,
                        "Invested_EUR": 900.0}])
    valuation.sync_holdings_meta(conn, df, {"EUNL.DE": 100.0}, date(2026, 10, 1))
    values = valuation.revalue_holdings(conn, {"EUNL.DE": 110.0})
    assert values["EUNL.DE"] == 1100.0


def test_sync_reminder_after_threshold():
    conn = get_connection()
    conn.execute("DELETE FROM holdings_meta")
    # v10.7.4 (R7): clear the 7-day throttle stamp so the reminder can fire.
    conn.execute("DELETE FROM meta")
    df = pd.DataFrame([{"Symbol": "EUNL.DE", "Current_Value_EUR": 1000.0,
                        "Invested_EUR": 900.0}])
    valuation.sync_holdings_meta(conn, df, {"EUNL.DE": 100.0}, date(2026, 1, 1))
    assert valuation.days_since_last_sync(conn, date(2026, 1, 20)) == 19
    assert valuation.sync_reminder_line(conn, date(2026, 1, 20)) is None
    line = valuation.sync_reminder_line(conn, date(2026, 3, 1))
    assert line is not None and "broker sync" in line


# ── Flows and Modified Dietz ──────────────────────────────────────────────────

def test_modified_dietz_excludes_deposit():
    start = date(2026, 9, 1)
    end = date(2026, 10, 1)
    mid = date(2026, 9, 16)
    # 200 EUR buy mid-period, no market move: return must be 0.
    ret = flows.modified_dietz(
        1000.0, 1200.0,
        [{"date": mid, "type": "buy", "amount_eur": 200.0}],
        start, end)
    assert abs(ret) < 1e-9


def test_modified_dietz_dividend_is_outflow():
    start = date(2026, 9, 1)
    end = date(2026, 10, 1)
    mid = date(2026, 9, 16)
    # 20 EUR dividend leaves invested, no market move: return must be 0.
    ret = flows.modified_dietz(
        1000.0, 980.0,
        [{"date": mid, "type": "dividend", "amount_eur": 20.0}],
        start, end)
    assert abs(ret) < 1e-9


def test_modified_dietz_market_move_only():
    start = date(2026, 9, 1)
    end = date(2026, 10, 1)
    ret = flows.modified_dietz(1000.0, 1100.0, [], start, end)
    assert abs(ret - 0.10) < 1e-9


def test_performance_line_mentions_deposits_and_market():
    start = date(2026, 9, 1)
    end = date(2026, 10, 1)
    line = flows.performance_line(
        1000.0, 1200.0,
        [{"date": date(2026, 9, 16), "type": "buy", "amount_eur": 200.0}],
        start, end)
    assert "you added" in line
    assert "market moved" in line
    assert "Return" in line


def test_sparplan_planned_then_actuals_reconciliation():
    conn = get_connection()
    conn.execute("DELETE FROM flows")
    flows.write_planned_sparplan_flows(
        conn, "2026-11",
        [{"symbol": "EUNL.DE", "amount_eur": 140.0}],
        date(2026, 11, 1))
    planned = flows.load_flows(conn)
    assert len(planned) == 1 and planned[0]["type"] == "buy"
    recon = flows.replace_auto_flows_with_actuals(
        conn, "2026-11",
        [{"symbol": "EUNL.DE", "amount_eur": 140.0, "date": date(2026, 11, 1)}])
    assert recon[0]["planned"] == 140.0
    assert recon[0]["actual"] == 140.0
    assert recon[0]["deviation"] == 0.0
    remaining = flows.load_flows(conn)
    assert len(remaining) == 1
    assert remaining[0]["note"] == "actual sparplan 2026-11"


# ── Retention ─────────────────────────────────────────────────────────────────

def test_downsample_daily_bars_is_idempotent():
    conn = get_connection()
    conn.execute("DELETE FROM market_history WHERE Symbol = 'TEST.DE'")
    old = date(2015, 1, 5)  # a Monday, well over 5 years ago
    rows = []
    for i in range(10):
        d = old + timedelta(days=i)
        rows.append((d.isoformat(), 10.0 + i, 20.0 + i, 5.0 + i, 15.0 + i, 100.0, "TEST.DE"))
    for d, o, h, low, c, v, sym in rows:
        conn.execute(
            "INSERT OR REPLACE INTO market_history "
            "(Date, Open, High, Low, Close, Volume, Symbol) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [d, o, h, low, c, v, sym])
    removed = retention.downsample_daily_bars(conn, date(2026, 10, 1))
    assert removed > 0
    after = conn.execute(
        "SELECT COUNT(*) FROM market_history WHERE Symbol = 'TEST.DE'").fetchone()[0]
    assert after < len(rows)
    # Idempotent: a second pass removes nothing.
    assert retention.downsample_daily_bars(conn, date(2026, 10, 1)) == 0


def test_prune_news_cache_removes_old_entries(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    import json
    import time

    now = time.time()
    cache = {
        "OLD": {"retrieved_at": now - 200 * 86400, "items": []},
        "NEW": {"retrieved_at": now - 1 * 86400, "items": []},
    }
    (tmp_path / "news_cache.json").write_text(json.dumps(cache), encoding="utf-8")
    removed = retention.prune_news_cache()
    assert removed == 1
    kept = json.loads((tmp_path / "news_cache.json").read_text(encoding="utf-8"))
    assert set(kept) == {"NEW"}


# ── Daily-run marker ──────────────────────────────────────────────────────────

def test_daily_marker_roundtrip(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    assert daily.read_last_daily_run() is None
    daily.write_last_daily_run(date(2026, 10, 1))
    assert daily.read_last_daily_run() == date(2026, 10, 1)
    assert daily.monitoring_gap_days(date(2026, 10, 4)) == 3

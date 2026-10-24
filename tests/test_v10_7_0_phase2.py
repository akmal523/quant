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

from datetime import date

from quant.engine import daily, flows

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


# ── Daily-run marker ──────────────────────────────────────────────────────────

def test_daily_marker_roundtrip(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    assert daily.read_last_daily_run() is None
    daily.write_last_daily_run(date(2026, 10, 1))
    assert daily.read_last_daily_run() == date(2026, 10, 1)
    assert daily.monitoring_gap_days(date(2026, 10, 4)) == 3

"""
test_v10_7_1_phase_d.py — Proof under load (v10.7.1, Part 4).

Intent: prove the system under load and end to end. 100 assets under 60
seconds; the monthly ritual; catch-up honesty; the app-open fallback; the
advice record; and the extended forbidden-token copy test.

Invariants: tests never touch the network or the real systemd.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pytest

from quant.data.database import get_connection
from quant.engine import alerts, daily, flows, plans
from quant.ui import copy as C

# ── Stress: 100 assets under 60 seconds ───────────────────────────────────────

def test_daily_run_100_assets_under_60s(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    holdings = [
        {"symbol": f"SYM{i:03d}", "name": f"Company {i}", "tier": "ALPHA",
         "value_eur": 1000.0, "current_weight": 0.01, "target_weight": 0.01,
         "conviction": 0.0}
        for i in range(100)
    ]
    start = time.time()
    result = daily.run_daily(
        date(2026, 10, 1),
        update_fn=lambda: None,
        holdings_fn=lambda conn: holdings,
        notify=False,
    )
    elapsed = time.time() - start
    assert result.status == "ok"
    assert elapsed < 60.0, f"100-asset daily run took {elapsed:.1f}s"


# ── Monthly end-to-end ────────────────────────────────────────────────────────

def test_monthly_end_to_end(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    conn = get_connection()
    conn.execute("DELETE FROM flows")
    conn.execute("DELETE FROM holdings_meta")
    conn.execute("DELETE FROM monthly_plans")

    legs = [{"symbol": "EUNL.DE", "name": "MSCI World", "amount_eur": 140.0,
             "kind": "long_term", "reason": "r", "fee_eur": 0.0}]
    plans.save_plan(conn, "2026-11", 200.0, legs,
                    approved_date=date(2026, 11, 1), execution_date=date(2026, 11, 1))
    flows.write_planned_sparplan_flows(
        conn, "2026-11", [{"symbol": "EUNL.DE", "amount_eur": 140.0}],
        date(2026, 11, 1))
    recon = plans.enter_actuals(
        conn, "2026-11",
        [{"symbol": "EUNL.DE", "amount_eur": 140.0, "date": date(2026, 11, 1)}],
        price_lookup={"EUNL.DE": 100.0})
    assert recon[0]["deviation"] == 0.0
    remaining = flows.load_flows(conn)
    assert len(remaining) == 1 and remaining[0]["note"] == "actual sparplan 2026-11"
    row = conn.execute(
        "SELECT shares FROM holdings_meta WHERE symbol = 'EUNL.DE'").fetchone()
    assert abs(row[0] - 1.4) < 1e-9
    assert plans.is_pending_sync()
    assert "Planned 140 EUNL.DE" in plans.reconciliation_line(recon[0])


# ── Catch-up end-to-end ───────────────────────────────────────────────────────

def test_catch_up_gap_phrase(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    daily.write_last_daily_run(date(2026, 9, 28))
    result = daily.run_daily(
        date(2026, 10, 1), update_fn=lambda: None,
        holdings_fn=lambda conn: [], notify=False)
    assert result.gap_days == 3
    artifact = (tmp_path / "daily_2026-10-01.md").read_text(encoding="utf-8")
    assert "Monitoring gap: no runs for 3 days" in artifact


# ── App-open fallback ─────────────────────────────────────────────────────────

def test_app_open_fallback_stale_and_current(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    # No marker -> stale.
    assert daily.staleness_status(date(2026, 10, 1)) is not None
    # Current marker -> not stale.
    daily.write_last_daily_run(date(2026, 9, 30))
    assert daily.staleness_status(date(2026, 10, 1)) is None
    # Old marker -> stale.
    daily.write_last_daily_run(date(2026, 9, 20))
    assert daily.staleness_status(date(2026, 10, 1)) is not None


# ── Advice record ─────────────────────────────────────────────────────────────

def test_advice_record_correct_and_format():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    alerts.evaluate_alerts(
        conn, [{"symbol": "AMZN", "tier": "ALPHA", "value_eur": 500.0,
                "structural": 30.0}], today=date(2026, 1, 1))
    alert_id = alerts.open_alerts(conn)[0]["id"]
    alerts.resolve_alert(conn, alert_id, "done", "sold", 100.0, today=date(2026, 1, 2))
    scored = alerts.score_resolved_alerts(conn, {"AMZN": 80.0}, today=date(2026, 2, 5))
    assert scored == 1
    row = conn.execute("SELECT verdict FROM alerts WHERE id = ?", [alert_id]).fetchone()
    assert row[0] == "correct"
    line = alerts.advice_record_line(conn)
    assert "Advice record" in line and "correct" in line


# ── Extended forbidden-token copy test ────────────────────────────────────────

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")
_PAGE_FILE = {
    C.PAGE_HOLDINGS: "pages/portfolio.py",
    C.PAGE_FIND: "pages/explore.py",
}


def _all_text(at: AppTest) -> str:
    chunks: list[str] = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


@pytest.mark.parametrize("page", [C.PAGE_HOLDINGS, C.PAGE_FIND])
def test_no_forbidden_tokens_on_redesigned_pages(page):
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(_PAGE_FILE[page]).run()
    assert not at.exception, f"{page} raised: {at.exception}"
    text = _all_text(at)
    for token in C.FORBIDDEN_TOKENS:
        assert token not in text, f"forbidden token {token!r} on {page}"
    for legacy in ("SECTOR", "SATELLITE", "CORE", "ACTIVE"):
        assert legacy not in text, f"legacy tier {legacy!r} on {page}"
    assert "Average entry price" not in text

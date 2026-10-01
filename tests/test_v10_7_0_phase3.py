"""
test_v10_7_0_phase3.py — Engine, alerts, notify, scheduler (v10.7.0, Phase 3).

Intent: lock the anti-nonsense laws, the five level-triggered alert conditions,
the notification payload rules, and the scheduler unit content. Tests never
touch the real systemd or the network.

Invariants:
  - Sizing laws are absolute (the 143.53 EUR historical failure never sells).
  - Alerts are level-triggered and never duplicated while open.
  - Notification payloads carry no totals and never the token.
"""
from __future__ import annotations

from datetime import date

from quant.data.database import get_connection
from quant.engine import alerts, daily, notify, scheduler, sizing

# ── Sizing laws (Section 6) ───────────────────────────────────────────────────

def test_fortress_never_sells():
    assert sizing.can_sell("FORTRESS", 1000.0, 200.0) is None


def test_sell_amount_sanity_and_rounding():
    # min(200, 143.53 - 1) = 142.53 -> round down to 5 -> 140.
    assert sizing.sell_amount(200.0, 143.53) == 140.0


def test_historical_failure_no_sell_for_small_drift():
    # 143.53 EUR position, 6.9 percent drift (~9.9 EUR) -> below 25 EUR -> None.
    drift = 0.069 * 143.53
    assert sizing.sell_amount(drift, 143.53) is None


def test_small_positions_untouchable():
    assert sizing.is_untouchable(50.0)
    assert sizing.can_sell("ALPHA", 50.0, 200.0) is None


def test_cooldown_blocks_sells_only():
    assert sizing.cooldown_blocks_sell(True) is True
    assert sizing.cooldown_blocks_sell(False) is False


def test_buy_rounding_and_fees():
    assert sizing.round_buy(147.0) == 140.0
    assert sizing.buy_fee(is_savings_plan=True) == 0.0
    assert sizing.buy_fee(is_savings_plan=False) == 1.0


def test_cash_hurdle():
    assert sizing.cash_hurdle(True) is True
    assert sizing.cash_hurdle(False) is False


# ── Alert conditions (Section 5) ──────────────────────────────────────────────

def _clear_alerts():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    return conn


def test_structural_break_fires():
    conn = _clear_alerts()
    created = alerts.evaluate_alerts(
        conn, [{"symbol": "AMZN", "name": "Amazon.com", "tier": "ALPHA",
                "value_eur": 500.0, "structural": 35.0}], today=date(2026, 10, 1))
    assert any(a["kind"] == alerts.KIND_STRUCTURAL for a in created)


def test_tactical_collapse_fires():
    conn = _clear_alerts()
    created = alerts.evaluate_alerts(
        conn, [{"symbol": "AMZN", "tier": "ALPHA", "value_eur": 500.0,
                "tactical": 40.0, "prev_tactical_7d": 70.0}], today=date(2026, 10, 1))
    assert any(a["kind"] == alerts.KIND_TACTICAL for a in created)


def test_position_crash_fires():
    conn = _clear_alerts()
    created = alerts.evaluate_alerts(
        conn, [{"symbol": "AMZN", "tier": "ALPHA", "value_eur": 800.0,
                "prev_value_7d": 1000.0}], today=date(2026, 10, 1))
    assert any(a["kind"] == alerts.KIND_CRASH for a in created)


def test_regime_flip_fires_once():
    conn = _clear_alerts()
    created = alerts.evaluate_alerts(
        conn, [], regime="bear", prev_regime="bull", today=date(2026, 10, 1))
    assert any(a["kind"] == alerts.KIND_REGIME for a in created)
    # A second evaluation does not duplicate the open market-level alert.
    again = alerts.evaluate_alerts(
        conn, [], regime="bear", prev_regime="bull", today=date(2026, 10, 2))
    assert not any(a["kind"] == alerts.KIND_REGIME for a in again)


def test_speculative_stop_loss_fires():
    conn = _clear_alerts()
    created = alerts.evaluate_alerts(
        conn, [{"symbol": "MEME", "tier": "SPECULATIVE", "value_eur": 40.0,
                "entry_price": 100.0, "current_price": 40.0}], today=date(2026, 10, 1))
    assert any(a["kind"] == alerts.KIND_SPECULATIVE for a in created)


def test_level_triggered_no_duplicate_open_alert():
    conn = _clear_alerts()
    holding = [{"symbol": "AMZN", "tier": "ALPHA", "value_eur": 500.0,
                "structural": 30.0}]
    first = alerts.evaluate_alerts(conn, holding, today=date(2026, 10, 1))
    second = alerts.evaluate_alerts(conn, holding, today=date(2026, 10, 2))
    assert len(first) == 1
    assert second == []


def test_resolve_and_self_scoring():
    conn = _clear_alerts()
    alerts.evaluate_alerts(
        conn, [{"symbol": "AMZN", "tier": "ALPHA", "value_eur": 500.0,
                "structural": 30.0}], today=date(2026, 1, 1))
    open_now = alerts.open_alerts(conn)
    assert open_now
    alert_id = open_now[0]["id"]
    assert alerts.resolve_alert(conn, alert_id, "done", "sold", 100.0,
                                today=date(2026, 1, 2))
    # 30 days later the price is lower -> a done sell is correct.
    scored = alerts.score_resolved_alerts(
        conn, {"AMZN": 80.0}, today=date(2026, 2, 5))
    assert scored == 1
    row = conn.execute("SELECT verdict FROM alerts WHERE id = ?", [alert_id]).fetchone()
    assert row[0] == "correct"


# ── Notifications (Section 4) ─────────────────────────────────────────────────

def test_alert_payload_has_no_totals_or_token():
    text = notify.format_alert({
        "symbol": "AMZN", "name": "Amazon.com", "action": "sell",
        "amount_eur": 75.0, "fee_eur": 1.0,
        "message": "tactics fell from 71 to 38 in 7 days.",
    })
    assert "ACTION FOR TOMORROW" in text
    assert "Amazon.com" in text
    assert "75" in text
    lowered = text.lower()
    assert "portfolio" not in lowered
    assert "total" not in lowered
    assert "token" not in lowered


def test_send_alert_failure_is_silent():
    # channel none -> no send, no raise.
    assert notify.send_alert({"symbol": "X", "action": "hold", "message": "m"},
                             {"channel": "none"}) is False


def test_notify_status_line_off():
    assert "off" in notify.status_line({"channel": "none"})


# ── Scheduler (Section 3.1) ───────────────────────────────────────────────────

def test_service_unit_uses_absolute_paths():
    unit = scheduler.build_service_unit("/venv/bin/python", "/proj")
    assert "ExecStart=/venv/bin/python -m quant.cli daily" in unit
    assert "WorkingDirectory=/proj" in unit


def test_timer_unit_has_two_slots_and_persistent():
    unit = scheduler.build_timer_unit("18:45", "07:45")
    assert "OnCalendar=*-*-* 18:45:00" in unit
    assert "OnCalendar=*-*-* 07:45:00" in unit
    assert "Persistent=true" in unit


# ── Daily job (Section 3) ─────────────────────────────────────────────────────

def test_weekend_is_skipped():
    result = daily.run_daily(date(2026, 10, 3))  # a Saturday
    assert result.status == "skipped"
    assert "non-trading day" in result.message


def test_gap_line_exact_phrase():
    line = daily.gap_line(3)
    assert line == "Monitoring gap: no runs for 3 days; conditions evaluated on the latest data."
    assert daily.gap_line(1) is None


def test_staleness_status_with_fake_clock(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    assert daily.staleness_status(date(2026, 10, 1)) is not None
    daily.write_last_daily_run(date(2026, 9, 30))
    # Wednesday 1 Oct, last run Tuesday 30 Sep -> current.
    assert daily.staleness_status(date(2026, 10, 1)) is None
    # Monday 5 Oct, last run Tuesday 30 Sep -> stale.
    assert daily.staleness_status(date(2026, 10, 5)) is not None


def test_morning_slot_does_not_recompute(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    (tmp_path / "daily_2026-10-01.md").write_text("x", encoding="utf-8")
    result = daily.run_morning(date(2026, 10, 1))
    assert result.status == "ok"
    assert "open action" in result.message

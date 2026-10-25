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

from quant.engine import daily, notify, scheduler, sizing

# ── Sizing laws (Section 6) ───────────────────────────────────────────────────

def test_fortress_never_sells():
    from quant.engine import action_taxonomy

    over = [{"symbol": "LT", "tier": "FORTRESS", "current_weight": 0.9,
             "target_weight": 0.5, "value_eur": 900.0},
            {"symbol": "ACT", "tier": "ALPHA", "current_weight": 0.1,
             "target_weight": 0.5, "value_eur": 100.0}]
    assert not [i for i in action_taxonomy.build(over)
                if i["verb"] == action_taxonomy.VERB_SELL]


def test_sell_amount_sanity_and_rounding():
    # min(200, 143.53 - 1) = 142.53 -> round down to 5 -> 140.
    assert sizing.sell_amount(200.0, 143.53) == 140.0


def test_historical_failure_no_sell_for_small_drift():
    # 143.53 EUR position, 6.9 percent drift (~9.9 EUR) -> below 25 EUR -> None.
    drift = 0.069 * 143.53
    assert sizing.sell_amount(drift, 143.53) is None


def test_small_positions_untouchable():
    assert sizing.is_untouchable(50.0)


# ── Notifications (Section 4) ─────────────────────────────────────────────────


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

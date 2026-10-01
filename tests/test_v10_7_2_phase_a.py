"""
test_v10_7_2_phase_a.py — Real-world hardening: locks and busy-database grace.

Covers (v10.7.2, Part 1):
  - dead-pid takeover with the exact doctor phrase;
  - alive-pid below the stale threshold retries then aborts gracefully;
  - alive-pid above the hard cap takes over with a warning;
  - busy-database retry succeeds when the blocker releases mid-backoff;
  - a failed run writes the marker and the morning slot retries;
  - no traceback text in user-facing output.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from quant import paths
from quant.engine import daily, lock
from quant.ui import copy as C


def _write_lock(tmp_path, pid, age_min, command="other"):
    started = (datetime.now() - timedelta(minutes=age_min)).isoformat(timespec="seconds")
    (tmp_path / ".runner.lock").write_text(
        json.dumps({"pid": pid, "started_at": started, "command": command}),
        encoding="utf-8")


# ── Lock takeover ─────────────────────────────────────────────────────────────

def test_dead_pid_takeover_exact_doctor_phrase(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    _write_lock(tmp_path, 999999, 5, command="old")

    res = lock.acquire("new")
    assert res.acquired is True
    assert res.taken_over is True
    assert res.stale_pid == 999999

    line = lock.status_line()
    assert line == "runner lock: taken over from stale process (pid 999999, age 5 min)"

    # The doctor renders the same exact phrase.
    from quant.cli import _cmd_doctor

    _cmd_doctor(None)
    out = capsys.readouterr().out
    assert "runner lock: taken over from stale process (pid 999999, age 5 min)" in out


def test_alive_pid_below_threshold_retries_then_aborts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    _write_lock(tmp_path, 1, 0, command="other")  # pid 1 is always alive

    sleeps: list[float] = []
    res = lock.acquire("new", attempts=4, total_seconds=0.4,
                       sleep_fn=lambda s: sleeps.append(s))
    assert res.acquired is False
    assert "held by live process" in res.message
    assert len(sleeps) == 3  # attempts - 1 backoff waits


def test_alive_pid_above_hard_cap_takes_over(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    _write_lock(tmp_path, 1, 300, command="runaway")  # above HARD_LOCK_MINUTES

    res = lock.acquire("new")
    assert res.acquired is True
    assert res.taken_over is True
    assert "runaway" in res.message


def test_permission_error_counts_as_alive(monkeypatch):
    def _raise(_pid, _sig):
        raise PermissionError("not permitted")

    monkeypatch.setattr(lock.os, "kill", _raise)
    assert lock.pid_alive(1234) is True


# ── Busy-database retry ───────────────────────────────────────────────────────

def test_busy_database_retry_succeeds_when_blocker_releases(monkeypatch):
    from quant.data import database

    class FakeConn:
        def close(self):
            pass

    calls = {"n": 0}

    def flaky(timeout=2.0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Conflicting lock on database file")
        return FakeConn()

    monkeypatch.setattr(database, "_connect_with_timeout", flaky)
    monkeypatch.setattr(database.time, "sleep", lambda _s: None)

    with database.write_connection(attempts=5, total_seconds=0.1) as conn:
        assert isinstance(conn, FakeConn)
    assert calls["n"] == 3


def test_write_connection_raises_after_budget(monkeypatch):
    from quant.data import database

    def always_locked(timeout=2.0):
        raise RuntimeError("Conflicting lock")

    monkeypatch.setattr(database, "_connect_with_timeout", always_locked)
    monkeypatch.setattr(database.time, "sleep", lambda _s: None)

    import pytest

    with pytest.raises(RuntimeError):
        with database.write_connection(attempts=3, total_seconds=0.1):
            pass


# ── Graceful failure ──────────────────────────────────────────────────────────

def test_failed_run_writes_marker_and_morning_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    from quant.data import database

    def boom():
        raise RuntimeError("Conflicting lock")

    monkeypatch.setattr(database, "get_connection", boom)

    res = daily.run_daily(date(2026, 10, 1))  # a Thursday
    assert res.status == "error"
    assert res.message == C.DB_BUSY_RETRY

    failed = daily.read_last_failed_run()
    assert failed is not None
    assert failed["reason"] == C.DB_BUSY_RETRY
    assert failed.get("at")

    # The morning slot retries (no artifact was written by the failed run).
    called = {"yes": False}

    def fake_run_daily(*_a, **_k):
        called["yes"] = True
        return daily.DailyResult("ok", "retried")

    monkeypatch.setattr(daily, "run_daily", fake_run_daily)
    res2 = daily.run_morning(date(2026, 10, 1))
    assert called["yes"] is True
    assert res2.status == "ok"


def test_no_traceback_in_user_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    from quant.cli import _cmd_daily

    monkeypatch.setattr(lock, "acquire",
                        lambda *a, **k: lock.LockResult(False, "held by live process"))

    rc = _cmd_daily(None)
    out = capsys.readouterr().out
    assert rc == 1
    assert C.DB_BUSY_RETRY in out
    assert "Traceback" not in out

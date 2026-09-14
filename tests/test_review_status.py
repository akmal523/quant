"""
test_review_status.py — review_status single source + doctor probes + heartbeat (H3.3).
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.ui.render as render  # noqa: E402
from quant import paths  # noqa: E402
from quant.ui import copy as C  # noqa: E402
from quant.ui import runner  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader", "success")


def _all_text(at: AppTest) -> str:
    return " ".join(str(getattr(el, "value", ""))
                    for attr in _TEXTLIKE for el in getattr(at, attr, []))


def _page(monkeypatch, page, review, history):
    monkeypatch.setattr(render, "latest_review", lambda ok_only=False: review)
    monkeypatch.setattr(render, "read_history", lambda: history)
    monkeypatch.setattr(render, "read_regime", lambda: {"state": "insufficient_history"})
    monkeypatch.setattr(render, "read_actions", lambda: [])
    monkeypatch.setattr(render, "load_portfolio",
                        lambda: pd.DataFrame([{"Symbol": "AMZN", "Amount_EUR": 10.0}]))
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(page).run()
    return at


def test_failed_review_shows_s4_and_health(monkeypatch):
    review = {"review_status": "failed", "error": "scoring exploded"}
    today = _all_text(_page(monkeypatch, "pages/today.py", review, pd.DataFrame()))
    assert C.LAST_REVIEW_FAILED in today
    settings = _all_text(_page(monkeypatch, "pages/settings.py", review, pd.DataFrame()))
    assert "scoring exploded" in settings


def test_failed_without_error_uses_generic(monkeypatch):
    review = {"review_status": "failed"}
    settings = _all_text(_page(monkeypatch, "pages/settings.py", review, pd.DataFrame()))
    assert C.HEALTH_REVIEW_FAILED in settings


def test_success_clears_failure(monkeypatch):
    hist = pd.DataFrame([{"review_ts": "2026-09-13", "value_eur": 100.0, "pnl_eur": 1.0}])
    review = {"review_status": "ok", "review_ts": "2026-09-13"}
    today = _all_text(_page(monkeypatch, "pages/today.py", review, hist))
    assert C.LAST_REVIEW_FAILED not in today
    settings = _all_text(_page(monkeypatch, "pages/settings.py", review, hist))
    assert C.HEALTH_REVIEW_FAILED not in settings


def test_doctor_has_probes(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    from quant.cli import _cmd_doctor

    _cmd_doctor(None)
    out = capsys.readouterr().out
    assert "search apple:" in out
    assert "search samsung:" in out
    assert "probe AAPL history:" in out
    assert "sentiment model:" in out


def test_heartbeat_refreshes_during_run(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(render, "_HEARTBEAT_INTERVAL", 0.05)
    monkeypatch.setattr(render, "read_actions", lambda: [])

    def slow(_cmd):
        time.sleep(0.25)
        return runner.RunResult("ok", "", None, 0, 0.25)

    monkeypatch.setattr(render.runner, "run", slow)
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/portfolio.py").run()
    at.session_state["_running"] = True
    at.run()
    # The patched run() never writes the heartbeat; only the refresh thread does.
    lock = tmp_path / ".runner.lock"
    assert lock.exists()

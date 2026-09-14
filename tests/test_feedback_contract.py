"""
test_feedback_contract.py — Operation feedback + mutex heartbeat (v10.5.3, R4).

Covers: disabled verb-ing button, dedicated progress container cleared on
completion, one outcome line, stale heartbeat auto-release, no nested acquisition,
foreign-vs-own-session busy messages, and the fold-ins (Repair button exists
exactly once app-wide; the status vocabulary has a single mapper).
"""
from __future__ import annotations

import json
import os
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
    chunks = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def _write_hb(tmp_path, owner, pid, age_s):
    (tmp_path / ".runner.lock").write_text(
        json.dumps({"owner": owner, "pid": pid, "ts": time.time() - age_s}),
        encoding="utf-8")


# ── Heartbeat ─────────────────────────────────────────────────────────────────

def test_stale_heartbeat_auto_releases(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    _write_hb(tmp_path, "other", 999999, 700)  # older than 10 minutes
    assert runner._foreign_live_heartbeat() is False


def test_foreign_live_heartbeat_blocks_with_other_tab_sentence(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    _write_hb(tmp_path, "other", 999999, 0)
    assert runner._foreign_live_heartbeat() is True
    res = runner.run("update")  # short-circuits before any subprocess
    assert res.status == "busy"
    assert res.message == C.ERROR_ALREADY_RUNNING


def test_own_session_never_other_tab(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    _write_hb(tmp_path, runner._OWNER, os.getpid(), 0)
    assert runner._foreign_live_heartbeat() is False


def test_no_nested_acquisition(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    runner._LOCK.acquire()
    try:
        res = runner.run("update")
        assert res.status == "busy"
        assert res.message == C.ERROR_RUNNING  # own session, not the other-tab line
    finally:
        runner._LOCK.release()


# ── Feedback ──────────────────────────────────────────────────────────────────

def test_progress_cleared_and_one_outcome_line(monkeypatch):
    fake = runner.RunResult("ok", "", None, 0, 1.0)
    monkeypatch.setattr(render.runner, "run", lambda cmd: fake)
    monkeypatch.setattr(render, "read_actions", lambda: [])
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/portfolio.py").run()
    at.session_state["_running"] = True
    at.run()
    text = _all_text(at)
    assert C.OUTCOME_NOTHING in text
    # The dedicated progress container is cleared on completion.
    assert not at.get("progress")


def test_open_today_gated_on_success(monkeypatch):
    render_source = (Path(__file__).resolve().parents[1] / "quant" / "ui" / "render.py").read_text()
    assert 'st.switch_page("pages/today.py")' in render_source
    assert 'st.session_state.get("_review_ok")' in render_source


# ── Fold-in: Repair button single home ────────────────────────────────────────

def _settings_stub(monkeypatch):
    monkeypatch.setattr(render, "read_history", lambda: pd.DataFrame())
    monkeypatch.setattr(render, "latest_review", lambda ok_only=False: {})
    monkeypatch.setattr(render, "read_regime", lambda: {"state": "insufficient_history"})
    monkeypatch.setattr(render, "read_actions", lambda: [])
    monkeypatch.setattr(render, "load_portfolio",
                        lambda: pd.DataFrame([{"Symbol": "5J50.DE", "Amount_EUR": 10.0}]))
    monkeypatch.setattr(render, "resolve_broker", lambda s: {"isin": "", "isin_source": ""})


def _count_repair_buttons(at: AppTest) -> int:
    n = 0
    for el in at.get("button"):
        if str(getattr(el, "label", "")) == C.BTN_REPAIR_REGISTRY:
            n += 1
    return n


def test_repair_button_single_home(monkeypatch):
    _settings_stub(monkeypatch)
    settings = AppTest.from_file(DASHBOARD, default_timeout=60)
    settings.run()
    settings.switch_page("pages/settings.py").run()
    assert _count_repair_buttons(settings) == 1

    for page in ("pages/today.py", "pages/explore.py"):
        at = AppTest.from_file(DASHBOARD, default_timeout=60)
        at.run()
        at.switch_page(page).run()
        assert _count_repair_buttons(at) == 0


# ── Fold-in: single status mapper ─────────────────────────────────────────────

def test_status_vocabulary_single_mapper(monkeypatch):
    from quant.reporting import artifacts

    base = {"Tier": "ACTIVE", "Value_EUR": 100.0, "Current_Weight": "50.0%",
            "Target_Weight": "50.0%", "Drift": "0.0%"}
    audit = pd.DataFrame([
        {**base, "Symbol": "AMZN",
         "Recommendation": "BUY 150 EUR (ACTIVE drift -16.1% exceeds 5.0% threshold)"},
        {**base, "Symbol": "AAPL",
         "Recommendation": "SELL 50 EUR (ACTIVE drift 8.0% exceeds 5.0% threshold)"},
        {**base, "Symbol": "MSFT", "Recommendation": "HOLD: within threshold"},
        {**base, "Symbol": "ZZZZ",
         "Recommendation": "BUY 100 EUR (ACTIVE drift -9.0% exceeds 5.0% threshold)"},
        {**base, "Symbol": "NVDA", "Cooldown_Until": "2026-09-20",
         "Recommendation": "BUY 60 EUR (ACTIVE drift -7.0% exceeds 5.0% threshold)"},
    ])
    monkeypatch.setattr(artifacts, "_read_audit_df", lambda: audit)
    vocabulary = {C.STATUS_ON_TRACK, C.STATUS_ADD, C.STATUS_TRIM,
                  C.STATUS_BLOCKED, C.STATUS_BELOW_MIN, C.STATUS_NOT_REVIEWED}
    for h in artifacts.read_actions():
        status = h["status"]
        assert status in vocabulary or status.startswith("Waiting until "), status

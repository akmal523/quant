"""
test_today_states.py — Today S0-S4 states + confidence contract (v10.5.3, R3).

Renders Today with injected artifacts (render-module seams) and asserts the exact
catalog copy for each state, the suppression footnote with the live config value,
and the regime-confidence boundaries.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.ui.render as render  # noqa: E402
from quant.analytics.scoring import regime_confidence  # noqa: E402
from quant.ui import copy as C  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    chunks = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def _today(monkeypatch, *, history, review, regime, actions, portfolio) -> AppTest:
    monkeypatch.setattr(render, "read_history", lambda: history)
    monkeypatch.setattr(render, "latest_review", lambda ok_only=False: review)
    monkeypatch.setattr(render, "read_regime", lambda: regime)
    monkeypatch.setattr(render, "read_actions", lambda: actions)
    monkeypatch.setattr(render, "load_portfolio", lambda: portfolio)
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/today.py").run()
    return at


_EMPTY = pd.DataFrame()
_PORT = pd.DataFrame([{"Symbol": "AMZN", "Amount_EUR": 100.0}])
_REGIME_EST = {"state": "estimated", "label": "rising", "confidence": "high",
               "prob": 0.8, "as_of": "2026-09-13", "error": None}


def test_s0_no_review(monkeypatch):
    at = _today(monkeypatch, history=_EMPTY, review={}, regime=dict(_REGIME_EST),
                actions=[], portfolio=_PORT)
    text = _all_text(at)
    assert C.GUIDE_NO_REVIEW in text
    assert "Market trend:" not in text
    # The holdings table renders every row "Not reviewed yet" (dataframe, not text).
    statuses = set()
    for el in getattr(at, "dataframe", []):
        frame = getattr(el, "value", None)
        if frame is not None and "Status" in frame:
            statuses.update(frame["Status"].astype(str))
    assert C.STATUS_NOT_REVIEWED in statuses


def test_s1_review_no_actions(monkeypatch):
    holdings = [{"symbol": "AMZN", "action": None, "blocked": False,
                 "status": C.STATUS_ON_TRACK, "value_eur": 100.0,
                 "current_weight": "100%", "target_weight": "100%", "drift": "0.0%",
                 "min_trade_eur": 50.0, "cooldown_until": None, "suppressed": None}]
    text = _all_text(_today(monkeypatch, history=_EMPTY,
                  review={"review_ts": "2026-09-13", "latest_bar": "2026-09-11"},
                  regime=dict(_REGIME_EST), actions=holdings, portfolio=_PORT))
    assert C.MARKET_TREND.format(label="rising", confidence="high") in text
    assert C.EMPTY_NOTHING_TO_DO in text


def test_s2_action_card(monkeypatch):
    holdings = [{"symbol": "AMZN", "action": "BUY MORE", "amount_eur": 150.0,
                 "blocked": False, "status": C.STATUS_ADD, "value_eur": 100.0,
                 "current_weight": "10.0%", "target_weight": "20.0%", "drift": "-10.0%",
                 "min_trade_eur": 50.0, "cooldown_until": None, "suppressed": None}]
    text = _all_text(_today(monkeypatch, history=_EMPTY,
                  review={"review_ts": "2026-09-13", "latest_bar": "2026-09-11"},
                  regime=dict(_REGIME_EST), actions=holdings, portfolio=_PORT))
    assert "Add about 150 EUR to" in text


def test_s3_below_min_footnote_uses_config_value(monkeypatch):
    holdings = [{"symbol": "AMZN", "action": None, "blocked": False,
                 "status": C.STATUS_BELOW_MIN, "value_eur": 100.0,
                 "current_weight": "20.0%", "target_weight": "0.0%", "drift": "20.0%",
                 "min_trade_eur": 50.0, "cooldown_until": None, "suppressed": "below_min"}]
    text = _all_text(_today(monkeypatch, history=_EMPTY,
                  review={"review_ts": "2026-09-13", "latest_bar": "2026-09-11"},
                  regime=dict(_REGIME_EST), actions=holdings, portfolio=_PORT))
    assert "below the 50 EUR minimum order size" in text


def test_s4_review_failed_card(monkeypatch):
    hist = pd.DataFrame([{"review_ts": "2026-09-12", "value_eur": 100.0}])
    text = _all_text(_today(monkeypatch, history=hist,
                            review={"review_status": "failed"}, regime=dict(_REGIME_EST),
                            actions=[], portfolio=_PORT))
    assert C.LAST_REVIEW_FAILED in text


def test_confidence_boundaries():
    assert regime_confidence(0.80) == "high"
    assert regime_confidence(0.65) == "medium"
    assert regime_confidence(0.60) == "low"

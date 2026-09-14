"""
test_chart_rules.py — Value chart contract (v10.5.3, R6; spec 3.1).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.ui.render as render  # noqa: E402
from quant.ui import copy as C  # noqa: E402
from quant.ui.render import _filter_range, _range_annotation, _rebase  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    return " ".join(str(getattr(el, "value", ""))
                    for attr in _TEXTLIKE for el in getattr(at, attr, []))


def _today(monkeypatch, hist) -> AppTest:
    monkeypatch.setattr(render, "read_history", lambda: hist)
    monkeypatch.setattr(render, "latest_review",
                        lambda ok_only=False: {"review_ts": "2026-09-11"})
    monkeypatch.setattr(render, "read_regime",
                        lambda: {"state": "estimated", "label": "rising", "confidence": "high"})
    monkeypatch.setattr(render, "read_actions", lambda: [])
    monkeypatch.setattr(render, "load_portfolio",
                        lambda: pd.DataFrame([{"Symbol": "AMZN", "Amount_EUR": 10.0}]))
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/today.py").run()
    return at


def test_two_points_shows_sentence(monkeypatch):
    hist = pd.DataFrame({"review_ts": ["2026-09-01", "2026-09-11"],
                         "value_eur": [100.0, 101.0]})
    assert C.CHART_BUILDING in _all_text(_today(monkeypatch, hist))


def test_three_points_renders_chart(monkeypatch):
    hist = pd.DataFrame({"review_ts": ["2026-08-01", "2026-09-01", "2026-09-11"],
                         "value_eur": [100.0, 103.0, 104.2]})
    at = _today(monkeypatch, hist)
    assert not at.exception
    assert C.CHART_BUILDING not in _all_text(at)


def test_filter_range_endpoints():
    df = pd.DataFrame({"review_ts": pd.to_datetime(
        ["2026-01-01", "2026-06-01", "2026-09-01"]), "value_eur": [100, 110, 120]})
    got = _filter_range(df, "3M")
    assert list(got["value_eur"]) == [110, 120]
    assert len(_filter_range(df, "Max")) == 3


def test_growth_rebase_first_value_is_100():
    s = pd.Series([50.0, 60.0, 75.0])
    rebased = _rebase(s)
    assert rebased.iloc[0] == 100.0
    assert rebased.iloc[-1] == 150.0


def test_annotation_format():
    df = pd.DataFrame({"review_ts": pd.to_datetime(["2026-06-01", "2026-09-01"]),
                       "value_eur": [100.0, 104.2]})
    assert _range_annotation(df) == "+4.2% since 1 Jun 2026 (4.20 EUR)"

"""
test_explore_states.py — Explore states (v10.5.3, R3).

Asserts scores absent/present with glossary visibility, the scores-none sentence,
and the missing-ISIN how-to-buy sentence with no repair button on the page.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.ui.render as render  # noqa: E402
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


def _explore(monkeypatch, scores, broker) -> str:
    monkeypatch.setattr(render, "load_index", lambda: [])
    monkeypatch.setattr(render, "search", lambda idx, q, n=10: [
        {"symbol": "AMZN", "label": "Amazon (AMZN)", "name": "Amazon", "isin": ""}])
    monkeypatch.setattr(render, "latest_review",
                        lambda ok_only=False: {"review_ts": "2026-09-13"})
    monkeypatch.setattr(render, "read_scores", lambda sym: scores)
    monkeypatch.setattr(render, "resolve_broker", lambda sym: broker)
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/explore.py").run()
    at.text_input[0].set_value("amazon").run()
    return _all_text(at)


def test_scores_absent_hides_glossary(monkeypatch):
    text = _explore(monkeypatch,
                    scores={"structural_grade": None, "tactical_grade": None,
                            "active_score": None},
                    broker={"isin": "US0231351067", "currency": "USD",
                            "isin_source": "user", "instrument_class": "EQUITY"})
    assert C.SCORES_NONE.format(name="AMZN") in text
    assert "how sound the asset is fundamentally" not in text


def test_scores_present_shows_glossary(monkeypatch):
    text = _explore(monkeypatch,
                    scores={"structural_grade": 70.0, "tactical_grade": 60.0,
                            "active_score": 67.0},
                    broker={"isin": "US0231351067", "currency": "USD",
                            "isin_source": "user", "instrument_class": "EQUITY"})
    assert f"Overall score: {C.fmt_score(67.0)}" in text
    assert "how sound the asset is fundamentally" in text


def test_missing_isin_sentence_no_button(monkeypatch):
    text = _explore(monkeypatch,
                    scores={"structural_grade": 70.0, "tactical_grade": 60.0,
                            "active_score": 67.0},
                    broker={"isin": "", "currency": "USD", "isin_source": "",
                            "instrument_class": "EQUITY"})
    assert C.HOW_TO_BUY_ISIN_MISSING.format(name="AMZN") in text
    assert C.BTN_REPAIR_REGISTRY not in text

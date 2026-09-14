"""
test_news_format.py — Explore news display (H3.4).

Weekday dates (no raw timezone), cap at 5 visible rows + an "Earlier items"
expander, and sentiment honesty when the model is absent.
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
    for el in getattr(at, "expander", []):
        chunks.append(str(getattr(el, "label", "")))
    return " ".join(chunks)


def _items(n: int):
    return [{"source": "Yahoo Finance", "headline": f"H{i}",
             "published_at": "2026-09-13T10:00:00+00:00", "score": 0.0}
            for i in range(n)]


def _explore_with_news(monkeypatch, items, sentiment_available):
    monkeypatch.setattr(render, "load_index", lambda: [])
    monkeypatch.setattr(render, "search", lambda idx, q, n=10: [
        {"symbol": "AMZN", "label": "Amazon (AMZN)", "name": "Amazon", "isin": ""}])
    monkeypatch.setattr(render, "load_news", lambda sym: items)
    monkeypatch.setattr(render, "_sentiment_available", lambda: sentiment_available)
    monkeypatch.setattr(render, "latest_review", lambda: {"review_ts": "2026-09-13"})
    monkeypatch.setattr(render, "read_scores",
                        lambda sym: {"structural_grade": None, "tactical_grade": None,
                                     "active_score": None})
    monkeypatch.setattr(render, "resolve_broker",
                        lambda sym: {"isin": "US0231351067", "currency": "USD",
                                     "isin_source": "user", "instrument_class": "EQUITY"})
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/explore.py").run()
    at.text_input[0].set_value("amazon").run()
    return at


def test_formatter_weekday_unit():
    assert C.fmt_weekday_date("2026-09-13") == "Sunday 13 Sep 2026"


def test_weekday_date_rendered_no_raw_timezone(monkeypatch):
    at = _explore_with_news(monkeypatch, _items(2), sentiment_available=True)
    text = _all_text(at)
    assert "Sunday 13 Sep 2026" in text
    assert "+0000" not in text and "+00:00" not in text


def test_cap_five_with_earlier_expander(monkeypatch):
    at = _explore_with_news(monkeypatch, _items(7), sentiment_available=True)
    text = _all_text(at)
    assert C.NEWS_EARLIER.format(n=2) in text      # 7 - 5 = 2 earlier items
    assert "H0" in text                            # first visible row


def test_sentiment_honesty_line_when_model_absent(monkeypatch):
    at = _explore_with_news(monkeypatch, _items(1), sentiment_available=False)
    text = _all_text(at)
    assert C.SENTIMENT_UNAVAILABLE in text
    assert "neutral" not in text


def test_sentiment_word_present_when_available(monkeypatch):
    at = _explore_with_news(monkeypatch, _items(1), sentiment_available=True)
    text = _all_text(at)
    assert "neutral" in text
    assert C.SENTIMENT_UNAVAILABLE not in text

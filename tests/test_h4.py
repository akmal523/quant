"""
test_h4.py — sentiment scoring write path (H4).

Symptom: the model was available yet every cache entry carried `scorer: default`
and no row ever showed a sentiment word. Root cause: `fetch_news_items` hardcoded
`default` and never invoked the scorer; `load_news` is the only cache writer.

Contract (recorded-source, boundary stub per D1):
  - model available -> fresh entries carry `scorer: "model"` + a real score and
    render the word.
  - model absent    -> entries stay `scorer: "default"`, score 0.0, no word.
  - scoring timeout -> entries stay `default`, no word, exactly one log line.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

import quant.data.news as newsmod  # noqa: E402

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.ui.render as render  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


class _Feed:
    """Live-shaped feed boundary (Yahoo RSS shape: title + published)."""

    def __init__(self, titles):
        self.entries = [{"title": t, "published": "Mon, 14 Sep 2026 14:41:24 +0000"}
                        for t in titles]


class _ModelStub:
    """Model boundary: one non-neutral headline -> a positive score."""

    def score_texts(self, texts):
        return [0.42 for _ in texts]


class _SlowStub:
    def score_texts(self, texts):
        time.sleep(5.0)
        return [0.0 for _ in texts]


def _fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(newsmod.paths, "OUTPUTS_DIR", tmp_path)
    return tmp_path


# ── write path: model available ──────────────────────────────────────────────

def test_model_scorer_writes_scorer_model(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    items = newsmod.load_news("AAPL", fetcher=lambda s, t=10: _Feed(["Big beat"]),
                              scorer=_ModelStub())
    assert items[0]["scorer"] == "model"
    assert items[0]["score"] == pytest.approx(0.42)
    st = newsmod.cache_scorer_stats()
    assert st["model"] == 1 and st["default"] == 0 and st["pos"] == 1


def test_absent_model_writes_scorer_default(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    items = newsmod.load_news("AAPL", fetcher=lambda s, t=10: _Feed(["H"]),
                              scorer_factory=lambda: None)
    assert items[0]["scorer"] == "default"
    assert items[0]["score"] == 0.0
    assert newsmod.cache_scorer_stats()["default"] == 1


def test_scoring_timeout_keeps_default_and_logs(tmp_path, monkeypatch, caplog):
    _fresh(tmp_path, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="quant.data.news"):
        items = newsmod.load_news("AAPL", fetcher=lambda s, t=10: _Feed(["H"]),
                                  scorer=_SlowStub(), score_timeout=0.01)
    assert items[0]["scorer"] == "default"
    assert items[0]["score"] == 0.0
    assert sum("timed out" in r.message for r in caplog.records) == 1


def test_cache_hit_does_not_rescore(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    now = time.time()
    newsmod.load_news("AAPL", fetcher=lambda s, t=10: _Feed(["H"]),
                      scorer=_ModelStub(), now=now)
    # A cache hit returns the stored entry without touching the scorer/fetcher.
    hit = newsmod.load_news("AAPL", fetcher=lambda s, t=10: _Feed(["SHOULD NOT"]),
                            scorer=_ModelStub(), now=now + 60)
    assert hit[0]["headline"] == "H"


# ── render: only a model score shows the word ────────────────────────────────

def _explore(monkeypatch, items):
    monkeypatch.setattr(render, "load_index", lambda: [])
    monkeypatch.setattr(render, "search", lambda idx, q, n=10: [
        {"symbol": "AMZN", "label": "Amazon (AMZN)", "name": "Amazon", "isin": ""}])
    monkeypatch.setattr(render, "latest_review",
                        lambda ok_only=False: {"review_ts": "2026-09-13"})
    monkeypatch.setattr(render, "read_scores",
                        lambda s: {"structural_grade": None, "tactical_grade": None,
                                   "active_score": None})
    monkeypatch.setattr(render, "load_news", lambda s: items)
    monkeypatch.setattr(render, "_sentiment_available", lambda: True)
    monkeypatch.setattr(render, "resolve_broker",
                        lambda s: {"isin": "US0231351067", "currency": "USD",
                                   "isin_source": "user", "instrument_class": "EQUITY"})
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/explore.py").run()
    at.text_input[0].set_value("amazon").run()
    chunks = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def _item(scorer, score):
    return {"source": "Yahoo Finance", "headline": "H",
            "published_at": "2026-09-13T10:00:00+00:00",
            "score": score, "scorer": scorer}


def test_model_entry_shows_word(monkeypatch):
    assert "positive" in _explore(monkeypatch, [_item("model", 0.42)])


def test_default_entry_hides_word(monkeypatch):
    assert "positive" not in _explore(monkeypatch, [_item("default", 0.0)])

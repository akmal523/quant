"""
test_h3_6.py — Failed-review shadowing, news dates, sentiment provenance (H3.6).

N1: data reads use the most recent SUCCESSFUL review; a failed latest attempt
    adds one freshness line and never shadows scores.
N2: the news date formatter parses RFC-2822 (the live cache shape) and ISO-8601.
N3: each news cache entry carries `scorer`; the word renders only for `model`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.data.news as newsmod  # noqa: E402
import quant.ui.render as render  # noqa: E402
from quant.reporting import artifacts  # noqa: E402
from quant.ui.copy import SCORES_AS_OF, fmt_weekday_date  # noqa: E402

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


# ── N1: failed review must not shadow data reads ──────────────────────────────

def test_failed_latest_does_not_shadow_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUTPUTS_DIR", str(tmp_path))
    ok = tmp_path / "run_2026-09-13_232300"
    ok.mkdir()
    (ok / "metrics.json").write_text(
        '{"review_status":"ok","review_ts":"2026-09-13T23:23:00"}', encoding="utf-8")
    pd.DataFrame([{"Symbol": "AMZN", "Structural_Grade": 70.0,
                   "Tactical_Grade": 60.0, "Active_Score": 67.0}]).to_parquet(
        ok / "scores.parquet")
    bad = tmp_path / "run_2026-09-13_232400"
    bad.mkdir()
    (bad / "metrics.json").write_text(
        '{"review_status":"failed","error":"boom"}', encoding="utf-8")

    assert artifacts.latest_review().get("review_status") == "failed"
    assert artifacts.latest_review(ok_only=True).get("review_status") == "ok"
    assert artifacts.read_scores("AMZN")["active_score"] == 67.0


def _explore(monkeypatch, latest_review_fn, scores, news):
    monkeypatch.setattr(render, "load_index", lambda: [])
    monkeypatch.setattr(render, "search", lambda idx, q, n=10: [
        {"symbol": "AMZN", "label": "Amazon (AMZN)", "name": "Amazon", "isin": ""}])
    monkeypatch.setattr(render, "latest_review", latest_review_fn)
    monkeypatch.setattr(render, "read_scores", lambda s: scores)
    monkeypatch.setattr(render, "load_news", lambda s: news)
    monkeypatch.setattr(render, "resolve_broker",
                        lambda s: {"isin": "US0231351067", "currency": "USD",
                                   "isin_source": "user", "instrument_class": "EQUITY"})
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/explore.py").run()
    at.text_input[0].set_value("amazon").run()
    return _all_text(at)


def test_explore_shows_scores_as_of_when_latest_failed(monkeypatch):
    def _lr(ok_only=False):
        return ({"review_status": "ok", "review_ts": "2026-09-13T23:23:00"}
                if ok_only else {"review_status": "failed"})

    text = _explore(monkeypatch, _lr,
                    {"structural_grade": 70.0, "tactical_grade": 60.0, "active_score": 67.0},
                    [])
    assert "Overall score: 67 / 100" in text
    assert SCORES_AS_OF.format(date="Sunday 13 Sep 2026, 23:23") in text


def test_explore_shows_catalogue_as_of_when_latest_ok(monkeypatch):
    # H3.8 (M9): the catalogue as-of line is always shown, one variant only.
    def _lr(ok_only=False):
        return {"review_status": "ok", "review_ts": "2026-09-13T23:23:00"}

    text = _explore(monkeypatch, _lr,
                    {"structural_grade": 70.0, "tactical_grade": 60.0, "active_score": 67.0},
                    [])
    assert "Overall score: 67 / 100" in text
    assert SCORES_AS_OF.format(date="Sunday 13 Sep 2026, 23:23") in text


# ── N2: news date parsing ────────────────────────────────────────────────────

def test_news_date_parses_rfc2822_and_iso():
    assert fmt_weekday_date("Mon, 14 Sep 2026 14:41:24 +0000") == "Monday 14 Sep 2026"
    assert fmt_weekday_date("2026-09-13T10:00:00+00:00") == "Sunday 13 Sep 2026"
    assert fmt_weekday_date("not a date") == ""


def test_news_row_renders_weekday_for_live_shaped_entry(monkeypatch):
    items = [{"source": "Yahoo Finance", "headline": "H",
              "published_at": "Mon, 14 Sep 2026 14:41:24 +0000",
              "score": 0.0, "scorer": "default"}]
    text = _explore(monkeypatch, lambda ok_only=False: {}, {}, items)
    assert "Monday 14 Sep 2026" in text
    assert "+0000" not in text


# ── N3: sentiment provenance ─────────────────────────────────────────────────

class _Feed:
    def __init__(self, titles):
        self.entries = [{"title": t, "published": "Mon, 14 Sep 2026 14:41:24 +0000"}
                        for t in titles]


def test_cache_entries_are_scorer_default(tmp_path, monkeypatch):
    # H4: inject a model-absent boundary so the default path is exercised
    # without loading FinBERT. Model-present behaviour lives in test_h4.py.
    monkeypatch.setattr(newsmod.paths, "OUTPUTS_DIR", tmp_path)
    items = newsmod.load_news("AAPL", fetcher=lambda s, t=10: _Feed(["H"]),
                              scorer_factory=lambda: None)
    assert items[0]["scorer"] == "default"
    st = newsmod.cache_scorer_stats()
    assert st["entries"] == 1 and st["default"] == 1 and st["model"] == 0


def test_default_scorer_drops_sentiment_word(monkeypatch):
    items = [{"source": "Yahoo Finance", "headline": "H",
              "published_at": "2026-09-13T10:00:00+00:00",
              "score": 0.0, "scorer": "default"}]
    text = _explore(monkeypatch, lambda ok_only=False: {}, {}, items)
    assert "neutral" not in text


def test_model_scorer_keeps_sentiment_word(monkeypatch):
    items = [{"source": "Yahoo Finance", "headline": "H",
              "published_at": "2026-09-13T10:00:00+00:00",
              "score": 0.0, "scorer": "model"}]
    text = _explore(monkeypatch, lambda ok_only=False: {}, {}, items)
    assert "neutral" in text

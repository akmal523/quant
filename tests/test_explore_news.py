"""
test_explore_news.py — On-demand news cache contract (v10.5.3, R6; spec 6.2/6.4).
"""
from __future__ import annotations

from quant.data import news as newsmod
from quant.ui import copy as C


class _Feed:
    def __init__(self, titles):
        self.entries = [{"title": t, "published": "2026-09-10"} for t in titles]


def test_cache_hit_does_not_refetch(tmp_path, monkeypatch):
    monkeypatch.setattr(newsmod.paths, "OUTPUTS_DIR", tmp_path)
    calls = {"n": 0}

    def fetcher(_sym, _t=10):
        calls["n"] += 1
        return _Feed(["H1"])

    first = newsmod.load_news("AAPL", fetcher=fetcher)
    second = newsmod.load_news("AAPL", fetcher=fetcher)
    assert first and first == second
    assert calls["n"] == 1


def test_fetch_failure_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(newsmod.paths, "OUTPUTS_DIR", tmp_path)

    def bad(_sym, _t=10):
        raise RuntimeError("boom")

    assert newsmod.load_news("AAPL", fetcher=bad) == []


def test_expired_cache_refetches(tmp_path, monkeypatch):
    monkeypatch.setattr(newsmod.paths, "OUTPUTS_DIR", tmp_path)
    calls = {"n": 0}

    def fetcher(_sym, _t=10):
        calls["n"] += 1
        return _Feed(["H"])

    newsmod.load_news("AAPL", fetcher=fetcher, now=0)
    newsmod.load_news("AAPL", fetcher=fetcher, now=100000)  # > 24 h later
    assert calls["n"] == 2


def test_list_row_uses_weekday_date(tmp_path, monkeypatch):
    monkeypatch.setattr(newsmod.paths, "OUTPUTS_DIR", tmp_path)

    def fetcher(_sym, _t=10):
        return _Feed(["Headline"])

    items = newsmod.load_news("AAPL", fetcher=fetcher)
    when = C.fmt_weekday_date(items[0]["published_at"])
    assert when == "Thursday 10 Sep 2026"


def test_atomic_write_crash_keeps_original(tmp_path, monkeypatch):
    monkeypatch.setattr(newsmod.paths, "OUTPUTS_DIR", tmp_path)
    newsmod._write_cache({"A": {"retrieved_at": 1, "items": ["x"]}})
    before = (tmp_path / "news_cache.json").read_text(encoding="utf-8")

    def _crash(*_a, **_k):
        raise RuntimeError("crash mid-write")

    monkeypatch.setattr(newsmod.json, "dump", _crash)
    newsmod._write_cache({"B": {"retrieved_at": 2, "items": []}})
    assert (tmp_path / "news_cache.json").read_text(encoding="utf-8") == before
    assert not list(tmp_path.glob("*.tmp"))


def test_outage_counter_three_failures_then_clear(tmp_path, monkeypatch):
    monkeypatch.setattr(newsmod.paths, "OUTPUTS_DIR", tmp_path)
    assert newsmod.outage_message() is None
    newsmod.record_fetch_result(False, now=1000.0)
    newsmod.record_fetch_result(False, now=1001.0)
    assert newsmod.outage_message() is None
    newsmod.record_fetch_result(False, now=1002.0)
    msg = newsmod.outage_message()
    assert msg and msg.startswith("News source unreachable since")
    newsmod.record_fetch_result(True)
    assert newsmod.outage_message() is None


def test_review_and_explore_share_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(newsmod.paths, "OUTPUTS_DIR", tmp_path)

    def fetcher(_sym, _t=10):
        return _Feed(["A", "B"])

    explore = newsmod.load_news("AAPL", fetcher=fetcher)
    review = newsmod.load_news("AAPL")  # cache hit; must not refetch
    assert review == explore

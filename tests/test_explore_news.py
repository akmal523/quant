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

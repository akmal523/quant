"""news.py — News fetch + on-demand cache (v10.5.3, R6/R7).

Intent: structured news headlines for Explore (on-demand, 24 h cache) and the
review's sentiment. The cache is a JSON file under outputs/, NOT the DuckDB store,
so the UI stays read-only on the database.

Invariants:
  - load_news returns [] on any fetch failure (renders the none-sentence; no Health
    item per symbol).
  - A cache hit within ttl_hours does not fetch.
  - fetch is injectable for hermetic tests.

Dependencies: feedparser, urllib, quant.paths.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import urllib.request

import feedparser

from quant import paths

logger = logging.getLogger(__name__)


def _cache_path() -> str:
    return os.path.join(str(paths.OUTPUTS_DIR), "news_cache.json")


def _read_cache() -> dict:
    try:
        with open(_cache_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write via temp file + rename so a crash mid-write cannot corrupt the file."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:  # noqa: BLE001
                pass
        raise


def _write_cache(cache: dict) -> None:
    try:
        _atomic_write_json(_cache_path(), cache)
    except Exception:  # noqa: BLE001
        pass


def _default_fetcher(symbol: str, timeout: int = 10):
    url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return feedparser.parse(resp.read())


def fetch_news_items(symbol: str, fetcher=None, timeout: int = 10) -> list[dict] | None:
    """Return up to 10 structured items, or None on fetch failure."""
    fetcher = fetcher or _default_fetcher
    try:
        feed = fetcher(symbol, timeout)
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] news fetch failed: %s", symbol, e)
        return None
    items: list[dict] = []
    for entry in getattr(feed, "entries", [])[:10]:
        items.append({
            "source": "Yahoo Finance",
            "headline": str(entry.get("title", "")),
            "published_at": str(entry.get("published", "")),
            "score": 0.0,
            # H3.6 (N3) + H4: provenance. The fetch path itself does not score;
            # load_news() scores headlines and upgrades this to "model" when a
            # scorer is available, else it stays "default" (never a silent score).
            "scorer": "default",
        })
    return items


def _normalize_items(items: list[dict]) -> list[dict]:
    """Read-side migration: entries lacking `scorer` are `default` (H3.6, N3)."""
    for it in items:
        if isinstance(it, dict):
            it.setdefault("scorer", "default")
    return items


# ── H4: sentiment scoring boundary ───────────────────────────────────────────
# Intent: the news cache must carry honest scorer provenance. The fetch path
# never invents a sentiment; load_news() scores headlines through FinBERT when
# the stack is importable, batching per symbol, and downgrades to "default" on
# any failure or timeout (P4: no silent score, one log line).
# Invariants: a "model" entry always carries a real model score; a "default"
# entry always has score 0.0 and renders no word. Scoring is injectable so tests
# stub the model at the boundary (D1 recorded-source style).
_default_scorer_value = None
_default_scorer_resolved = False


def _build_default_scorer():
    """Construct a FinBERTBatchScorer when transformers+torch are importable."""
    try:
        import importlib.util

        if (importlib.util.find_spec("transformers") is None
                or importlib.util.find_spec("torch") is None):
            return None
        from quant.analytics.sentiment import FinBERTBatchScorer

        return FinBERTBatchScorer()
    except Exception as e:  # noqa: BLE001
        logger.warning("news scorer unavailable; entries stay default: %s", e)
        return None


def _default_scorer_cached():
    """Resolve the default scorer once per process (model load is expensive)."""
    global _default_scorer_value, _default_scorer_resolved
    if not _default_scorer_resolved:
        _default_scorer_value = _build_default_scorer()
        _default_scorer_resolved = True
    return _default_scorer_value


def _score_items(items: list[dict], scorer, timeout: float) -> None:
    """Batch-score headlines in place. Any failure keeps entries at default."""
    if scorer is None or not items:
        return
    headlines = [str(it.get("headline", "")) for it in items]
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as _FutTimeout

    ex = ThreadPoolExecutor(max_workers=1)
    try:
        scores = ex.submit(scorer.score_texts, headlines).result(timeout=timeout)
    except _FutTimeout:
        logger.warning("news scoring timed out after %ss; entries stay default", timeout)
        ex.shutdown(wait=False)
        return
    except Exception as e:  # noqa: BLE001
        logger.warning("news scoring failed; entries stay default: %s", e)
        ex.shutdown(wait=False)
        return
    ex.shutdown(wait=False)
    if not isinstance(scores, list) or len(scores) != len(items):
        logger.warning("news scoring returned unexpected shape; entries stay default")
        return
    for it, s in zip(items, scores):
        it["score"] = float(s)
        it["scorer"] = "model"


def cache_scorer_stats() -> dict:
    """Distribution of news-cache entries by scorer (H3.6, N3). {} when absent."""
    cache = _read_cache()
    stats = {"entries": 0, "model": 0, "default": 0, "pos": 0, "neg": 0, "neu": 0}
    for entry in cache.values():
        items = entry.get("items", []) if isinstance(entry, dict) else []
        for it in items:
            if not isinstance(it, dict):
                continue
            stats["entries"] += 1
            if it.get("scorer") == "model":
                stats["model"] += 1
            else:
                stats["default"] += 1
            s = it.get("score", 0) or 0
            if s > 0:
                stats["pos"] += 1
            elif s < 0:
                stats["neg"] += 1
            else:
                stats["neu"] += 1
    return stats


def load_news(symbol: str, fetcher=None, now: float | None = None,
              ttl_hours: float = 24.0, scorer=None, scorer_factory=None,
              score_timeout: float = 10.0) -> list[dict]:
    """Return cached-or-fetched news. A fetch failure yields [] and stores nothing.

    H4: on a cache miss the freshly fetched headlines are scored through the
    model when one is available (``scorer`` explicit, else ``scorer_factory``,
    else the cached default built from the installed stack). Scoring failures or
    a timeout leave every entry at ``scorer: "default"`` with one log line.
    """
    now = time.time() if now is None else now
    cache = _read_cache()
    entry = cache.get(symbol)
    if entry and (now - float(entry.get("retrieved_at", 0))) < ttl_hours * 3600:
        return _normalize_items(entry.get("items", []))
    items = fetch_news_items(symbol, fetcher=fetcher)
    if items is None:
        return []
    if scorer is None:
        scorer = (scorer_factory() if scorer_factory is not None
                  else _default_scorer_cached())
    _score_items(items, scorer, score_timeout)
    cache[symbol] = {"retrieved_at": now, "items": items}
    _write_cache(cache)
    return _normalize_items(items)


def fetch_news_headlines(symbol: str) -> str:
    """Legacy joined-headlines helper (review sentiment path)."""
    items = load_news(symbol)
    if not items:
        return ""
    return " | ".join(i["headline"] for i in items[:5])


def _state_path() -> str:
    return os.path.join(str(paths.OUTPUTS_DIR), "news_state.json")


def _read_state() -> dict:
    try:
        with open(_state_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def record_fetch_result(success: bool, now: float | None = None) -> dict:
    """Persist the consecutive review-fetch failure counter (reset on success)."""
    state = _read_state()
    now = time.time() if now is None else now
    if success:
        state = {"consecutive_failures": 0, "since": None}
    else:
        n = int(state.get("consecutive_failures", 0)) + 1
        state = {"consecutive_failures": n, "since": state.get("since") or now}
    try:
        _atomic_write_json(_state_path(), state)
    except Exception:  # noqa: BLE001
        pass
    return state


def outage_message() -> str | None:
    """Health line after >=3 consecutive review-fetch failures, else None."""
    state = _read_state()
    if int(state.get("consecutive_failures", 0)) >= 3 and state.get("since"):
        from datetime import datetime

        from quant.ui import copy as C

        d = datetime.fromtimestamp(float(state["since"]))
        since = f"{d.strftime('%A')} {d.day} {d.strftime('%b')}"
        return C.HEALTH_NEWS_OUTAGE.format(since=since)
    return None

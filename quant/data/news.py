"""
news.py — headlines per symbol, scored by FinBERT, stored as ``nlp_evidence``.

Run by ``refresh`` (so by the daily job and Run analysis) for the held symbols
and the current funnel survivors:

  1. fetch the Yahoo Finance RSS headlines of each symbol;
  2. skip titles already stored for that symbol;
  3. score the new titles in one FinBERT batch: tone = P(positive) - P(negative)
     in -1..1 (the label order is read from the model config);
  4. insert them into ``nlp_evidence`` and prune rows older than 60 days.

Without the optional ``news`` extra (torch + transformers) headlines are still
stored and shown, unscored; the sentiment score component then stays empty
instead of being faked.

v10.8.10 replaces: a JSON news cache, a FinBERT scorer that skipped every text
under 50 characters (so almost every headline scored a flat 0), a "news pillar"
that then declared itself absent and stopped loading the model for good, and
SEC 8-K filing text. ``nlp_evidence`` had had no writer since v10.5.

Invariants: ``ingest`` never raises; fetcher and scorer are injectable.
"""
from __future__ import annotations

import logging
import urllib.request
from collections.abc import Callable
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
KEEP_DAYS = 60
MAX_PER_SYMBOL = 15


def model_available() -> bool:
    """True when the optional news stack (torch + transformers) is installed."""
    import importlib.util

    return (importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None)


def _rss(symbol: str, timeout: int = 10) -> bytes:
    req = urllib.request.Request(
        f"https://finance.yahoo.com/rss/headline?s={symbol}",
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _published(value: Any) -> datetime | None:
    try:
        return parsedate_to_datetime(str(value)).replace(tzinfo=None)
    except (TypeError, ValueError, IndexError):
        return None


def fetch_headlines(symbol: str, fetch: Callable[[str], bytes] = _rss) -> list[dict]:
    """``[{title, published_at}]`` for a symbol; [] on any failure."""
    import feedparser

    try:
        feed = feedparser.parse(fetch(symbol))
    except Exception as e:  # noqa: BLE001
        logger.warning("[%s] news fetch failed: %s", symbol, e)
        return []
    out = []
    for entry in feed.entries[:MAX_PER_SYMBOL]:
        title = " ".join(str(entry.get("title", "")).split())
        if title:
            out.append({"title": title, "published_at": _published(entry.get("published"))})
    return out


class HeadlineScorer:
    """FinBERT tone for short texts (loaded once, on first use)."""

    def __init__(self, model_name: str = FINBERT_MODEL) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(2)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        self._pos, self._neg = labels["positive"], labels["negative"]

    def score(self, texts: list[str]) -> list[float]:
        """Tone in -1..1 per text: P(positive) - P(negative)."""
        out: list[float] = []
        for i in range(0, len(texts), 32):
            enc = self.tokenizer(texts[i:i + 32], padding=True, truncation=True,
                                 max_length=64, return_tensors="pt")
            with self._torch.no_grad():
                probs = self._torch.softmax(self.model(**enc).logits, dim=-1)
            out += (probs[:, self._pos] - probs[:, self._neg]).tolist()
        return [round(float(s), 4) for s in out]


def _default_scorer() -> Any:
    if not model_available():
        return None
    try:
        return HeadlineScorer()
    except Exception as e:  # noqa: BLE001
        logger.warning("news scorer unavailable (%s); headlines stored unscored", e)
        return None


def ingest(conn: Any, symbols: list[str], fetch: Callable[[str], bytes] = _rss,
           scorer: Any = "default", today: date | None = None) -> dict:
    """Fetch, score and store new headlines. Returns ``{symbols, new, scored}``."""
    today = today or date.today()
    stats = {"symbols": 0, "new": 0, "scored": 0}
    fresh: list[tuple[str, dict]] = []
    for sym in symbols:
        items = fetch_headlines(sym, fetch)
        if not items:
            continue
        stats["symbols"] += 1
        try:
            known = {r[0] for r in conn.execute(
                "SELECT title FROM nlp_evidence WHERE symbol = ?", [sym]).fetchall()}
        except Exception:  # noqa: BLE001
            known = set()
        fresh += [(sym, it) for it in items if it["title"] not in known]
    if not fresh:
        return stats
    if scorer == "default":
        scorer = _default_scorer()
    scores: list[float | None] = [None] * len(fresh)
    if scorer is not None:
        try:
            scores = list(scorer.score([it["title"] for _s, it in fresh]))
        except Exception as e:  # noqa: BLE001
            logger.warning("news scoring failed (%s); headlines stored unscored", e)
    try:
        for (sym, it), score in zip(fresh, scores, strict=True):
            conn.execute(
                "INSERT INTO nlp_evidence (symbol, source, title, published_at, score, "
                "confidence) VALUES (?, 'news', ?, ?, ?, ?)",
                [sym, it["title"], it["published_at"], score,
                 None if score is None else "model"])
        conn.execute("DELETE FROM nlp_evidence WHERE COALESCE(published_at, retrieved_at) "
                     "< ?", [datetime.combine(today, datetime.min.time())
                             - timedelta(days=KEEP_DAYS)])
    except Exception as e:  # noqa: BLE001
        logger.warning("news store failed: %s", e)
        return stats
    stats["new"] = len(fresh)
    stats["scored"] = sum(1 for s in scores if s is not None)
    return stats


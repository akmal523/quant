"""
news.py — RSS headline fetcher with async batch support.

Fetch hierarchy per ticker:
  1. Google News RSS via aiohttp (async, concurrent batch)
  2. Google News RSS via urllib (sync fallback)
  3. yfinance .news (last resort)

Async batch fetch reduces total news acquisition time from O(N) sequential
to O(1) network round-trips for all tickers combined.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) "
    "AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
]

_GNEWS_BASE = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def _parse_rss_body(body: bytes, max_items: int) -> list[dict]:
    """Parse RSS XML bytes into list of headline dicts."""
    root = ET.fromstring(body)
    items = []
    for item in root.findall(".//item")[:max_items]:
        pub_date = item.findtext("pubDate") or ""
        # Convert RFC-822 pubDate to Unix timestamp for time-weighting in sentiment
        try:
            import email.utils
            ts = str(email.utils.parsedate_to_datetime(pub_date).timestamp())
        except Exception:
            ts = str(time.time())

        items.append({
            "title":               (item.findtext("title")       or "").strip(),
            "summary":             (item.findtext("description") or "").strip(),
            "published":           pub_date,
            "published_timestamp": ts,
        })
    return items


# ── Async fetch (primary) ─────────────────────────────────────────────────────

async def _fetch_rss_async(
    session,       # aiohttp.ClientSession
    query: str,
    max_items: int = 10,
) -> list[dict]:
    """
    Async Google News RSS fetch using aiohttp.
    Called from get_headlines_batch_async() — not meant for direct use.
    """
    q   = urllib.parse.quote_plus(query)
    url = _GNEWS_BASE.format(q=q)
    ua  = random.choice(_USER_AGENTS)

    for attempt in range(3):
        try:
            async with session.get(url, headers={"User-Agent": ua}, timeout=10) as resp:
                if resp.status != 200:
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status
                    )
                body = await resp.read()
            return _parse_rss_body(body, max_items)

        except ET.ParseError:
            logger.debug("[RSS/async] XML parse error for '%s'.", query)
            break
        except Exception as exc:
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.debug("[RSS/async] Error for '%s' (attempt %d): %s. Retry in %.1fs",
                         query, attempt + 1, exc, wait)
            await asyncio.sleep(wait)

    return []


async def get_headlines_batch_async(
    fetch_specs: list[tuple[str, str, int]],
) -> dict[str, list[dict]]:
    """
    Fetch headlines for multiple tickers concurrently in a single aiohttp session.

    Args:
      fetch_specs: List of (symbol, query_name, max_items) tuples.

    Returns:
      Dict mapping symbol → list of headline dicts.
    """
    try:
        import aiohttp
    except ImportError:
        logger.warning("[News] aiohttp not installed. Falling back to sequential sync fetch.")
        return {
            sym: get_recent_headlines(sym, name, max_items)
            for sym, name, max_items in fetch_specs
        }

    results: dict[str, list[dict]] = {}

    async with aiohttp.ClientSession() as session:
        tasks = {
            sym: asyncio.create_task(
                _fetch_rss_async(session, name.strip() if name.strip() else sym, max_items)
            )
            for sym, name, max_items in fetch_specs
        }
        for sym, task in tasks.items():
            try:
                items = await task
                results[sym] = items if items else []
            except Exception as exc:
                logger.debug("[News/async] Task failed for %s: %s", sym, exc)
                results[sym] = []

    return results


# ── Sync fetch (fallback) ─────────────────────────────────────────────────────

def _fetch_rss(query: str, max_items: int = 10) -> list[dict]:
    """Synchronous Google News RSS fetch with exponential backoff retry."""
    q   = urllib.parse.quote_plus(query)
    url = _GNEWS_BASE.format(q=q)

    for attempt in range(3):
        ua  = random.choice(_USER_AGENTS)
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read()
            return _parse_rss_body(body, max_items)

        except HTTPError as exc:
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.debug("[RSS] HTTP %d for '%s' (attempt %d). Retry in %.1fs",
                         exc.code, query, attempt + 1, wait)
            time.sleep(wait)

        except URLError as exc:
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.debug("[RSS] URLError for '%s': %s. Retry in %.1fs", query, exc, wait)
            time.sleep(wait)

        except ET.ParseError:
            logger.debug("[RSS] XML parse error for '%s'.", query)
            break

    return []


def _yfinance_fallback(symbol: str) -> list[dict]:
    """yfinance .news — last-resort fallback."""
    try:
        import yfinance as yf
        raw = yf.Ticker(symbol).news or []
        return [
            {
                "title":               item.get("title",   ""),
                "summary":             item.get("summary", ""),
                "published":           str(item.get("providerPublishTime", "")),
                "published_timestamp": str(item.get("providerPublishTime", time.time())),
            }
            for item in raw[:10]
        ]
    except Exception:
        return []


def get_recent_headlines(symbol: str, name: str = "", max_items: int = 10) -> list[dict]:
    """
    Synchronous single-ticker headline fetch.
    Used as fallback when async batch is unavailable.
    """
    query = name.strip() if name.strip() else symbol
    items = _fetch_rss(query, max_items)
    if not items:
        logger.debug("[News] RSS empty for %s — falling back to yfinance.", symbol)
        items = _yfinance_fallback(symbol)
    return items

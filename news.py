"""
news.py — RSS headline fetcher.
Primary:  Google News RSS (User-Agent rotation, exponential-backoff retry).
Fallback: yfinance .news property.
"""
from __future__ import annotations
import logging
import random
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

# Rotate User-Agent strings to reduce 429 / block probability.
_USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) "
    "AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

_GNEWS_BASE = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def _fetch_rss(query: str, max_items: int = 10) -> list[dict]:
    """
    Fetch Google News RSS for a query.
    Retries up to 3 times with exponential back-off and UA rotation.
    """
    q   = urllib.parse.quote_plus(query)
    url = _GNEWS_BASE.format(q=q)

    for attempt in range(3):
        ua  = random.choice(_USER_AGENTS)
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read()
            root  = ET.fromstring(body)
            items = []
            for item in root.findall(".//item")[:max_items]:
                items.append({
                    "title":     (item.findtext("title")       or "").strip(),
                    "summary":   (item.findtext("description") or "").strip(),
                    "published": (item.findtext("pubDate")     or ""),
                })
            return items

        except HTTPError as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.debug(f"[RSS] HTTP {e.code} for '{query}' (attempt {attempt+1}). "
                         f"Retry in {wait:.1f}s")
            time.sleep(wait)

        except URLError as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            logger.debug(f"[RSS] URLError for '{query}': {e}. Retry in {wait:.1f}s")
            time.sleep(wait)

        except ET.ParseError:
            logger.debug(f"[RSS] XML parse error for '{query}'.")
            break

    return []


def _yfinance_fallback(symbol: str) -> list[dict]:
    """
    yfinance .news — last-resort fallback when RSS returns nothing.
    """
    try:
        import yfinance as yf
        raw = yf.Ticker(symbol).news or []
        return [
            {
                "title":     item.get("title",   ""),
                "summary":   item.get("summary", ""),
                "published": str(item.get("providerPublishTime", "")),
            }
            for item in raw[:10]
        ]
    except Exception:
        return []


def get_recent_headlines(symbol: str, name: str = "", max_items: int = 10) -> list[dict]:
    """
    Fetch recent headlines for an asset.
      1. Google News RSS using the company name (more semantic than ticker).
      2. Falls back to yfinance .news on empty result.
    """
    query = name.strip() if name.strip() else symbol
    items = _fetch_rss(query, max_items)
    if not items:
        logger.debug(f"[News] RSS empty for {symbol!r}; falling back to yfinance.")
        items = _yfinance_fallback(symbol)
    return items

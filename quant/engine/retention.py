"""
retention.py — Data retention inside the daily run (v10.7.0, Section 13).

Intent: keep the store bounded. Daily bars older than 5 years are downsampled
to weekly bars (OHLC via first/max/min/last, volume summed); the news cache is
pruned to 90 days. Upsert by (symbol, date) everywhere.

Invariants:
  - run_retention never raises; returns a summary dict.
  - Downsampling is idempotent: a week that already holds one row stays one row.
  - The news cache is a JSON file under outputs/; a missing file is a no-op.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from typing import Any

from quant import paths

RETENTION_DAILY_YEARS = 5
NEWS_CACHE_MAX_DAYS = 90


def _iso_week(value: Any) -> tuple[int, int]:
    """(ISO year, ISO week) for a date or ISO date string."""
    if isinstance(value, date):
        d = value
    else:
        d = date.fromisoformat(str(value)[:10])
    iso = d.isocalendar()
    return (iso[0], iso[1])


def downsample_daily_bars(
    conn,
    today: date | None = None,
    years: int = RETENTION_DAILY_YEARS,
) -> int:
    """Downsample daily bars older than ``years`` to weekly bars.

    Returns the number of rows removed (daily rows minus weekly rows). Idempotent:
    re-running on already-weekly data leaves one row per week.
    """
    today = today or date.today()
    cutoff = (today - timedelta(days=365 * years)).isoformat()
    try:
        rows = conn.execute(
            "SELECT Symbol, Date, Open, High, Low, Close, Volume FROM market_history "
            "WHERE Date < ? ORDER BY Symbol, Date",
            [cutoff],
        ).fetchall()
    except Exception:  # noqa: BLE001
        return 0
    if not rows:
        return 0

    groups: dict[tuple[str, tuple[int, int]], list] = {}
    for symbol, d, o, h, low, c, v in rows:
        groups.setdefault((str(symbol), _iso_week(d)), []).append((d, o, h, low, c, v))

    weekly: list[tuple] = []
    for (symbol, _week), items in groups.items():
        items.sort(key=lambda x: x[0])
        first = items[0]
        last = items[-1]
        weekly.append(
            (
                symbol,
                last[0],
                first[1],
                max(x[2] for x in items if x[2] is not None),
                min(x[3] for x in items if x[3] is not None),
                last[4],
                sum((x[5] or 0) for x in items),
            )
        )

    conn.execute("DELETE FROM market_history WHERE Date < ?", [cutoff])
    for symbol, d, o, h, low, c, v in weekly:
        conn.execute(
            "INSERT OR REPLACE INTO market_history "
            "(Date, Open, High, Low, Close, Volume, Symbol) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [d, o, h, low, c, v, symbol],
        )
    return len(rows) - len(weekly)


def prune_news_cache(
    today: date | None = None,
    max_days: int = NEWS_CACHE_MAX_DAYS,
) -> int:
    """Remove news-cache entries older than ``max_days``. Returns rows removed."""
    path = os.path.join(str(paths.OUTPUTS_DIR), "news_cache.json")
    if not os.path.exists(path):
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:  # noqa: BLE001
        return 0
    if not isinstance(cache, dict):
        return 0
    if today is None:
        now = datetime.now().timestamp()
    else:
        now = datetime(today.year, today.month, today.day).timestamp()
    cutoff = now - max_days * 86400
    kept = {
        key: value
        for key, value in cache.items()
        if float((value or {}).get("retrieved_at", 0) or 0) >= cutoff
    }
    removed = len(cache) - len(kept)
    if removed:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(kept, f)
        except Exception:  # noqa: BLE001
            return 0
    return removed


def run_retention(conn, today: date | None = None) -> dict:
    """Run the retention job. Never raises; returns a summary dict."""
    today = today or date.today()
    return {
        "bars_downsampled": downsample_daily_bars(conn, today),
        "news_pruned": prune_news_cache(today),
    }

"""
cash_rate.py — Trade Republic cash interest rate (v10.5.1).

Intent: the cash APY is a dated fact, not a constant. Trade Republic raised its
rate to 2.5 percent on 16 Sep 2026. Keep a dated schedule with a documented
source per row, expose ``current_cash_apy(as_of)``, and offer an optional live
fetch that falls back to the schedule and never raises.

Invariants:
  - ``current_cash_apy(as_of)`` returns the rate effective on that date.
  - ``fetch_live_cash_apy()`` never raises; returns None on any failure.
  - The dated schedule is the deterministic source of truth for tests.
  - Pure lookups; the only I/O is the optional live fetch.

Dependencies: datetime, json, re, urllib (live fetch only).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime

# Documented public source for the Trade Republic cash rate. The live fetch is
# best-effort; the dated schedule below is the fallback and the test oracle.
SOURCE_URL = "https://www.traderepublic.com/en-de/interest"

# Fallback rate used when no schedule row applies (kept in sync with config).
FALLBACK_APY = 0.025


@dataclass(frozen=True)
class CashRate:
    """One dated cash-rate fact with its source."""

    effective_date: date
    apy: float
    source_url: str


# Dated schedule. Newest last. Each row is a reviewable, sourced fact.
CASH_RATE_SCHEDULE: list[CashRate] = [
    CashRate(date(2024, 1, 1), 0.0225, SOURCE_URL),
    CashRate(date(2026, 9, 16), 0.025, SOURCE_URL),
]


def _as_date(value: date | datetime | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return date.today()


def schedule_apy(as_of: date | datetime | str | None = None) -> float:
    """Return the scheduled APY effective on ``as_of`` (default today)."""
    when = _as_date(as_of)
    applicable = [r for r in CASH_RATE_SCHEDULE if r.effective_date <= when]
    if not applicable:
        return CASH_RATE_SCHEDULE[0].apy if CASH_RATE_SCHEDULE else FALLBACK_APY
    return max(applicable, key=lambda r: r.effective_date).apy


def current_rate(as_of: date | datetime | str | None = None) -> CashRate:
    """Return the applicable schedule row (apy + effective_date) for ``as_of``.

    Intent (v10.5.2, A4): the UI helper text must render the live schedule value
    AND its effective date, not a bare constant. Pure lookup; never raises.
    """
    when = _as_date(as_of)
    applicable = [r for r in CASH_RATE_SCHEDULE if r.effective_date <= when]
    if not applicable:
        if CASH_RATE_SCHEDULE:
            return CASH_RATE_SCHEDULE[0]
        return CashRate(when, FALLBACK_APY, SOURCE_URL)
    return max(applicable, key=lambda r: r.effective_date)


def _default_fetcher(url: str) -> str:
    """Fetch a URL's HTML with a browser User-Agent."""
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _default_parser(html: str) -> float | None:
    """Extract the first percentage (e.g. '2.5 %' or '2,5 %') from HTML."""
    m = re.search(r"(\d+[.,]\d+)\s*%", html)
    if not m:
        return None
    return float(m.group(1).replace(",", ".")) / 100.0


def fetch_live_cash_apy(
    url: str = SOURCE_URL,
    fetcher=None,
    parser=None,
) -> float | None:
    """Best-effort live fetch of the cash APY. Never raises; None on failure.

    Intent: allow the rate to track the broker without a code change, while the
    dated schedule remains the deterministic fallback. ``fetcher`` and
    ``parser`` are injectable for hermetic tests.
    """
    fetcher = fetcher or _default_fetcher
    parser = parser or _default_parser
    try:
        html = fetcher(url)
        rate = parser(html)
        if rate is None or not (0.0 <= rate <= 0.20):
            return None
        return rate
    except Exception:  # noqa: BLE001
        return None


def current_cash_apy(
    as_of: date | datetime | str | None = None,
    live: bool = False,
) -> float:
    """Return the cash APY to use. Live fetch first (if requested), else schedule."""
    if live:
        rate = fetch_live_cash_apy()
        if rate is not None:
            return rate
    return schedule_apy(as_of)


def update_cash_rate(
    apy: float,
    effective_date: date | datetime | str,
    source_url: str = SOURCE_URL,
) -> list[CashRate]:
    """Append a new dated rate to the in-memory schedule. Returns the schedule.

    Intent: a single, reviewable way to record a rate change. Persist with
    ``save_schedule`` if a durable copy is wanted.
    """
    when = _as_date(effective_date)
    CASH_RATE_SCHEDULE[:] = [r for r in CASH_RATE_SCHEDULE if r.effective_date != when]
    CASH_RATE_SCHEDULE.append(CashRate(when, float(apy), source_url))
    CASH_RATE_SCHEDULE.sort(key=lambda r: r.effective_date)
    return CASH_RATE_SCHEDULE


def save_schedule(path: str) -> None:
    """Persist the schedule to JSON (reviewable, version-controllable)."""
    payload = [
        {"effective_date": r.effective_date.isoformat(), "apy": r.apy,
         "source_url": r.source_url}
        for r in CASH_RATE_SCHEDULE
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_schedule(path: str) -> list[CashRate]:
    """Load a schedule from JSON, replacing the in-memory one. Returns it."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    CASH_RATE_SCHEDULE[:] = [
        CashRate(date.fromisoformat(r["effective_date"]), float(r["apy"]),
                 r.get("source_url", SOURCE_URL))
        for r in rows
    ]
    CASH_RATE_SCHEDULE.sort(key=lambda r: r.effective_date)
    return CASH_RATE_SCHEDULE


def as_dicts() -> list[dict]:
    """Return the schedule as plain dicts (for diagnostics)."""
    return [asdict(r) for r in CASH_RATE_SCHEDULE]

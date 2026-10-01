"""
news_pillar.py — diagnose the dead news model, then demote it honestly (v10.7.2).

Intent: the heavy FinBERT stack (torch, transformers) is loaded on an old laptop
and may contribute nothing: every score is the neutral default, and tactical
grades already apply the no-data penalty. This module diagnoses WHY (read-only,
never imports torch) and persists a first-class ``news_pillar`` status
(``active`` | ``absent``), recomputed on the Friday run. When absent, the scorer
path must not import torch at all (the performance win), and the UI/briefing show
one honest line.

Invariants:
  - The diagnostic never imports torch/transformers (importlib.util.find_spec only).
  - ``recompute_status`` never raises; returns the status dict.
  - ``absent`` when zero model-scored items in the last 30 days.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime

from quant import paths

STATE_NAME = "news_pillar.json"
WINDOW_DAYS = 30
MIN_TEXT_CHARS = 50

STATUS_ACTIVE = "active"
STATUS_ABSENT = "absent"


def state_path() -> str:
    """Absolute path of the persisted news-pillar status."""
    return os.path.join(str(paths.OUTPUTS_DIR), STATE_NAME)


def _cache_path() -> str:
    """Absolute path of the news cache (read directly; no import of news)."""
    return os.path.join(str(paths.OUTPUTS_DIR), "news_cache.json")


def _read_cache() -> dict:
    try:
        with open(_cache_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


# ── Persisted status ──────────────────────────────────────────────────────────

def read_status() -> dict:
    """The persisted news-pillar status. Defaults to active when never computed."""
    try:
        with open(state_path(), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("status") in (STATUS_ACTIVE, STATUS_ABSENT):
            return data
    except Exception:  # noqa: BLE001
        pass
    return {"status": STATUS_ACTIVE, "as_of": None, "model_scored_30d": None}


def write_status(status: str, model_scored_30d: int | None = None,
                 as_of: date | None = None) -> None:
    """Persist the news-pillar status. Never raises."""
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(state_path(), "w", encoding="utf-8") as f:
            json.dump({
                "status": status,
                "model_scored_30d": model_scored_30d,
                "as_of": (as_of or date.today()).isoformat(),
            }, f)
    except Exception:  # noqa: BLE001
        pass


def is_absent() -> bool:
    """True when the news pillar is demoted to absent."""
    return read_status().get("status") == STATUS_ABSENT


def enable() -> None:
    """Force the news pillar active (``quant news-doctor --enable``)."""
    write_status(STATUS_ACTIVE, model_scored_30d=None, as_of=date.today())


# ── Model availability (no torch import) ──────────────────────────────────────

def model_available() -> bool:
    """True when transformers + torch are importable. Never imports them."""
    try:
        import importlib.util

        return (importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("torch") is not None)
    except Exception:  # noqa: BLE001
        return False


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_published(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value.strip())
    except Exception:  # noqa: BLE001
        return None


def _within_window(value, now: datetime, days: int = WINDOW_DAYS) -> bool:
    """True when the published date is within the window (unparseable = recent)."""
    dt = _parse_published(value)
    if dt is None:
        return True
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt >= (now - timedelta(days=days))


# ── Diagnostic ────────────────────────────────────────────────────────────────

def _default_reason(headline: str, model_ok: bool) -> str:
    from quant.ui import copy as C

    if len(str(headline).strip()) < MIN_TEXT_CHARS:
        return C.NEWS_DOCTOR_REASON_SHORT
    if not model_ok:
        return C.NEWS_DOCTOR_REASON_UNAVAILABLE
    return C.NEWS_DOCTOR_REASON_NOT_INVOKED


def diagnostic(symbols: list[str], now: datetime | None = None) -> list[dict]:
    """Per-symbol news-pillar counts. Read-only; never imports torch."""
    now = now or datetime.now()
    model_ok = model_available()
    cache = _read_cache()
    out: list[dict] = []
    for symbol in symbols:
        entry = cache.get(symbol) or {}
        items = entry.get("items", []) if isinstance(entry, dict) else []
        total = long_enough = model = default = 0
        reasons: dict[str, int] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            if not _within_window(it.get("published_at"), now):
                continue
            total += 1
            headline = str(it.get("headline", ""))
            if len(headline.strip()) >= MIN_TEXT_CHARS:
                long_enough += 1
            if it.get("scorer") == "model":
                model += 1
            else:
                default += 1
                reason = _default_reason(headline, model_ok)
                reasons[reason] = reasons.get(reason, 0) + 1
        dominant = max(reasons, key=reasons.get) if reasons else ""
        out.append({
            "symbol": symbol,
            "total": total,
            "long_enough": long_enough,
            "model": model,
            "default": default,
            "dominant_reason": dominant,
        })
    return out


def diagnostic_lines(symbols: list[str], now: datetime | None = None) -> list[str]:
    """Plain diagnostic lines, one per symbol."""
    from quant.ui import copy as C

    lines: list[str] = []
    for row in diagnostic(symbols, now=now):
        if row["total"] == 0:
            lines.append(C.NEWS_DOCTOR_NONE.format(symbol=row["symbol"]))
        else:
            lines.append(C.NEWS_DOCTOR_LINE.format(
                symbol=row["symbol"], total=row["total"], long=row["long_enough"],
                model=row["model"], default=row["default"],
                reason=row["dominant_reason"] or "n/a"))
    return lines


def count_model_scored_30d(now: datetime | None = None) -> int:
    """Model-scored news items across the whole cache in the last 30 days."""
    now = now or datetime.now()
    cache = _read_cache()
    count = 0
    for entry in cache.values():
        items = entry.get("items", []) if isinstance(entry, dict) else []
        for it in items:
            if not isinstance(it, dict):
                continue
            if it.get("scorer") != "model":
                continue
            if _within_window(it.get("published_at"), now):
                count += 1
    return count


def recompute_status(today: date | None = None, now: datetime | None = None) -> dict:
    """Recompute and persist the news-pillar status (the Friday run). Never raises.

    Absent when zero model-scored items in the last 30 days; active otherwise.
    """
    today = today or date.today()
    try:
        model_count = count_model_scored_30d(now=now)
    except Exception:  # noqa: BLE001
        model_count = 0
    status = STATUS_ACTIVE if model_count > 0 else STATUS_ABSENT
    write_status(status, model_scored_30d=model_count, as_of=today)
    return read_status()


def summary_line(now: datetime | None = None) -> str:
    """One condensed line for the doctor."""
    from quant.ui import copy as C

    status = read_status()
    model_count = status.get("model_scored_30d")
    if model_count is None:
        model_count = count_model_scored_30d(now=now)
    return C.NEWS_DOCTOR_STATUS.format(status=status.get("status"), model=model_count)

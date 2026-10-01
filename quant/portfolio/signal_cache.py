"""
signal_cache.py — Weekly signal cadence (v10.6.2).

Intent: Alpha signals are generated on Friday and cached. Monday through
Thursday the system only monitors; it serves the cached Friday signals so the
user is not nudged to trade mid-week. If no cache exists yet, the first call
generates and caches.

Invariants:
  - ``is_signal_day`` is True only on Friday.
  - ``save_signal_cache`` writes atomically (temp file + ``os.replace``).
  - ``load_signal_cache`` never raises; returns an empty dict on any failure.
  - ``signals_for_today`` always returns (signals, as_of); never raises.

Dependencies: quant.paths, quant.portfolio.alpha (is_signal_day).
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

from quant import paths
from quant.portfolio.alpha import is_signal_day

_CACHE_NAME = "signal_cache.json"


def cache_path() -> str:
    """Return the signal-cache path under the outputs dir."""
    return str(Path(paths.OUTPUTS_DIR) / _CACHE_NAME)


def save_signal_cache(signals: list[dict], as_of: str) -> str:
    """Persist the generated signals atomically. Returns the written path.

    Invariants: writes ``{"as_of": ..., "generated_at": ..., "signals": [...]}``;
    atomic (temp file in the same dir + ``os.replace``); never leaves a partial
    file.
    """
    path = cache_path()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": str(as_of),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "signals": signals or [],
    }
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, target)
    except Exception:  # noqa: BLE001
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def load_signal_cache(max_age_days: int = 7) -> dict | None:
    """Load the cached signals if fresh. Returns None when missing or stale.

    Intent (v10.6.3): a cache older than ``max_age_days`` (by file mtime or by
    its ``as_of`` date) is invalidated so a stale Friday signal is never served
    the following week. Invariants: never raises; returns None when the file is
    missing, unreadable, stale, or not a dict.
    """
    path = cache_path()
    if not os.path.exists(path):
        return None
    try:
        age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
        if age_days > max_age_days:
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        as_of = data.get("as_of")
        if as_of:
            try:
                cache_date = datetime.fromisoformat(str(as_of))
                if (datetime.now() - cache_date).days > max_age_days:
                    return None
            except ValueError:
                pass
        return data
    except Exception:  # noqa: BLE001
        return None


def invalidate_signal_cache() -> None:
    """Delete the signal cache if present. Never raises."""
    path = cache_path()
    try:
        if os.path.exists(path):
            os.unlink(path)
    except OSError:
        pass


def signals_for_today(
    today: date,
    generate_fn: Callable[[], list[dict]],
) -> tuple[list[dict], str]:
    """Return today's signals and their as-of date.

    Intent: on Friday, call ``generate_fn()`` and cache the result; Monday
    through Thursday, serve the cached signals. If no cache exists on a
    non-signal day, generate once so the user is never left without signals.
    Invariants: returns (signals, as_of); never raises; ``generate_fn`` is a
    zero-argument callable returning a list of signal dicts.
    """
    if is_signal_day(today):
        signals = list(generate_fn() or [])
        as_of = today.isoformat()
        try:
            save_signal_cache(signals, as_of)
        except Exception:  # noqa: BLE001
            pass
        return signals, as_of

    cached = load_signal_cache()
    if cached and cached.get("signals"):
        return list(cached["signals"]), str(cached.get("as_of", ""))

    # No cache yet on a non-signal day: generate once so the UI is not empty.
    signals = list(generate_fn() or [])
    as_of = today.isoformat()
    try:
        save_signal_cache(signals, as_of)
    except Exception:  # noqa: BLE001
        pass
    return signals, as_of

"""
cache.py — Disk-based caching for expensive calculations (v10.6.5).

Intent: repeated calculations (for example the structural grade for the same
fundamentals) are recomputed every run. This module caches a JSON-serializable
result on disk, keyed by the function name and its arguments, with a max age.

Invariants:
  - ``disk_cache`` returns the wrapped function's value unchanged.
  - A missing, stale, or corrupted cache entry recomputes; a write failure is
    swallowed (the result is still returned).
  - The cache directory is created lazily (importing this module writes nothing).

Dependencies: hashlib, json, time, quant.paths.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

from quant import paths

T = TypeVar("T")


def cache_dir() -> Path:
    """Return the cache directory (created lazily)."""
    d = Path(paths.DATA_DIR) / "cache"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    return d


def _compute_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Compute a stable cache key from the function name and arguments."""
    key_data = {
        "func": func_name,
        "args": str(args),
        "kwargs": str(sorted(kwargs.items())),
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()


def disk_cache(max_age_days: int = 7) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Cache a function's JSON-serializable result on disk.

    Parameters
    ----------
    max_age_days : int
        Entries older than this are ignored and recomputed.

    Returns
    -------
    Callable
        The decorated function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            cache_file = cache_dir() / f"{_compute_cache_key(func.__name__, args, kwargs)}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, encoding="utf-8") as fh:
                        cached = json.load(fh)
                    age_days = (time.time() - float(cached["timestamp"])) / 86400.0
                    if age_days <= max_age_days:
                        return cached["result"]
                except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
                    pass  # corrupted or unreadable: recompute

            result = func(*args, **kwargs)
            try:
                with open(cache_file, "w", encoding="utf-8") as fh:
                    json.dump({"timestamp": time.time(), "result": result}, fh)
            except Exception:  # noqa: BLE001
                pass  # write failure is non-fatal
            return result

        return wrapper

    return decorator


def clear_cache() -> int:
    """Delete all cache entries. Returns the number removed. Never raises."""
    removed = 0
    try:
        for cache_file in cache_dir().glob("*.json"):
            try:
                cache_file.unlink()
                removed += 1
            except OSError:
                pass
    except Exception:  # noqa: BLE001
        pass
    return removed


def get_cache_stats() -> dict:
    """Return cache statistics: entry count, total size, and location."""
    try:
        files = list(cache_dir().glob("*.json"))
        total_size = sum(f.stat().st_size for f in files)
    except Exception:  # noqa: BLE001
        files, total_size = [], 0
    return {
        "num_entries": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 4),
        "cache_dir": str(cache_dir()),
    }

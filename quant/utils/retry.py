"""
retry.py — Retry logic with exponential backoff (v10.6.5).

Intent: external calls (yfinance, SEC EDGAR) fail transiently. This decorator
retries a bounded number of times with exponential backoff, then re-raises the
last error.

Invariants: pure (no I/O beyond ``time.sleep``); the wrapped function's return
value is passed through unchanged.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a function with exponential backoff.

    Parameters
    ----------
    max_retries : int
        Maximum number of retry attempts after the first try.
    initial_delay : float
        Seconds to wait before the first retry.
    max_delay : float
        Upper bound on the delay between retries.
    backoff_factor : float
        Multiplier applied to the delay after each retry.
    exceptions : tuple[type[Exception], ...]
        Exception types that trigger a retry.

    Returns
    -------
    Callable
        The decorated function.

    Examples
    --------
    >>> @retry_with_backoff(max_retries=3, exceptions=(ConnectionError,))
    ... def fetch():
    ...     return do_request()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:  # noqa: BLE001
                    last_exception = e
                    if attempt == max_retries:
                        break
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator

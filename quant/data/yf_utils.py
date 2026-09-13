"""
yf_utils.py — Timeout-guarded, rate-limit-aware yfinance helpers.

Intent: yfinance's history()/download() calls use requests with no default
timeout. When Yahoo throttles the IP (common with parallel workers hitting a
1000+ universe), a call can hang forever OR raise YFRateLimitError. This module:
  - bounds every call with a hard timeout (daemon thread), and
  - batches downloads (yf.download) to minimise request volume, and
  - applies an exponential-backoff circuit breaker when Yahoo rate-limits.

Invariants:
  - Never blocks longer than `timeout` seconds per attempt (plus cooldown).
  - Returns None / omits a symbol on timeout, rate-limit, or exception.
  - The worker thread is a daemon, so it can never keep the process alive.

Dependencies: yfinance (lazy), pandas, threading.
"""
from __future__ import annotations

import contextlib
import logging
import os
import random
import threading
import time

import pandas as pd


@contextlib.contextmanager
def _silence_yf_output():
    """Suppress yfinance's delisted-ticker noise (stderr + logger) for one call.

    Intent: the broad universe contains bad/class-share tickers Yahoo does not
    carry; yfinance prints '$SYM: possibly delisted' to stderr (and logs it at
    ERROR). These are non-fatal (the symbol is simply omitted) but flood the
    console and appear as [ERROR] in main.py's log. Raise the yfinance logger
    and redirect stderr to devnull for the duration of the fetch.
    Invariants: restores the logger level and stderr on exit, even on exception.
    """
    yf_logger = logging.getLogger("yfinance")
    prev_level = yf_logger.level
    yf_logger.setLevel(logging.CRITICAL)
    stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(stderr_fd, 2)
        os.close(devnull)
        os.close(stderr_fd)
        yf_logger.setLevel(prev_level)

# Default per-attempt timeout (seconds).
DEFAULT_TIMEOUT = 15.0
DEFAULT_RETRIES = 2
DEFAULT_BACKOFF = 1.0
# Number of tickers per yf.download() request. Kept moderate so each request
# stays under Yahoo's rate limits while still batching heavily.
DEFAULT_CHUNK_SIZE = 50
# Hard cap on any single rate-limit cooldown so a thread never sleeps forever.
MAX_COOLDOWN = 90.0

# ── Global rate-limit circuit breaker ─────────────────────────────────────────
# When Yahoo returns YFRateLimitError, every worker should back off together
# instead of hammering. `_RATE_LIMIT_UNTIL` is the epoch time before which no
# request should be attempted.
_RATE_LOCK = threading.Lock()
_RATE_LIMIT_UNTIL = 0.0


def _is_rate_limit(err: BaseException) -> bool:
    """True if `err` is yfinance's YFRateLimitError (checked by name, no import)."""
    return type(err).__name__ == "YFRateLimitError"


def _wait_for_cooldown() -> None:
    """Block until the global rate-limit cooldown (if any) has elapsed."""
    with _RATE_LOCK:
        wait = _RATE_LIMIT_UNTIL - time.time()
    if wait > 0:
        time.sleep(min(wait, MAX_COOLDOWN))


def _note_rate_limit(seconds: float) -> None:
    """Extend the global cooldown to at least `seconds` from now."""
    global _RATE_LIMIT_UNTIL
    with _RATE_LOCK:
        _RATE_LIMIT_UNTIL = max(_RATE_LIMIT_UNTIL, time.time() + seconds)


def rate_limited() -> bool:
    """True if a rate-limit cooldown is currently active."""
    with _RATE_LOCK:
        return time.time() < _RATE_LIMIT_UNTIL


# Consecutive rate-limit hits before the whole run gives up. Yahoo has likely
# temporarily banned the IP, and further requests only deepen the ban, so it is
# far better to abort with a clear message than to grind through 1000 retries.
_MAX_RATE_LIMIT_HITS = 3
_RATE_HITS = 0
_ABORT = False


def aborted() -> bool:
    """True once the run has given up due to persistent rate limiting."""
    return _ABORT


def _record_rate_limit() -> None:
    global _RATE_HITS, _ABORT
    with _RATE_LOCK:
        _RATE_HITS += 1
        if _RATE_HITS >= _MAX_RATE_LIMIT_HITS and not _ABORT:
            _ABORT = True
            print(" [!] Yahoo rate limit persists - aborting further fetches. "
                  "Wait a few minutes (or switch network) and re-run.")


def _reset_rate_limit_hits() -> None:
    global _RATE_HITS
    with _RATE_LOCK:
        _RATE_HITS = 0


def _normalize_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Drop empty/NaN rows and require a usable Close column."""
    if df is None or df.empty:
        return None
    df = df.dropna(how="all")
    if "Close" not in df.columns:
        return None
    df = df.dropna(subset=["Close"])
    return df if not df.empty else None


def _extract_frame(data: pd.DataFrame | None, ticker: str, single: bool) -> pd.DataFrame | None:
    """Pull one ticker's OHLCV frame out of a (possibly MultiIndex) download."""
    if data is None or data.empty:
        return None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker not in data.columns.get_level_values(0):
                return None
            df = data[ticker].copy()
        else:
            # Flat columns only make sense for a single-ticker request.
            if not single:
                return None
            df = data.copy()
    except Exception:
        return None
    return _normalize_frame(df)


def history_with_timeout(
    ticker: str,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    **kwargs,
) -> pd.DataFrame | None:
    """Fetch a single ticker's yfinance history with a hard per-attempt timeout.

    Intent: guarantee the caller is never blocked indefinitely by a hung Yahoo
    request, and back off correctly when Yahoo rate-limits. Runs the blocking
    yfinance call in a daemon thread and joins with a timeout.

    Args:
        ticker: yahoo ticker symbol.
        timeout: max seconds to wait per attempt.
        retries: extra attempts after the first (total attempts = retries + 1).
        backoff: base seconds between attempts (rate-limit backoff doubles it).
        **kwargs: forwarded to yf.Ticker(ticker).history() (period/start/auto_adjust).

    Returns:
        DataFrame on success (may be empty), or None if every attempt timed out,
        was rate-limited, or raised.
    """
    import yfinance as yf

    last_df: pd.DataFrame | None = None
    for attempt in range(retries + 1):
        if aborted():
            return None
        _wait_for_cooldown()
        result: list = [None]
        error: list = [None]

        def _worker() -> None:
            try:
                result[0] = yf.Ticker(ticker).history(**kwargs)
            except Exception as e:  # noqa: BLE001
                error[0] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout)

        if t.is_alive():
            # Timed out; the daemon thread is abandoned.
            if attempt < retries:
                time.sleep(backoff + random.uniform(0, 0.5))
                continue
            return None

        if error[0] is not None and _is_rate_limit(error[0]):
            _record_rate_limit()
            wait = backoff * (2 ** attempt) + random.uniform(0, 1.0)
            _note_rate_limit(wait)
            if attempt < retries and not aborted():
                time.sleep(min(wait, MAX_COOLDOWN))
                continue
            return None

        df = result[0]
        if df is not None and not df.empty:
            _reset_rate_limit_hits()
            return df
        if df is not None:
            last_df = df
        if error[0] is not None and attempt >= retries:
            return last_df
        if attempt < retries:
            time.sleep(backoff + random.uniform(0, 0.5))

    return last_df


def download_batch(
    tickers: list[str],
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    progress: bool = False,
) -> dict[str, pd.DataFrame]:
    """Batch-download OHLCV history for many tickers via yf.download().

    Intent: replace N per-ticker requests with ceil(N / chunk_size) chunked
    requests. This is both far faster and far less likely to trip Yahoo's rate
    limiter than the per-ticker path. Failed/empty tickers are simply omitted.

    Args:
        tickers: list of yahoo symbols (deduplicated, empties dropped).
        period / start / end / interval / auto_adjust: forwarded to yf.download.
        chunk_size: tickers per request.
        timeout / retries / backoff: per-chunk timeout and retry policy.

    Returns:
        {ticker: DataFrame} for every ticker that returned usable data.
    """
    import yfinance as yf

    syms = [t for t in dict.fromkeys(tickers) if t]
    out: dict[str, pd.DataFrame] = {}
    if not syms:
        return out

    chunks = [syms[i:i + chunk_size] for i in range(0, len(syms), chunk_size)]
    for chunk in chunks:
        if aborted():
            break
        for attempt in range(retries + 1):
            if aborted():
                break
            _wait_for_cooldown()
            try:
                with _silence_yf_output():
                    data = yf.download(
                        list(chunk), period=period, start=start, end=end,
                        interval=interval, auto_adjust=auto_adjust,
                        group_by="ticker", threads=True, progress=progress,
                        timeout=timeout,
                    )
            except Exception as e:  # noqa: BLE001
                if _is_rate_limit(e):
                    _record_rate_limit()
                    wait = backoff * (2 ** attempt) + random.uniform(0, 1.0)
                    _note_rate_limit(wait)
                    if attempt < retries and not aborted():
                        time.sleep(min(wait, MAX_COOLDOWN))
                        continue
                    break
                if attempt < retries:
                    time.sleep(backoff + random.uniform(0, 0.5))
                    continue
                break

            single = len(chunk) == 1
            for t in chunk:
                df = _extract_frame(data, t, single)
                if df is not None:
                    out[t] = df
            _reset_rate_limit_hits()
            break

    return out

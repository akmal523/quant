"""
lock.py — the shared runner lock with ownership and takeover (v10.7.2, Part 1.1).

Intent: a laptop that sleeps mid-run and runs an app plus a timer plus manual
commands makes lock collisions a weekly event, not an edge case. ONE helper owns
the runner lock file so every writer (``quant daily``, ``quant run``, the UI
runner, ``quant backup``) respects the same lock. The lock file carries ``pid``,
``started_at`` (ISO), and ``command`` (plus the legacy ``owner``/``ts`` fields the
UI heartbeat used).

Acquisition rules:
  - Lock absent: take it.
  - Lock present, pid NOT alive (``os.kill(pid, 0)`` raises ``ProcessLookupError``):
    take over immediately; record the exact doctor phrase
    "runner lock: taken over from stale process (pid <pid>, age <N> min)".
  - Lock present, pid alive, age at or below ``STALE_LOCK_MINUTES``: retry with
    backoff (5 attempts over about 60 seconds), then abort gracefully.
  - Lock present, pid alive, age above ``HARD_LOCK_MINUTES``: take over with a
    warning line (assume a runaway process); record the takeover.
  - ``PermissionError`` from the liveness probe means alive: treat as alive.

Invariants:
  - Never raises; returns a ``LockResult``.
  - ``release`` removes the lock only when this process owns it.
  - The takeover phrase is persisted so ``quant doctor`` can render it later.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime

from quant import paths
from quant.config import (
    HARD_LOCK_MINUTES,
    LOCK_RETRY_ATTEMPTS,
    LOCK_RETRY_TOTAL_SECONDS,
)

logger = logging.getLogger(__name__)

LOCK_NAME = ".runner.lock"
STATE_NAME = "lock_state.json"


def lock_path() -> str:
    """Absolute path of the runner lock file."""
    return os.path.join(str(paths.OUTPUTS_DIR), LOCK_NAME)


def _state_path() -> str:
    """Absolute path of the persisted takeover record."""
    return os.path.join(str(paths.OUTPUTS_DIR), STATE_NAME)


# ── Read / write ──────────────────────────────────────────────────────────────

def read_lock() -> dict | None:
    """Read the lock file as a dict; None when absent or unreadable."""
    try:
        with open(lock_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def write_lock(command: str, *, refresh: bool = False) -> None:
    """Write the lock file for this process. Never raises.

    ``refresh`` keeps the original ``started_at`` when this process already owns
    the lock (the UI heartbeat refresh path); otherwise it stamps a new start.
    """
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        existing = read_lock()
        started = None
        if refresh and existing and existing.get("pid") == os.getpid():
            started = existing.get("started_at")
        started = started or datetime.now().isoformat(timespec="seconds")
        payload = {
            "pid": os.getpid(),
            "started_at": started,
            "command": command,
            # Legacy fields kept so older readers (and the UI heartbeat) work.
            "owner": existing.get("owner") if (refresh and existing) else None,
            "ts": time.time(),
        }
        with open(lock_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:  # noqa: BLE001
        pass


def release() -> None:
    """Remove the lock file when this process owns it. Never raises."""
    existing = read_lock()
    if existing and existing.get("pid") == os.getpid():
        try:
            os.remove(lock_path())
        except Exception:  # noqa: BLE001
            pass


# ── Liveness / age ────────────────────────────────────────────────────────────

def pid_alive(pid: int | None) -> bool:
    """True when ``pid`` is a live process. PermissionError means alive."""
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:  # noqa: BLE001
        # Conservative: an unknown probe result is treated as alive.
        return True
    return True


def lock_age_minutes(lock: dict | None, now: float | None = None) -> float | None:
    """Age of the lock in minutes from ``started_at`` (falls back to ``ts``)."""
    if not lock:
        return None
    now = time.time() if now is None else now
    started = lock.get("started_at")
    if started:
        try:
            dt = datetime.fromisoformat(str(started))
            return max(0.0, (now - dt.timestamp()) / 60.0)
        except ValueError:
            pass
    try:
        return max(0.0, (now - float(lock.get("ts", 0))) / 60.0)
    except (TypeError, ValueError):
        return None


# ── Takeover record (so the doctor can render the exact phrase) ───────────────

def takeover_line(pid: int | None, age_min: float | None) -> str:
    """The exact doctor phrase for a stale-process takeover."""
    age = int(age_min) if age_min is not None else 0
    return f"runner lock: taken over from stale process (pid {pid}, age {age} min)"


def _record_takeover(pid: int | None, age_min: float | None, reason: str) -> None:
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(_state_path(), "w", encoding="utf-8") as f:
            json.dump({
                "pid": pid,
                "age_min": age_min,
                "reason": reason,
                "at": datetime.now().isoformat(timespec="seconds"),
            }, f)
    except Exception:  # noqa: BLE001
        pass


def read_takeover() -> dict | None:
    """The last recorded takeover, or None."""
    try:
        with open(_state_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


# ── Acquisition ───────────────────────────────────────────────────────────────

@dataclass
class LockResult:
    """Outcome of a lock acquisition attempt."""

    acquired: bool
    message: str
    taken_over: bool = False
    stale_pid: int | None = None
    age_min: float | None = None


def _backoff_delays(attempts: int, total_seconds: float) -> list[float]:
    """Evenly spaced delays so the retry budget totals about ``total_seconds``."""
    sleeps = max(0, attempts - 1)
    if sleeps == 0:
        return []
    return [total_seconds / sleeps] * sleeps


def acquire(
    command: str,
    *,
    blocking: bool = True,
    attempts: int = LOCK_RETRY_ATTEMPTS,
    total_seconds: float = LOCK_RETRY_TOTAL_SECONDS,
    sleep_fn=time.sleep,
    now_fn=time.time,
) -> LockResult:
    """Acquire the runner lock. Never raises; returns a ``LockResult``.

    ``blocking=False`` (the UI path) returns immediately when a live foreign
    owner holds the lock, instead of retrying for the full budget.

    Re-entrancy: a process that already owns the lock (or a child that inherited
    it via ``QUANT_LOCK_HELD=1``) acquires immediately, so ``quant all`` and the
    UI's spawned subprocess never deadlock against their own lock.
    """
    if os.environ.get("QUANT_LOCK_HELD") == "1":
        return LockResult(True, "runner lock: inherited from parent")
    existing = read_lock()
    if existing and existing.get("pid") == os.getpid():
        return LockResult(True, "runner lock: already held by this process")

    delays = _backoff_delays(attempts, total_seconds)
    for i in range(max(1, attempts)):
        lock = read_lock()
        if not lock:
            write_lock(command)
            return LockResult(True, "runner lock: acquired")

        pid = lock.get("pid")
        age = lock_age_minutes(lock, now_fn())

        if not pid_alive(pid):
            # Dead owner: take over immediately.
            line = takeover_line(pid, age)
            logger.warning(line)
            _record_takeover(pid, age, "dead_pid")
            write_lock(command)
            return LockResult(True, line, taken_over=True, stale_pid=pid, age_min=age)

        if age is not None and age > HARD_LOCK_MINUTES:
            # Live but far past the hard cap: assume a runaway process.
            line = (f"runner lock: taken over from runaway process "
                    f"(pid {pid}, age {int(age)} min)")
            logger.warning(line)
            _record_takeover(pid, age, "runaway")
            write_lock(command)
            return LockResult(True, line, taken_over=True, stale_pid=pid, age_min=age)

        # Live owner below the hard cap: respect it.
        if not blocking:
            return LockResult(
                False,
                f"runner lock: held by live process (pid {pid}, "
                f"age {int(age) if age is not None else 0} min)",
                stale_pid=pid, age_min=age)
        if i < len(delays):
            sleep_fn(delays[i])
            continue
        return LockResult(
            False,
            f"runner lock: held by live process (pid {pid}, "
            f"age {int(age) if age is not None else 0} min)",
            stale_pid=pid, age_min=age)

    return LockResult(False, "runner lock: could not be acquired")


# ── Doctor / status ───────────────────────────────────────────────────────────

def is_foreign_live() -> bool:
    """True when a DIFFERENT live process holds the lock (UI busy check)."""
    lock = read_lock()
    if not lock:
        return False
    if lock.get("pid") == os.getpid():
        return False
    return pid_alive(lock.get("pid"))


def status_line() -> str:
    """One plain doctor line for the runner lock.

    Prefers the exact takeover phrase when a takeover was recorded and the lock
    is not currently held by a live foreign process; otherwise reports presence
    or absence.
    """
    lock = read_lock()
    if lock and lock.get("pid") != os.getpid() and pid_alive(lock.get("pid")):
        age = lock_age_minutes(lock)
        return (f"runner lock: present pid={lock.get('pid')} "
                f"age={int(age) if age is not None else 0} min")
    takeover = read_takeover()
    if takeover:
        return takeover_line(takeover.get("pid"), takeover.get("age_min"))
    if lock:
        age = lock_age_minutes(lock)
        return (f"runner lock: present pid={lock.get('pid')} "
                f"age={int(age) if age is not None else 0} min")
    return "runner lock: none"

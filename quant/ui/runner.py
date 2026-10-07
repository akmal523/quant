"""
runner.py — UI-triggered run orchestrator + heartbeat (v10.5.3, R4).

Intent: the app must not spawn overlapping refresh/review runs. ONE in-process
mutex serializes them; a foreign, still-live run is detected via a heartbeat file
(outputs/.runner.lock) carrying owner+pid+timestamp. A heartbeat older than 10
minutes is stale and auto-released. Runs execute as
``sys.executable -m quant.cli <command>``; the UI shows a verb-ing label, a
progress element, and one numbered outcome line.

Invariants:
  - At most one run at a time per process (non-blocking lock); no nested acquire.
  - If a session disconnects mid-run, the orphaned subprocess may finish and
    write artifacts while the lock waits out its stale window; harmless
    (runs are idempotent, artifacts are per-run) and bounded by 600 s.
  - The heartbeat is written on acquire, refreshed at the end, removed on release.
  - "A review is already running in another tab." only for a LIVE FOREIGN heartbeat.
  - A run on empty/absent DB never raises; returns a RunResult.

Dependencies: subprocess, sys, threading, time, uuid, quant.engine.lock,
quant.ui.copy.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass

from quant.ui import copy as ui_copy

# One lock per process: serializes UI-triggered runs across tabs.
_LOCK = threading.Lock()

# This process's identity (kept for backward compatibility; the shared lock
# tracks ownership by pid).
_OWNER = uuid.uuid4().hex

# UI command buttons -> CLI subcommands (P2: command names never reach the UI).
REFRESH = "update"
REVIEW = "run"
SAVE_AND_REVIEW = "all"
REPAIR = "repair"
SCHEDULE = "schedule"


@dataclass
class RunResult:
    """Outcome of a UI-triggered run."""

    status: str            # "ok" | "error" | "busy"
    message: str
    log_path: str | None
    returncode: int
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def is_running() -> bool:
    """True if a run currently holds the orchestrator lock."""
    return _LOCK.locked()


# ── Heartbeat / runner lock (v10.7.2: one shared lock) ────────────────────────
# The file lock is owned by quant.engine.lock so every writer (daily, run, the
# UI, backup) respects the same lock. The UI keeps its in-process mutex for
# same-process serialization and uses the shared lock's non-blocking mode so a
# foreign live owner yields the other-tab sentence immediately.

def _heartbeat_path() -> str:
    from quant.engine import lock as lock_mod

    return lock_mod.lock_path()


def _read_heartbeat() -> dict | None:
    from quant.engine import lock as lock_mod

    return lock_mod.read_lock()


def _write_heartbeat(command: str = "ui") -> None:
    from quant.engine import lock as lock_mod

    lock_mod.write_lock(command, refresh=True)


def _clear_heartbeat() -> None:
    from quant.engine import lock as lock_mod

    lock_mod.release()


def _heartbeat_live(hb: dict | None) -> bool:
    from quant.engine import lock as lock_mod

    if not hb:
        return False
    return lock_mod.pid_alive(hb.get("pid"))


def _foreign_live_heartbeat() -> bool:
    """True iff a DIFFERENT live process holds the lock.

    v10.7.2: liveness is by pid (a dead pid is a stale lock to take over), not by
    a heartbeat timestamp. This process's own lock returns False.
    """
    from quant.engine import lock as lock_mod

    return lock_mod.is_foreign_live()


def _latest_log_path() -> str | None:
    from quant.reporting.artifacts import latest_run

    run_dir = latest_run()
    if not run_dir:
        return None
    log = os.path.join(run_dir, "pipeline.log")
    return log if os.path.exists(log) else None


def run_repair() -> RunResult:
    """Run the ISIN registry repair in-process under the mutex. Never raises.

    Intent (A6): the Settings Health Repair button re-checks after the repair, so
    the repair runs in this process and shares the orchestrator lock. Writes only
    data/broker_registry.csv, never the DB.
    """
    from quant.engine import lock as lock_mod

    if not _LOCK.acquire(blocking=False):
        return RunResult("busy", ui_copy.ERROR_RUNNING, None, -1)
    t0 = time.time()
    try:
        res = lock_mod.acquire(REPAIR, blocking=False)
        if not res.acquired:
            return RunResult("busy", ui_copy.ERROR_ALREADY_RUNNING, None, -1)
        from quant.data.registry_repair import repair_isins

        summary = repair_isins()
        message = (f"filled {summary['filled']} ISINs from market data, "
                   f"{summary['not_found']} not found.")
        return RunResult("ok", message, None, 0, time.time() - t0)
    except Exception:  # noqa: BLE001
        return RunResult("error", ui_copy.ERROR_REFRESH_FAILED, None, -1,
                         time.time() - t0)
    finally:
        lock_mod.release()
        _LOCK.release()


def run(command: str = REFRESH) -> RunResult:
    """Run a pipeline command under the mutex + shared lock. Never raises.

    One acquisition covers the whole command (``all`` = update then run in one
    subprocess), so nested acquisition is impossible by construction.
    """
    from quant.engine import lock as lock_mod

    if not _LOCK.acquire(blocking=False):
        # Same process already running: never the foreign-session sentence.
        return RunResult("busy", ui_copy.ERROR_RUNNING, None, -1)
    t0 = time.time()
    try:
        res = lock_mod.acquire(command, blocking=False)
        if not res.acquired:
            return RunResult("busy", ui_copy.ERROR_ALREADY_RUNNING, None, -1)
        # The UI holds the lock; tell the spawned subprocess it inherited it so
        # it does not deadlock against its own parent.
        child_env = os.environ.copy()
        child_env["QUANT_LOCK_HELD"] = "1"
        proc = subprocess.run(
            [sys.executable, "-m", "quant.cli", command],
            capture_output=True,
            text=True,
            env=child_env,
        )
        # Refresh at the update->run boundary / end of the sequence.
        _write_heartbeat(command)
        log_path = _latest_log_path()
        if proc.returncode == 0:
            return RunResult("ok", "", log_path, 0, time.time() - t0)
        return RunResult("error", ui_copy.ERROR_REFRESH_FAILED, log_path,
                         proc.returncode, time.time() - t0)
    except Exception:  # noqa: BLE001
        return RunResult("error", ui_copy.ERROR_REFRESH_FAILED, _latest_log_path(),
                         -1, time.time() - t0)
    finally:
        lock_mod.release()
        _LOCK.release()

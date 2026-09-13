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
  - The heartbeat is written on acquire, refreshed at the end, removed on release.
  - "A review is already running in another tab." only for a LIVE FOREIGN heartbeat.
  - A run on empty/absent DB never raises; returns a RunResult.

Dependencies: subprocess, sys, threading, json, time, uuid, quant.ui.copy.
"""
from __future__ import annotations

import json
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

# A heartbeat older than this is stale and may be overwritten (spec 4.4).
_STALE_SECONDS = 600.0
# This process's heartbeat identity.
_OWNER = uuid.uuid4().hex

# UI command buttons -> CLI subcommands (P2: command names never reach the UI).
REFRESH = "update"
REVIEW = "run"
SAVE_AND_REVIEW = "all"
REPAIR = "repair"


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


# ── Heartbeat (spec 4.4) ──────────────────────────────────────────────────────

def _heartbeat_path() -> str:
    from quant import paths

    return os.path.join(str(paths.OUTPUTS_DIR), ".runner.lock")


def _read_heartbeat() -> dict | None:
    try:
        with open(_heartbeat_path(), encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _write_heartbeat() -> None:
    try:
        path = _heartbeat_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"owner": _OWNER, "pid": os.getpid(), "ts": time.time()}, f)
    except Exception:  # noqa: BLE001
        pass


def _clear_heartbeat() -> None:
    hb = _read_heartbeat()
    if hb and hb.get("owner") == _OWNER:
        try:
            os.remove(_heartbeat_path())
        except Exception:  # noqa: BLE001
            pass


def _heartbeat_live(hb: dict | None) -> bool:
    if not hb:
        return False
    try:
        return (time.time() - float(hb.get("ts", 0))) < _STALE_SECONDS
    except (TypeError, ValueError):
        return False


def _foreign_live_heartbeat() -> bool:
    """True iff a DIFFERENT session holds a still-live heartbeat.

    Stale heartbeats (>10 min) and this process's own heartbeat return False, so
    a killed run never blocks the next press.
    """
    hb = _read_heartbeat()
    return (_heartbeat_live(hb)
            and hb.get("owner") != _OWNER
            and hb.get("pid") != os.getpid())


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
    if not _LOCK.acquire(blocking=False):
        return RunResult("busy", ui_copy.ERROR_RUNNING, None, -1)
    t0 = time.time()
    try:
        if _foreign_live_heartbeat():
            return RunResult("busy", ui_copy.ERROR_ALREADY_RUNNING, None, -1)
        _write_heartbeat()
        from quant.data.registry_repair import repair_isins

        summary = repair_isins()
        message = (f"filled {summary['filled']} ISINs from market data, "
                   f"{summary['not_found']} not found.")
        return RunResult("ok", message, None, 0, time.time() - t0)
    except Exception:  # noqa: BLE001
        return RunResult("error", ui_copy.ERROR_REFRESH_FAILED, None, -1,
                         time.time() - t0)
    finally:
        _clear_heartbeat()
        _LOCK.release()


def run(command: str = REFRESH) -> RunResult:
    """Run a pipeline command under the mutex + heartbeat. Never raises.

    One acquisition covers the whole command (``all`` = update then run in one
    subprocess), so nested acquisition is impossible by construction.
    """
    if not _LOCK.acquire(blocking=False):
        # Same process already running: never the foreign-session sentence.
        return RunResult("busy", ui_copy.ERROR_RUNNING, None, -1)
    t0 = time.time()
    try:
        if _foreign_live_heartbeat():
            return RunResult("busy", ui_copy.ERROR_ALREADY_RUNNING, None, -1)
        _write_heartbeat()
        proc = subprocess.run(
            [sys.executable, "-m", "quant.cli", command],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        # Refresh at the update->run boundary / end of the sequence.
        _write_heartbeat()
        log_path = _latest_log_path()
        if proc.returncode == 0:
            return RunResult("ok", "", log_path, 0, time.time() - t0)
        return RunResult("error", ui_copy.ERROR_REFRESH_FAILED, log_path,
                         proc.returncode, time.time() - t0)
    except Exception:  # noqa: BLE001
        return RunResult("error", ui_copy.ERROR_REFRESH_FAILED, _latest_log_path(),
                         -1, time.time() - t0)
    finally:
        _clear_heartbeat()
        _LOCK.release()

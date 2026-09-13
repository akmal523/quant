"""
runner.py — Single in-process orchestrator for UI-triggered runs (v10.5.1, spec 4).

Intent: the app must not spawn overlapping refresh/review runs. One mutex
serializes them; a second tab gets a plain "already running" message. Runs
execute as ``sys.executable -m quant.cli <command>``, inherit the environment,
and stream to the run log; the UI shows a progress bar and one status line.

Invariants:
  - At most one run at a time per process (non-blocking lock).
  - Subprocess uses sys.executable and inherits the environment.
  - Returns a ``RunResult``; never raises.

Dependencies: subprocess, sys, threading, quant.ui.copy, quant.reporting.artifacts.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass

from quant.ui import copy as ui_copy

# One lock per process: serializes UI-triggered runs across tabs.
_LOCK = threading.Lock()

# UI command buttons -> CLI subcommands (P2: command names never reach the UI).
REFRESH = "update"
REVIEW = "run"
SAVE_AND_REVIEW = "all"


@dataclass
class RunResult:
    """Outcome of a UI-triggered run."""

    status: str            # "ok" | "error" | "busy"
    message: str
    log_path: str | None
    returncode: int

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def is_running() -> bool:
    """True if a run currently holds the orchestrator lock."""
    return _LOCK.locked()


def _latest_log_path() -> str | None:
    from quant.reporting.artifacts import latest_run

    run_dir = latest_run()
    if not run_dir:
        return None
    log = os.path.join(run_dir, "pipeline.log")
    return log if os.path.exists(log) else None


def run(command: str = REFRESH) -> RunResult:
    """Run a pipeline command under the mutex. Never raises.

    Returns status "busy" if another run is in progress, "ok" on exit 0, else
    "error" with the catalog remedy sentence.
    """
    if not _LOCK.acquire(blocking=False):
        return RunResult("busy", ui_copy.ERROR_ALREADY_RUNNING, None, -1)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "quant.cli", command],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        log_path = _latest_log_path()
        if proc.returncode == 0:
            return RunResult("ok", "", log_path, 0)
        return RunResult("error", ui_copy.ERROR_REFRESH_FAILED, log_path, proc.returncode)
    except Exception:  # noqa: BLE001
        return RunResult("error", ui_copy.ERROR_REFRESH_FAILED, _latest_log_path(), -1)
    finally:
        _LOCK.release()

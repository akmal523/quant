"""
artifacts.py — Run artifacts + structured logging.

Intent: each run writes a timestamped artifact directory (outputs/run_<ts>/)
with factor scores, NLP scores, run config, and metrics. Enables debugging why
a signal appeared. Also provides a structured JSON logger.
Invariants: artifact dir created idempotently; writes are atomic-ish (parquet).
Dependencies: pandas, polars, json, logging.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import pandas as pd

OUTPUTS_DIR = "outputs"


def new_run_dir() -> str:
    """Create and return outputs/run_<YYYY-MM-DD_HHMMSS>/."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(OUTPUTS_DIR, f"run_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


def latest_run_dir() -> str | None:
    """Return the most recent outputs/run_<ts>/ directory, or None.

    Intent (Phase 5 / v10.2): the dashboard sidebar shows the last run timestamp
    and reads the latest factor_scores.parquet from the newest artifact dir.
    Invariants: returns None if no run artifacts exist.
    """
    if not os.path.isdir(OUTPUTS_DIR):
        return None
    runs = [d for d in os.listdir(OUTPUTS_DIR) if d.startswith("run_")]
    if not runs:
        return None
    return os.path.join(OUTPUTS_DIR, max(runs))


def save_artifact(run_dir: str, name: str, df: pd.DataFrame) -> str:
    """Save a DataFrame to run_dir/<name>.parquet. Returns full path."""
    path = os.path.join(run_dir, f"{name}.parquet")
    df.to_parquet(path, index=False)
    return path


def save_config(run_dir: str, config: dict) -> str:
    """Save run configuration as JSON."""
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return path


def save_metrics(run_dir: str, metrics: dict) -> str:
    """Save run metrics as JSON."""
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


# ── Structured Logging ────────────────────────────────────────────────────────

class StructuredLogger:
    """JSON-structured logger wrapper. Emits one JSON object per record."""

    def __init__(self, name: str = "quant") -> None:
        self.logger = logging.getLogger(name)

    def _emit(self, level: int, event: str, **fields) -> None:
        record = {"event": event, **fields}
        self.logger.log(level, json.dumps(record, default=str))

    def info(self, event: str, **fields) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields) -> None:
        self._emit(logging.ERROR, event, **fields)


# Module-level singleton for convenience.
structured_log = StructuredLogger()
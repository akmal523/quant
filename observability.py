"""
observability.py — Structured Observability (Part 3, Gap #3).

Intent: collect timing, errors, and step status across the pipeline so you can
debug slow/failing steps. Wraps each pipeline stage in a timed context manager
and produces a human-readable summary plus a JSON export for the dashboard.

Invariants:
  - step() records duration_ms and status (success/error/skipped).
  - summary() returns a multi-line string.
  - to_json() returns a JSON-serializable dict.

Dependencies: time, contextlib, dataclasses.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class StepMetrics:
    name: str
    duration_ms: float = 0.0
    status: str = "success"
    symbols_processed: int = 0
    errors: list = field(default_factory=list)


class ObservabilityCollector:
    """Collects timing, errors, and data quality metrics across the pipeline."""

    def __init__(self):
        self.steps: list[dict] = []

    @contextmanager
    def step(self, name: str) -> Generator:
        """Time a pipeline step, recording success/error status."""
        step_start = time.perf_counter()
        record = {"name": name, "start": step_start}
        self.steps.append(record)
        try:
            yield
            record["status"] = "success"
        except Exception as e:  # noqa: BLE001
            record["status"] = "error"
            record["error"] = str(e)
            raise
        finally:
            record["duration_ms"] = (time.perf_counter() - step_start) * 1000

    def summary(self) -> str:
        """Generate a human-readable summary."""
        total_ms = sum(s.get("duration_ms", 0) for s in self.steps)
        errors = [s for s in self.steps if s.get("status") == "error"]

        lines = [
            f"Pipeline completed in {total_ms/1000:.1f}s",
            f"Steps: {len(self.steps)} total, {len(errors)} errors",
        ]

        sorted_steps = sorted(self.steps, key=lambda x: x.get("duration_ms", 0), reverse=True)
        lines.append("\nTop 5 slowest steps:")
        for s in sorted_steps[:5]:
            lines.append(f"  {s['name']}: {s.get('duration_ms', 0)/1000:.2f}s")

        if errors:
            lines.append(f"\n{len(errors)} errors:")
            for e in errors:
                lines.append(f"  {e['name']}: {e.get('error', 'unknown')}")

        return "\n".join(lines)

    def to_json(self) -> dict:
        """Export metrics for the dashboard."""
        return {
            "total_duration_ms": sum(s.get("duration_ms", 0) for s in self.steps),
            "steps": self.steps,
            "error_count": len([s for s in self.steps if s.get("status") == "error"]),
        }
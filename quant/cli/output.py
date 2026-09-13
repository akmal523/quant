"""
output.py — Terse CLI output + run log (v10.5.0, spec 3.1).

Intent: default CLI output contains only decision-relevant aggregate lines
(R4). Per-symbol and per-step detail goes to outputs/run_<ts>/pipeline.log and
is echoed to stdout only under --verbose. Parallel workers must not print
interleaved per-symbol lines; progress is a single rewritten carriage-return
line.

Invariants:
  - `line()` always prints (aggregate lines).
  - `detail()` writes the log file always; stdout only when verbose.
  - `progress()` rewrites one line via carriage return; `end_progress()` closes it.
  - No emoji, no exclamation marks (R5).

Dependencies: quant.reporting.artifacts (run dir), sys, os.
"""
from __future__ import annotations

import os
import sys

from quant.reporting.artifacts import new_run_dir


class Reporter:
    """Routes aggregate lines to stdout and detail lines to the run log."""

    def __init__(self, verbose: bool = False, log_path: str | None = None) -> None:
        self.verbose = verbose
        self.log_path = log_path
        self._log_fh = None
        self._progress_open = False
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self._log_fh = open(log_path, "a", encoding="utf-8")

    # ── Aggregate lines (always visible) ──────────────────────────────────────
    def line(self, msg: str = "") -> None:
        self._close_progress()
        print(msg)
        self._log(msg)

    # ── Detail lines (log always; stdout only when verbose) ───────────────────
    def detail(self, msg: str) -> None:
        self._log(msg)
        if self.verbose:
            self._close_progress()
            print(msg)

    # ── Single rewritten progress line ────────────────────────────────────────
    def progress(self, msg: str) -> None:
        if self.verbose:
            return
        sys.stdout.write("\r" + msg.ljust(60))
        sys.stdout.flush()
        self._progress_open = True

    def end_progress(self) -> None:
        self._close_progress()

    def _close_progress(self) -> None:
        if self._progress_open:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._progress_open = False

    def _log(self, msg: str) -> None:
        if self._log_fh:
            self._log_fh.write(msg + "\n")
            self._log_fh.flush()

    def close(self) -> None:
        self._close_progress()
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None


# ── Module-level singleton ────────────────────────────────────────────────────
# The CLI configures this once; pipeline modules import `reporter` and call
# reporter.detail(...) instead of print(...) for per-symbol lines.
reporter = Reporter()


def configure(verbose: bool = False, log_path: str | None = None) -> Reporter:
    """(Re)configure the global reporter IN PLACE. Called by the CLI entry point.

    Invariant: the module-level ``reporter`` object identity is preserved, so
    modules that did ``from quant.cli.output import reporter`` keep working.
    Also attaches a logging FileHandler so per-step ``logger.info`` lines land
    in the same pipeline.log (spec 3.1).
    """
    reporter.verbose = verbose
    reporter._close_progress()
    if reporter._log_fh:
        reporter._log_fh.close()
        reporter._log_fh = None
    reporter.log_path = log_path
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        reporter._log_fh = open(log_path, "a", encoding="utf-8")

        import logging
        root = logging.getLogger()
        # Drop any FileHandler from a previous configure() call.
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",
        ))
        root.addHandler(fh)
    return reporter


def default_log_path() -> str:
    """Return outputs/run_<ts>/pipeline.log for the current run dir."""
    return os.path.join(new_run_dir(), "pipeline.log")

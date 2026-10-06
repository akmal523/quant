#!/usr/bin/env python3
"""check_no_user_state.py — fail if user state is tracked in git (v10.8.0, 1.1).

Intent: the owner's real positions, cash, tiers, backups and database must never
be committed. This guard runs in pre-commit and CI. It reads ``git ls-files`` and
exits nonzero when a tracked path matches a user-state pattern.

Invariants: read-only; never modifies the index; exits 0 when git is unavailable
(so a non-git checkout does not break the hook).
"""
from __future__ import annotations

import re
import subprocess
import sys

# Paths that hold the owner's real state. Examples under data/examples/ are fine.
_PATTERNS = [
    re.compile(r"^data/portfolio\.csv$"),
    re.compile(r"^data/account\.yaml$"),
    re.compile(r"^data/tiers\.csv$"),
    re.compile(r"^data/backups/"),
    re.compile(r"^data/notify\.toml$"),
    re.compile(r".*\.duckdb$"),
    re.compile(r".*\.duckdb\.wal$"),
]


def tracked_user_state(paths: list[str]) -> list[str]:
    """Return the tracked paths that match a user-state pattern."""
    return [p for p in paths if any(rx.search(p) for rx in _PATTERNS)]


def main() -> int:
    try:
        out = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout
    except Exception:  # noqa: BLE001
        return 0
    bad = tracked_user_state(out.splitlines())
    if bad:
        print("User state must not be tracked in git:")
        for p in bad:
            print(f"  {p}")
        print("Run: git rm --cached <path>  (keeps the file on disk)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

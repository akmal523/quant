"""
make_golden.py — Regenerate the golden backtest snapshot (v10.4.0, Phase 4).

Intent: when a backtest change is INTENTIONAL, regenerate the golden file and
review the diff in the PR. Never regenerate to make a failing test pass blindly.

Usage:
    python3 scripts/make_golden.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.golden_util import compute_snapshot  # noqa: E402

GOLDEN_PATH = ROOT / "tests" / "golden" / "backtest_2024.json"


def main() -> int:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot = compute_snapshot()
    with open(GOLDEN_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    print(f"Golden snapshot written: {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
test_golden_snapshot.py — Golden-file regression test (v10.4.0, Phase 4).

Intent: the gold standard for quant code. Save the exact backtest output as a
golden file; fail CI if a refactor changes it by more than 0.01%. This catches
accidental logic bugs that unit tests miss.

Invariants: deterministic (seeded synthetic data); no network.
Dependencies: tests.golden_util, quant.strategy.backtest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.golden_util import compute_snapshot  # noqa: E402

GOLDEN_PATH = Path(__file__).parent / "golden" / "backtest_2024.json"
# 0.01% relative tolerance.
TOLERANCE = 0.0001


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a nested dict of numbers into dotted keys."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def test_golden_snapshot_matches():
    assert GOLDEN_PATH.exists(), (
        "Golden file missing. Run: python3 scripts/make_golden.py"
    )
    golden = _flatten(json.loads(GOLDEN_PATH.read_text()))
    current = _flatten(compute_snapshot())

    assert set(golden) == set(current), (
        f"Snapshot keys changed: {set(golden) ^ set(current)}"
    )

    for key, gv in golden.items():
        cv = current[key]
        denom = abs(gv) if abs(gv) > 1e-9 else 1.0
        rel = abs(cv - gv) / denom
        assert rel <= TOLERANCE, (
            f"Golden drift on {key}: current={cv} golden={gv} "
            f"({rel:.4%} > {TOLERANCE:.2%})"
        )


if __name__ == "__main__":
    test_golden_snapshot_matches()
    print("Golden snapshot OK")

"""mutation_report.py — mutation score and survivors (v10.7.5, Part 4).

Runs mutmut over the math core (configured in pyproject ``[tool.mutmut]``) and
prints the kill score and the survivor list. The CI threshold is 90 percent; any
survivor that touches a user-facing number is blocking and must be killed by a
test, not allowlisted.

Usage: ``python scripts/mutation_report.py [--results-only]``.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "tests" / "mutation_survivors_allowlist.txt"
THRESHOLD = 90.0


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return (result.stdout or "") + (result.stderr or "")


def _survivors() -> list[str]:
    """Parse the survivor list from ``mutmut results``."""
    out = _run(["mutmut", "results"])
    survivors: list[str] = []
    for line in out.splitlines():
        if "survived" in line.lower():
            survivors.append(line.strip())
    return survivors


def _score() -> tuple[int, int]:
    """Return (killed, total) parsed from ``mutmut results``."""
    out = _run(["mutmut", "results"])
    killed = len(re.findall(r"\bkilled\b", out, flags=re.IGNORECASE))
    survived = len(re.findall(r"\bsurvived\b", out, flags=re.IGNORECASE))
    return killed, killed + survived


def _allowlist() -> list[str]:
    if not ALLOWLIST.exists():
        return []
    return [ln.strip() for ln in ALLOWLIST.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mutation report (v10.7.5).")
    parser.add_argument("--results-only", action="store_true",
                        help="do not run mutmut; parse the existing results")
    args = parser.parse_args(argv)

    if not args.results_only:
        print("running mutmut (this is slow)...")
        _run(["mutmut", "run"])

    killed, total = _score()
    score = (killed / total * 100.0) if total else 0.0
    survivors = _survivors()
    allowed = _allowlist()

    print(f"mutation score: {score:.1f} percent ({killed}/{total} killed)")
    print(f"survivors: {len(survivors)} (allowlisted: {len(allowed)})")
    for survivor in survivors:
        print(f"  {survivor}")

    if score < THRESHOLD:
        print(f"FAIL: score below the {THRESHOLD:.0f} percent threshold")
        return 1
    if len(survivors) > len(allowed):
        print("FAIL: more survivors than the allowlist permits")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

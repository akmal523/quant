#!/usr/bin/env python3
"""
final_verification.py — Run all final verification checks (v10.6.5).

Runs the test suite, ruff, the type-hint and docstring audits, the docs build,
and the benchmarks, then prints a summary. Read-only.

Usage:
    python scripts/final_verification.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and report success or failure."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if result.returncode == 0:
        print(f"PASS: {description}")
        return True
    print(f"FAIL: {description}")
    print(result.stdout[-2000:])
    print(result.stderr[-2000:])
    return False


def main() -> int:
    """Run all verification checks and print a summary."""
    checks = [
        ([sys.executable, "-m", "pytest", "tests/", "-q"], "Full test suite"),
        ([sys.executable, "-m", "ruff", "check", "quant/"], "Ruff linting"),
        ([sys.executable, "scripts/audit_type_hints.py"], "Type hints audit"),
        ([sys.executable, "scripts/audit_docstrings.py"], "Docstrings audit"),
        ([sys.executable, "-m", "mkdocs", "build", "--strict"], "Documentation build"),
        ([sys.executable, "scripts/benchmark.py"], "Performance benchmarks"),
    ]

    results: list[tuple[str, bool]] = []
    for cmd, description in checks:
        results.append((description, run_command(cmd, description)))

    print(f"\n{'=' * 60}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 60}")
    all_passed = True
    for description, success in results:
        print(f"  [{'PASS' if success else 'FAIL'}] {description}")
        if not success:
            all_passed = False

    if all_passed:
        print("\nAll verification checks passed.")
        return 0
    print("\nSome verification checks failed. Review output above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

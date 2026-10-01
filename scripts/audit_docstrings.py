#!/usr/bin/env python3
"""
audit_docstrings.py — Audit docstring coverage for public functions (v10.6.5).

Scans all Python files under quant/ and reports public functions (not starting
with an underscore) that are missing a docstring or have a very short one.
Read-only.

Usage:
    python scripts/audit_docstrings.py
"""
from __future__ import annotations

import ast
from pathlib import Path

_MIN_LEN = 20


def _iter_public_functions(tree: ast.AST):
    """Yield module-level functions and class methods (skip nested helpers)."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield item


def check_file(file_path: Path) -> list[dict]:
    """Return missing-docstring issues for one file."""
    issues: list[dict] = []
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return issues

    for node in _iter_public_functions(tree):
        if node.name.startswith("_"):
            continue
        docstring = ast.get_docstring(node)
        if not docstring:
            issues.append({"file": str(file_path), "line": node.lineno,
                           "function": node.name, "issue": "missing docstring"})
        elif len(docstring) < _MIN_LEN:
            issues.append({"file": str(file_path), "line": node.lineno,
                           "function": node.name, "issue": "docstring too short"})
    return issues


def main() -> int:
    """Audit quant/ and report issues."""
    quant_dir = Path(__file__).resolve().parents[1] / "quant"
    all_issues: list[dict] = []
    for py_file in quant_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        all_issues.extend(check_file(py_file))

    if not all_issues:
        print("All public functions have docstrings.")
        return 0

    print(f"Found {len(all_issues)} missing or incomplete docstrings:\n")
    for issue in all_issues:
        print(f"  {issue['file']}:{issue['line']}  {issue['function']}: {issue['issue']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

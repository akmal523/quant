#!/usr/bin/env python3
"""
audit_type_hints.py — Audit type-hint coverage for public functions (v10.6.5).

Scans all Python files under quant/ and reports public functions (not starting
with an underscore) that are missing a return annotation or an argument
annotation. Read-only.

Usage:
    python scripts/audit_type_hints.py
"""
from __future__ import annotations

import ast
from pathlib import Path


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
    """Return missing-type-hint issues for one file."""
    issues: list[dict] = []
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, OSError):
        return issues

    for node in _iter_public_functions(tree):
        if node.name.startswith("_"):
            continue
        if node.returns is None:
            issues.append({"file": str(file_path), "line": node.lineno,
                           "function": node.name, "issue": "missing return type"})
        for arg in node.args.args:
            if arg.arg in ("self", "cls"):
                continue
            if arg.annotation is None:
                issues.append({"file": str(file_path), "line": node.lineno,
                               "function": node.name,
                               "issue": f"argument '{arg.arg}' missing type"})
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
        print("All public functions have complete type hints.")
        return 0

    print(f"Found {len(all_issues)} missing type hints:\n")
    for issue in all_issues:
        print(f"  {issue['file']}:{issue['line']}  {issue['function']}: {issue['issue']}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

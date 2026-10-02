"""test_audit_stubs.py — the static auditor must stay green (v10.7.5, Part 1).

Intent: make stubs impossible to ship. The auditor exits nonzero on any blocking
finding; this test runs it in-process and also enforces the Tier W ratchet (a
warn count may only shrink).

Invariants:
  - Zero blocking findings on the current tree.
  - No Tier W kind count exceeds tests/warn_baseline.txt.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_stubs", ROOT / "scripts" / "audit_stubs.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve the module via sys.modules during class creation.
    sys.modules["audit_stubs"] = module
    spec.loader.exec_module(module)
    return module


def test_no_blocking_findings():
    mod = _load_auditor()
    blocking = [f for f in mod.run() if f.blocking]
    assert not blocking, "\n".join(
        f"{f.file}:{f.line}: {f.kind}: {f.message}" for f in blocking)


def test_warn_counts_do_not_grow():
    mod = _load_auditor()
    counts = mod._warn_counts(mod.run())
    baseline = mod._read_baseline()
    for kind, count in counts.items():
        assert count <= baseline.get(kind, 0), (
            f"Tier W '{kind}' grew from {baseline.get(kind, 0)} to {count}; "
            "fix the fallback or register it in the fallback registry")

"""test_mutation_allowlist.py — the survivor allowlist may only shrink (v10.7.5).

Intent: mutation survivors are debt. The allowlist may only shrink; growing it
requires killing the mutant with a test (or, for a non-user-facing survivor,
lowering the baseline deliberately in review).

Invariants: the allowlist entry count never exceeds the recorded baseline.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "tests" / "mutation_survivors_allowlist.txt"

# Update ONLY when the allowlist shrinks (or a reviewed, non-user-facing survivor
# is added). Never raise this to hide a new survivor.
BASELINE = 0


def _entries() -> list[str]:
    if not ALLOWLIST.exists():
        return []
    return [ln.strip() for ln in ALLOWLIST.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def test_allowlist_only_shrinks():
    assert len(_entries()) <= BASELINE, (
        f"mutation survivor allowlist grew to {len(_entries())} (baseline {BASELINE}); "
        "kill the mutant with a test instead")

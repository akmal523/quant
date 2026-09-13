"""
test_identifiers.py — ISIN validation contract (v10.5.2, F1).

Test vectors come ONLY from repository files (data/broker_registry.csv) plus
constructed invalid values. No vector is recalled from memory (F1).
"""
from __future__ import annotations

import csv
from pathlib import Path

from quant.data.identifiers import is_valid_isin

REGISTRY = Path(__file__).resolve().parents[1] / "data" / "broker_registry.csv"


def _repo_isins() -> list[str]:
    with open(REGISTRY, newline="", encoding="utf-8") as f:
        return [r["isin"].strip() for r in csv.DictReader(f) if (r.get("isin") or "").strip()]


def test_accepts_every_isin_already_in_the_repo():
    values = _repo_isins()
    assert len(values) >= 20
    for v in values:
        assert is_valid_isin(v), f"repo ISIN rejected: {v}"


def test_rejects_wrong_check_digit():
    # IE00B4L5Y983 is a repo vector; flip the check digit.
    assert not is_valid_isin("IE00B4L5Y984")


def test_rejects_wrong_length():
    assert not is_valid_isin("IE00B4L5Y98")


def test_rejects_lowercase():
    assert not is_valid_isin("ie00b4l5y983")


def test_rejects_non_isin_strings():
    assert not is_valid_isin("")
    assert not is_valid_isin(None)
    assert not is_valid_isin("not-an-isin")

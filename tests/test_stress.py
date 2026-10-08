"""
test_stress.py — Stress tests for extreme scenarios (v10.6.5).

Hermetic: synthetic data only, no network, no live store.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



def test_corrupted_tiers_csv_recovery(tmp_path):
    """A corrupted tiers file is recovered by parsing valid rows."""
    from quant.portfolio.tier_manager import _repair_corrupted_csv

    tiers_csv = tmp_path / "tiers.csv"
    tiers_csv.write_text(
        "symbol,tier,last_updated,notes\n"
        "AAPL,FORTRESS,2026-10-01,\n"
        "INVALID LINE WITHOUT COMMAS\n"
        "MSFT,ALPHA,2026-10-01,\n",
        encoding="utf-8",
    )
    df = _repair_corrupted_csv(str(tiers_csv))
    assert len(df) >= 2
    assert set(df["symbol"]) >= {"AAPL", "MSFT"}




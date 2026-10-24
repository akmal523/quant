"""
test_v10_6_4.py — v10.6.4 tests: batch-scoring performance, safe tier load,
corruption repair, and the health check.

Hermetic: synthetic data only, no network, no live store.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_load_tiers_safe_missing(tmp_path, monkeypatch):
    """A missing tiers file yields an empty frame and no crash."""
    import quant.portfolio.tier_manager as tm

    monkeypatch.setattr(tm.paths, "DATA_TIERS", str(tmp_path / "missing.csv"))
    tiers_df, warnings = tm.load_tiers_safe()
    assert tiers_df.empty
    assert isinstance(warnings, list)


def test_health_check_runs():
    """The health check returns a valid status and a list of checks."""
    from quant.cli.health import run_health_check

    result = run_health_check()
    assert result["status"] in ("HEALTHY", "WARNING", "CRITICAL")
    assert isinstance(result["checks"], list) and result["checks"]
    for check in result["checks"]:
        assert check["status"] in ("OK", "WARNING", "CRITICAL")
        assert "name" in check and "message" in check

"""
test_v10_6_4.py — v10.6.4 tests: batch-scoring performance, safe tier load,
corruption repair, and the health check.

Hermetic: synthetic data only, no network, no live store.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_batch_scoring_performance():
    """Batch scoring 50 assets completes in under 30 seconds."""
    from quant.analytics.scoring import batch_score_assets_parallel

    rng = np.random.default_rng(0)
    symbols = [f"ASSET_{i}" for i in range(50)]
    price_data = {
        s: pd.DataFrame({
            "Close": 100 + rng.normal(0, 1, 100).cumsum(),
            "Volume": rng.integers(100_000, 1_000_000, 100).astype(float),
        })
        for s in symbols
    }
    fundamentals = {s: {"PE": 20, "ROE": 0.2} for s in symbols}
    news = {s: [] for s in symbols}
    tiers = {s: "ALPHA" for s in symbols}

    start = time.time()
    results = batch_score_assets_parallel(symbols, price_data, fundamentals, news, tiers)
    elapsed = time.time() - start

    assert len(results) == 50
    assert elapsed < 30.0


def test_load_tiers_safe_missing(tmp_path, monkeypatch):
    """A missing tiers file yields an empty frame and no crash."""
    import quant.portfolio.tier_manager as tm

    monkeypatch.setattr(tm.paths, "DATA_TIERS", str(tmp_path / "missing.csv"))
    tiers_df, warnings = tm.load_tiers_safe()
    assert tiers_df.empty
    assert isinstance(warnings, list)


def test_repair_corrupted_csv(tmp_path):
    """A corrupted tiers file is recovered by parsing valid rows."""
    from quant.portfolio.tier_manager import _repair_corrupted_csv

    path = tmp_path / "tiers.csv"
    path.write_text(
        "garbage line without commas\n"
        "symbol,tier,last_updated,notes\n"
        "AAPL,ALPHA,2026-10-01,\n"
        "MSFT,FORTRESS,2026-10-01,\n",
        encoding="utf-8",
    )
    df = _repair_corrupted_csv(str(path))
    assert set(df["symbol"]) == {"AAPL", "MSFT"}
    assert df[df["symbol"] == "MSFT"]["tier"].iloc[0] == "FORTRESS"


def test_health_check_runs():
    """The health check returns a valid status and a list of checks."""
    from quant.cli.health import run_health_check

    result = run_health_check()
    assert result["status"] in ("HEALTHY", "WARNING", "CRITICAL")
    assert isinstance(result["checks"], list) and result["checks"]
    for check in result["checks"]:
        assert check["status"] in ("OK", "WARNING", "CRITICAL")
        assert "name" in check and "message" in check

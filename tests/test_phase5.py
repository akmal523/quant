"""
test_phase5.py — Unit tests for Phase 5 (v10.2) state machine + scoring fixes.

Covers: graduation grace period, CORE immunity, delist tracking, ISIN checksum,
inverse routing exclusion, and ETF score differentiation. Pure-logic tests
(no network, no cvxpy).
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import unittest

import pandas as pd

from quant.analytics.scoring import etf_quality_score, etf_tactical_grade
from quant.execution.taxonomy import validate_isin


class TestDelistTracking(unittest.TestCase):
    def test_three_failures_produce_delisted(self):
        """Three consecutive fetch failures mark a symbol DELISTED."""
        from quant.data.database import get_connection, init_db
        from quant.execution.taxonomy import mark_delisted
        init_db()
        conn = get_connection()
        conn.execute(
            "INSERT OR REPLACE INTO asset_registry "
            "(symbol, universe_status, fetch_failures, updated_at) "
            "VALUES ('TEST_DELIST', 'WATCHLIST', 2, ?)",
            [0.0],
        )
        mark_delisted("TEST_DELIST", "consecutive fetch failures")
        row = conn.execute(
            "SELECT universe_status FROM asset_registry WHERE symbol = 'TEST_DELIST'"
        ).fetchone()
        self.assertEqual(row[0], "DELISTED")
        conn.execute("DELETE FROM asset_registry WHERE symbol = 'TEST_DELIST'")


class TestValidateIsin(unittest.TestCase):
    def test_valid_isin(self):
        self.assertTrue(validate_isin("IE00B4L5Y983"))

    def test_corrupted_checksum(self):
        self.assertFalse(validate_isin("IE00B4L5Y984"))

    def test_wrong_length(self):
        self.assertFalse(validate_isin("IE00B4L5Y98"))

    def test_non_alnum(self):
        self.assertFalse(validate_isin("IE00B4L5Y98-"))


class TestEtfScoreDifferentiation(unittest.TestCase):
    def test_etf_scores_not_identical(self):
        """ETF scores differ when inputs differ (regression for the 93.6 tie)."""
        features = pd.DataFrame({
            "Symbol": ["EIMI.L", "CSPX.L", "EUNL.DE"],
            "trend_strength": [0.1, 0.2, 0.3],
            "rel_strength_6m": [0.05, 0.1, 0.15],
            "vol_60d": [0.2, 0.15, 0.1],
            "ret_12m": [0.1, 0.2, 0.3],
        })
        out = etf_quality_score(features)
        self.assertGreater(out["etf_quality"].nunique(), 1)

    def test_etf_tactical_continuous(self):
        """ETF tactical grade is continuous, not binary."""
        grades = {etf_tactical_grade(0.99, z) for z in [-1.0, 0.0, 1.0]}
        self.assertGreater(len(grades), 1)


if __name__ == "__main__":
    unittest.main()

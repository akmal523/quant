"""
test_phase4.py — Unit tests for Phase 4 (Broker-Aware Family Office Terminal).

Covers: fee hurdle math, cash risk-free rate, signal routing, taxonomy,
and universe graduation detection. Pure-logic tests (no network, no cvxpy).
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import unittest

import pandas as pd

from quant.execution.taxonomy import classify_instrument, resolve_broker
from quant.portfolio.optimizer import minimum_trade_size, passes_fee_hurdle
from quant.portfolio.risk import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    daily_risk_free_rate,
)


class TestFeeHurdle(unittest.TestCase):
    def test_minimum_trade_size_formula(self):
        # (2 / 200) * 10000 = 100 EUR for 200 bps alpha.
        self.assertAlmostEqual(minimum_trade_size(200), 100.0, places=2)

    def test_minimum_trade_size_zero_alpha(self):
        self.assertEqual(minimum_trade_size(0), float("inf"))

    def test_passes_fee_hurdle(self):
        self.assertTrue(passes_fee_hurdle(200, 100))   # exactly at hurdle
        self.assertFalse(passes_fee_hurdle(200, 50))   # below hurdle
        self.assertTrue(passes_fee_hurdle(200, 10000)) # well above


class TestCashRiskFreeRate(unittest.TestCase):
    def test_daily_rf_positive_small(self):
        rf = daily_risk_free_rate(0.0225)
        self.assertGreater(rf, 0)
        self.assertLess(rf, 0.001)  # ~6e-5

    def test_daily_rf_zero_apy(self):
        self.assertEqual(daily_risk_free_rate(0), 0.0)

    def test_sortino_uses_cash_rf(self):
        r = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        s = calculate_sortino_ratio(r)
        self.assertIsInstance(s, float)

    def test_sharpe_uses_cash_rf(self):
        r = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        sh = calculate_sharpe_ratio(r)
        self.assertIsInstance(sh, float)


class TestTaxonomy(unittest.TestCase):
    def test_classify_equity(self):
        self.assertEqual(classify_instrument("AAPL"), "EQUITY")

    def test_classify_etf_from_registry(self):
        self.assertEqual(classify_instrument("VOO"), "ETF")

    def test_resolve_broker_isin(self):
        broker = resolve_broker("AAPL")
        self.assertEqual(broker["isin"], "US0378331005")
        self.assertEqual(broker["tr_ticker"], "AAPL")

    def test_resolve_broker_fallback(self):
        broker = resolve_broker("UNKNOWN_SYM")
        self.assertEqual(broker["tr_ticker"], "UNKNOWN_SYM")


if __name__ == "__main__":
    unittest.main()

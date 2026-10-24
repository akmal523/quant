"""
test_phase4.py — Unit tests for Phase 4 (Broker-Aware Family Office Terminal).

Covers: fee hurdle math, cash risk-free rate, signal routing, taxonomy,
and universe graduation detection. Pure-logic tests (no network, no cvxpy).
"""
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import unittest

from quant.execution.taxonomy import classify_instrument, resolve_broker
from quant.portfolio.risk import (
    daily_risk_free_rate,
)


class TestCashRiskFreeRate(unittest.TestCase):
    def test_daily_rf_positive_small(self):
        rf = daily_risk_free_rate(0.0225)
        self.assertGreater(rf, 0)
        self.assertLess(rf, 0.001)  # ~6e-5

    def test_daily_rf_zero_apy(self):
        self.assertEqual(daily_risk_free_rate(0), 0.0)


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

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

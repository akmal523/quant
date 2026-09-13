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

from quant.portfolio.risk import daily_risk_free_rate, calculate_sortino_ratio, calculate_sharpe_ratio
from quant.portfolio.optimizer import minimum_trade_size, passes_fee_hurdle
from quant.execution.routing import route_signal, build_execution_instruction
from quant.execution.taxonomy import classify_instrument, resolve_broker
from quant.execution.discovery import detect_graduation


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


class TestSignalRouting(unittest.TestCase):
    def test_etf_routes_sparplan(self):
        self.assertEqual(route_signal(85, 50, "ETF"), "SPARPLAN")

    def test_high_tactical_routes_active(self):
        self.assertEqual(route_signal(60, 80, "EQUITY"), "ACTIVE")

    def test_high_structural_routes_sparplan(self):
        self.assertEqual(route_signal(85, 55, "EQUITY"), "SPARPLAN")

    def test_low_grades_hold(self):
        self.assertEqual(route_signal(40, 40, "EQUITY"), "HOLD")

    def test_execution_instruction_sparplan(self):
        inst = build_execution_instruction("VUSA.DE", "SPARPLAN", 100, 50, 100,
                                           "IE00B5BMR087", "VUSA.DE")
        self.assertEqual(inst["action"], "SPARPLAN")
        self.assertIn("0 EUR buy fee", inst["instruction"])


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


class TestDiscovery(unittest.TestCase):
    def _make_df(self, closes, volumes):
        idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
        return pd.DataFrame({"Close": closes, "Volume": volumes}, index=idx)

    def test_52w_high_graduates(self):
        closes = list(range(100, 200))  # monotonic rising -> new high
        vols = [1000] * 100
        df = self._make_df(closes, vols)
        ok, reason = detect_graduation("X", df)
        self.assertTrue(ok)
        self.assertIn("52-week high", reason)

    def test_volume_anomaly_graduates(self):
        closes = [100] * 100
        vols = [1000] * 99 + [5000]  # 5x the 20d avg
        df = self._make_df(closes, vols)
        ok, reason = detect_graduation("X", df)
        self.assertTrue(ok)
        self.assertIn("volume", reason)

    def test_no_anomaly(self):
        closes = [100] * 100
        vols = [1000] * 100
        df = self._make_df(closes, vols)
        ok, _ = detect_graduation("X", df)
        self.assertFalse(ok)

    def test_insufficient_data(self):
        df = self._make_df([100, 101], [1000, 1000])
        ok, reason = detect_graduation("X", df)
        self.assertFalse(ok)
        self.assertIn("insufficient", reason)


if __name__ == "__main__":
    unittest.main()
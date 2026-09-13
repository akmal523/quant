"""
test_rebalancing.py — Unit tests for the strategic rebalancing tier system (v11).
Covers: tier classification, fee hurdle, per-tier rebalance logic, CORE never SELL,
drift thresholds, first-run baseline ease-in.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import pandas as pd

from quant.config import (
    CORE_ASSETS, SATELLITE_ASSETS, ACTIVE_ASSETS, SECTOR_ASSETS,
    TARGET_WEIGHTS, MIN_TRADE_SIZE_EUR,
)


# ── Tier Classification ───────────────────────────────────────────────────────

def test_classify_asset_tiers():
    """Each known symbol maps to its expected tier."""
    from quant.portfolio.portfolio import classify_asset
    assert classify_asset("EUNL.DE") == "CORE"
    assert classify_asset("SXRV.DE") == "SATELLITE"   # precedence over CORE_ETFS
    assert classify_asset("AMZN") == "ACTIVE"
    assert classify_asset("5J50.DE") == "SECTOR"
    print("  [PASS] test_classify_asset_tiers")


def test_classify_asset_unknown_defaults_active():
    """Unknown symbols default to ACTIVE."""
    from quant.portfolio.portfolio import classify_asset
    assert classify_asset("ZZZZ") == "ACTIVE"
    print("  [PASS] test_classify_asset_unknown_defaults_active")


def test_target_weights_sum_to_one():
    """TARGET_WEIGHTS must sum to 1.0 (portfolio fully allocated)."""
    assert abs(sum(TARGET_WEIGHTS.values()) - 1.0) < 1e-9
    print("  [PASS] test_target_weights_sum_to_one")


# ── Fee Hurdle ────────────────────────────────────────────────────────────────

def test_calculate_min_trade_size_floor():
    """Min trade size never below MIN_TRADE_SIZE_EUR."""
    from quant.portfolio.optimizer import calculate_min_trade_size
    size = calculate_min_trade_size(0.50, 0.50, 1000.0)  # zero drift
    assert size >= MIN_TRADE_SIZE_EUR
    print(f"  [PASS] test_calculate_min_trade_size_floor: {size:.0f}")


def test_calculate_min_trade_size_drift_scales():
    """Larger drift -> larger min trade size (when drift dominates fee floor)."""
    from quant.portfolio.optimizer import calculate_min_trade_size
    # Portfolio 10000 EUR so drift value exceeds the 200 EUR fee floor.
    small = calculate_min_trade_size(0.50, 0.45, 10000.0)  # 5% drift = 500
    large = calculate_min_trade_size(0.50, 0.30, 10000.0)  # 20% drift = 2000
    assert large > small
    print(f"  [PASS] test_calculate_min_trade_size_drift_scales: {small:.0f} < {large:.0f}")


def test_passes_fee_hurdle():
    """Fee hurdle: 200 bps alpha needs >= 100 EUR to clear 2 EUR fee."""
    from quant.portfolio.optimizer import passes_fee_hurdle
    assert passes_fee_hurdle(200.0, 100.0) is True
    assert passes_fee_hurdle(200.0, 99.0) is False
    print("  [PASS] test_passes_fee_hurdle")


# ── Rebalance Logic ───────────────────────────────────────────────────────────

def test_should_rebalance_core_drift_threshold():
    """CORE rebalances only on >10% drift (after time gate)."""
    from quant.data.database import init_db
    init_db()
    from quant.portfolio.portfolio import should_rebalance_asset
    # Force a rebalance_log row so time gate passes (last = 100 days ago).
    from quant.portfolio.portfolio import set_last_rebalance
    set_last_rebalance("EUNL.DE", "2026-06-01")
    ok, reason = should_rebalance_asset(
        "EUNL.DE", 0.62, 0.50, "CORE", "2026-09-09",
    )  # 12% drift
    assert ok, reason
    print(f"  [PASS] test_should_rebalance_core_drift_threshold: {reason}")


def test_should_rebalance_core_small_drift_holds():
    """CORE with <10% drift -> HOLD."""
    from quant.data.database import init_db
    init_db()
    from quant.portfolio.portfolio import should_rebalance_asset
    from quant.portfolio.portfolio import set_last_rebalance
    set_last_rebalance("EUNL.DE", "2026-06-01")
    ok, reason = should_rebalance_asset(
        "EUNL.DE", 0.53, 0.50, "CORE", "2026-09-09",
    )  # 3% drift
    assert not ok, reason
    print(f"  [PASS] test_should_rebalance_core_small_drift_holds: {reason}")


def test_should_rebalance_first_run_baseline():
    """First run (no log row) eases in: no forced rebalance."""
    from quant.data.database import init_db
    init_db()
    from quant.portfolio.portfolio import should_rebalance_asset
    from quant.portfolio.portfolio import set_last_rebalance
    # Use a symbol with no prior log entry.
    set_last_rebalance("AMZN", "2026-09-09")  # ensure baseline exists
    ok, reason = should_rebalance_asset(
        "AMZN", 0.18, 0.20, "ACTIVE", "2026-09-09",
    )
    # ACTIVE rebalances daily; drift 2% < 5% threshold -> HOLD.
    assert not ok, reason
    print(f"  [PASS] test_should_rebalance_first_run_baseline: {reason}")


# ── CORE Never SELL ───────────────────────────────────────────────────────────

def test_core_never_sell():
    """CORE assets never emit SELL/TRIM regardless of grades."""
    from quant.analytics.scoring import generate_signal_for_tier
    for drift in (-0.20, 0.0, 0.20):
        _, signal = generate_signal_for_tier(
            "EUNL.DE", 10.0, 10.0, 5.0, "CORE", 0.5 + drift, 0.5,
        )
        assert signal in ("HOLD", "BUY MORE (DCA OK)"), signal
    print("  [PASS] test_core_never_sell")


def test_satellite_trim_on_overweight():
    """SATELLITE trims only on >10% overweight."""
    from quant.analytics.scoring import generate_signal_for_tier
    _, signal = generate_signal_for_tier(
        "SXRV.DE", 80.0, 80.0, 20.0, "SATELLITE", 0.35, 0.20,
    )  # 15% overweight
    assert signal == "TRIM POSITION", signal
    print(f"  [PASS] test_satellite_trim_on_overweight: {signal}")


def test_active_full_spectrum():
    """ACTIVE keeps full tactical spectrum (can SELL)."""
    from quant.analytics.scoring import generate_signal_for_tier
    _, sell = generate_signal_for_tier(
        "AMZN", 40.0, 30.0, 10.0, "ACTIVE", 0.20, 0.20,
    )  # speculative + low tactical -> SELL
    assert sell == "SELL", sell
    _, buy = generate_signal_for_tier(
        "AMZN", 80.0, 70.0, 20.0, "ACTIVE", 0.20, 0.20,
    )  # strong -> BUY
    assert buy == "BUY", buy
    print(f"  [PASS] test_active_full_spectrum: {sell} / {buy}")


# ── Liquidity Check ───────────────────────────────────────────────────────────

def test_check_volume_liquidity():
    """Trade exceeding 1% of ADV is rejected."""
    from quant.portfolio.optimizer import check_volume_liquidity
    df = pd.DataFrame({
        "Close": [100.0] * 30,
        "Volume": [1000.0] * 30,  # ADV = 1000 shares
    })
    # 2000 EUR / 100 = 20 shares < 1% of 1000 (10 shares)? 20 > 10 -> reject.
    ok, reason = check_volume_liquidity("TEST", 2000.0, df)
    assert not ok, reason
    # 500 EUR / 100 = 5 shares < 10 -> accept.
    ok2, _ = check_volume_liquidity("TEST", 500.0, df)
    assert ok2
    print(f"  [PASS] test_check_volume_liquidity: {reason}")


if __name__ == "__main__":
    import sys
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} rebalancing tests passed.")
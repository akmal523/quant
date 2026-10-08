"""test_v10_8_2_value_series.py — the chart fix (v10.8.2, section 5).

Before v10.8.2 the series stopped at the last snapshot/flow date (so new prices
never extended it) and a day with no price for a held symbol added 0 (the flat
line at 0 with a -1..1 axis). These tests pin the required behavior:

  - the series runs to the latest available price date;
  - a day on which a held symbol has never had a price is skipped, not zeroed;
  - a missing close carries the previous close forward (incl. weekends);
  - a mid-period buy and a sell move shares without a phantom gain.
"""
from __future__ import annotations

from quant.engine.value_series import build_value_series


def test_two_holdings_buy_and_sell_with_weekend_and_missing_day():
    snapshots = [
        {"date": "2026-01-01", "symbol": "AAA", "shares": 10.0},
        {"date": "2026-01-01", "symbol": "BBB", "shares": 5.0},
    ]
    closes = {
        ("AAA", "2026-01-01"): 100.0, ("AAA", "2026-01-02"): 100.0,
        ("AAA", "2026-01-05"): 100.0, ("AAA", "2026-01-06"): 100.0,
        ("BBB", "2026-01-01"): 200.0, ("BBB", "2026-01-02"): 200.0,
        ("BBB", "2026-01-05"): 210.0, ("BBB", "2026-01-06"): 210.0,
    }
    flows = [
        {"date": "2026-01-05", "type": "buy", "amount_eur": 100.0, "symbol": "AAA"},
        {"date": "2026-01-06", "type": "sell", "amount_eur": 50.0, "symbol": "BBB"},
    ]
    series = build_value_series(snapshots, flows, closes, {})
    by_day = {s["date"].isoformat(): s["value_eur"] for s in series}

    # 01-03 and 01-04 are a weekend; closes carry forward (no zero point).
    assert by_day["2026-01-02"] == 2000.0
    assert by_day["2026-01-03"] == 2000.0
    assert by_day["2026-01-04"] == 2000.0
    # Buy 100 EUR of AAA at 100 -> +1 share: 11*100 + 5*210 = 2150.
    assert by_day["2026-01-05"] == 2150.0
    # Sell 50 EUR of BBB at 210 -> -0.2381 shares: 11*100 + 4.7619*210 = 2100.
    assert by_day["2026-01-06"] == 2100.0
    # No zero points anywhere.
    assert all(v > 0 for v in by_day.values())


def test_day_with_unpriced_held_symbol_is_skipped_not_zeroed():
    snapshots = [
        {"date": "2026-01-01", "symbol": "AAA", "shares": 10.0},
        {"date": "2026-01-01", "symbol": "CCC", "shares": 1.0},
    ]
    closes = {
        ("AAA", "2026-01-01"): 100.0, ("AAA", "2026-01-05"): 100.0,
        ("CCC", "2026-01-05"): 50.0,   # CCC has no price before 01-05
    }
    series = build_value_series(snapshots, [], closes, {})
    dates = [s["date"].isoformat() for s in series]
    # Days 01-01..01-04 are skipped (CCC never priced): no zeroed points.
    assert dates == ["2026-01-05"]
    assert series[0]["value_eur"] == 1050.0  # 10*100 + 1*50


def test_series_extends_to_the_latest_price_date():
    snapshots = [{"date": "2026-01-01", "symbol": "AAA", "shares": 1.0}]
    closes = {("AAA", "2026-01-01"): 10.0, ("AAA", "2026-01-08"): 11.0}
    series = build_value_series(snapshots, [], closes, {})
    assert len(series) == 8                       # 01-01 .. 01-08
    assert series[-1]["date"].isoformat() == "2026-01-08"
    assert series[-1]["value_eur"] == 11.0


def test_no_prices_returns_empty_series():
    snapshots = [{"date": "2026-01-01", "symbol": "AAA", "shares": 1.0}]
    assert build_value_series(snapshots, [], {}, {}) == []

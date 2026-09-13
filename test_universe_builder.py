"""
test_universe_builder.py — Unit tests for the broad universe loader (Plan 3, Phase 1).

Covers the pure symbol-extraction helpers. Network-dependent index loading is
not exercised here (needs live Wikipedia access).
"""
from __future__ import annotations

import pandas as pd


def test_normalize():
    """Symbols are uppercased and whitespace-stripped."""
    from universe_builder import _normalize
    assert _normalize("  amzn ") == "AMZN"
    assert _normalize("eunl.de") == "EUNL.DE"
    assert _normalize("") == ""
    print("  [PASS] test_normalize")


def test_extract_symbols_from_table():
    """Ticker column is detected regardless of exact header casing."""
    from universe_builder import _extract_symbols_from_table
    df = pd.DataFrame({"Ticker symbol": ["AAPL", "MSFT", "NVDA"]})
    assert _extract_symbols_from_table(df) == ["AAPL", "MSFT", "NVDA"]
    df2 = pd.DataFrame({"Company": ["Apple"], "Symbol": ["aapl"]})
    assert _extract_symbols_from_table(df2) == ["AAPL"]
    print("  [PASS] test_extract_symbols_from_table")


def test_extract_symbols_skips_empty():
    """Empty/NaN symbols are dropped; no ticker column returns []."""
    from universe_builder import _extract_symbols_from_table
    df = pd.DataFrame({"Ticker symbol": ["AAPL", None, "", "MSFT"]})
    assert _extract_symbols_from_table(df) == ["AAPL", "MSFT"]
    df2 = pd.DataFrame({"Company": ["Apple"]})  # no ticker column
    assert _extract_symbols_from_table(df2) == []
    print("  [PASS] test_extract_symbols_skips_empty")


def test_broad_etfs_present():
    """The curated broad-ETF list is non-empty and includes core ETFs."""
    from universe_builder import BROAD_ETFS
    assert len(BROAD_ETFS) > 20
    assert "EUNL.DE" in BROAD_ETFS.values()
    assert "IWDA.AS" in BROAD_ETFS.values()
    print(f"  [PASS] test_broad_etfs_present: {len(BROAD_ETFS)} ETFs")


if __name__ == "__main__":
    test_normalize()
    test_extract_symbols_from_table()
    test_extract_symbols_skips_empty()
    test_broad_etfs_present()
    print("\nAll universe_builder tests passed.")
"""
test_universe_builder.py — Unit tests for the broad universe loader (Plan 3, Phase 1).

Covers the pure symbol-extraction helpers. Network-dependent index loading is
not exercised here (needs live Wikipedia access).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


def test_normalize():
    """Symbols are uppercased and whitespace-stripped."""
    from quant.data.universe_builder import _normalize
    assert _normalize("  amzn ") == "AMZN"
    assert _normalize("eunl.de") == "EUNL.DE"
    assert _normalize("") == ""
    print("  [PASS] test_normalize")


def test_broad_etfs_present():
    """The curated broad-ETF list is non-empty and includes core ETFs."""
    from quant.data.universe_builder import BROAD_ETFS
    assert len(BROAD_ETFS) > 20
    assert "EUNL.DE" in BROAD_ETFS.values()
    assert "IWDA.AS" in BROAD_ETFS.values()
    print(f"  [PASS] test_broad_etfs_present: {len(BROAD_ETFS)} ETFs")

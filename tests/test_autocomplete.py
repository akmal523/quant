"""
test_autocomplete.py — Instrument search contract (v10.5.1, P3d).

Asserts: "amazon" resolves to AMZN; "world" offers EUNL.DE and SWRD.L; an ISIN
paste resolves exactly.
"""
from __future__ import annotations

from quant.ui.search import search


def _index():
    return [
        {"symbol": "AMZN", "name": "Amazon.com Inc", "isin": "US0231351067",
         "keywords": ["amazon"]},
        {"symbol": "EUNL.DE", "name": "MSCI World Xetra (EUNL)", "isin": "IE00B4L5Y983",
         "keywords": ["world"]},
        {"symbol": "SWRD.L", "name": "MSCI World (SWRD)", "isin": "IE00BFY0GT14",
         "keywords": ["world"]},
        {"symbol": "GLD", "name": "Gold Shares (GLD)", "isin": "US78463V1070",
         "keywords": ["gold"]},
    ]


def test_amazon_resolves_to_amzn():
    results = search(_index(), "amazon")
    assert results
    assert results[0]["symbol"] == "AMZN"


def test_world_offers_eunl_and_swrd():
    results = search(_index(), "world", limit=10)
    symbols = {r["symbol"] for r in results}
    assert "EUNL.DE" in symbols
    assert "SWRD.L" in symbols


def test_isin_paste_resolves_exactly():
    results = search(_index(), "US0231351067")
    assert results
    assert results[0]["symbol"] == "AMZN"
    assert results[0]["score"] == 100.0


def test_exact_symbol_ranks_first():
    results = search(_index(), "GLD")
    assert results[0]["symbol"] == "GLD"


def test_empty_query_returns_nothing():
    assert search(_index(), "") == []

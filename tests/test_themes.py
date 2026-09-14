"""
test_themes.py — Theme search + zero-match copy (H3.4).

Themes are prose tags (data/themes.csv), not identifiers (F1 does not apply).
"space" resolves through "aerospace"/"defence"; "samsung" is a zero match.
"""
from __future__ import annotations

from quant.ui import copy as C
from quant.ui.search import (
    load_index,
    load_themes,
    resolve_theme_symbols,
    search,
)


def test_themes_file_loads_and_resolves():
    themes = load_themes()
    assert "gold" in themes and "space" in themes
    resolved = resolve_theme_symbols(themes)
    assert "5J50.DE" in resolved["space"]          # space -> aerospace -> 5J50.DE
    assert "DFEN" in resolved["space"]             # space -> defence


def test_space_offers_aerospace_rows():
    symbols = {r["symbol"] for r in search(load_index(), "space", 10)}
    assert {"5J50.DE", "DFEN"} <= symbols


def test_gold_still_matches_the_family():
    symbols = {r["symbol"] for r in search(load_index(), "gold", 10)}
    assert {"GLD", "GDX", "GDXJ"} <= symbols


def test_samsung_is_a_zero_match():
    assert search(load_index(), "samsung", 10) == []


def test_empty_query_returns_nothing():
    assert search(load_index(), "", 10) == []


def test_zero_match_sentence_is_exact():
    assert C.EMPTY_NO_MATCHES.format(query="samsung") == (
        "No instrument matches samsung. Try a company or fund name, a symbol, "
        "an ISIN, or a theme such as gold or defence.")

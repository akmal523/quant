"""
search.py — Instrument search index (v10.5.3, R5).

Intent: resolve a typed query to instruments by display name, name, symbol, ISIN,
or keyword tag. "amazon" -> AMZN, "gold" -> the gold family, "world" -> the world
ETFs. Labels use the "Name (TICKER)" form with duplicate-paren removal.

Invariants:
  - search(records, query, limit) ranks exact symbol/ISIN first.
  - label_for never doubles the parens: "Name (TICKER)" is appended once.
  - Pure over an in-memory list; load_index() reads via a read-only connection.

Dependencies: quant.data.universe_builder (seed), quant.data.database (read path).
"""
from __future__ import annotations

import os
import re

from quant import paths
from quant.data.names import clean_display_name
from quant.data.universe_builder import BROAD_ETFS

THEMES_PATH = paths.DATA_THEMES

# Curated keyword tags per symbol (themes that do not appear in the name).
TAG_KEYWORDS: dict[str, list[str]] = {
    "GLD": ["gold"], "SLV": ["silver"], "GDX": ["gold", "miners"],
    "GDXJ": ["gold", "miners"], "SGLN.L": ["silver"],
    "URTH": ["world"], "IWDA.AS": ["world"], "EUNL.DE": ["world"],
    "SWRD.L": ["world"], "IQQQ.DE": ["water"], "GLUG.L": ["water", "clean water"],
    "PHO": ["water"],
}


def label_for(name: str, symbol: str) -> str:
    """Return the canonical "Name (TICKER)" label with duplicate-paren removal."""
    # B2: never trust the registry value is pre-cleaned; clean at render time.
    name = clean_display_name((name or "").strip(), symbol).strip()
    if not name or name.upper() == (symbol or "").upper():
        return symbol
    if re.search(rf"\({re.escape(symbol)}\)$", name, re.IGNORECASE):
        return name
    return f"{name} ({symbol})"


def load_themes(path: str | None = None) -> dict[str, list[str]]:
    """Load data/themes.csv -> {theme: [tokens]}. Tokens are symbols or themes.

    Themes are prose tags (H3.4), not identifiers (F1 does not apply). A token
    that names another theme links to it (theme-to-theme). The symbols column is
    an UNQUOTED comma list (per the file format), so parse on the FIRST comma
    (pandas would mis-split the 2-column layout). Best-effort.
    """
    path = path or THEMES_PATH
    if not os.path.exists(path):
        return {}
    out: dict[str, list[str]] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "," not in line:
                    continue
                theme, rest = line.split(",", 1)
                theme = theme.strip().lower()
                if not theme or theme == "theme":      # skip the header row
                    continue
                toks = [t.strip() for t in rest.split(",") if t.strip()]
                if toks:
                    out[theme] = toks
    except Exception:  # noqa: BLE001
        return {}
    return out


def resolve_theme_symbols(themes: dict[str, list[str]]) -> dict[str, set[str]]:
    """Resolve each theme to its symbol set (recursive; cycle-safe)."""
    cache: dict[str, set[str]] = {}

    def _resolve(name: str, seen: frozenset[str]) -> set[str]:
        if name in cache:
            return cache[name]
        if name in seen:
            return set()
        out: set[str] = set()
        for tok in themes.get(name, []):
            low = tok.strip().lower()
            if low in themes:
                out |= _resolve(low, seen | {name})
            else:
                out.add(tok.strip().upper())
        cache[name] = out
        return out

    for name in themes:
        _resolve(name, frozenset())
    return cache


def symbol_themes(themes: dict[str, list[str]]) -> dict[str, set[str]]:
    """Inverse map: symbol -> set of theme names that (transitively) include it."""
    inv: dict[str, set[str]] = {}
    for theme, syms in resolve_theme_symbols(themes).items():
        for s in syms:
            inv.setdefault(s.upper(), set()).add(theme)
    return inv


def _record(symbol: str, name: str = "", isin: str = "", keywords=None,
            display_name: str = "", themes=None) -> dict:
    nm = (name or symbol or "").strip()
    kws = [k.lower() for k in (keywords or [])]
    th = [t.lower() for t in (themes or [])]
    return {
        "symbol": (symbol or "").strip().upper(),
        "name": nm,
        "display_name": (display_name or nm).strip(),
        "isin": (isin or "").strip().upper(),
        "keywords": kws + th,
        "themes": th,
    }


def _static_seed(sym_themes: dict[str, set[str]] | None = None) -> list[dict]:
    """Seed records from the curated broad-ETF list (always available)."""
    sym_themes = sym_themes or {}
    return [_record(sym, name, keywords=TAG_KEYWORDS.get(sym),
                    themes=sym_themes.get(sym.upper()))
            for name, sym in BROAD_ETFS.items()]


def search(records: list[dict], query: str, limit: int = 10) -> list[dict]:
    """Rank records against a query. Returns the top ``limit`` matches."""
    q = (query or "").strip().lower()
    if not q:
        return []

    scored: list[tuple[float, dict]] = []
    for r in records:
        sym = r["symbol"].lower()
        nm = r["name"].lower()
        dn = str(r.get("display_name") or r["name"]).lower()
        isin = r["isin"].lower()
        score = 0.0
        if q == sym or (isin and q == isin):
            score = 100.0
        elif q in dn or q in nm:
            score = 60.0
        elif q in sym:
            score = 50.0
        else:
            for kw in r.get("keywords", []):
                if q in kw:
                    score = 40.0
                    break
        if score == 0.0:
            tokens = [t for t in q.replace("-", " ").split() if t]
            text = f"{sym} {nm} {dn} {' '.join(r.get('keywords', []))}"
            if tokens and all(t in text for t in tokens):
                score = 20.0
        if score > 0:
            label = label_for(r.get("display_name") or r["name"], r["symbol"])
            scored.append((score, {**r, "score": score, "label": label}))

    scored.sort(key=lambda x: (-x[0], x[1]["symbol"]))
    return [rec for _, rec in scored[:limit]]


def load_index() -> list[dict]:
    """Build the search index: static ETF seed + asset_registry + themes.

    Corpus = display_name + name + symbol + ISIN + themes, all substring and
    case-insensitive (H3.4).
    """
    try:
        sym_themes = symbol_themes(load_themes())
    except Exception:  # noqa: BLE001
        sym_themes = {}
    records = {r["symbol"]: r for r in _static_seed(sym_themes)}
    try:
        from quant.data.database import read_only_connection

        with read_only_connection() as conn:
            rows = conn.execute(
                "SELECT symbol, COALESCE(display_name, name) AS nm, isin "
                "FROM asset_registry"
            ).fetchall()
        for sym, nm, isin in rows:
            s = (sym or "").strip().upper()
            if not s:
                continue
            records[s] = _record(s, nm or s, isin or "", TAG_KEYWORDS.get(s),
                                 nm or s, sym_themes.get(s))
    except Exception:  # noqa: BLE001
        pass
    return list(records.values())


def resolve(query: str, limit: int = 10) -> list[dict]:
    """Convenience: search the live index."""
    return search(load_index(), query, limit)

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

import re

from quant.data.universe_builder import BROAD_ETFS

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
    name = (name or "").strip()
    if not name or name.upper() == (symbol or "").upper():
        return symbol
    if re.search(rf"\({re.escape(symbol)}\)$", name, re.IGNORECASE):
        return name
    return f"{name} ({symbol})"


def _record(symbol: str, name: str = "", isin: str = "", keywords=None,
            display_name: str = "") -> dict:
    nm = (name or symbol or "").strip()
    return {
        "symbol": (symbol or "").strip().upper(),
        "name": nm,
        "display_name": (display_name or nm).strip(),
        "isin": (isin or "").strip().upper(),
        "keywords": [k.lower() for k in (keywords or [])],
    }


def _static_seed() -> list[dict]:
    """Seed records from the curated broad-ETF list (always available)."""
    return [_record(sym, name, keywords=TAG_KEYWORDS.get(sym))
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
    """Build the search index from asset_registry (+ static ETF seed)."""
    records = {r["symbol"]: r for r in _static_seed()}
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
            records[s] = _record(s, nm or s, isin or "", TAG_KEYWORDS.get(s), nm or s)
    except Exception:  # noqa: BLE001
        pass
    return list(records.values())


def resolve(query: str, limit: int = 10) -> list[dict]:
    """Convenience: search the live index."""
    return search(load_index(), query, limit)

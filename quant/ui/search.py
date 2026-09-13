"""
search.py — Instrument search index (v10.5.1, spec 5.3).

Intent: resolve a typed query to instruments by name, symbol, ISIN, or keyword.
"amazon" -> AMZN, "world" -> EUNL.DE / SWRD.L, an ISIN paste resolves exactly.

Invariants:
  - search(records, query, limit) ranks exact symbol/ISIN matches first.
  - Name matches outrank symbol substring matches, which outrank keywords.
  - Pure over an in-memory list; load_index() reads via a read-only connection.

Dependencies: quant.data.database (read path only).
"""
from __future__ import annotations

from quant.data.universe_builder import BROAD_ETFS


def _record(symbol: str, name: str = "", isin: str = "", keywords=None) -> dict:
    return {
        "symbol": (symbol or "").strip().upper(),
        "name": (name or symbol or "").strip(),
        "isin": (isin or "").strip().upper(),
        "keywords": [k.lower() for k in (keywords or [])],
    }


def _static_seed() -> list[dict]:
    """Seed records from the curated broad-ETF list (always available)."""
    return [_record(sym, name) for name, sym in BROAD_ETFS.items()]


def search(records: list[dict], query: str, limit: int = 10) -> list[dict]:
    """Rank records against a query. Returns the top ``limit`` matches.

    Each result adds a ``score`` and a ``label`` ("Name (SYMBOL)").
    """
    q = (query or "").strip().lower()
    if not q:
        return []

    scored: list[tuple[float, dict]] = []
    for r in records:
        sym = r["symbol"].lower()
        name = r["name"].lower()
        isin = r["isin"].lower()
        score = 0.0
        if q == sym or (isin and q == isin):
            score = 100.0
        elif q in name:
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
            text = f"{sym} {name} {' '.join(r.get('keywords', []))}"
            if tokens and all(t in text for t in tokens):
                score = 20.0
        if score > 0:
            label = f"{r['name']} ({r['symbol']})" if r["name"] else r["symbol"]
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
                "SELECT symbol, name, isin FROM asset_registry"
            ).fetchall()
        for sym, name, isin in rows:
            s = (sym or "").strip().upper()
            if not s:
                continue
            records[s] = _record(s, name or s, isin or "")
    except Exception:  # noqa: BLE001
        pass
    return list(records.values())


def resolve(query: str, limit: int = 10) -> list[dict]:
    """Convenience: search the live index."""
    return search(load_index(), query, limit)

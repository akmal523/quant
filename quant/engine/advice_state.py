"""
advice_state.py — the stateful advice log (v10.8.4, T5).

Intent: advice is a persisted object, not a pure re-computation. Before v10.8.4
the same single sell advice repeated verbatim every day forever, with no record
of how long it had been open. This module reconciles the freshly computed advice
against the ``advice_log`` table:

  - an item not yet open is inserted with ``open_since = today``;
  - an item already open keeps its original ``open_since`` (so the UI can say
    "open since <date>");
  - an item no longer produced is closed (status ``done``), so the list stops
    repeating it and History shows the advice log.

Identity is ``(verb, symbol, horizon)``: the same recommendation for the same
thing in the same time bucket is one open object.

Only the writers (daily job, confirm, refresh, via
:func:`quant.engine.today_list.persist`) call :func:`sync`; the render path calls
the read-only :func:`annotate` (v10.8.10, R-C4).

Invariants:
  - ``sync`` never raises; on failure it returns the items unchanged.
  - Only ``open`` rows are returned by :func:`open_items`; a done row disappears
    from the active list but stays in History.
"""
from __future__ import annotations

from datetime import date
from typing import Any

STATUS_OPEN = "open"
STATUS_DONE = "done"

_KEY_FIELDS = ("verb", "symbol", "horizon")


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _key(item: dict) -> tuple:
    return tuple(str(item.get(f) or "") for f in _KEY_FIELDS)


def sync(conn: Any, items: list[dict] | None, as_of: date | None = None) -> list[dict]:
    """Persist ``items`` and return them enriched with ``id``/``open_since``.

    ``items`` are the actionable records (each has verb, symbol, horizon,
    amount_eur, reason, class). Items no longer present are closed.
    """
    as_of = as_of or date.today()
    items = list(items or [])
    if conn is None:
        return items
    try:
        existing = conn.execute(
            "SELECT id, verb, symbol, horizon, open_since FROM advice_log "
            "WHERE status = ?", [STATUS_OPEN]).fetchall()
    except Exception:  # noqa: BLE001
        return items
    open_by_key: dict = {}
    for rid, verb, symbol, horizon, open_since in existing:
        open_by_key[(str(verb or ""), str(symbol or ""), str(horizon or ""))] = (
            int(rid), open_since)

    seen: set = set()
    out: list[dict] = []
    for item in items:
        k = _key(item)
        seen.add(k)
        found = open_by_key.get(k)
        if found is None:
            rid = _insert(conn, item, as_of)
            enriched = dict(item)
            enriched["id"] = rid
            enriched["open_since"] = as_of
            out.append(enriched)
        else:
            rid, open_since = found
            try:
                conn.execute(
                    "UPDATE advice_log SET amount_eur = ?, reason = ?, class = ? "
                    "WHERE id = ?",
                    [item.get("amount_eur"), str(item.get("reason") or ""),
                     str(item.get("class") or ""), rid])
            except Exception:  # noqa: BLE001
                pass
            enriched = dict(item)
            enriched["id"] = rid
            enriched["open_since"] = open_since or as_of
            out.append(enriched)

    # Close the items no longer produced.
    for k, (rid, _open_since) in open_by_key.items():
        if k not in seen:
            try:
                conn.execute(
                    "UPDATE advice_log SET status = ? WHERE id = ?",
                    [STATUS_DONE, rid])
            except Exception:  # noqa: BLE001
                pass
    return out


def annotate(items: list[dict] | None, as_of: date | None = None,
             conn: Any = None) -> list[dict]:
    """Read-only twin of :func:`sync` for the render path (v10.8.10, R-C4).

    Enriches each item with the ``id`` / ``open_since`` of its open log row; an
    item the writers have not persisted yet gets ``id=None`` and
    ``open_since=as_of``. Never writes, never raises.
    """
    as_of = as_of or date.today()
    open_by_key = {_key(r): r for r in open_items(conn)}
    out: list[dict] = []
    for item in items or []:
        found = open_by_key.get(_key(item))
        enriched = dict(item)
        enriched["id"] = found["id"] if found else None
        enriched["open_since"] = (found.get("open_since") if found else None) or as_of
        out.append(enriched)
    return out


def mark_item_done(conn: Any, item: dict, as_of: date | None = None) -> bool:
    """Mark one item done, logging it first if the writers have not yet (user click)."""
    if conn is None:
        return False
    rid = item.get("id")
    if rid is None:
        found = {_key(r): r for r in open_items(conn)}.get(_key(item))
        rid = found["id"] if found else _insert(conn, item, as_of or date.today())
    return mark_done(conn, rid)


def _insert(conn: Any, item: dict, as_of: date) -> int | None:
    try:
        row = conn.execute(
            "INSERT INTO advice_log (verb, symbol, amount_eur, reason, "
            "horizon, class, created_at, open_since, status) "
            "VALUES (?, ?, ?, ?, ?, ?, now(), ?, ?) RETURNING id",
            [str(item.get("verb") or ""), str(item.get("symbol") or ""),
             item.get("amount_eur"), str(item.get("reason") or ""),
             str(item.get("horizon") or ""), str(item.get("class") or ""),
             as_of, STATUS_OPEN]).fetchone()
        return int(row[0]) if row else None
    except Exception:  # noqa: BLE001
        return None


def mark_done(conn: Any, advice_id: int | None) -> bool:
    """Mark one advice done (it stops repeating; History keeps it)."""
    if conn is None or advice_id is None:
        return False
    try:
        conn.execute("UPDATE advice_log SET status = ? WHERE id = ?",
                     [STATUS_DONE, int(advice_id)])
        return True
    except Exception:  # noqa: BLE001
        return False


def open_items(conn: Any = None) -> list[dict]:
    """The currently open advice rows."""
    rows = _query(conn, "SELECT id, verb, symbol, amount_eur, reason, horizon, "
                        "class, open_since FROM advice_log WHERE status = ? "
                        "ORDER BY open_since", [STATUS_OPEN])
    return [{"id": int(r[0]), "verb": r[1], "symbol": r[2], "amount_eur": r[3],
             "reason": r[4], "horizon": r[5], "class": r[6], "open_since": r[7]}
            for r in rows]


def history(conn: Any = None, limit: int = 200) -> list[dict]:
    """The advice log for the History page (newest first)."""
    rows = _query(conn, "SELECT id, verb, symbol, amount_eur, reason, horizon, "
                        "class, created_at, open_since, status FROM advice_log "
                        "ORDER BY created_at DESC LIMIT ?", [int(limit)])
    return [{"id": int(r[0]), "verb": r[1], "symbol": r[2], "amount_eur": r[3],
             "reason": r[4], "horizon": r[5], "class": r[6], "created_at": r[7],
             "open_since": r[8], "status": r[9]} for r in rows]


def _query(conn: Any, sql: str, params: list | None = None) -> list:
    try:
        if conn is not None:
            return conn.execute(sql, params or []).fetchall()
        from quant.data.database import read_only_connection

        with read_only_connection() as c:
            return c.execute(sql, params or []).fetchall()
    except Exception:  # noqa: BLE001
        return []

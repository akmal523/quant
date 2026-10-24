"""
mtm.py — the daily mark-to-market valuation store (v10.8.4, T1; v10.8.7, R-C1).

Intent: the product must show ONE value per datum. Before v10.8.4 a holding's
value came from three places (the broker CSV, the ``holdings_meta`` estimate, a
review artifact), so the chart lagged, the header delta was unlabeled and the
holdings table disagreed with itself. This module is the single writer and reader
of the ``valuation`` / ``portfolio_total`` stores:

  valuation(date, symbol, shares, price_eur, fx, value_eur, source)
  portfolio_total(date, value_eur, cash_eur)

It is written by the daily job, by confirm & save, and by the on-demand refresh;
it is idempotent per (date, symbol). Shares are reconstructed from the ledger
(recorded buys/sells plus corporate-action split factors) as of each date.

v10.8.7 (R-C1): uninvested cash is NOT an entity. ``portfolio_total`` is the sum of
invested position values only. The ``cash_eur`` column is frozen audit data: it is
written NULL and never read (the physical column drop is deferred). The flows->cash
computation is deleted; the ledger flows remain for audit only.

Invariants:
  - Pure helpers (reconstruct_shares) do no I/O when the rows are passed in; the
    I/O wrappers take an explicit connection.
  - Never raises on a missing price: the symbol is skipped (its last stored value
    is untouched), never zeroed.
  - value_eur = shares * price_eur * fx, rounded to 2 decimals.
"""
from __future__ import annotations

import bisect
import logging
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)

SOURCE_DAILY = "daily"
SOURCE_CONFIRM = "confirm"
SOURCE_REFRESH = "refresh"
SOURCE_BACKFILL = "backfill"


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Ledger reconstruction (pure over rows) ────────────────────────────────────

def reconstruct_from_rows(opening: dict, trades: list, splits: list,
                          as_of: date) -> dict[str, float]:
    """Reconstruct shares per symbol as of ``as_of``, in CURRENT share units. Pure.

    Prices in ``market_history`` are split-adjusted to today's share basis (Yahoo
    ``auto_adjust`` + a full refetch whenever a split appears), so every lot and
    trade is restated to today's units: its shares are multiplied by the factor
    of each split that took effect AFTER the lot/trade date. Shares x adjusted
    close is then the right value on every day, before and after a split.

    Args:
        opening: ``{symbol: (base_date, base_shares)}`` — the opening lot (the
            broker's synced position). A trade dated on or before ``base_date`` is
            already reflected and is ignored.
        trades: ``[(trade_date, symbol, action, shares)]`` (buy/sell).
        splits: ``[(effective_date, symbol, factor)]`` (factor 2.0 for a 2:1).
        as_of: the valuation date.

    Returns ``{symbol: shares}`` (positive shares only).
    """
    later: dict[str, list[tuple[date, float]]] = {}
    for row in splits or []:
        eff, sym, factor = (list(row) + [None, None, None])[:3]
        ed = _as_date(eff)
        f = _num(factor, 1.0)
        if ed is not None and f > 0:
            later.setdefault(str(sym), []).append((ed, f))

    def _restate(sym: str, when: date | None, shares: float) -> float:
        for ed, f in later.get(sym, []):
            if when is None or ed > when:
                shares *= f
        return shares

    out: dict[str, tuple[date | None, float]] = {}
    for sym, base in (opening or {}).items():
        base_date, base_shares = base
        bd = _as_date(base_date)
        out[str(sym)] = (bd, _restate(str(sym), bd, _num(base_shares)))

    for row in trades or []:
        t_date, sym, action, shares = (list(row) + [None, None, None, None])[:4]
        td = _as_date(t_date)
        if td is None or td > as_of:
            continue
        sym = str(sym)
        signed = _num(shares)
        if str(action).lower() == "sell":
            signed = -signed
        elif str(action).lower() != "buy":
            continue
        base_date, current = out.get(sym, (None, 0.0))
        # A trade on or before the opening lot's date is already in that lot.
        if base_date is not None and td <= base_date:
            continue
        out[sym] = (base_date, current + _restate(sym, td, signed))

    return {sym: round(v, 6) for sym, (_d, v) in out.items() if v > 0}


# ── Ledger reads (explicit connection) ────────────────────────────────────────

def _opening_lots(conn: Any) -> dict:
    """The opening lot per symbol: {symbol: (sync_date, shares)}.

    Source: ``holdings_meta`` (the broker CSV sync). A symbol with no row has no
    opening lot (its shares come solely from recorded trades).
    """
    out: dict = {}
    try:
        rows = conn.execute(
            "SELECT symbol, shares, sync_date FROM holdings_meta").fetchall()
    except Exception:  # noqa: BLE001
        return out
    for sym, shares, sync_date in rows:
        out[str(sym)] = (sync_date, _num(shares))
    return out


def _trade_rows(conn: Any) -> list:
    """Recorded trades: [(date, symbol, action, shares)]."""
    try:
        return conn.execute(
            "SELECT date, symbol, action, shares FROM trades ORDER BY date").fetchall()
    except Exception:  # noqa: BLE001
        return []


def _split_rows(conn: Any) -> list:
    """Corporate-action split factors: [(effective_date, symbol, factor)].

    Reads the ``splits`` table when present; empty otherwise (a DB predating it).
    """
    try:
        return conn.execute(
            "SELECT effective_date, symbol, factor FROM splits ORDER BY effective_date"
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []


# ── Price / FX resolution ─────────────────────────────────────────────────────


def _fx_rate(symbol: str) -> float:
    try:
        from quant.data.currency import get_fx_to_eur

        return float(get_fx_to_eur(symbol))
    except Exception:  # noqa: BLE001
        return 1.0


# ── Per-trading-day backfill (v10.8.5, T2/D3/D7) ──────────────────────────────

def _price_history(conn: Any, symbols: set) -> dict:
    """{symbol: [(date, native_close), ...] sorted} for the given symbols."""
    if not symbols:
        return {}
    try:
        placeholders = ",".join("?" for _ in symbols)
        rows = conn.execute(
            f"SELECT Symbol, Date, Close FROM market_history "
            f"WHERE Symbol IN ({placeholders}) AND Close IS NOT NULL",
            list(symbols)).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    out: dict = {}
    for sym, day, close in rows:
        d = _as_date(day)
        if d is None or close is None:
            continue
        out.setdefault(str(sym), []).append((d, float(close)))
    for sym in out:
        out[sym].sort(key=lambda t: t[0])
    return out


def _close_on_or_before(series: list, day: date) -> float | None:
    """The close on ``day`` or the most recent before it (None when none yet)."""
    if not series:
        return None
    idx = bisect.bisect_right(series, (day, float("inf"))) - 1
    return series[idx][1] if idx >= 0 else None


def _first_held_day(opening: dict, trades: list) -> date | None:
    """The earliest day the portfolio held anything."""
    days = [_as_date(b[0]) for b in opening.values() if b and b[0] is not None]
    days += [_as_date(t[0]) for t in trades if t and t[0] is not None]
    days = [d for d in days if d is not None]
    return min(days) if days else None


def backfill(conn: Any, price_lookup: Any = None, fx_lookup: Any = None,
             source: str = SOURCE_BACKFILL, start: date | None = None,
             end: date | None = None) -> dict:
    """Write a valuation row for EVERY trading day the portfolio was held (T2).

    Shares are reconstructed from the ledger as of each day; value = shares *
    close * fx. v10.8.7 (R-C1): cash is not tracked. The series is continuous (one
    point per trading day, no forward-filled total) and its tail equals the latest
    close. Idempotent per (date, symbol). Returns ``{days, rows}``. Never raises.
    """
    opening = _opening_lots(conn)
    trades = _trade_rows(conn)
    splits = _split_rows(conn)
    symbols = set(opening) | {str(t[1]) for t in trades if len(t) > 1 and t[1]}
    if not symbols:
        return {"days": 0, "rows": 0}

    hist = _price_history(conn, symbols)
    days = sorted({d for s in hist.values() for d, _c in s})
    if not days and callable(price_lookup):
        base = _first_held_day(opening, trades) or date.today()
        days = [base]
    if not days:
        return {"days": 0, "rows": 0}

    start = _as_date(start) or _first_held_day(opening, trades)
    end = _as_date(end) or days[-1]
    rows = 0
    day_count = 0
    for d in days:
        if start and d < start:
            continue
        if end and d > end:
            continue
        shares = reconstruct_from_rows(opening, trades, splits, d)
        total = 0.0
        for sym, sh in shares.items():
            if callable(price_lookup):
                price = price_lookup(sym)
            else:
                price = _close_on_or_before(hist.get(sym, []), d)
            if price is None:
                continue
            fx = float(fx_lookup(sym)) if callable(fx_lookup) else _fx_rate(sym)
            fx = fx or 1.0
            value = round(sh * float(price) * fx, 2)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO valuation "
                    "(date, symbol, shares, price_eur, fx, value_eur, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [d, sym, sh, float(price), fx, value, str(source)])
                rows += 1
                total += value
            except Exception:  # noqa: BLE001
                continue
        try:
            conn.execute(
                "INSERT OR REPLACE INTO portfolio_total (date, value_eur, cash_eur) "
                "VALUES (?, ?, NULL)", [d, round(total, 2)])
            day_count += 1
        except Exception:  # noqa: BLE001
            pass
    return {"days": day_count, "rows": rows}


def refresh(as_of: date | None = None, source: str = SOURCE_REFRESH) -> dict:
    """Open a write connection and backfill the valuation store. Summary."""
    from quant.data.database import write_connection

    as_of = as_of or date.today()
    try:
        with write_connection() as conn:
            res = backfill(conn, source=source)
        return {"ok": True, "date": as_of, "rows": res["rows"],
                "days": res["days"]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "date": as_of, "rows": 0, "days": 0, "error": str(e)}


# ── Readers ───────────────────────────────────────────────────────────────────

def read_total_series(conn: Any = None) -> list[dict]:
    """The daily portfolio-total series oldest first (invested value only).

    Returns ``[{date, value_eur}]``; empty when nothing is stored. v10.8.7 (R-C1):
    ``cash_eur`` is frozen audit data and is never read.
    """
    rows = _query(conn, "SELECT date, value_eur FROM portfolio_total "
                        "ORDER BY date")
    return [{"date": r[0], "value_eur": _num(r[1])} for r in rows]


def read_series(symbol: str | None = None, start: Any = None, end: Any = None,
                conn: Any = None) -> list[dict]:
    """Valuation rows from the store (v10.8.6, B2).

    Without ``symbol``: the daily portfolio totals ``[{date, value_eur}]``
    (invested only; v10.8.7, R-C1). With ``symbol``: the per-symbol valuation rows
    ``[{date, symbol, shares, price_eur, fx, value_eur}]`` within ``[start, end]``.
    Only stored dates are returned (no forward-fill). Empty when nothing is stored.
    """
    if symbol is None:
        return read_total_series(conn)
    sql = ("SELECT date, symbol, shares, price_eur, fx, value_eur FROM valuation "
           "WHERE symbol = ?")
    params: list = [str(symbol)]
    if start is not None:
        sql += " AND date >= ?"
        params.append(_as_date(start))
    if end is not None:
        sql += " AND date <= ?"
        params.append(_as_date(end))
    sql += " ORDER BY date"
    rows = _query(conn, sql, params)
    return [{"date": r[0], "symbol": str(r[1]), "shares": _num(r[2]),
             "price_eur": _num(r[3]), "fx": _num(r[4]), "value_eur": _num(r[5])}
            for r in rows]


def held_symbols(as_of: Any = None, conn: Any = None) -> list[str]:
    """The symbols held on the latest stored valuation date <= ``as_of`` (B3)."""
    target = _as_date(as_of) or date.today()
    rows = _query(conn, "SELECT MAX(date) FROM valuation WHERE date <= ?", [target])
    if not rows or rows[0][0] is None:
        return []
    d = _as_date(rows[0][0])
    srows = _query(conn, "SELECT DISTINCT symbol FROM valuation WHERE date = ? "
                         "ORDER BY symbol", [d])
    return [str(r[0]) for r in srows]


def latest_total(conn: Any = None) -> dict:
    """The most recent ``portfolio_total`` row (invested value only), or {}."""
    rows = _query(conn, "SELECT date, value_eur FROM portfolio_total "
                        "ORDER BY date DESC LIMIT 1")
    if not rows:
        return {}
    r = rows[0]
    return {"date": r[0], "value_eur": _num(r[1])}


def day_delta(conn: Any = None) -> tuple[float, float, date | None]:
    """The last day's change: ``(eur, pct, latest_date)`` from stored totals.

    The home header delta is exactly ``total(D) - total(D-1)`` (invested value plus
    cash). It is computed by the ONE pure helper
    :func:`quant.reporting.artifacts.compute_day_delta`, so the number the header
    shows and the number the invariant test checks can never drift (v10.8.6, B6).
    Returns ``(0.0, 0.0, latest_date_or_None)`` when fewer than two days are stored.
    """
    series = read_total_series(conn)
    if not series:
        return 0.0, 0.0, None
    latest_d = _as_date(series[-1]["date"])
    try:
        from quant.reporting.artifacts import compute_day_delta

        eur, pct = compute_day_delta(series)
    except Exception:  # noqa: BLE001
        eur, pct = 0.0, 0.0
    return eur, pct, latest_d


# ── Startup catch-up (v10.8.6, B1/B2) ─────────────────────────────────────────

def _mark_valuation_stale(error: Any) -> None:
    """One log line + mark the ``valuation`` stage stale in run_manifest."""
    logger.warning("valuation catch-up failed: %s", error)
    try:
        from quant.data.database import connect_with_retry
        from quant.engine import manifest

        conn = connect_with_retry(attempts=1)
        try:
            run = manifest.latest_run_id(conn) or manifest.new_run_id()
            manifest.record_stage(conn, run, "valuation", manifest.STATUS_STALE,
                                  error=str(error))
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass


def catch_up(retries: int = 3, base_delay: float = 0.5) -> dict:
    """Catch the valuation store up to the latest price (v10.8.6, B1/B2).

    Runs once per process start (the caller guards it); never inside a rerun. When
    the latest valuation date is behind the latest ``market_history`` date, backfill
    with bounded retry on lock contention. On final failure: one log line + mark the
    ``valuation`` stage ``stale`` in ``run_manifest``. Never raises.
    """
    import time

    from quant.data.database import connect_with_retry

    last_err: Exception | None = None
    tries = max(1, int(retries))
    for attempt in range(tries):
        conn = None
        try:
            conn = connect_with_retry(attempts=1)
            price = conn.execute("SELECT MAX(Date) FROM market_history").fetchall()
            latest_price = _as_date(price[0][0]) if price and price[0][0] else None
            val = conn.execute("SELECT MAX(date) FROM valuation").fetchall()
            latest_val = _as_date(val[0][0]) if val and val[0][0] else None
            if latest_price is None:
                return {"ok": True, "skipped": True, "reason": "no prices"}
            if latest_val is not None and latest_val >= latest_price:
                return {"ok": True, "skipped": True, "latest": str(latest_val)}
            res = backfill(conn, source=SOURCE_REFRESH, end=latest_price)
            return {"ok": True, "rows": res.get("rows", 0),
                    "days": res.get("days", 0), "latest": str(latest_price)}
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < tries - 1:
                time.sleep(base_delay * (2 ** attempt))
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
    _mark_valuation_stale(last_err)
    return {"ok": False, "error": str(last_err)}


def latest_price_date(conn: Any = None) -> date | None:
    """The most recent stored valuation date, or None."""
    rows = _query(conn, "SELECT MAX(date) FROM valuation")
    if not rows or rows[0][0] is None:
        return None
    return _as_date(rows[0][0])


def read_symbol_series(conn: Any = None) -> dict:
    """The per-symbol value series from the valuation store.

    Returns ``{symbol: [{date, value_eur}, ...]}`` oldest first, for the Growth
    chart's one-line-per-held-asset view. Empty when nothing is stored.
    """
    rows = _query(conn, "SELECT date, symbol, value_eur FROM valuation "
                        "ORDER BY date")
    out: dict = {}
    for d, sym, v in rows:
        out.setdefault(str(sym), []).append(
            {"date": _as_date(d), "value_eur": _num(v)})
    return out


def _query(conn: Any, sql: str, params: list | None = None) -> list:
    """Run a read query on ``conn`` or a short-lived read-only connection."""
    try:
        if conn is not None:
            return conn.execute(sql, params or []).fetchall()
        from quant.data.database import read_only_connection

        with read_only_connection() as c:
            return c.execute(sql, params or []).fetchall()
    except Exception:  # noqa: BLE001
        return []

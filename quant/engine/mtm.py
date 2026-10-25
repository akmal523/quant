"""
mtm.py — the daily mark-to-market valuation store (v10.8.4, T1; v10.8.7, R-C1).

Intent: the product must show ONE value per datum. Before v10.8.4 a holding's
value came from three places (the broker CSV, the ``holdings_meta`` estimate, a
review artifact), so the chart lagged, the header delta was unlabeled and the
holdings table disagreed with itself. This module is the single writer and reader
of the ``valuation`` / ``portfolio_total`` stores:

  valuation(date, symbol, shares, price_eur, fx, value_eur, source)
  portfolio_total(date, value_eur, cash_eur)

It is written by the daily job, by confirm & save, and by the on-demand refresh,
and rebuilt from scratch each time. Shares on a day are the latest saved
holdings table on or before it (position_snapshots), restated to today's units
by later stock splits; value = shares x close x FX.

v10.8.7 (R-C1): uninvested cash is NOT an entity. ``portfolio_total`` is the sum of
invested position values only. The ``cash_eur`` column is frozen audit data: it is
written NULL and never read (the physical column drop is deferred). The flows->cash
computation is deleted; the ledger flows remain for audit only.

Invariants:
  - Pure helpers (restate, shares_on) do no I/O; the I/O wrappers take an
    explicit connection.
  - A missing price skips the symbol for that day, never zeroes it.
  - backfill lets a database error propagate; its callers (confirm, the daily
    job, refresh) catch it, and the rebuild is atomic per call.
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

# The SQL lives at module level: keyword and column case cannot change what a
# query does, so these strings are not mutation targets.
SQL_SNAPSHOTS = ("SELECT snapshot_date, symbol, shares FROM position_snapshots "
                 "ORDER BY snapshot_date")
SQL_SPLITS = "SELECT effective_date, symbol, factor FROM splits ORDER BY effective_date"
SQL_PRICES = ("SELECT Symbol, Date, Close FROM market_history "
              "WHERE Symbol IN ({}) AND Close IS NOT NULL")
SQL_CLEAR_VALUATION = "DELETE FROM valuation"
SQL_CLEAR_TOTAL = "DELETE FROM portfolio_total"
SQL_WRITE_VALUATION = ("INSERT OR REPLACE INTO valuation "
                       "(date, symbol, shares, price_eur, fx, value_eur, source) "
                       "VALUES (?, ?, ?, ?, ?, ?, ?)")
SQL_WRITE_TOTAL = ("INSERT OR REPLACE INTO portfolio_total (date, value_eur, cash_eur) "
                   "VALUES (?, ?, NULL)")
SQL_TOTALS = "SELECT date, value_eur FROM portfolio_total ORDER BY date"
SQL_LATEST_TOTAL = "SELECT date, value_eur FROM portfolio_total ORDER BY date DESC LIMIT 1"
SQL_LAST_PRICE = "SELECT MAX(Date) FROM market_history"
SQL_LAST_VALUATION = "SELECT MAX(date) FROM valuation"
SQL_SYMBOL_SERIES = "SELECT date, symbol, value_eur FROM valuation ORDER BY date"


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

def restate(shares: float, symbol: str, since: date | None, splits: list) -> float:
    """Shares counted at ``since``, restated to today's units by later splits. Pure.

    Prices in ``market_history`` are split-adjusted to today's share basis, so a
    count taken before a split is multiplied by that split's factor (2.0 for 2:1).
    """
    for row in splits or []:
        eff, sym, factor = (list(row) + [None, None, None])[:3]
        ed, f = _as_date(eff), _num(factor)        # unknown factor -> 0 -> skipped
        if str(sym) == str(symbol) and ed is not None and f > 0 and (
                since is None or ed > since):
            shares *= f
    return shares


def shares_on(snapshots: list, splits: list, day: date) -> dict[str, float]:
    """Shares per symbol held on ``day``, in today's units. Pure.

    ``snapshots`` is ``[(snapshot_date, {symbol: shares}), ...]`` sorted by date:
    each is the whole holdings table as saved that day (Confirm and save), so
    the latest one on or before ``day`` is what was held. Before the first one,
    nothing was held.
    """
    held: tuple | None = None
    for snap_day, table in snapshots:
        if snap_day <= day:
            held = (snap_day, table)
    if held is None:
        return {}
    snap_day, table = held
    return {sym: round(restate(sh, sym, snap_day, splits), 6)
            for sym, sh in table.items() if sh > 0}


def _snapshots(conn: Any) -> list:
    """``[(date, {symbol: shares})]`` from position_snapshots, oldest first.

    No snapshot yet (nothing saved) -> the current holdings table as today's.
    """
    try:
        rows = conn.execute(SQL_SNAPSHOTS).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    by_day: dict = {}
    for d, sym, sh in rows:
        by_day.setdefault(_as_date(d), {})[str(sym)] = _num(sh)
    if by_day:
        return sorted(by_day.items())
    from quant.engine.positions import positions_now

    table = {p["symbol"]: p["shares"] for p in positions_now(None) if p["shares"] > 0}
    return [(date.today(), table)] if table else []


def _split_rows(conn: Any) -> list:
    """Corporate-action split factors: [(effective_date, symbol, factor)].

    Reads the ``splits`` table when present; empty otherwise (a DB predating it).
    """
    try:
        return conn.execute(SQL_SPLITS).fetchall()
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
    """{symbol: [(date, native_close), ...] sorted} for the given symbols.

    An empty symbol set is invalid SQL (``IN ()``) and lands in the same empty
    fallback as an unreadable table.
    """
    try:
        rows = conn.execute(
            SQL_PRICES.format(",".join("?" for _ in symbols)), list(symbols)).fetchall()
    except Exception:  # noqa: BLE001
        return {}
    out: dict = {}
    for sym, day, close in rows:           # Close IS NOT NULL in the query
        d = _as_date(day)
        if d is None:
            continue
        out.setdefault(str(sym), []).append((d, float(close)))
    for sym in out:
        out[sym].sort()                    # (date, close): one close per date
    return out


def _close_on_or_before(series: list, day: date) -> float | None:
    """The close on ``day`` or the most recent before it (None when none yet)."""
    idx = bisect.bisect_right(series, day, key=lambda t: t[0])
    return series[idx - 1][1] if idx else None


def backfill(conn: Any, fx_lookup: Any = None, source: str = SOURCE_BACKFILL) -> dict:
    """Rebuild the store: one valuation row per held symbol per trading day.

    v10.8.10: shares come from the saved holdings tables (position_snapshots),
    not from a reconstructed ledger (opening lots + diff trades), which one bad
    row could corrupt for good. The store is rebuilt from scratch on every call
    (small: days x holdings), so it heals itself. value = shares x close x FX.
    ``fx_lookup(symbol)`` replaces the live EUR rate (tests). Returns
    ``{days, rows}``.
    """
    snapshots = _snapshots(conn)
    if not snapshots:
        return {"days": 0, "rows": 0}
    splits = _split_rows(conn)
    symbols = {s for _d, table in snapshots for s in table}
    hist = _price_history(conn, symbols)
    first = snapshots[0][0]
    days = sorted({d for s in hist.values() for d, _c in s if d >= first})
    conn.execute(SQL_CLEAR_VALUATION)
    conn.execute(SQL_CLEAR_TOTAL)
    fx = fx_lookup if callable(fx_lookup) else _fx_rate
    fx_cache: dict = {}
    rows = 0
    for d in days:
        total = 0.0
        for sym, sh in shares_on(snapshots, splits, d).items():
            price = _close_on_or_before(hist.get(sym, []), d)
            if price is None:
                continue
            if sym not in fx_cache:
                fx_cache[sym] = float(fx(sym)) or 1.0
            value = round(sh * price * fx_cache[sym], 2)
            conn.execute(SQL_WRITE_VALUATION,
                         [d, sym, sh, price, fx_cache[sym], value, str(source)])
            rows += 1
            total += value
        # The parts are whole cents; rounding only drops float noise, so 2 or 3
        # places are the same.
        conn.execute(SQL_WRITE_TOTAL, [d, round(total, 2)])  # pragma: no mutate
    return {"days": len(days), "rows": rows}


# ── Readers ───────────────────────────────────────────────────────────────────

def read_total_series(conn: Any = None) -> list[dict]:
    """The daily portfolio-total series oldest first (invested value only).

    Returns ``[{date, value_eur}]``; empty when nothing is stored. v10.8.7 (R-C1):
    ``cash_eur`` is frozen audit data and is never read.
    """
    rows = _query(conn, SQL_TOTALS)
    return [{"date": r[0], "value_eur": _num(r[1])} for r in rows]


def latest_total(conn: Any = None) -> dict:
    """The most recent ``portfolio_total`` row (invested value only), or {}."""
    rows = _query(conn, SQL_LATEST_TOTAL)
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
    from quant.reporting.artifacts import compute_day_delta

    series = read_total_series(conn)
    if not series:
        return 0.0, 0.0, None
    eur, pct = compute_day_delta(series)
    return eur, pct, _as_date(series[-1]["date"])


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
    import contextlib
    import time

    from quant.data.database import connect_with_retry

    last_err: Exception | None = None  # pragma: no mutate (always set before use)
    tries = max(1, int(retries))
    for attempt in range(tries):
        try:
            with contextlib.closing(connect_with_retry(attempts=1)) as conn:
                latest_price = _as_date(conn.execute(SQL_LAST_PRICE).fetchone()[0])
                latest_val = _as_date(conn.execute(SQL_LAST_VALUATION).fetchone()[0])
                if latest_price is None:
                    return {"ok": True, "skipped": True, "reason": "no prices"}
                if latest_val is not None and latest_val >= latest_price:
                    return {"ok": True, "skipped": True, "latest": str(latest_val)}
                res = backfill(conn, source=SOURCE_REFRESH)
                return {"ok": True, "rows": res["rows"], "days": res["days"],
                        "latest": str(latest_price)}
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < tries - 1:
                time.sleep(base_delay * (2 ** attempt))
    _mark_valuation_stale(last_err)
    return {"ok": False, "error": str(last_err)}


def latest_price_date(conn: Any = None) -> date | None:
    """The most recent stored valuation date, or None."""
    rows = _query(conn, SQL_LAST_VALUATION)
    return _as_date(rows[0][0]) if rows else None


def read_symbol_series(conn: Any = None) -> dict:
    """The per-symbol value series from the valuation store.

    Returns ``{symbol: [{date, value_eur}, ...]}`` oldest first, for the Growth
    chart's one-line-per-held-asset view. Empty when nothing is stored.
    """
    rows = _query(conn, SQL_SYMBOL_SERIES)
    out: dict = {}
    for d, sym, v in rows:
        out.setdefault(str(sym), []).append(
            {"date": _as_date(d), "value_eur": _num(v)})
    return out


def _query(conn: Any, sql: str) -> list:
    """Run a read query on ``conn`` or a short-lived read-only connection."""
    try:
        if conn is not None:
            return conn.execute(sql).fetchall()
        from quant.data.database import read_only_connection

        with read_only_connection() as c:
            return c.execute(sql).fetchall()
    except Exception:  # noqa: BLE001
        return []

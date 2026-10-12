"""
mtm.py — the daily mark-to-market valuation store (v10.8.4, T1).

Intent: the product must show ONE value per datum. Before v10.8.4 a holding's
value came from three places (the broker CSV, the ``holdings_meta`` estimate, a
review artifact), so the chart lagged, the header delta was unlabeled and the
holdings table disagreed with itself. This module is the single writer and reader
of the ``valuation`` / ``portfolio_total`` stores:

  valuation(date, symbol, shares, price_eur, fx, value_eur, source)
  portfolio_total(date, value_eur, cash_eur)

It is written by the daily job, by confirm & save, and by the on-demand refresh;
it is idempotent per (date, symbol). Shares are reconstructed from the ledger
(recorded buys/sells plus corporate-action split factors) as of each date; cash is
tracked from the recorded flows. Every chart, delta, holdings table and class
weight reads this store and nothing else.

Invariants:
  - Pure helpers (reconstruct_shares, cash_from_flows) do no I/O when the rows are
    passed in; the I/O wrappers take an explicit connection.
  - Never raises on a missing price: the symbol is skipped (its last stored value
    is untouched), never zeroed.
  - value_eur = shares * price_eur * fx, rounded to 2 decimals.
"""
from __future__ import annotations

from datetime import date
from typing import Any

DEFAULT_OPENING_CASH = 0.0
SOURCE_DAILY = "daily"
SOURCE_CONFIRM = "confirm"
SOURCE_REFRESH = "refresh"


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
    """Reconstruct shares per symbol as of ``as_of``. Pure; no I/O.

    Args:
        opening: ``{symbol: (base_date, base_shares)}`` — the opening lot (the
            broker's synced position). A trade dated on or before ``base_date`` is
            already reflected and is ignored.
        trades: ``[(trade_date, symbol, action, shares)]`` (buy/sell).
        splits: ``[(effective_date, symbol, factor)]`` (factor 2.0 for a 2:1).
        as_of: the valuation date.

    Returns ``{symbol: shares}`` (positive shares only).
    """
    out: dict[str, tuple[date | None, float]] = {}
    for sym, base in (opening or {}).items():
        base_date, base_shares = base
        out[str(sym)] = (_as_date(base_date), _num(base_shares))

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
        out[sym] = (base_date, current + signed)

    shares = {sym: value for sym, (_d, value) in out.items()}

    for row in splits or []:
        eff, sym, factor = (list(row) + [None, None, None])[:3]
        ed = _as_date(eff)
        factor = _num(factor, 1.0)
        if ed is None or ed > as_of or factor <= 0:
            continue
        sym = str(sym)
        if sym in shares:
            shares[sym] *= factor

    return {sym: round(v, 6) for sym, v in shares.items() if v > 0}


def cash_from_flows(opening_cash: float, flows: list, as_of: date) -> float:
    """Cash as of ``as_of``: opening cash plus the recorded flows. Pure.

    A buy moves money OUT of cash into invested (cash decreases); a sell and a
    dividend move money back to cash (cash increases).
    """
    cash = _num(opening_cash)
    for row in flows or []:
        f_date, f_type, amount = (list(row) + [None, None, None])[:3]
        fd = _as_date(f_date)
        if fd is None or fd > as_of:
            continue
        kind = str(f_type).lower()
        amount = _num(amount)
        if kind == "buy":
            cash -= amount
        elif kind in ("sell", "dividend"):
            cash += amount
    return cash


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


def _flow_rows(conn: Any) -> list:
    """Recorded flows: [(date, type, amount_eur)]."""
    try:
        return conn.execute(
            "SELECT date, type, amount_eur FROM flows ORDER BY date").fetchall()
    except Exception:  # noqa: BLE001
        return []


def reconstruct_shares(conn: Any, as_of: date | None = None,
                       opening: dict | None = None,
                       trades: list | None = None,
                       splits: list | None = None) -> dict[str, float]:
    """Shares per symbol as of ``as_of``, reconstructed from the ledger.

    ``opening`` / ``trades`` / ``splits`` may be injected (tests, pure use);
    otherwise they are read from the connection.
    """
    as_of = as_of or date.today()
    if opening is None:
        opening = _opening_lots(conn)
    if trades is None:
        trades = _trade_rows(conn)
    if splits is None:
        splits = _split_rows(conn)
    return reconstruct_from_rows(opening, trades, splits, as_of)


def cash_as_of(conn: Any, as_of: date | None = None,
               opening_cash: float | None = None,
               flows: list | None = None) -> float:
    """Cash as of ``as_of``: opening cash (account) plus recorded flows."""
    as_of = as_of or date.today()
    if opening_cash is None:
        opening_cash = _account_cash()
    if flows is None:
        flows = _flow_rows(conn)
    return round(cash_from_flows(opening_cash, flows, as_of), 2)


def _account_cash() -> float:
    """The user's operational cash from account.yaml (0.0 on any failure)."""
    try:
        from quant.portfolio.account import load_account

        return float(load_account().cash_eur or 0.0)
    except Exception:  # noqa: BLE001
        return DEFAULT_OPENING_CASH


# ── Price / FX resolution ─────────────────────────────────────────────────────

def _native_price(symbol: str, as_of: date, conn: Any) -> float | None:
    try:
        from quant.data.currency import _native_close

        return _native_close(symbol, as_of=as_of, conn=conn)
    except Exception:  # noqa: BLE001
        return None


def _fx_rate(symbol: str) -> float:
    try:
        from quant.data.currency import get_fx_to_eur

        return float(get_fx_to_eur(symbol))
    except Exception:  # noqa: BLE001
        return 1.0


# ── The writer ────────────────────────────────────────────────────────────────

def write_valuation(conn: Any, as_of: date | None = None,
                    price_lookup: Any = None, fx_lookup: Any = None,
                    source: str = SOURCE_DAILY,
                    opening_cash: float | None = None) -> int:
    """Write one valuation row per held symbol for ``as_of`` (idempotent).

    ``price_lookup`` and ``fx_lookup`` are optional callables ``(symbol) -> float``
    (tests inject them). By default the native close comes from ``market_history``
    and the FX multiplier from :func:`quant.data.currency.get_fx_to_eur`.

    Also upserts the day's ``portfolio_total`` (sum of value_eur + cash). Returns
    the number of per-symbol rows written. Never raises on a missing price.
    """
    as_of = as_of or date.today()
    shares = reconstruct_shares(conn, as_of)
    total = 0.0
    written = 0
    for sym, sh in shares.items():
        native = price_lookup(sym) if callable(price_lookup) else \
            _native_price(sym, as_of, conn)
        if native is None:
            continue
        fx = float(fx_lookup(sym)) if callable(fx_lookup) else _fx_rate(sym)
        native = float(native)
        fx = float(fx) if fx else 1.0
        value = round(sh * native * fx, 2)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO valuation "
                "(date, symbol, shares, price_eur, fx, value_eur, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [as_of, sym, sh, native, fx, value, str(source)])
            written += 1
            total += value
        except Exception:  # noqa: BLE001
            continue
    cash = cash_as_of(conn, as_of, opening_cash=opening_cash)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO portfolio_total (date, value_eur, cash_eur) "
            "VALUES (?, ?, ?)", [as_of, round(total, 2), cash])
    except Exception:  # noqa: BLE001
        pass
    return written


def refresh(as_of: date | None = None, source: str = SOURCE_REFRESH) -> dict:
    """Open a write connection and write today's valuation. Returns a summary."""
    from quant.data.database import write_connection

    as_of = as_of or date.today()
    try:
        with write_connection() as conn:
            written = write_valuation(conn, as_of, source=source)
        return {"ok": True, "date": as_of, "rows": written}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "date": as_of, "rows": 0, "error": str(e)}


# ── Readers ───────────────────────────────────────────────────────────────────

def read_series(conn: Any = None) -> list[dict]:
    """The daily total series from ``portfolio_total`` oldest first.

    Returns ``[{date, value_eur, cash_eur}]``; empty when nothing is stored.
    """
    rows = _query(conn, "SELECT date, value_eur, cash_eur FROM portfolio_total "
                        "ORDER BY date")
    return [{"date": r[0], "value_eur": _num(r[1]), "cash_eur": _num(r[2])}
            for r in rows]


def latest_total(conn: Any = None) -> dict:
    """The most recent ``portfolio_total`` row, or an empty dict."""
    rows = _query(conn, "SELECT date, value_eur, cash_eur FROM portfolio_total "
                        "ORDER BY date DESC LIMIT 1")
    if not rows:
        return {}
    r = rows[0]
    return {"date": r[0], "value_eur": _num(r[1]), "cash_eur": _num(r[2])}


def day_delta(conn: Any = None) -> tuple[float, float, date | None]:
    """The last day's change: ``(eur, pct, latest_date)`` from stored totals.

    The home header delta is exactly ``total(D) - total(D-1)`` where the total is
    the invested value plus cash. Returns ``(0.0, 0.0, latest_date_or_None)`` when
    fewer than two days are stored.
    """
    rows = _query(conn, "SELECT date, value_eur, cash_eur FROM portfolio_total "
                        "ORDER BY date DESC LIMIT 2")
    if len(rows) < 2:
        d = _as_date(rows[0][0]) if rows else None
        return 0.0, 0.0, d
    latest_d = _as_date(rows[0][0])
    prev = _num(rows[1][1]) + _num(rows[1][2])
    last = _num(rows[0][1]) + _num(rows[0][2])
    if not prev:
        return 0.0, 0.0, latest_d
    return round(last - prev, 2), (last / prev - 1.0) * 100.0, latest_d


def holdings_at(conn: Any, as_of: date | None = None) -> list[dict]:
    """The stored valuation rows for ``as_of`` (or the latest stored date)."""
    if as_of is None:
        rows = _query(conn, "SELECT MAX(date) FROM valuation")
        if not rows or rows[0][0] is None:
            return []
        as_of = _as_date(rows[0][0])
    rows = _query(conn, "SELECT symbol, shares, price_eur, fx, value_eur, source "
                        "FROM valuation WHERE date = ? ORDER BY symbol", [as_of])
    return [{"symbol": str(r[0]), "shares": _num(r[1]), "price_eur": _num(r[2]),
             "fx": _num(r[3]), "value_eur": _num(r[4]), "source": str(r[5])}
            for r in rows]


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


def cash_series(conn: Any = None) -> list[dict]:
    """The cash series from ``portfolio_total`` oldest first."""
    rows = _query(conn, "SELECT date, cash_eur FROM portfolio_total ORDER BY date")
    return [{"date": _as_date(r[0]), "cash_eur": _num(r[1])} for r in rows]


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

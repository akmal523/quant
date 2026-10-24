"""
artifacts.py — the read helpers the pages use, plus the per-run log directory.

Reads: the regime estimate, a symbol's score row, the valuation store
(``portfolio_total``), the value series and the chart math (day delta, growth
view, value hover, tick dates). ``new_run_dir`` creates the
``outputs/run_<ts>/`` folder for a run's ``pipeline.log``. Never raise.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any, NamedTuple

import pandas as pd

from quant import paths

OUTPUTS_DIR = str(paths.OUTPUTS_DIR)


_RUN_DIR: str | None = None


def new_run_dir() -> str:
    """Create and return outputs/run_<YYYY-MM-DD_HHMMSS>/.

    Invariant (v10.5.0): idempotent per process. The first call creates the
    directory; later calls return the same path so one run writes all artifacts
    (factor scores, telemetry, pipeline.log, web/) into a single dir. Before
    v10.5.0 each call minted a new timestamped dir, scattering one run's output.
    """
    global _RUN_DIR
    if _RUN_DIR is not None:
        return _RUN_DIR
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = os.path.join(OUTPUTS_DIR, f"run_{ts}")
    os.makedirs(path, exist_ok=True)
    _RUN_DIR = path
    return path


# ── v10.5.3 (spec 1.1): the five UI artifact accessors ────────────────────────
# Intent: NO UI page touches run directories or parquet paths directly. Every
# artifact read goes through one of these five helpers, so a "the artifact did
# not reach the UI" bug is impossible by construction.
# Invariants: never raise; return an empty/neutral value when the artifact is
# absent (a missing artifact is a real state the UI renders, not an exception).

_REGIME_DEFAULT: dict = {
    "state": "insufficient_history",
    "label": None,
    "prob": None,
    "confidence": None,
    "as_of": None,
    "error": None,
}


def _analysis_regime() -> dict:
    """The regime stored by the analysis chain's regime stage, or {}."""
    try:
        from quant.data.database import read_only_connection
        from quant.engine.analysis import REGIME_META_KEY

        with read_only_connection() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?",
                               [REGIME_META_KEY]).fetchone()
        data = json.loads(row[0]) if row and row[0] else {}
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def read_regime() -> dict:
    """The market regime stored by the analysis chain; neutral default when absent.

    Label: bull / bear / chop (consumers normalise with
    :func:`quant.portfolio.regime_constraints.regime_key`).
    """
    stored = _analysis_regime()
    return {**_REGIME_DEFAULT, **stored} if stored else dict(_REGIME_DEFAULT)


def read_score_row(symbol: str) -> dict:
    """The full score row for a symbol from the ``scores`` store (v10.8.7, T2).

    Returns ``{composite, confidence, structural_grade, tactical_grade,
    sentiment, stewardship, present, total}``; empty when nothing is stored.
    """
    try:
        from quant.data.database import read_only_connection

        with read_only_connection() as conn:
            res = conn.execute(
                "SELECT composite, confidence, structural_grade, tactical_grade, "
                "sentiment, stewardship, present_components, total_components "
                "FROM scores WHERE symbol = ?", [symbol]).fetchone()
    except Exception:  # noqa: BLE001
        return {}
    if not res:
        return {}
    return {
        "composite": res[0], "confidence": str(res[1] or "low"),
        "structural_grade": res[2], "tactical_grade": res[3],
        "sentiment": res[4], "stewardship": res[5],
        "present": int(res[6] or 0), "total": int(res[7] or 0),
    }


def read_portfolio_total() -> pd.DataFrame:
    """The daily portfolio total from the valuation store (v10.8.4, T1).

    The single source for the home chart and the day delta. Columns ``date`` and
    ``value_eur`` (invested only; v10.8.7, R-C1); empty when nothing is stored.
    Never raises.
    """
    empty = pd.DataFrame(columns=["date", "value_eur"])
    try:
        from quant.engine import mtm

        series = mtm.read_series()
    except Exception:  # noqa: BLE001
        return empty
    if not series:
        return empty
    return pd.DataFrame(series)


def read_value_series() -> pd.DataFrame:
    """The daily invested-value series (v10.8.0; store v10.8.4, T1).

    Reads the mark-to-market ``portfolio_total`` store (the single source) and
    falls back to the legacy snapshot+flow builder only when the store is empty,
    so a pre-v10.8.4 database still charts. Independent of reviews. Returns a
    frame with columns ``date`` and ``value_eur`` (empty when there is no data).
    Never raises.
    """
    store = read_portfolio_total()
    if not store.empty:
        return store[["date", "value_eur"]].copy()

    from quant.data.currency import get_fx_to_eur
    from quant.data.database import read_only_connection
    from quant.engine.value_series import build_value_series

    empty = pd.DataFrame(columns=["date", "value_eur"])
    try:
        with read_only_connection() as conn:
            snaps = conn.execute(
                "SELECT snapshot_date, symbol, shares FROM position_snapshots"
            ).fetchall()
            flows = conn.execute(
                "SELECT date, type, amount_eur, symbol FROM flows"
            ).fetchall()
            symbols = {str(s[1]) for s in snaps}
            closes: dict = {}
            if symbols:
                placeholders = ",".join("?" for _ in symbols)
                rows = conn.execute(
                    f"SELECT Symbol, Date, Close FROM market_history "
                    f"WHERE Symbol IN ({placeholders})", list(symbols)).fetchall()
                closes = {(str(r[0]), str(r[1])[:10]): float(r[2])
                          for r in rows if r[2] is not None}
    except Exception:  # noqa: BLE001
        return empty

    snap_dicts = [{"date": s[0], "symbol": s[1], "shares": s[2]} for s in snaps]
    flow_dicts = [{"date": f[0], "type": f[1], "amount_eur": f[2], "symbol": f[3]}
                  for f in flows]
    fx = {sym: get_fx_to_eur(sym) for sym in symbols}
    series = build_value_series(snap_dicts, flow_dicts, closes, fx)
    if not series:
        return empty
    return pd.DataFrame(series)


# ── Pure chart / valuation helpers (v10.8.6, Defect B) ────────────────────────
# Pure: no streamlit, no database writes; inputs -> outputs. The chart and the
# header delta call these instead of recomputing inline (R1: one value per datum).
# Inline, untestable logic is how the "+0.0%" delta and flat steps survived two
# iterations, so these guarantees are named and tested.


class Point(NamedTuple):
    """One chart point: a date and its value (normalized in the Growth view)."""

    date: date
    value: float


def _coerce_date(value: Any) -> date | None:
    """A ``datetime.date`` from a date/datetime/Timestamp/ISO string, else None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception:  # noqa: BLE001
        return None


def _value_at(rows: list[dict], day: date | None) -> float | None:
    """The stored value exactly on ``day`` (no forward-fill), else None."""
    if day is None:
        return None
    for row in rows or []:
        if _coerce_date(row.get("date")) == day:
            v = row.get("value_eur")
            return float(v) if v is not None else None
    return None


def chart_tick_dates(dates, max_ticks: int = 6) -> list[date]:
    """Unique, evenly sampled X-axis tick dates (v10.8.6, B5). Pure.

    Deduplicates and limits to ``max_ticks`` evenly spaced entries. The renderer
    uses these as ``tickvals`` so the axis can never show a duplicate date.
    """
    seen: list[date] = []
    for d in dates or []:
        c = _coerce_date(d)
        if c is not None and c not in seen:
            seen.append(c)
    unique = sorted(seen)
    if len(unique) <= max_ticks:
        return unique
    step = max(1, len(unique) // max_ticks)
    return unique[::step]


def format_chart_ticks(dates, max_ticks: int = 6) -> list[str]:
    """Unique, readable X-axis labels, evenly sampled (v10.8.6, B5).

    Deduplicates, limits to ``max_ticks`` evenly spaced entries, and formats each
    as ``DD Mon``. Pure.
    """
    return [d.strftime("%d %b") for d in chart_tick_dates(dates, max_ticks)]


def compute_day_delta(series, as_of=None) -> tuple[float, float]:
    """The day change from a portfolio-total series (v10.8.6, B6).

    ``series`` is ``[{date, value_eur}, ...]``; the total is the invested value
    only (v10.8.7, R-C1). Returns ``(delta_eur, delta_pct)`` =
    ``total(D) - total(D-1)`` where D is ``as_of`` (default: the last point). No
    forward-fill, no fallback. Pure.
    """
    pts: list[tuple[date, float]] = []
    for row in series or []:
        d = _coerce_date(row.get("date"))
        if d is None:
            continue
        total = float(row.get("value_eur") or 0)
        pts.append((d, total))
    pts.sort(key=lambda t: t[0])
    if len(pts) < 2:
        return 0.0, 0.0
    idx = len(pts) - 1
    if as_of is not None:
        target = _coerce_date(as_of)
        earlier = [i for i, (d, _t) in enumerate(pts) if target is not None and d <= target]
        if not earlier:
            return 0.0, 0.0
        idx = earlier[-1]
    if idx <= 0:
        return 0.0, 0.0
    prev = pts[idx - 1][1]
    today = pts[idx][1]
    delta = round(today - prev, 2)
    pct = (today / prev - 1.0) * 100.0 if prev else 0.0
    return delta, pct


def _normalize_100(rows) -> list[Point]:
    """A series rebased to 100 at its first point (empty when no rows). Pure."""
    pts: list[tuple[date, float]] = []
    for r in rows or []:
        d = _coerce_date(r.get("date"))
        if d is not None:
            pts.append((d, float(r.get("value_eur") or 0.0)))
    pts.sort(key=lambda t: t[0])
    if not pts:
        return []
    base = pts[0][1]
    if not base:
        return [Point(d, 100.0) for d, _v in pts]
    return [Point(d, v / base * 100.0) for d, v in pts]


def compute_growth_view(portfolio, per_symbol: dict, held_symbols) -> dict:
    """Normalized-to-100 series per line for the Growth view (v10.8.6, B3/5).

    ``portfolio`` is ``[{date, value_eur}, ...]``; ``per_symbol`` is
    ``{symbol: [{date, value_eur}, ...]}``; ``held_symbols`` are the lines to draw.
    Returns ``{"portfolio": [Point, ...], <sym>: [Point, ...]}``, each rebased to
    100 at its first point. Pure.
    """
    out: dict = {"portfolio": _normalize_100(portfolio)}
    for sym in held_symbols or []:
        pts = _normalize_100((per_symbol or {}).get(sym))
        if pts:
            out[str(sym)] = pts
    return out


def compute_value_view_hover(as_of, portfolio, per_symbol: dict,
                             held_symbols) -> dict:
    """The Value-view hover payload for ``as_of`` (v10.8.6, B3/6; v10.8.7, R-C1).

    Returns ``{"total": <invested sum>, <sym>: <value>, ...}`` where the symbol
    values sum to the invested total. Cash is not an entity (R-C1), so there is no
    cash key. Only stored values are used (no forward-fill). Pure.
    """
    day = _coerce_date(as_of)
    out: dict = {}
    for sym in held_symbols or []:
        v = _value_at((per_symbol or {}).get(sym), day)
        if v is not None:
            out[str(sym)] = round(float(v), 2)
    out["total"] = round(sum(out.values()), 2)
    return out

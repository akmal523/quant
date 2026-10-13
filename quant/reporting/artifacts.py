"""
artifacts.py — Run artifacts + structured logging.

Intent: each run writes a timestamped artifact directory (outputs/run_<ts>/)
with factor scores, NLP scores, run config, and metrics. Enables debugging why
a signal appeared. Also provides a structured JSON logger.
Invariants: artifact dir created idempotently; writes are atomic-ish (parquet).
Dependencies: pandas, polars, json, logging.
"""
from __future__ import annotations

import json
import logging
import os
import re
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


def current_run_dir() -> str | None:
    """Return the run dir created in this process, or None if none yet."""
    return _RUN_DIR


def latest_run() -> str | None:
    """Return the most recent run dir (spec 4.3 single artifact accessor).

    Intent: the dashboard Explorer must read scores through ONE helper so it can
    never point at a stale or wrong artifact path. Alias of latest_run_dir().
    """
    return latest_run_dir()


def latest_run_dir() -> str | None:
    """Return the most recent outputs/run_<ts>/ directory, or None.

    Intent (Phase 5 / v10.2): the dashboard sidebar shows the last run timestamp
    and reads the latest factor_scores.parquet from the newest artifact dir.
    Invariants: returns None if no run artifacts exist.
    """
    if not os.path.isdir(OUTPUTS_DIR):
        return None
    runs = [d for d in os.listdir(OUTPUTS_DIR) if d.startswith("run_")]
    if not runs:
        return None
    return os.path.join(OUTPUTS_DIR, max(runs))


def save_artifact(run_dir: str, name: str, df: pd.DataFrame) -> str:
    """Save a DataFrame to run_dir/<name>.parquet. Returns full path."""
    path = os.path.join(run_dir, f"{name}.parquet")
    df.to_parquet(path, index=False)
    return path


def save_config(run_dir: str, config: dict) -> str:
    """Save run configuration as JSON."""
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return path


def save_metrics(run_dir: str, metrics: dict) -> str:
    """Save run metrics as JSON."""
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
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


# H3.7 (L1): a review lives in a TIMESTAMPED dir that carries metrics.json.
# outputs/run_latest and update-only dirs must never shadow a review.
_RUN_RE = re.compile(r"^run_\d{4}-\d{2}-\d{2}_\d{6}$")


def _review_run_dirs() -> list[str]:
    """Timestamped run dirs carrying metrics.json, newest first (H3.7, L1)."""
    if not os.path.isdir(OUTPUTS_DIR):
        return []
    runs = sorted((d for d in os.listdir(OUTPUTS_DIR) if _RUN_RE.match(d)),
                  reverse=True)
    return [os.path.join(OUTPUTS_DIR, d) for d in runs
            if os.path.exists(os.path.join(OUTPUTS_DIR, d, "metrics.json"))]


def _read_metrics(run: str) -> dict | None:
    path = os.path.join(run, "metrics.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def latest_review(ok_only: bool = False) -> dict:
    """Return the latest review's metrics.json as a dict ({} when absent).

    H3.6/H3.7 (N1/L1): reads only timestamped dirs that carry metrics.json, so
    an update-only dir or outputs/run_latest cannot shadow a review. A review
    without `review_status` is legacy and counts as ok; only "failed" excludes.
    """
    for run in _review_run_dirs():
        m = _read_metrics(run)
        if m is None:
            continue
        if ok_only and m.get("review_status") == "failed":
            continue
        return m
    return {}


def latest_ok_run_dir() -> str | None:
    """Newest timestamped review dir that is ok under the L1 rule (else None)."""
    for run in _review_run_dirs():
        m = _read_metrics(run)
        if m is not None and m.get("review_status") != "failed":
            return run
    return None


def latest_ok_review_ts() -> datetime | None:
    """Timestamp parsed from the newest ok review dir name (H3.7, L3)."""
    run = latest_ok_run_dir()
    if not run:
        return None
    name = os.path.basename(run)                 # run_YYYY-MM-DD_HHMMSS
    try:
        return datetime.strptime(name[4:], "%Y-%m-%d_%H%M%S")
    except ValueError:
        return None


def _update_state_path() -> str:
    return os.path.join(OUTPUTS_DIR, "update_state.json")


def write_update_state(payload: dict) -> None:
    """Persist the last successful update's timestamp/counts (H3.7, L2)."""
    try:
        with open(_update_state_path(), "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception:  # noqa: BLE001
        pass


def read_update_state() -> dict:
    """Read outputs/update_state.json ({} when absent)."""
    path = _update_state_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def read_regime() -> dict:
    """Return the regime block (spec 1.2). Neutral default when absent.

    H3.6 (N1): reads the most recent SUCCESSFUL review (ok_only).
    """
    review = latest_review(ok_only=True)
    regime = review.get("regime")
    if isinstance(regime, dict):
        return {**_REGIME_DEFAULT, **regime}
    return dict(_REGIME_DEFAULT)


def _read_audit_df() -> pd.DataFrame:
    path = os.path.join(OUTPUTS_DIR, "portfolio_audit.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def read_actions() -> list[dict]:
    """Return per-holding dicts merging the audit row with its canonical action.

    Intent: the holdings table and the action cards read ONE object (spec 2.1 /
    A3). Includes the audit columns the table needs (value, weights) plus the
    action fields (action, amount, blocked, status, remedy).
    """
    from quant.config import MIN_TRADE_SIZE_EUR, rebalance_threshold
    from quant.reporting.actions import build_actions
    from quant.ui import copy as ui_copy

    audit = _read_audit_df()
    if audit.empty:
        return []

    # v10.8.0 (Phase 1): overlay the LIVE positions so the Overview table, the
    # steps, the action cards and My holdings read the same value, weight and
    # tier. Before this, the Overview read the frozen review artifact while My
    # holdings read the live broker CSV, so the two disagreed.
    try:
        from quant.engine.positions import positions_now
        from quant.portfolio.tier_manager import load_tiers_safe, tier_map

        live = {p["symbol"]: p for p in positions_now()}
        tmap = tier_map(load_tiers_safe()[0])
        total = sum(float(p["value_eur"]) for p in live.values()) or 1.0
        if live:
            audit = audit.copy()
            for i, r in audit.iterrows():
                sym = str(r.get("Symbol", ""))
                p = live.get(sym)
                if p:
                    audit.at[i, "Value_EUR"] = float(p["value_eur"])
                    audit.at[i, "Current_Weight"] = (
                        f"{float(p['value_eur']) / total * 100:.1f}%")
                if sym in tmap:
                    audit.at[i, "Tier"] = tmap[sym]
    except Exception:  # noqa: BLE001
        pass

    def _pct(value) -> float:
        try:
            return float(str(value).rstrip("%") or 0) / 100.0
        except (TypeError, ValueError):
            return 0.0

    by_sym = {a["symbol"]: a for a in build_actions(audit)}
    total_value = float(audit["Value_EUR"].sum()) if "Value_EUR" in audit else 0.0
    out: list[dict] = []
    for _, r in audit.iterrows():
        sym = str(r.get("Symbol", ""))
        a = by_sym.get(sym, {})
        rec = str(r.get("Recommendation", "") or "")
        tier = str(r.get("Tier", "ACTIVE"))
        threshold = rebalance_threshold(tier)
        drift_frac = _pct(r.get("Drift", ""))
        cooldown = r.get("Cooldown_Until")
        if isinstance(cooldown, float) and cooldown != cooldown:
            cooldown = None
        status = a.get("status") or ui_copy.status_for(
            rec, cooldown_until=cooldown, drift_frac=drift_frac, threshold=threshold)
        suppressed = None
        # S3: over threshold but the move is below the minimum order size.
        if not a.get("action") and not a.get("blocked") and abs(drift_frac) > threshold:
            if abs(drift_frac) * total_value < MIN_TRADE_SIZE_EUR:
                suppressed = "below_min"
                status = ui_copy.STATUS_BELOW_MIN
        out.append({
            "symbol": sym,
            "name": str(r.get("Name", "") or sym),
            "tier": tier,
            "value_eur": float(r.get("Value_EUR", 0) or 0),
            "current_weight": str(r.get("Current_Weight", "")),
            "target_weight": str(r.get("Target_Weight", "")),
            "drift": str(r.get("Drift", "")),
            "action": a.get("action"),
            "kind": a.get("kind"),
            "amount_eur": a.get("amount_eur"),
            "blocked": bool(a.get("blocked", False)),
            "remedy": a.get("remedy"),
            "status": status,
            "min_trade_eur": a.get("min_trade_eur", MIN_TRADE_SIZE_EUR),
            "cooldown_until": cooldown,
            "suppressed": suppressed,
        })
    return out


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


def read_scores(symbol: str) -> dict:
    """Return {structural_grade, tactical_grade, active_score} for a symbol.

    Source order (v10.8.7, T2): the ``scores`` store, then the latest run's
    scores artifact, then asset_registry. Invariants: returns the keys with None
    when nothing is known.
    """
    row = read_score_row(symbol)
    if row:
        return {
            "structural_grade": row.get("structural_grade"),
            "tactical_grade": row.get("tactical_grade"),
            "active_score": row.get("composite"),
        }
    keys = {"structural_grade": None, "tactical_grade": None, "active_score": None}
    # H3.6 (N1): read the most recent SUCCESSFUL review, not the latest attempt.
    run = latest_ok_run_dir()
    if run:
        path = os.path.join(run, "scores.parquet")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                m = df[df["Symbol"].astype(str) == str(symbol)]
                if not m.empty:
                    r = m.iloc[0]
                    return {
                        "structural_grade": float(r.get("Structural_Grade", 0) or 0),
                        "tactical_grade": float(r.get("Tactical_Grade", 0) or 0),
                        "active_score": float(r.get("Active_Score", 0) or 0),
                    }
            except Exception:  # noqa: BLE001
                pass
    try:
        from quant.data.database import read_only_connection

        with read_only_connection() as conn:
            res = conn.execute(
                "SELECT structural_grade, tactical_grade, active_score "
                "FROM asset_registry WHERE symbol = ?", [symbol]
            ).fetchone()
        if res:
            return {
                "structural_grade": float(res[0] or 0),
                "tactical_grade": float(res[1] or 0),
                "active_score": float(res[2] or 0),
            }
    except Exception:  # noqa: BLE001
        pass
    return keys


def read_history() -> pd.DataFrame:
    """Return the portfolio value history (one row per review).

    Single accessor for the Today value chart (spec 1.1). Delegates to the
    history module so the table definition stays in one place.
    """
    from quant.portfolio.history import load_history

    return load_history()


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


# ── Structured Logging ────────────────────────────────────────────────────────

class StructuredLogger:
    """JSON-structured logger wrapper. Emits one JSON object per record."""

    def __init__(self, name: str = "quant") -> None:
        self.logger = logging.getLogger(name)

    def _emit(self, level: int, event: str, **fields) -> None:
        record = {"event": event, **fields}
        self.logger.log(level, json.dumps(record, default=str))

    def info(self, event: str, **fields) -> None:
        """Emit an INFO record."""
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields) -> None:
        """Emit a WARNING record."""
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields) -> None:
        """Emit an ERROR record."""
        self._emit(logging.ERROR, event, **fields)


# Module-level singleton for convenience.
structured_log = StructuredLogger()

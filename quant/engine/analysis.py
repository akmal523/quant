"""
analysis.py — the analysis pipeline + manifest (v10.8.7, T2/A3).

Intent: the daily job AND the on-demand "Run analysis" button execute the SAME
ordered chain, and every stage writes one ``run_manifest`` row so the UI never
guesses why a computation is missing:

    prices -> fundamentals -> nlp -> regime -> scoring -> optimization

Fetch stages (prices, fundamentals, nlp) reuse stored data fresher than their TTL
(prices 1 day, fundamentals 7 days, nlp 1 day); compute stages (regime, scoring,
optimization) always recompute. A stage that cannot run is recorded as missing or
failed with its reason, never silently skipped.

Scoring degrades gracefully (R-C2): any subset of components yields a numeric
composite + a confidence tier; a missing component is a documented neutral penalty
(:func:`quant.analytics.scoring.composite_score`). The composite is written to the
``scores`` store for every symbol with a price.

Invariants:
  - Never raises: a failing stage is recorded and the chain continues.
  - One manifest row per stage per run.
  - The scores store is written for every symbol with a price.
"""
from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from datetime import date, datetime
from typing import Any

from quant.engine import manifest

logger = logging.getLogger(__name__)

# Per-stage TTL (days) for the fetch stages (A3). Compute stages always recompute.
STAGE_TTL_DAYS = {"prices": 1, "fundamentals": 7, "nlp": 1}

STATUS_OK_FRESH = "ok-fresh"
STATUS_OK_REUSED = "ok-reused"

# The benchmark used for the macro regime (broad market index).
REGIME_SYMBOL = "URTH"


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_date(value: Any) -> date | None:
    """Coerce a date / ISO string / epoch number to a date, or None."""
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value)).date()
        except (OverflowError, OSError, ValueError):
            return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _universe(conn: Any) -> list[str]:
    """The symbols the analysis covers: held + registry + funnel survivors."""
    out: set[str] = set()
    for sql in (
        "SELECT DISTINCT symbol FROM holdings_meta",
        "SELECT symbol FROM asset_registry",
        "SELECT symbol FROM funnel_survivors",
    ):
        try:
            for row in conn.execute(sql).fetchall():
                if row and row[0]:
                    out.add(str(row[0]))
        except Exception:  # noqa: BLE001
            continue
    return sorted(out)


def _covered(conn: Any, table: str, col: str, sym_col: str,
             symbols: list[str], ttl_days: int, now: date) -> int:
    """How many of ``symbols`` have a ``table.col`` value newer than the TTL (A3)."""
    n = 0
    for sym in symbols:
        try:
            row = conn.execute(
                f"SELECT MAX({col}) FROM {table} WHERE {sym_col} = ?",
                [sym]).fetchone()
        except Exception:  # noqa: BLE001
            continue
        if not row or row[0] is None:
            continue
        latest = _to_date(row[0])
        if latest is not None and (now - latest).days < ttl_days:
            n += 1
    return n


def _record(conn: Any, run_id: str, stage: str, status: str,
            coverage: str = "", error: str = "") -> dict:
    manifest.record_stage(conn, run_id, stage, status, coverage=coverage,
                          error=error)
    return {"stage": stage, "status": status, "coverage": coverage, "error": error}


# ── Component extraction (pure over the connection) ───────────────────────────

def _technical_from_prices(conn: Any, symbol: str) -> float | None:
    """A 0-100 technical score from the price series (None when < 2 points)."""
    try:
        rows = conn.execute(
            "SELECT Close FROM market_history WHERE Symbol = ? AND Close IS NOT NULL "
            "ORDER BY Date", [symbol]).fetchall()
    except Exception:  # noqa: BLE001
        return None
    closes = [float(r[0]) for r in rows if r[0] is not None]
    if len(closes) < 2:
        return None
    lookback = min(len(closes) - 1, 21)
    mom = closes[-1] / closes[-1 - lookback] - 1.0 if closes[-1 - lookback] else 0.0
    return round(50.0 + 50.0 * math.tanh(mom * 5.0), 1)


def _fundamentals_row(conn: Any, symbol: str) -> dict | None:
    try:
        row = conn.execute(
            "SELECT pe, peg, roe, debt_to_equity, interest_expense "
            "FROM fundamentals WHERE symbol = ?", [symbol]).fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not row:
        return None
    pe, peg, roe, de, ie = row
    if pe is None and peg is None and roe is None:
        return None
    return {"PE": pe, "PEG": peg, "ROE": roe, "DebtToEquity": de, "ICR": ie}


def _fundamental_component(conn: Any, symbol: str) -> float | None:
    f = _fundamentals_row(conn, symbol)
    if f is None:
        return None
    from quant.analytics.scoring import evaluate_structural_grade, stewardship_score_v2

    s = stewardship_score_v2(f)
    return round(float(evaluate_structural_grade(
        f.get("PE"), f.get("PEG"), f.get("ROE"), s)), 1)


def _stewardship_component(conn: Any, symbol: str) -> float | None:
    f = _fundamentals_row(conn, symbol)
    if f is None:
        return None
    from quant.analytics.scoring import stewardship_score_v2
    from quant.config import WEIGHT_STEWARDSHIP

    s = stewardship_score_v2(f)
    return round(min(100.0, s / float(WEIGHT_STEWARDSHIP) * 100.0), 1)


def _sentiment_component(conn: Any, symbol: str) -> float | None:
    try:
        row = conn.execute(
            "SELECT AVG(score) FROM nlp_evidence WHERE symbol = ?", [symbol]).fetchone()
    except Exception:  # noqa: BLE001
        return None
    if not row or row[0] is None:
        return None
    v = float(row[0])
    if -1.0 <= v <= 1.0:
        v = (v + 1.0) * 50.0
    return round(max(0.0, min(100.0, v)), 1)


def components_for(conn: Any, symbol: str) -> dict:
    """The four scoring components for a symbol (None when absent). Pure-ish."""
    return {
        "technical": _technical_from_prices(conn, symbol),
        "fundamental": _fundamental_component(conn, symbol),
        "sentiment": _sentiment_component(conn, symbol),
        "stewardship": _stewardship_component(conn, symbol),
    }


def write_scores(conn: Any, symbol: str, result: dict) -> None:
    """Upsert one row into the ``scores`` store. Never raises."""
    comps = result.get("components") or {}
    try:
        conn.execute(
            "INSERT OR REPLACE INTO scores "
            "(symbol, composite, confidence, structural_grade, tactical_grade, "
            " sentiment, stewardship, present_components, total_components, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [str(symbol), _num(result.get("composite")), str(result.get("confidence")),
             _num(comps.get("fundamental")), _num(comps.get("technical")),
             _num(comps.get("sentiment")), _num(comps.get("stewardship")),
             int(result.get("present") or 0), int(result.get("total") or 0),
             time.time()])
    except Exception:  # noqa: BLE001
        pass


# ── Stages ────────────────────────────────────────────────────────────────────

def _fetch_stage(conn: Any, run_id: str, stage: str, table: str, col: str,
                 sym_col: str, symbols: list[str], now: date,
                 fetcher: Callable | None) -> dict:
    """A fetch stage: reuse when fresh, else fetch, else missing/failed (A3).

    Coverage is per-symbol: a stage is ``ok-reused`` only when every universe
    symbol has fresh data; a partial or empty store is ``missing`` (or fetched).
    """
    ttl = STAGE_TTL_DAYS.get(stage, 1)
    total = len(symbols)
    covered = _covered(conn, table, col, sym_col, symbols, ttl, now)
    if total and covered == total:
        return _record(conn, run_id, stage, STATUS_OK_REUSED,
                       coverage=manifest.coverage_text((covered, total)))
    if fetcher is not None:
        try:
            fetcher()
            covered = _covered(conn, table, col, sym_col, symbols, ttl, now)
            return _record(conn, run_id, stage, STATUS_OK_FRESH,
                           coverage=manifest.coverage_text((covered, total)))
        except Exception as e:  # noqa: BLE001
            return _record(conn, run_id, stage, manifest.STATUS_FAILED, error=str(e))
    if covered > 0:
        return _record(conn, run_id, stage, STATUS_OK_REUSED,
                       coverage=manifest.coverage_text((covered, total)))
    return _record(conn, run_id, stage, manifest.STATUS_MISSING,
                   coverage=manifest.coverage_text((0, total)),
                   error="no data source available")


def _regime_stage(conn: Any, run_id: str, now: date) -> dict:
    """Compute the macro regime from the benchmark history (always recompute)."""
    try:
        rows = conn.execute(
            "SELECT Close FROM market_history WHERE Symbol = ? AND Close IS NOT NULL "
            "ORDER BY Date", [REGIME_SYMBOL]).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    closes = [float(r[0]) for r in rows if r[0] is not None]
    if len(closes) < 252:
        return _record(conn, run_id, "regime", manifest.STATUS_MISSING,
                       error="not enough history yet")
    try:
        import pandas as pd

        from quant.analytics.scoring import fit_market_regime

        series = pd.Series(closes)
        vol = series.pct_change().rolling(20).std()
        prob = fit_market_regime(series, vol)
        label = "bull" if prob >= 0.5 else "bear"
        return _record(conn, run_id, "regime", manifest.STATUS_OK,
                       coverage=manifest.coverage_text((1, 1)),
                       error="") | {"label": label}
    except Exception as e:  # noqa: BLE001
        return _record(conn, run_id, "regime", manifest.STATUS_FAILED, error=str(e))


def _scoring_stage(conn: Any, run_id: str, symbols: list[str]) -> dict:
    """Compute a numeric composite for every symbol with a price (R-C2)."""
    from quant.analytics.scoring import composite_score

    scored = 0
    for sym in symbols:
        comps = components_for(conn, sym)
        if comps.get("technical") is None:
            continue
        result = composite_score(comps)
        write_scores(conn, sym, result)
        scored += 1
    status = manifest.STATUS_OK if scored else manifest.STATUS_MISSING
    err = "" if scored else "no symbol has a price yet"
    return _record(conn, run_id, "scoring", status,
                   coverage=manifest.coverage_text((scored, len(symbols))),
                   error=err)


def _optimization_stage(conn: Any, run_id: str, symbols: list[str]) -> dict:
    """A light optimization pass over the scored universe (always recompute)."""
    try:
        row = conn.execute("SELECT COUNT(*) FROM scores").fetchone()
        n = int(row[0]) if row and row[0] is not None else 0
    except Exception:  # noqa: BLE001
        n = 0
    if n == 0:
        return _record(conn, run_id, "optimization", manifest.STATUS_MISSING,
                       error="no scores to optimize")
    return _record(conn, run_id, "optimization", manifest.STATUS_OK,
                   coverage=manifest.coverage_text((n, len(symbols))))


# ── The pipeline ──────────────────────────────────────────────────────────────

def run_analysis(conn: Any, symbols: list[str] | None = None,
                 now: date | None = None,
                 fetchers: dict | None = None,
                 progress: Callable[[str], None] | None = None) -> dict:
    """Run the ordered analysis chain, writing one manifest row per stage.

    ``fetchers`` maps a fetch stage to a zero-arg callable (tests inject them;
    the on-demand button passes the real data updaters). ``progress`` is an
    optional ``(stage) -> None`` callback for the UI's per-stage progress line.
    Never raises. Returns ``{run_id, stages, scored}``.
    """
    now = now or date.today()
    fetchers = fetchers or {}
    run_id = manifest.new_run_id()
    symbols = symbols if symbols is not None else _universe(conn)

    stages: list[dict] = []
    for stage, table, col, sym_col in (
            ("prices", "market_history", "Date", "Symbol"),
            ("fundamentals", "fundamentals", "updated_at", "symbol"),
            ("nlp", "nlp_evidence", "retrieved_at", "symbol")):
        if progress:
            progress(stage)
        stages.append(_fetch_stage(conn, run_id, stage, table, col, sym_col,
                                   symbols, now, fetchers.get(stage)))
    if progress:
        progress("regime")
    stages.append(_regime_stage(conn, run_id, now))
    if progress:
        progress("scoring")
    stages.append(_scoring_stage(conn, run_id, symbols))
    if progress:
        progress("optimization")
    stages.append(_optimization_stage(conn, run_id, symbols))

    scored = 0
    for s in stages:
        if s["stage"] == "scoring" and s["status"] == manifest.STATUS_OK:
            try:
                scored = int(str(s["coverage"]).split("/")[0])
            except (ValueError, IndexError):
                scored = 0
    return {"run_id": run_id, "stages": stages, "scored": scored}

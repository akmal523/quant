"""
manifest.py — the analysis-pipeline validity contract (v10.8.4, T7).

Intent: every run (daily or on-demand) writes one ``run_manifest`` row per stage
so the UI never guesses why a computation is missing. An empty state reads the
manifest and names the failed/stale/missing stage and its error; a literal None
in the UI is forbidden. Scoring never silently skips: a missing NLP input yields a
score with a documented confidence penalty and a ``confidence`` flag (None only
when the whole stage failed, and then the manifest says why).

Stages (fixed vocabulary): prices, fundamentals, nlp, regime, scoring,
optimization, valuation.

Invariants:
  - ``record_stage`` upserts on (run_id, stage); never raises.
  - Status is one of ok / stale / failed / missing.
  - The reader returns the most recent run's rows, one per stage.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

STATUS_OK = "ok"
STATUS_STALE = "stale"
STATUS_FAILED = "failed"
STATUS_MISSING = "missing"

STAGES = ("prices", "fundamentals", "nlp", "regime", "scoring", "optimization",
          "valuation", "names")

# Human stage names (R6: the UI never shows a raw stage key).
STAGE_WORDS = {
    "prices": "Prices",
    "fundamentals": "Fundamentals",
    "nlp": "News sentiment",
    "regime": "Market trend",
    "scoring": "Scoring",
    "optimization": "Optimization",
    "valuation": "Valuation",
    "names": "Display names",
}


def new_run_id(when: datetime | None = None) -> str:
    """A stable run id from a timestamp (second resolution)."""
    when = when or datetime.now()
    return when.strftime("%Y-%m-%d_%H%M%S")


def stage_word(stage: str) -> str:
    """Plain-word name for a stage key."""
    return STAGE_WORDS.get(str(stage), str(stage))


def status_word(status: str) -> str:
    """Plain-word status for a stage (v10.8.7, A3: ok-fresh / ok-reused)."""
    return {
        STATUS_OK: "ok",
        STATUS_STALE: "stale",
        STATUS_FAILED: "failed",
        STATUS_MISSING: "missing",
        "ok-fresh": "ok (fresh)",
        "ok-reused": "ok (reused)",
    }.get(str(status), "missing")


def record_stage(conn: Any, run_id: str, stage: str, status: str,
                 coverage: str = "", error: str = "",
                 started_at: datetime | None = None,
                 finished_at: datetime | None = None) -> None:
    """Upsert one manifest row. Never raises."""
    try:
        conn.execute(
            "INSERT OR REPLACE INTO run_manifest "
            "(run_id, stage, status, started_at, finished_at, coverage, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [str(run_id), str(stage), str(status),
             started_at or datetime.now(), finished_at or datetime.now(),
             str(coverage), str(error or "")])
    except Exception:  # noqa: BLE001
        pass


def latest_run_id(conn: Any = None) -> str | None:
    """The most recent run id in the manifest, or None."""
    rows = _query(conn, "SELECT run_id FROM run_manifest "
                        "ORDER BY finished_at DESC LIMIT 1")
    return str(rows[0][0]) if rows else None


def read_manifest(conn: Any = None, run_id: str | None = None) -> list[dict]:
    """The manifest rows for a run (default: the latest run), one per stage.

    Returns ``[{stage, status, started_at, finished_at, coverage, error}]`` in
    the fixed stage order; empty when nothing is recorded.
    """
    run = run_id or latest_run_id(conn)
    if not run:
        return []
    rows = _query(conn, "SELECT stage, status, started_at, finished_at, coverage, "
                        "error FROM run_manifest WHERE run_id = ?", [run])
    by_stage = {str(r[0]): r for r in rows}
    out: list[dict] = []
    for stage in STAGES:
        r = by_stage.get(stage)
        if r is None:
            out.append({"stage": stage, "status": STATUS_MISSING,
                        "started_at": None, "finished_at": None,
                        "coverage": "", "error": ""})
        else:
            out.append({"stage": stage, "status": str(r[1]),
                        "started_at": r[2], "finished_at": r[3],
                        "coverage": str(r[4] or ""), "error": str(r[5] or "")})
    return out


def stage_status(stage: str, conn: Any = None) -> str:
    """The latest status for one stage (missing when unrecorded)."""
    for row in read_manifest(conn):
        if row["stage"] == stage:
            return row["status"]
    return STATUS_MISSING


def coverage_text(line: str | list | tuple | None) -> str:
    """Format a coverage value as ``n/N symbols`` (T7/R3)."""
    if line is None:
        return ""
    if isinstance(line, list | tuple) and len(line) == 2:
        return f"{int(line[0])}/{int(line[1])} symbols"
    return str(line)


def _fmt_ts(value: Any) -> str:
    """A timestamp as '13 Oct 2026, 17:43' (empty when unknown)."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%d %b %Y, %H:%M")
    return str(value)


def coverage_line(conn: Any = None) -> str:
    """The ONE coverage line: ``coverage: <n>/<N> symbols, as of <ts>`` (R-P2).

    Reads the run manifest's scoring stage. A bare "-" is forbidden: when no
    coverage is recorded the line is a reasoned empty state, never a dash.
    """
    rows = {r["stage"]: r for r in read_manifest(conn)}
    scoring = rows.get("scoring", {})
    cov = str(scoring.get("coverage") or "").strip()
    if not cov or cov == "-":
        return "coverage: not available yet"
    when = _fmt_ts(scoring.get("finished_at"))
    if when:
        return f"coverage: {cov}, as of {when}"
    return f"coverage: {cov}"


def confidence_flag(present: int, total: int) -> str:
    """The data-confidence flag carried on every score (T7).

    ``low`` when most inputs are missing (the documented penalty applies); the
    score still exists. A stage that FAILED is a manifest failure, not a low flag.
    """
    try:
        total = int(total)
        present = int(present)
    except (TypeError, ValueError):
        return "low"
    if total <= 0:
        return "low"
    ratio = present / total
    if ratio >= 0.8:
        return "high"
    if ratio >= 0.5:
        return "medium"
    return "low"


def empty_state_reason(stage: str, conn: Any = None) -> str:
    """The 'Not available — <reason>' sentence for a missing/failed stage (R2).

    Reads the manifest so the reason is never guessed.
    """
    rows = {r["stage"]: r for r in read_manifest(conn)}
    row = rows.get(stage)
    word = stage_word(stage)
    if row is None or row["status"] in (STATUS_MISSING, ""):
        return f"{word}: not available yet."
    if row["status"] == STATUS_FAILED:
        detail = row.get("error") or "the analysis did not complete"
        return f"{word}: not available - {detail}."
    if row["status"] == STATUS_STALE:
        return f"{word}: out of date."
    return ""


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

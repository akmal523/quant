"""
artifacts.py — Run artifacts + structured logging.

Intent: each run writes a timestamped artifact directory (outputs/run_<ts>/)
with factor scores, NLP scores, run config, and metrics. Enables debugging why
a signal appeared. Also provides a structured JSON logger.
Invariants: artifact dir created idempotently; writes are atomic-ish (parquet).
Dependencies: pandas, polars, json, logging.
"""
from __future__ import annotations

from quant import paths
import json
import logging
import os
import re
from datetime import datetime

import pandas as pd

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
    from quant.config import MIN_TRADE_SIZE_EUR, REBALANCE_DRIFT_TIERS
    from quant.reporting.actions import build_actions
    from quant.ui import copy as ui_copy

    audit = _read_audit_df()
    if audit.empty:
        return []

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
        threshold = REBALANCE_DRIFT_TIERS.get(tier, 0.05)
        drift_frac = _pct(r.get("Drift", ""))
        cooldown = r.get("Cooldown_Until")
        if isinstance(cooldown, float) and cooldown != cooldown:
            cooldown = None
        status = a.get("status") or ui_copy.status_for(rec, cooldown_until=cooldown)
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
            "amount_eur": a.get("amount_eur"),
            "blocked": bool(a.get("blocked", False)),
            "remedy": a.get("remedy"),
            "status": status,
            "min_trade_eur": a.get("min_trade_eur", MIN_TRADE_SIZE_EUR),
            "cooldown_until": cooldown,
            "suppressed": suppressed,
        })
    return out


def read_scores(symbol: str) -> dict:
    """Return {structural_grade, tactical_grade, active_score} for a symbol.

    Source order: the latest run's scores artifact, then asset_registry.
    Invariants: returns the keys with None when nothing is known.
    """
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


# ── Structured Logging ────────────────────────────────────────────────────────

class StructuredLogger:
    """JSON-structured logger wrapper. Emits one JSON object per record."""

    def __init__(self, name: str = "quant") -> None:
        self.logger = logging.getLogger(name)

    def _emit(self, level: int, event: str, **fields) -> None:
        record = {"event": event, **fields}
        self.logger.log(level, json.dumps(record, default=str))

    def info(self, event: str, **fields) -> None:
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields) -> None:
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields) -> None:
        self._emit(logging.ERROR, event, **fields)


# Module-level singleton for convenience.
structured_log = StructuredLogger()
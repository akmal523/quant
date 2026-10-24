"""
daily.py — the daily job body, catch-up, and last-run marker (v10.7.0, Section 3).

Intent: merely owning a powered-on laptop is enough. The systemd timer runs
``quant daily`` at 18:45 and 07:45. Each run is idempotent and guarded by the
runner lock; weekends are skipped. A run: refresh (prices with missed days,
fundamentals, news), the analysis chain (regime, scores), the valuation store,
the ONE list (persisted to the advice log), the day's digest written to
``outputs/daily_<date>.md``, and, from the morning slot, the digest sent to
Telegram once per price date.

Catch-up: if the previous run was more than one trading day ago, the run is a
catch-up; every summary artifact gains the exact line
"Monitoring gap: no runs for N days; conditions evaluated on the latest data."

Invariants:
  - run_daily never raises; returns a DailyResult.
  - A weekend writes nothing and returns status "skipped".
  - The heavy steps (update, scoring) are injectable for tests.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from quant import paths

MARKER_NAME = ".daily_last_run"
FAILED_MARKER_NAME = ".daily_last_failed"


@dataclass
class DailyResult:
    """Outcome of a daily run."""

    status: str  # "ok" | "skipped" | "error"
    message: str
    items: int = 0
    gap_days: int | None = None
    invested_eur: float | None = None
    digest_sent: bool = False


def marker_path() -> str:
    """Absolute path of the last-daily-run marker file."""
    return os.path.join(str(paths.OUTPUTS_DIR), MARKER_NAME)


def read_last_daily_run() -> date | None:
    """The date of the last successful daily run, or None when never run."""
    try:
        with open(marker_path(), encoding="utf-8") as f:
            return date.fromisoformat(f.read().strip()[:10])
    except Exception:  # noqa: BLE001
        return None


def write_last_daily_run(when: date | None = None) -> None:
    """Record the date of a successful daily run. Never raises."""
    when = when or date.today()
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(marker_path(), "w", encoding="utf-8") as f:
            f.write(when.isoformat())
    except Exception:  # noqa: BLE001
        pass


# ── v10.7.2 (Part 1.3): the last-failed-run marker ────────────────────────────
# When the daily job cannot acquire the database after retries it must NOT crash
# with a traceback: it writes this marker with the plain reason, exits nonzero
# quietly, and lets the morning slot retry. The doctor renders the reason + time.

def failed_marker_path() -> str:
    """Absolute path of the last-failed-run marker file."""
    return os.path.join(str(paths.OUTPUTS_DIR), FAILED_MARKER_NAME)


def write_last_failed_run(reason: str, when: datetime | None = None) -> None:
    """Record a failed run with its plain reason. Never raises."""
    when = when or datetime.now()
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(failed_marker_path(), "w", encoding="utf-8") as f:
            json.dump({"reason": reason, "at": when.isoformat(timespec="seconds")}, f)
    except Exception:  # noqa: BLE001
        pass


def read_last_failed_run() -> dict | None:
    """The last failed run as {reason, at}, or None when there is none."""
    try:
        with open(failed_marker_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def clear_last_failed_run() -> None:
    """Remove the failed-run marker after a successful run. Never raises."""
    try:
        os.remove(failed_marker_path())
    except Exception:  # noqa: BLE001
        pass


def monitoring_gap_days(today: date | None = None) -> int | None:
    """Calendar days since the last run; None when never run."""
    last = read_last_daily_run()
    if last is None:
        return None
    today = today or date.today()
    return max(0, (today - last).days)


def is_trading_day(when: date) -> bool:
    """Weekend-only rule: Saturday and Sunday are non-trading days."""
    return when.weekday() < 5


def previous_trading_day(when: date) -> date:
    """The trading day before ``when`` (weekend-only rule)."""
    prev = when - timedelta(days=1)
    while not is_trading_day(prev):
        prev -= timedelta(days=1)
    return prev


def staleness_status(today: date | None = None) -> str | None:
    """One-line status when the last run is older than the previous trading day.

    Returns None when the data is current. Used by the app-open fallback.
    """
    today = today or date.today()
    last = read_last_daily_run()
    if last is None:
        return "Updating market data, no previous run."
    if last < previous_trading_day(today):
        return f"Updating market data, last run was {(today - last).days} days ago"
    return None


def gap_line(gap_days: int | None) -> str | None:
    """The exact monitoring-gap sentence, or None when there is no gap."""
    from quant.ui import copy as ui_copy

    if gap_days is None or gap_days <= 1:
        return None
    return ui_copy.MONITORING_GAP.format(n=gap_days)


def _write_daily_artifact(today: date, result: DailyResult, digest: str) -> str:
    """Write the day's digest (with the gap line when catching up). Returns the path."""
    path = os.path.join(str(paths.OUTPUTS_DIR), f"daily_{today.isoformat()}.md")
    lines = [f"# Daily {today.isoformat()}", ""]
    gap = gap_line(result.gap_days)
    if gap:
        lines += [gap, ""]
    lines.append(digest)
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:  # noqa: BLE001
        pass
    return path


def run_daily(
    today: date | None = None,
    *,
    update_fn: Callable[[], Any] | None = None,
    notify: bool = True,
    conn: Any = None,
) -> DailyResult:
    """Run the daily job. Idempotent; never raises.

    update_fn: the market-data refresh (default: none, for tests).
    notify: send the morning digest (it still sends at most once per price date).
    """
    today = today or date.today()
    if not is_trading_day(today):
        return DailyResult("skipped", "non-trading day, skipped")

    from quant.data.database import get_connection

    if conn is None:
        try:
            conn = get_connection()
        except Exception:  # noqa: BLE001
            # v10.7.2 (Part 1.3): never crash with a traceback. Write the plain
            # reason, exit quietly, and let the morning slot retry.
            from quant.ui import copy as ui_copy

            write_last_failed_run(ui_copy.DB_BUSY_RETRY)
            return DailyResult("error", ui_copy.DB_BUSY_RETRY)
    gap = monitoring_gap_days(today)

    try:
        if update_fn is not None:
            update_fn()
    except Exception:  # noqa: BLE001
        pass

    # v10.8.7 (T2): the analysis chain (regime, scores, run manifest), then the
    # mark-to-market valuation store (idempotent).
    try:
        from quant.engine import analysis, mtm
        from quant.engine import manifest as manifest_mod

        analysis.run_analysis(conn)
        res = mtm.backfill(conn, source=mtm.SOURCE_DAILY)
        written = res.get("rows", 0)
        run_id = manifest_mod.latest_run_id(conn) or manifest_mod.new_run_id()
        manifest_mod.record_stage(
            conn, run_id, "valuation", manifest_mod.STATUS_OK,
            coverage=manifest_mod.coverage_text((written, written)))
    except Exception:  # noqa: BLE001
        pass

    # The ONE list from the fresh data: persisted (advice log), written to the
    # daily artifact, and sent as the morning digest (v10.8.10).
    from quant.engine import notify as notify_mod
    from quant.engine.holdings_view import holdings_view
    from quant.engine.mtm import latest_total
    from quant.engine.today_list import build_items, persist
    from quant.reporting.artifacts import read_regime

    persist(conn)
    holdings = holdings_view(conn)
    items = build_items(holdings, regime=read_regime())
    total = latest_total(conn)
    result = DailyResult(
        status="ok",
        message=f"daily run complete: {len(items)} item(s) on the list",
        items=len(items),
        gap_days=gap,
        invested_eur=total.get("value_eur") if total else None,
    )
    try:
        snap = notify_mod.digest_snapshot(conn, items, holdings, today)
        _write_daily_artifact(today, result, notify_mod.build_digest(snap))
    except Exception:  # noqa: BLE001
        _write_daily_artifact(today, result, "")
    if notify:
        result.digest_sent = notify_mod.send_daily_digest(conn, items, holdings, today)

    write_last_daily_run(today)
    clear_last_failed_run()
    return result

"""
daily.py — the daily job body, catch-up, and last-run marker (v10.7.0, Section 3).

Intent: merely owning a powered-on laptop is enough. The daily job is idempotent
and guarded by the existing runner lock. It skips weekends, backfills missed
bars (the range fetch does that), revalues holdings, recomputes scores, evaluates
level-triggered alerts, writes artifacts, and dispatches notifications. Silence
is the default: if no condition fires, it says nothing to the user.

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
from dataclasses import dataclass, field
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
    alerts_created: int = 0
    alerts_notified: int = 0
    gap_days: int | None = None
    invested_eur: float | None = None
    new_alerts: list = field(default_factory=list)


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


def _latest_close(conn, symbol: str) -> float | None:
    """Latest close for a symbol in EUR (v10.8.0, 2.2); None when absent."""
    from quant.data.currency import price_in_eur

    return price_in_eur(symbol, conn=conn)


def _default_holdings(conn) -> list[dict]:
    """Build holding dicts from the broker CSV + tiers (no heavy scoring)."""
    try:
        from quant.portfolio.portfolio import load_portfolio
        from quant.portfolio.tier_manager import load_tiers_safe, tier_map

        df = load_portfolio()
        tiers_df, _ = load_tiers_safe()
        tmap = tier_map(tiers_df)
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    if df is None or getattr(df, "empty", True):
        return out
    for _, row in df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol:
            continue
        value = float(row.get("Current_Value_EUR", 0) or 0)
        entry = float(row.get("Avg_Entry_Price", 0) or 0)
        out.append({
            "symbol": symbol,
            "name": symbol,
            "tier": str(tmap.get(symbol, "ALPHA")).upper(),
            "value_eur": value,
            "entry_price": entry,
            "current_price": _latest_close(conn, symbol),
        })
    return out


def _write_daily_artifact(today: date, result: DailyResult, alerts: list[dict]) -> str:
    """Write a minimal daily markdown artifact. Returns the path."""
    from quant.ui import copy as ui_copy

    path = os.path.join(str(paths.OUTPUTS_DIR), f"daily_{today.isoformat()}.md")
    lines = [f"# Daily {today.isoformat()}", ""]
    gap = gap_line(result.gap_days)
    if gap:
        lines.append(gap)
        lines.append("")
    if alerts:
        lines.append(f"## {ui_copy.SEC_ALERTS}")
        lines.append("")
        for alert in alerts:
            lines.append(f"- {alert.get('message', '')}")
        lines.append("")
    else:
        lines.append(ui_copy.NOTHING_URGENT)
        lines.append("")
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:  # noqa: BLE001
        pass
    return path


def run_daily(
    today: date | None = None,
    *,
    update_fn: Callable[[], Any] | None = None,
    holdings_fn: Callable[[Any], list[dict]] | None = None,
    regime: str | None = None,
    prev_regime: str | None = None,
    notify: bool = True,
    conn: Any = None,
) -> DailyResult:
    """Run the daily job. Idempotent; never raises.

    update_fn: the market-data updater (default: none, for tests).
    holdings_fn: builds holding dicts (default: broker CSV + tiers).
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

    # Revalue holdings and write the invested value series.
    invested = None
    try:
        from quant.engine import valuation

        values = valuation.revalue_holdings(conn, lambda s: _latest_close(conn, s))
        if values:
            invested = sum(values.values())
            valuation.write_value_history(conn, today, invested)
    except Exception:  # noqa: BLE001
        pass

    # Evaluate alerts.
    new_alerts: list[dict] = []
    try:
        from quant.engine import alerts as alerts_mod

        holdings = (holdings_fn(conn) if holdings_fn else _default_holdings(conn))
        new_alerts = alerts_mod.evaluate_alerts(
            conn, holdings, regime=regime, prev_regime=prev_regime, today=today)
    except Exception:  # noqa: BLE001
        new_alerts = []

    result = DailyResult(
        status="ok",
        message="daily run complete",
        alerts_created=len(new_alerts),
        gap_days=gap,
        invested_eur=invested,
        new_alerts=new_alerts,
    )

    # Retention: keep the store bounded (Section 13).
    try:
        from quant.engine import retention

        retention.run_retention(conn, today)
    except Exception:  # noqa: BLE001
        pass

    # v10.7.2 (Part 2.3): recompute the news-pillar status on the Friday run.
    # Absent when zero model-scored items in the last 30 days; the daily job then
    # never imports torch (the performance win on the old laptop).
    try:
        if today.weekday() == 4:
            from quant.engine import news_pillar

            news_pillar.recompute_status(today)
    except Exception:  # noqa: BLE001
        pass

    # Write the artifact (with the gap line when catching up).
    _write_daily_artifact(today, result, new_alerts)

    # Dispatch notifications for new, unnotified alerts.
    if notify and new_alerts:
        try:
            from quant.engine import notify as notify_mod

            sent = 0
            for alert in new_alerts:
                if notify_mod.send_alert(alert):
                    sent += 1
            result.alerts_notified = sent
        except Exception:  # noqa: BLE001
            pass

    write_last_daily_run(today)
    clear_last_failed_run()
    return result


def run_morning(today: date | None = None, **kwargs: Any) -> DailyResult:
    """The 07:45 lightweight slot: at most one heavy computation per day.

    If today's artifact already exists, do not recompute; only send pending
    alerts plus the one-line morning summary. Otherwise run the full job first.
    """
    today = today or date.today()
    artifact = os.path.join(str(paths.OUTPUTS_DIR), f"daily_{today.isoformat()}.md")
    if os.path.exists(artifact):
        from quant.data.database import get_connection
        from quant.engine import alerts as alerts_mod

        conn = kwargs.get("conn") or get_connection()
        open_now = alerts_mod.open_alerts(conn)
        return DailyResult(
            status="ok",
            message=f"Today: {len(open_now)} open action(s).",
            alerts_created=0,
        )
    return run_daily(today, **kwargs)

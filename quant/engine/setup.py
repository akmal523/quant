"""
setup.py — the first-week experience (v10.7.2, Part 4).

Intent: a brand-new user reaches their first approved monthly decision by
following ``quant setup`` and one docs page, with zero prior knowledge. Each step
is idempotent, shows its current status first, and offers a skip. ``--check``
prints the statuses non-interactively (used by tests and CI).

Invariants:
  - ``check_steps`` is pure status computation (no prompting, no writes).
  - ``run_interactive`` never raises; a skip leaves the step untouched.
  - The checklist text is identical to docs/first_week.md.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from quant import paths

# The first-week checklist. IDENTICAL to docs/first_week.md (guarded by a test).
FIRST_WEEK_CHECKLIST = [
    "Set up the Telegram bot (quant notify-setup).",
    "Install the schedule (quant schedule).",
    "Make the first Monthly decision (enter your budget, approve the split).",
    "Execute it in Trade Republic (savings plan and one-off buy).",
    "Enter what you bought (one line), so the math stays honest.",
    "Refresh your broker values if the last sync is older than 35 days.",
    "Live your life.",
    "Glance at the summary on Fridays.",
    "A Telegram alert means place the order in the broker app; it executes at "
    "market open.",
]


@dataclass
class Step:
    """One setup step: a status plus an optional action."""

    key: str
    title: str
    status: str
    done: bool
    action: Callable[[], None] | None = None


def _market_step() -> Step:
    from quant.reporting.artifacts import read_update_state
    from quant.ui import copy as C

    state = read_update_state()
    if state.get("ts"):
        return Step("market", C.SETUP_MARKET,
                    C.SETUP_MARKET_DONE.format(date=C.fmt_ts(state.get("ts"))),
                    True, _refresh_market)
    return Step("market", C.SETUP_MARKET, C.SETUP_MARKET_TODO, False, _refresh_market)


def _tiers_step() -> Step:
    from quant.ui import copy as C

    n = 0
    try:
        with open(paths.DATA_TIERS, encoding="utf-8") as f:
            rows = [ln for ln in f.read().splitlines() if ln.strip()]
        n = max(0, len(rows) - 1)  # minus the header
    except Exception:  # noqa: BLE001
        n = 0
    if n > 0:
        return Step("tiers", C.SETUP_TIERS, C.SETUP_TIERS_DONE.format(n=n), True, None)
    return Step("tiers", C.SETUP_TIERS, C.SETUP_TIERS_TODO, False, None)


def _schedule_step() -> Step:
    from quant.engine import scheduler
    from quant.ui import copy as C

    if scheduler.is_installed():
        return Step("schedule", C.SETUP_SCHEDULE, C.SETUP_SCHEDULE_DONE, True,
                    scheduler.install)
    return Step("schedule", C.SETUP_SCHEDULE, C.SETUP_SCHEDULE_TODO, False,
                scheduler.install)


def _notify_step() -> Step:
    from quant.engine import notify
    from quant.ui import copy as C

    channel = str(notify.load_config().get("channel", "none")).lower()
    if channel in ("telegram", "email"):
        return Step("notify", C.SETUP_NOTIFY, C.SETUP_NOTIFY_DONE, True, None)
    return Step("notify", C.SETUP_NOTIFY, C.SETUP_NOTIFY_TODO, False, None)


def _backup_step() -> Step:
    from quant.engine import backup
    from quant.ui import copy as C

    last = backup.last_backup_at()
    if last is not None:
        return Step("backup", C.SETUP_BACKUP,
                    C.SETUP_BACKUP_DONE.format(date=C.fmt_date(last.date())), True,
                    _run_backup)
    return Step("backup", C.SETUP_BACKUP, C.SETUP_BACKUP_TODO, False, _run_backup)


def _checklist_step() -> Step:
    from quant.ui import copy as C

    return Step("checklist", C.SETUP_CHECKLIST, C.SETUP_CHECKLIST_HINT, True, None)


def check_steps() -> list[Step]:
    """The six setup steps with their current status. Pure; never raises."""
    steps: list[Step] = []
    for fn in (_market_step, _tiers_step, _schedule_step, _notify_step,
               _backup_step, _checklist_step):
        try:
            steps.append(fn())
        except Exception:  # noqa: BLE001
            steps.append(Step("unknown", "Step", "unavailable", False, None))
    return steps


def check_lines() -> list[str]:
    """Plain status lines for ``quant setup --check``."""
    from quant.ui import copy as C

    lines = [C.SETUP_HEADER]
    for i, step in enumerate(check_steps(), 1):
        marker = C.SETUP_DONE if step.done else C.SETUP_TODO
        lines.append(C.SETUP_STEP_LINE.format(
            n=i, title=step.title, status=f"{marker} ({step.status})"))
    return lines


def checklist_lines() -> list[str]:
    """The first-week checklist as numbered lines."""
    return [f"{i}. {item}" for i, item in enumerate(FIRST_WEEK_CHECKLIST, 1)]


# ── Actions (used by the interactive flow) ────────────────────────────────────

def _refresh_market() -> None:
    from quant.data.data_updater import main as updater_main

    updater_main()


def _run_backup() -> None:
    from quant.engine import backup

    backup.create_backup()


def run_interactive(input_fn: Callable[[str], str] = input,
                    print_fn: Callable[[str], None] = print) -> int:
    """Walk the six steps, showing status first and offering a skip. Never raises."""
    from quant.ui import copy as C

    for i, step in enumerate(check_steps(), 1):
        print_fn(C.SETUP_STEP_LINE.format(n=i, title=step.title, status=step.status))
        if step.done or step.action is None:
            continue
        try:
            answer = input_fn(f"{step.title}: {C.SETUP_ACTION} / {C.SETUP_SKIP}? ").strip().lower()
        except EOFError:
            answer = C.SETUP_SKIP
        if answer in ("y", "yes", C.SETUP_ACTION):
            try:
                step.action()
            except Exception as e:  # noqa: BLE001
                print_fn(f"{step.title}: failed ({type(e).__name__})")
    print_fn("")
    print_fn(C.SETUP_CHECKLIST)
    for line in checklist_lines():
        print_fn(line)
    return 0

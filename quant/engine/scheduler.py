"""
scheduler.py — systemd user timer install/status (v10.7.0, Section 3.1).

Intent: merely owning a powered-on laptop is enough. A systemd user timer runs
the daily job at 18:45 and a lightweight morning slot at 07:45. Persistent=true
fires a missed slot once at the next wake or boot. loginctl enable-linger keeps
timers alive across logout and reboot.

Invariants:
  - The unit-file builders are pure and unit-tested; tests never touch systemd.
  - install/uninstall/status never raise; they return a status string.
  - Absolute paths to the venv python and project dir are read at install time.
"""
from __future__ import annotations

import os
import subprocess
import sys

from quant import paths
from quant.config import SCHEDULE_DAILY_SLOT, SCHEDULE_MORNING_SLOT

SERVICE_NAME = "quant-daily.service"
TIMER_NAME = "quant-daily.timer"
BOT_SERVICE_NAME = "quant-bot.service"


def systemd_user_dir() -> str:
    """The systemd user unit directory."""
    return os.path.expanduser("~/.config/systemd/user")


def build_service_unit(python_path: str, project_dir: str) -> str:
    """The service unit content. Absolute paths, no hardcoding."""
    return (
        "[Unit]\n"
        "Description=Quant-AI daily job\n"
        "After=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"WorkingDirectory={project_dir}\n"
        f"ExecStart={python_path} -m quant.cli daily\n"
    )


def build_bot_unit(python_path: str, project_dir: str) -> str:
    """The Telegram bot service (v10.9.0): long-running, restarted on failure."""
    return (
        "[Unit]\n"
        "Description=Quant-AI Telegram bot\n"
        "After=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={project_dir}\n"
        f"ExecStart={python_path} -m quant.cli bot\n"
        "Restart=on-failure\n"
        "RestartSec=30\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def build_timer_unit(
    daily_slot: str = SCHEDULE_DAILY_SLOT,
    morning_slot: str = SCHEDULE_MORNING_SLOT,
) -> str:
    """The timer unit content with two OnCalendar slots and Persistent=true."""
    return (
        "[Unit]\n"
        "Description=Quant-AI daily timer\n"
        "\n"
        "[Timer]\n"
        f"OnCalendar=*-*-* {daily_slot}:00\n"
        f"OnCalendar=*-*-* {morning_slot}:00\n"
        "Persistent=true\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )


def _run(cmd: list[str]) -> bool:
    try:
        return subprocess.call(cmd) == 0
    except Exception:  # noqa: BLE001
        return False


def install() -> str:
    """Install the timer + service and enable lingering. Returns a status line."""
    unit_dir = systemd_user_dir()
    try:
        os.makedirs(unit_dir, exist_ok=True)
        with open(os.path.join(unit_dir, SERVICE_NAME), "w", encoding="utf-8") as f:
            # v10.8.1: the service runs from the code checkout, not the data dir.
            f.write(build_service_unit(sys.executable, str(paths.CODE_ROOT)))
        with open(os.path.join(unit_dir, TIMER_NAME), "w", encoding="utf-8") as f:
            f.write(build_timer_unit())
    except Exception as e:  # noqa: BLE001
        return f"install failed: {type(e).__name__}"
    _run(["systemctl", "--user", "daemon-reload"])
    _run(["systemctl", "--user", "enable", "--now", TIMER_NAME])
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if user:
        _run(["loginctl", "enable-linger", user])
    return f"installed: {TIMER_NAME} at {SCHEDULE_DAILY_SLOT} and {SCHEDULE_MORNING_SLOT}"


def uninstall() -> str:
    """Disable and remove the timer + service. Returns a status line."""
    _run(["systemctl", "--user", "disable", "--now", TIMER_NAME])
    unit_dir = systemd_user_dir()
    for name in (TIMER_NAME, SERVICE_NAME):
        try:
            os.remove(os.path.join(unit_dir, name))
        except Exception:  # noqa: BLE001
            pass
    _run(["systemctl", "--user", "daemon-reload"])
    return "uninstalled"


def install_bot() -> str:
    """Install and start the Telegram bot service. Returns a status line."""
    unit_dir = systemd_user_dir()
    try:
        os.makedirs(unit_dir, exist_ok=True)
        with open(os.path.join(unit_dir, BOT_SERVICE_NAME), "w", encoding="utf-8") as f:
            f.write(build_bot_unit(sys.executable, str(paths.CODE_ROOT)))
    except Exception as e:  # noqa: BLE001
        return f"install failed: {type(e).__name__}"
    _run(["systemctl", "--user", "daemon-reload"])
    _run(["systemctl", "--user", "enable", "--now", BOT_SERVICE_NAME])
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if user:
        _run(["loginctl", "enable-linger", user])
    return f"installed: {BOT_SERVICE_NAME}"


def uninstall_bot() -> str:
    """Stop and remove the Telegram bot service. Returns a status line."""
    _run(["systemctl", "--user", "disable", "--now", BOT_SERVICE_NAME])
    try:
        os.remove(os.path.join(systemd_user_dir(), BOT_SERVICE_NAME))
    except Exception:  # noqa: BLE001
        pass
    _run(["systemctl", "--user", "daemon-reload"])
    return "uninstalled"


def is_bot_installed() -> bool:
    """True when the bot service unit file exists."""
    return os.path.exists(os.path.join(systemd_user_dir(), BOT_SERVICE_NAME))


def bot_running() -> bool:
    """True when systemd reports the bot service active. Never raises."""
    try:
        res = subprocess.run(["systemctl", "--user", "is-active", BOT_SERVICE_NAME],
                             capture_output=True, text=True, timeout=5)
        return res.stdout.strip() == "active"
    except Exception:  # noqa: BLE001
        return False


def restart_bot() -> str:
    """Restart the bot service so it runs the current code (after an update)."""
    if not is_bot_installed():
        return "not installed; run quant bot --install"
    _run(["systemctl", "--user", "restart", BOT_SERVICE_NAME])
    return "restarted" if bot_running() else "restart failed; see journalctl --user -u quant-bot"


def bot_status_line() -> str:
    """One plain line: running in the background, stopped, or not installed."""
    if not is_bot_installed():
        return "not running in the background"
    return "running in the background" if bot_running() else "installed but stopped"


def is_installed() -> bool:
    """True when the timer unit file exists."""
    return os.path.exists(os.path.join(systemd_user_dir(), TIMER_NAME))


def status_line() -> str:
    """One plain line: installed yes/no, slots, last run, next expected run."""
    from quant.engine.daily import read_last_daily_run

    last = read_last_daily_run()
    last_str = last.isoformat() if last else "none"
    if not is_installed():
        return "not installed; run quant schedule"
    return (
        f"installed, slots {SCHEDULE_DAILY_SLOT} and {SCHEDULE_MORNING_SLOT}, "
        f"last run {last_str}"
    )

"""quant — command-line interface.

Intent: one console entry point (``quant``; ``python -m quant.cli`` without an
activated environment). Six user commands (``dash``, ``refresh``, ``daily``,
``upgrade``, ``backup``, ``doctor``) plus hidden maintenance commands.

State Transition:
  quant refresh   -> fetch market data + funnel survivors (writes DuckDB), then
                     persist the ONE list
  quant run       -> score, audit the holdings table, write the review, run the
                     analysis chain, persist the ONE list
  quant all       -> refresh then run
  quant daily     -> the timer's job (quant.engine.daily.run_daily)

Invariants:
  - Importing this package has NO side effects (heavy imports are lazy).
  - Exit code 0 on success; non-zero on failure (fail fast).

Dependencies: argparse; pipeline modules imported lazily inside commands.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence

from quant.cli import output


def _lock_or_fail(command: str) -> bool:
    """Acquire the shared runner lock or report a plain, retryable failure.

    v10.7.2 (Part 1): every writer respects one lock. When a live owner holds it
    past the retry budget, write the last-failed-run marker and return False so
    the caller exits nonzero quietly (the morning slot retries).
    """
    from quant.engine import lock as lock_mod

    res = lock_mod.acquire(command, blocking=True)
    if res.acquired:
        return True
    from quant.engine.daily import write_last_failed_run
    from quant.ui import copy as ui_copy

    write_last_failed_run(ui_copy.DB_BUSY_RETRY)
    print(ui_copy.DB_BUSY_RETRY)
    return False


def _cmd_update(_args: argparse.Namespace) -> int:
    """Fetch market data + run the funnel (step 1). Returns an exit code."""
    from quant.data.data_updater import main as updater_main
    from quant.engine import lock as lock_mod

    if not _lock_or_fail("update"):
        return 1
    try:
        rc = updater_main()
        _persist_today_list()
        return rc
    finally:
        lock_mod.release()


def _persist_today_list() -> None:
    """Writers only (R-C4): rebuild the ONE list and sync the advice log."""
    try:
        from quant.engine.today_list import persist

        persist()
    except Exception:  # noqa: BLE001
        pass


def _cmd_run(_args: argparse.Namespace) -> int:
    """Run the analysis chain on the stored data, then persist the ONE list.

    v10.8.10: this used to run a second, older review pipeline (process pool,
    HMM, FinBERT on filing text, audit CSV, run artifacts) whose output no page
    read and whose workers failed on the database lock. ``quant run`` is now the
    same chain the daily job runs: regime, scores, run manifest.
    """
    from quant.data.database import init_db, write_connection
    from quant.engine import analysis
    from quant.engine import lock as lock_mod

    if not _lock_or_fail("run"):
        return 1
    try:
        init_db()
        with write_connection() as conn:
            res = analysis.run_analysis(conn)
        for stage in res["stages"]:
            output.reporter.line(f"  {stage['stage']}: {stage['status']} "
                                 f"{stage.get('coverage') or ''}".rstrip())
        _persist_today_list()
        return 0
    finally:
        lock_mod.release()


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Read-only diagnosis. No writes, no secrets; safe to paste publicly."""
    import os

    from quant import paths
    from quant.data.database import read_only_connection
    from quant.data.names import probe_metadata, read_names_state

    _verbose = bool(getattr(_args, "verbose", False)) if _args is not None else False

    print(f"db: {paths.DB_FILE}")
    # F-series: a fresh install has no schema yet. Initialize it so counts are
    # real (0, not -1) and market_history resolves instead of raising a
    # CatalogException. init_db also seeds the writable data dir.
    try:
        from quant.data.database import init_db

        init_db()
    except Exception as e:  # noqa: BLE001
        print(f"db init: failed ({type(e).__name__}: {e})")
    try:
        with read_only_connection() as conn:
            def _count(sql: str) -> int:
                try:
                    return int(conn.execute(sql).fetchone()[0])
                except Exception:  # noqa: BLE001
                    return -1

            total = _count("SELECT COUNT(*) FROM asset_registry")
            miss_dn = _count("SELECT COUNT(*) FROM asset_registry "
                             "WHERE display_name IS NULL OR trim(display_name) = ''")
            miss_nm = _count("SELECT COUNT(*) FROM asset_registry "
                             "WHERE name IS NULL OR trim(name) = ''")
            miss_cur = _count("SELECT COUNT(*) FROM asset_registry "
                              "WHERE currency IS NULL OR trim(currency) = ''")
            miss_isin = _count("SELECT COUNT(*) FROM asset_registry "
                               "WHERE isin IS NULL OR trim(isin) = ''")
            if _verbose:
                try:
                    from quant.data.registry_repair import working_universe_sources

                    bd = working_universe_sources(conn)
                    for _name in ("broad_etfs", "core_active", "portfolio",
                                  "broker_registry", "curated", "survivors"):
                        print(f"  {_name}: {len(bd.get(_name, set()))}")
                    print(f"  working universe: "
                          f"{len(set().union(*bd.values())) if bd else 0}")
                except Exception:  # noqa: BLE001
                    pass
        print(f"registry rows: {total} (working universe)")
        print(f"missing display_name: {miss_dn}")
        print(f"missing name: {miss_nm}")
        print(f"missing currency: {miss_cur}")
        print(f"missing isin: {miss_isin}")
    except Exception as e:  # noqa: BLE001
        print(f"registry: unreadable ({type(e).__name__}: {e})")

    state = read_names_state()
    if state:
        print(f"names state: ts={state.get('ts')} rows={state.get('rows_total')} "
              f"filled={state.get('filled')} still_missing={state.get('still_missing')} "
              f"skipped={state.get('skipped_reason')}")
    else:
        print("names state: none (backfill has not run)")

    for sym in ("AMZN", "AAPL", "EUNL.DE"):
        pr = probe_metadata(sym)
        if pr.get("error"):
            print(f"probe {sym}: ERROR {pr['error']}")
        else:
            print(f"probe {sym}: longName={pr.get('long_name')!r} "
                  f"currency={pr.get('currency')!r}")

    # v10.8.10: news = the scored headlines in nlp_evidence (written by refresh).
    try:
        from quant.data.news import model_available

        with read_only_connection() as conn:
            n, scored, newest = conn.execute(
                "SELECT COUNT(*), COUNT(score), MAX(retrieved_at) FROM nlp_evidence"
            ).fetchone()
        model = "on" if model_available() else "off (install the news extra)"
        print(f"news: {n} headlines, {scored} scored, newest {newest or 'never'}; "
              f"model {model}")
    except Exception as e:  # noqa: BLE001
        print(f"news: unreadable ({e})")

    # v10.7.4 (Part 5.2): portfolio data issues and unpriceable holdings.
    try:
        from quant.portfolio.portfolio import portfolio_issues

        for issue in portfolio_issues():
            print(issue)
    except Exception as e:  # noqa: BLE001
        print(f"portfolio issues: unreadable ({e})")
    try:
        from quant.data.database import read_only_connection
        from quant.portfolio.portfolio import load_portfolio

        held = list(load_portfolio(paths.DATA_PORTFOLIO).get("Symbol", []))
        with read_only_connection() as conn:
            priced = {r[0] for r in conn.execute(
                "SELECT DISTINCT Symbol FROM market_history").fetchall()}
        unpriceable = [s for s in held if s not in priced]
        if unpriceable:
            print(f"unpriceable: {sorted(unpriceable)}")
    except Exception as e:  # noqa: BLE001
        print(f"unpriceable: unreadable ({e})")

    try:
        from quant.ui.search import load_index, search

        idx = load_index()
        for q in ("apple", "amazon", "gold", "space", "samsung"):
            res = search(idx, q, 5)
            if res:
                print(f"search {q}: " + ", ".join(r["label"] for r in res))
            else:
                print(f"search {q}: no match")
    except Exception as e:  # noqa: BLE001
        print(f"search probes: unreadable ({e})")

    # H3.5 (F1-F3): the Explore card fields, read through the SAME helper the
    # page uses, so a title/subtitle regression is diagnosable forever.
    try:
        from quant.ui.cards import explore_card_fields

        for sym in ("AMZN", "5J50.DE", "SGLN.L"):
            f = explore_card_fields(sym)
            print(f"explore card {sym}: title={f['name']!r} "
                  f"subtitle={f['subtitle']!r} details={f['has_details']}")
    except Exception as e:  # noqa: BLE001
        print(f"explore cards: unreadable ({e})")

    try:
        with read_only_connection() as conn:
            for sym in ("AMZN", "AAPL", "EUNL.DE"):
                n = conn.execute(
                    "SELECT COUNT(*) FROM market_history WHERE Symbol = ?", [sym]
                ).fetchone()[0]
                print(f"probe {sym} history: {n} bars")
    except Exception as e:  # noqa: BLE001
        print(f"history probes: unreadable ({e})")

    try:
        import importlib.util

        avail = (importlib.util.find_spec("transformers") is not None
                 and importlib.util.find_spec("torch") is not None)
        print(f"sentiment model: {'available' if avail else 'not available'}")
    except Exception:  # noqa: BLE001
        print("sentiment model: not available")

    try:
        from quant.engine import lock as lock_mod

        print(lock_mod.status_line())
    except Exception as e:  # noqa: BLE001
        print(f"runner lock: unreadable ({e})")

    # ── v10.7.0: automation and data-retention status (Section 13) ───────────
    try:
        size_mb = os.path.getsize(paths.DB_FILE) / (1024 * 1024)
        print(f"database size: {size_mb:.2f} MB")
    except Exception as e:  # noqa: BLE001
        print(f"database size: unreadable ({e})")

    try:
        from quant.engine.daily import monitoring_gap_days, read_last_daily_run

        last = read_last_daily_run()
        print(f"last daily run: {last.isoformat() if last else 'none'}")
        gap = monitoring_gap_days()
        print(f"monitoring gap: {gap if gap is not None else 'n/a'} days")
    except Exception as e:  # noqa: BLE001
        print(f"last daily run: unreadable ({e})")

    # v10.7.2 (Part 1.3): the last failed run with its plain reason and time.
    try:
        from quant.engine.daily import read_last_failed_run

        failed = read_last_failed_run()
        if failed:
            print(f"last failed run: {failed.get('at')} - {failed.get('reason')}")
        else:
            print("last failed run: none")
    except Exception as e:  # noqa: BLE001
        print(f"last failed run: unreadable ({e})")

    # v10.7.2 (Part 3.1): the last backup line.
    try:
        from quant.engine import backup

        print(backup.backup_line())
    except Exception as e:  # noqa: BLE001
        print(f"backup: unreadable ({e})")

    # v10.8.10: stale pre-10.8.1 copies of user state left in the code folder.
    from quant.data.user_data import leftover_legacy_state

    leftover = leftover_legacy_state()
    if leftover:
        print(f"old copies in {paths.CODE_ROOT} (already migrated to {paths.PROJECT_ROOT}; "
              f"not used, safe to delete after checking): {', '.join(leftover)}")

    try:
        with read_only_connection() as conn:
            _row = conn.execute("SELECT MAX(snapshot_date) FROM position_snapshots"
                                ).fetchone()
        _last = _row[0] if _row else None
        print(f"last confirm and save: {_last or 'never'}")
    except Exception as e:  # noqa: BLE001
        print(f"last confirm and save: unreadable ({e})")

    try:
        from quant.engine.scheduler import status_line

        print(f"scheduler: {status_line()}")
    except Exception:  # noqa: BLE001
        print("scheduler: not installed; run quant schedule")

    try:
        from quant.engine.notify import status_line as notify_status

        print(f"notifications: {notify_status()}")
    except Exception:  # noqa: BLE001
        print("notifications: off; run quant notify-setup")
    return 0


def _cmd_backup(args: argparse.Namespace) -> int:
    """Archive the user-owned state (v10.7.2, Part 3.1)."""
    from quant.engine import backup
    from quant.ui import copy as C

    result = backup.create_backup(
        custom_dir=getattr(args, "dir", None),
        include_secrets=getattr(args, "include_secrets", False))
    if not result["ok"]:
        print(f"backup failed: {result.get('error')}")
        return 1
    if result["warning"]:
        print(C.BACKUP_WARNING_SECRETS)
    print(C.BACKUP_DONE.format(path=result["path"], size=f"{result['size_mb']:.2f}"))
    print(C.BACKUP_MEMBERS.format(members=", ".join(result["members"])))
    print(C.BACKUP_RESTORE_HINT)
    return 0


def _cmd_dash(args: argparse.Namespace) -> int:
    """Launch the local interactive workspace (Streamlit).

    A1 (v10.5.2): bind localhost by default; ``--lan`` binds 0.0.0.0 so a phone
    on the same Wi-Fi can reach it. Never expose to the internet without an
    authenticating reverse proxy.
    """
    from quant import paths

    # B1/B4: backfill display names/ISINs before the app opens so a pre-existing
    # DB shows friendly names without a manual update. The ONLY write-enabled
    # connection outside update/run/publish (app startup).
    try:
        from quant.data.names import ensure_display_names

        ensure_display_names()
    except Exception:  # noqa: BLE001
        pass

    # F-series: the dashboard script lives in the PACKAGE, not the writable root
    # (which is the user data dir when installed).
    dash = os.path.join(str(paths.PACKAGE_DIR), "dashboard.py")
    address = "0.0.0.0" if getattr(args, "lan", False) else "127.0.0.1"
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", dash,
         "--server.address", address]
    )


def _cmd_all(args: argparse.Namespace) -> int:
    """Full daily cycle: update then run."""
    rc = _cmd_update(args)
    if rc != 0:
        return rc
    return _cmd_run(args)


def _cmd_validate_tiers(_args: argparse.Namespace) -> int:
    """Validate data/tiers.csv and report issues (v10.6.3)."""
    from quant.portfolio.portfolio import load_portfolio
    from quant.portfolio.tier_manager import load_tiers, validate_tiers_csv

    portfolio_df = load_portfolio()
    tiers_df = load_tiers()
    is_valid, errors = validate_tiers_csv(tiers_df, portfolio_df)
    if is_valid:
        print("tiers.csv is valid.")
        return 0
    print("tiers.csv has issues:")
    for e in errors:
        print(f"  - {e}")
    print("Run 'quant repair-tiers' to attempt automatic repair.")
    return 1


def _cmd_repair_tiers(_args: argparse.Namespace) -> int:
    """Repair common issues in data/tiers.csv (v10.6.3)."""
    from quant.portfolio.portfolio import load_portfolio
    from quant.portfolio.tier_manager import (
        load_tiers,
        repair_tiers_csv,
        save_tiers,
        validate_tiers_csv,
    )

    portfolio_df = load_portfolio()
    tiers_df = load_tiers()
    repaired = repair_tiers_csv(tiers_df, portfolio_df)
    is_valid, errors = validate_tiers_csv(repaired, portfolio_df)
    if is_valid:
        save_tiers(repaired)
        print("tiers.csv repaired.")
        return 0
    print("Repair failed. Manual intervention required:")
    for e in errors:
        print(f"  - {e}")
    return 1


def _cmd_health_check(_args: argparse.Namespace) -> int:
    """Run the system health check (v10.6.4)."""
    from quant.cli.health import run_health_check

    result = run_health_check()
    print(f"System health: {result['status']}")
    for check in result["checks"]:
        print(f"  {check['name']:<20} {check['status']:<9} {check['message']}")
    return 1 if result["status"] == "CRITICAL" else 0


def _cmd_schedule(args: argparse.Namespace) -> int:
    """Install, remove, or report the local daily timer (v10.7.0, Section 3.1)."""
    from quant.engine import scheduler

    if getattr(args, "off", False):
        print(scheduler.uninstall())
        return 0
    if getattr(args, "status", False):
        print(scheduler.status_line())
        return 0
    print(scheduler.install())
    return 0


def _cmd_notify_setup(_args: argparse.Namespace) -> int:
    """Interactive notification setup (v10.7.0, Section 4.1)."""
    from quant.engine import notify

    channel = input("Channel (telegram/email/none): ").strip().lower()
    config: dict = {"channel": channel or "none"}
    if channel == "telegram":
        print("Create a bot via BotFather; get your chat id via a user-info bot.")
        config["telegram_bot_token"] = input("Bot token: ").strip()
        config["telegram_chat_id"] = input("Chat id: ").strip()
    elif channel == "email":
        config["email_from"] = input("From address: ").strip()
        config["email_password"] = input("App password: ").strip()
        config["email_to"] = input("To address: ").strip()
    ok = notify.send_test(config)
    config["tested"] = bool(ok)
    notify.save_config(config)
    print("test message sent" if ok else "test failed; configuration saved anyway")
    return 0


def _cmd_daily(_args: argparse.Namespace) -> int:
    """Run the daily job (used by the timer). One line to stdout (Section 3.2)."""
    from quant.engine import daily
    from quant.engine import lock as lock_mod

    if not _lock_or_fail("daily"):
        return 1
    try:
        def _update() -> None:
            from quant.data.data_updater import main as updater_main

            updater_main()

        result = daily.run_daily(update_fn=_update)
        print(result.message)
        # A DB-busy failure exits nonzero quietly; the morning slot retries.
        return 1 if result.status == "error" else 0
    finally:
        lock_mod.release()


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``quant`` argument parser."""
    from quant import __version__

    parser = argparse.ArgumentParser(
        prog="quant",
        description="Quant-AI systematic equity pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"quant-ai {__version__}")
    # Global flag: default output is terse (R4). Detail goes to the run log.
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show per-symbol and per-step detail on stdout",
    )
    # v10.8.2 (section 9): six user commands; the rest are hidden internals (still
    # dispatchable, used by the scheduler and tests) and never appear in --help.
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    hidden = argparse.SUPPRESS

    # ── The six user commands ─────────────────────────────────────────────────
    dash = sub.add_parser("dash", help="launch the local interactive workspace")
    dash.add_argument("--lan", action="store_true",
                      help="also serve on the local network (phone on the same Wi-Fi)")
    sub.add_parser("refresh", help="fetch market data and analyse your holdings")
    sub.add_parser("daily", help="run the daily check (used by the timer)")
    sub.add_parser("upgrade", help="update the software to the latest version")
    backup_p = sub.add_parser("backup", help="archive your data (tar.gz)")
    backup_p.add_argument("--dir", default=None, help="output directory")
    backup_p.add_argument("--include-secrets", action="store_true",
                          help="also include the notification token file")
    sub.add_parser("doctor", help="read-only diagnosis (safe to paste)")

    # ── Hidden internal commands ──────────────────────────────────────────────
    sub.add_parser("run", help=hidden)
    sub.add_parser("all", help=hidden)
    sub.add_parser("validate-tiers", help=hidden)
    sub.add_parser("repair-tiers", help=hidden)
    sub.add_parser("health-check", help=hidden)
    sched = sub.add_parser("schedule", help=hidden)
    sched.add_argument("--off", action="store_true")
    sched.add_argument("--status", action="store_true")
    sub.add_parser("notify-setup", help=hidden)

    # argparse lists every subparser in --help; keep only the six user commands
    # (hidden subparsers remain dispatachable, used by the scheduler and tests).
    _user = {"dash", "refresh", "daily", "upgrade", "backup", "doctor"}
    actions = getattr(sub, "_choices_actions", [])
    sub._choices_actions = [a for a in actions if getattr(a, "dest", None) in _user]
    return parser


def _cmd_upgrade(args) -> int:
    """Update the software: fast-forward pull, reinstall, migrate (v10.8.0).

    Aborts safely on local changes. Prints the old and new version.
    """
    import subprocess
    import sys

    from quant import __version__, paths

    # v10.8.1: the git checkout is CODE_ROOT, not the (now per-user) data root.
    repo = paths.CODE_ROOT
    if not (repo / ".git").exists():
        output.reporter.line("Not a git checkout; update manually.")
        return 2
    # v10.8.1 (A2): back up the per-user data dir before touching the software.
    try:
        from quant.engine import backup as _backup

        _backup.create_backup()
    except Exception:  # noqa: BLE001
        pass
    # v10.8.1 (A2): copy any tracked legacy user-state files forward before pull.
    try:
        from quant.data.user_data import migrate_legacy_user_data

        migrate_legacy_user_data()
    except Exception:  # noqa: BLE001
        pass
    status = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo),
                            capture_output=True, text=True)
    if status.stdout.strip():
        output.reporter.line("Local changes present; commit or stash them first.")
        return 2
    old = __version__
    pull = subprocess.run(["git", "pull", "--ff-only"], cwd=str(repo),
                          capture_output=True, text=True)
    if pull.returncode != 0:
        output.reporter.line("Could not fast-forward; resolve the branch first.")
        return 2
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[dashboard]"],
                   cwd=str(repo))
    from quant.data.database import init_db

    init_db()
    from quant import __version__ as new_version

    output.reporter.line(f"Updated {old} to {new_version}.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point.

    Exit codes (spec 3.1): 0 success, 1 data-quality gate failure, 2 config error.
    """
    if argv is None:
        import sys

        argv = sys.argv[1:]
    # v10.8.0: `update` is a hidden alias for `refresh`.
    argv = ["refresh" if a == "update" else a for a in argv]
    parser = build_parser()
    args = parser.parse_args(argv)

    # Only commands that produce a run create a run dir + pipeline.log; the
    # others (dash, doctor, ...) must NOT mint a new (empty) run dir, or
    # latest_run() would point at it instead of the last real run.
    log_path = (
        output.default_log_path()
        if args.command in ("refresh", "update", "run", "all")
        else None
    )
    output.configure(verbose=getattr(args, "verbose", False), log_path=log_path)

    dispatch = {
        "refresh": _cmd_update,
        "update": _cmd_update,
        "upgrade": _cmd_upgrade,
        "run": _cmd_run,
        "all": _cmd_all,
        "dash": _cmd_dash,
        "doctor": _cmd_doctor,
        "backup": _cmd_backup,
        "validate-tiers": _cmd_validate_tiers,
        "repair-tiers": _cmd_repair_tiers,
        "health-check": _cmd_health_check,
        "schedule": _cmd_schedule,
        "notify-setup": _cmd_notify_setup,
        "daily": _cmd_daily,
    }
    try:
        return dispatch[args.command](args)
    except Exception as e:  # noqa: BLE001
        # Configuration / unexpected error: one error line + one remedy line.
        output.reporter.line(f"error: {e}")
        output.reporter.line("remedy: check the run log and configuration, then retry.")
        return 2
    finally:
        output.reporter.close()


__all__ = ["main", "build_parser"]

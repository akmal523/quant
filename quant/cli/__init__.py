"""quant — command-line interface.

Intent: one console entry point (``quant``) for the two-step pipeline plus
reconciliation, replacing direct ``python3 main.py`` / ``python3 data_updater.py``
invocations. The root wrapper scripts remain as thin compatibility shims.

State Transition:
  quant update    -> fetch market data + funnel survivors (writes DuckDB)
  quant run       -> score, audit the portfolio, emit reports
  quant reconcile -> diff the theoretical portfolio vs the broker CSV export
  quant all       -> update then run (full daily cycle)

Invariants:
  - Importing this package has NO side effects (heavy imports are lazy).
  - Exit code 0 on success; non-zero on failure (fail fast).

Dependencies: argparse; pipeline modules imported lazily inside commands.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from quant.cli import output


def _cmd_update(_args: argparse.Namespace) -> int:
    """Fetch market data + run the funnel (step 1). Returns an exit code."""
    from quant.data.data_updater import main as updater_main

    return updater_main()


def _cmd_run(_args: argparse.Namespace) -> int:
    """Score, audit, and report (step 2). Returns an exit code.

    H3.3: a failed review writes review_status=failed (+ error) to the run artifact
    so Today/Health/Reviews read ONE field; no portfolio_history row is written.
    """
    from quant.main import main as pipeline_main

    try:
        return pipeline_main()
    except Exception as e:  # noqa: BLE001
        try:
            from quant.reporting.artifacts import new_run_dir, save_metrics

            save_metrics(new_run_dir(), {"review_status": "failed", "error": str(e)})
        except Exception:  # noqa: BLE001
            pass
        raise


def _cmd_publish(_args: argparse.Namespace) -> int:
    """Render the static Published Briefing from the latest run artifacts."""
    from quant.reporting.web import publish

    return publish()


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Read-only diagnosis. No writes, no secrets; safe to paste publicly."""
    import json
    import os
    import time

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

    try:
        from quant.data import news as _news

        cache_path = _news._cache_path()
        if os.path.exists(cache_path):
            data = json.load(open(cache_path, encoding="utf-8"))
            ages = [time.time() - float(v.get("retrieved_at", 0)) for v in data.values()]
            print(f"news cache: {len(data)} symbols, newest {min(ages) / 3600:.1f} h")
        else:
            print("news cache: none")
    except Exception as e:  # noqa: BLE001
        print(f"news cache: unreadable ({e})")

    # H3.6 (N3): sentiment provenance distribution over the news cache.
    try:
        from quant.data.news import cache_scorer_stats

        _st = cache_scorer_stats()
        print(f"sentiment cache: {_st['entries']} entries, scorer model {_st['model']} / "
              f"default {_st['default']}, pos {_st['pos']} neg {_st['neg']} neu {_st['neu']}")
    except Exception as e:  # noqa: BLE001
        print(f"sentiment cache: unreadable ({e})")

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

    # H3.8 (M4): the advice engine, diagnosable like everything else.
    try:
        import pandas as _pd

        from quant.reporting.actions import build_actions

        _fix = _pd.DataFrame([
            {"Symbol": "EUNL.DE", "Tier": "CORE", "Drift": "-16.1%",
             "Target_Weight": "50.0%",
             "Recommendation": "BUY 150 EUR (CORE drift -16.1% exceeds 10.0% threshold)"},
            {"Symbol": "SXRV.DE", "Tier": "SATELLITE", "Drift": "10.5%",
             "Target_Weight": "20.0%",
             "Recommendation": "SELL 100 EUR (SATELLITE drift 10.5% exceeds 5.0% threshold)"},
        ])
        _acts = [a for a in build_actions(_fix) if not a.get("blocked")]
        print(f"actions oracle: {len(_acts)} actions on fixture")
    except Exception as e:  # noqa: BLE001
        print(f"actions oracle: unreadable ({e})")

    try:
        lock_path = os.path.join(str(paths.OUTPUTS_DIR), ".runner.lock")
        if os.path.exists(lock_path):
            hb = json.load(open(lock_path, encoding="utf-8"))
            age = time.time() - float(hb.get("ts", 0))
            if age > 600:
                print(f"runner lock: stale (auto-release on next operation), age={age:.0f}s")
            else:
                print(f"runner lock: present pid={hb.get('pid')} age={age:.0f}s")
        else:
            print("runner lock: none")
    except Exception as e:  # noqa: BLE001
        print(f"runner lock: unreadable ({e})")
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


def _cmd_reconcile(args: argparse.Namespace) -> int:
    """Diff the theoretical portfolio against the broker CSV export."""
    import pandas as pd

    from quant import paths
    from quant.execution.reconciliation import (
        build_theoretical_snapshot,
        reconcile,
        save_snapshot,
    )
    from quant.portfolio.portfolio import load_portfolio

    today = dt.date.today().isoformat()
    portfolio_df = load_portfolio(str(paths.DATA_PORTFOLIO))
    if portfolio_df.empty:
        print("[!] portfolio.csv is empty or missing; nothing to reconcile.")
        return 1

    scan_path = Path(paths.OUTPUTS_DIR) / "market_scan_v8.csv"
    if scan_path.exists():
        try:
            scan_df = pd.read_csv(scan_path)
        except Exception:  # noqa: BLE001
            scan_df = pd.DataFrame(columns=["Symbol", "Current_Price"])
    else:
        scan_df = pd.DataFrame(columns=["Symbol", "Current_Price"])

    theoretical = build_theoretical_snapshot(portfolio_df, scan_df, today)
    save_snapshot(theoretical)

    broker_path = Path(args.broker)
    broker_df = pd.read_csv(broker_path, comment="#") if broker_path.exists() else pd.DataFrame()
    result = reconcile(theoretical, broker_df)

    print("=" * 78)
    print(f" BROKER RECONCILIATION — {today}")
    print("=" * 78)
    if result.empty:
        print(" No positions to reconcile.")
        return 0

    breaks = result[result["flag"].astype(str).str.startswith("[!]")]
    print(result.to_string(index=False))
    print("-" * 78)
    if breaks.empty:
        print(" OK: no reconciliation breaks.")
        return 0
    print(f" [!] {len(breaks)} reconciliation break(s) detected:")
    for _, r in breaks.iterrows():
        print(f"   {r['symbol']}: {r['flag']} (deviation {r['deviation_eur']} EUR)")
    return 2


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


def _cmd_suggest_rebalance(_args: argparse.Namespace) -> int:
    """Suggest tier reassignments to fix allocation violations (v10.6.4)."""
    from quant.portfolio.autobalance import analyze_tier_allocations, suggest_rebalance
    from quant.portfolio.portfolio import load_portfolio
    from quant.portfolio.tier_manager import load_tiers

    portfolio_df = load_portfolio()
    tiers_df = load_tiers()
    analysis = analyze_tier_allocations(portfolio_df, tiers_df)

    print("Current tier allocations:")
    print(f"Total portfolio value: {analysis['total_value_eur']:.2f} EUR")
    for tier, alloc in analysis["allocations"].items():
        status = "VIOLATED" if alloc["violated"] else "OK"
        limit_str = f"{alloc['limit']:.1%}" if alloc["limit"] else "no limit"
        print(f"  {tier:<12} {alloc['value_eur']:>10.2f} EUR  "
              f"{alloc['pct']:>6.1%}  limit {limit_str:<10} {status}")

    if not analysis["violations"]:
        print("All tier allocations are within limits. No rebalancing needed.")
        return 0

    suggestions = suggest_rebalance(portfolio_df, tiers_df)
    if not suggestions:
        print("Violations detected but no suitable reassignments found. "
              "Manual review required.")
        return 0

    print(f"Suggested reassignments ({len(suggestions)}):")
    for i, s in enumerate(suggestions, 1):
        print(f"  {i}. {s['symbol']}: {s['current_tier']} -> {s['suggested_tier']} "
              f"({s['value_eur']:.2f} EUR)")
        print(f"     {s['reason']}")
    print("These are suggestions only. No changes have been made.")
    print("Use 'quant apply-rebalance --symbols SYM1,SYM2' to apply specific suggestions.")
    return 0


def _cmd_apply_rebalance(args: argparse.Namespace) -> int:
    """Apply approved tier reassignments (v10.6.4)."""
    from quant.portfolio.autobalance import apply_rebalance_suggestions, suggest_rebalance
    from quant.portfolio.portfolio import load_portfolio
    from quant.portfolio.tier_manager import load_tiers, save_tiers

    portfolio_df = load_portfolio()
    tiers_df = load_tiers()
    approved = [s.strip() for s in str(args.symbols).split(",") if s.strip()]
    suggestions = suggest_rebalance(portfolio_df, tiers_df)
    approved_suggestions = [s for s in suggestions if s["symbol"] in approved]

    if not approved_suggestions:
        print("No suggestions found for the specified symbols.")
        return 0

    if getattr(args, "dry_run", False):
        print("Dry run - changes that would be applied:")
        for s in approved_suggestions:
            print(f"  {s['symbol']}: {s['current_tier']} -> {s['suggested_tier']}")
        print("No changes made (dry-run mode).")
        return 0

    updated = apply_rebalance_suggestions(tiers_df, suggestions, approved)
    save_tiers(updated)
    print(f"Applied {len(approved_suggestions)} reassignment(s):")
    for s in approved_suggestions:
        print(f"  {s['symbol']}: {s['current_tier']} -> {s['suggested_tier']}")
    print("Tier changes do not execute trades. Buy or sell manually in Trade Republic.")
    return 0


def _cmd_autobalance_wizard(_args: argparse.Namespace) -> int:
    """Interactively review and apply rebalancing suggestions (v10.6.4)."""
    from quant.portfolio.autobalance import (
        analyze_tier_allocations,
        apply_rebalance_suggestions,
        suggest_rebalance,
    )
    from quant.portfolio.portfolio import load_portfolio
    from quant.portfolio.tier_manager import load_tiers, save_tiers

    portfolio_df = load_portfolio()
    tiers_df = load_tiers()
    analysis = analyze_tier_allocations(portfolio_df, tiers_df)
    if not analysis["violations"]:
        print("All tier allocations are within limits. No rebalancing needed.")
        return 0

    suggestions = suggest_rebalance(portfolio_df, tiers_df)
    if not suggestions:
        print("Violations detected but no suitable reassignments found.")
        return 0

    print(f"Found {len(suggestions)} rebalancing suggestion(s).")
    approved: list[str] = []
    for i, s in enumerate(suggestions, 1):
        print(f"--- Suggestion {i}/{len(suggestions)} ---")
        print(f"Symbol: {s['symbol']}")
        print(f"Move: {s['current_tier']} -> {s['suggested_tier']}")
        print(f"Value: {s['value_eur']:.2f} EUR")
        print(f"Reason: {s['reason']}")
        try:
            response = input("Apply this suggestion? [y/N]: ").strip().lower()
        except EOFError:
            response = "n"
        if response in ("y", "yes"):
            approved.append(s["symbol"])
            print("Approved.")
        else:
            print("Skipped.")

    if not approved:
        print("No suggestions approved. No changes made.")
        return 0

    updated = apply_rebalance_suggestions(tiers_df, suggestions, approved)
    save_tiers(updated)
    print(f"Applied {len(approved)} reassignment(s).")
    print("Tier changes do not execute trades. Buy or sell manually in Trade Republic.")
    return 0


def _cmd_clear_cache(_args: argparse.Namespace) -> int:
    """Clear all cached calculations (v10.6.5)."""
    from quant.analytics.cache import clear_cache, get_cache_stats

    before = get_cache_stats()
    removed = clear_cache()
    print(f"Cleared {removed} cache entries ({before['total_size_mb']:.2f} MB).")
    return 0


def _cmd_cache_stats(_args: argparse.Namespace) -> int:
    """Show cache statistics (v10.6.5)."""
    from quant.analytics.cache import get_cache_stats

    stats = get_cache_stats()
    print("Cache statistics:")
    print(f"  Entries: {stats['num_entries']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    print(f"  Location: {stats['cache_dir']}")
    return 0


def _cmd_health_check(_args: argparse.Namespace) -> int:
    """Run the system health check (v10.6.4)."""
    from quant.cli.health import run_health_check

    result = run_health_check()
    print(f"System health: {result['status']}")
    for check in result["checks"]:
        print(f"  {check['name']:<20} {check['status']:<9} {check['message']}")
    return 1 if result["status"] == "CRITICAL" else 0


def _cmd_weekly_report(args: argparse.Namespace) -> int:
    """Generate the Weekly Friday Report (Markdown + self-contained HTML).

    v10.6.2 (R-PDF-1): the HTML carries print CSS; the user prints to PDF from
    the browser. No PDF library is used.
    """
    from quant.reporting.weekly_report import save_weekly_report

    as_of = getattr(args, "as_of", None) or dt.date.today().isoformat()
    emergency = getattr(args, "emergency", None)
    md_path, html_path = save_weekly_report(as_of, emergency_amount=emergency)
    print(f"weekly report: {md_path}")
    print(f"weekly report: {html_path}")
    print("Open the HTML file in a browser and print to PDF.")
    return 0


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
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("update", help="fetch market data + run the funnel")
    sub.add_parser("run", help="score, audit, and report")
    reconcile = sub.add_parser("reconcile", help="diff portfolio vs the broker export")
    reconcile.add_argument(
        "--broker",
        default=None,
        help="Broker CSV export (Symbol, Value_EUR). Defaults to data/portfolio.csv.",
    )
    sub.add_parser("all", help="run update then run (full daily cycle)")
    sub.add_parser("publish", help="render the static Published Briefing")
    sub.add_parser("doctor", help="read-only diagnosis (no writes, safe to paste)")
    sub.add_parser("validate-tiers", help="validate data/tiers.csv and report issues")
    sub.add_parser("repair-tiers", help="repair common issues in data/tiers.csv")
    sub.add_parser("suggest-rebalance",
                   help="suggest tier reassignments to fix allocation violations")
    apply_reb = sub.add_parser("apply-rebalance", help="apply approved tier reassignments")
    apply_reb.add_argument("--symbols", required=True,
                           help="comma-separated symbols to rebalance")
    apply_reb.add_argument("--dry-run", action="store_true",
                           help="preview changes without applying")
    sub.add_parser("autobalance-wizard",
                   help="interactively review and apply rebalancing suggestions")
    sub.add_parser("health-check", help="run the system health check")
    sub.add_parser("clear-cache", help="clear all cached calculations")
    sub.add_parser("cache-stats", help="show cache statistics")
    weekly = sub.add_parser(
        "weekly-report", help="generate the Weekly Friday Report (Markdown + HTML)"
    )
    weekly.add_argument(
        "--as-of", default=None, help="report date (YYYY-MM-DD); defaults to today"
    )
    weekly.add_argument(
        "--emergency", type=float, default=None,
        help="cash amount (EUR) for the emergency sell order",
    )
    dash = sub.add_parser("dash", help="launch the local interactive workspace")
    dash.add_argument(
        "--lan", action="store_true",
        help="also serve on the local network (phone on the same Wi-Fi)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point.

    Exit codes (spec 3.1): 0 success, 1 data-quality gate failure, 2 config error.
    """
    from quant import paths

    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "broker", None) is None:
        args.broker = str(paths.DATA_PORTFOLIO)

    # Only commands that produce a run create a run dir + pipeline.log. publish,
    # reconcile, and dash must NOT mint a new (empty) run dir, or latest_run()
    # would point at it instead of the last real run.
    log_path = (
        output.default_log_path()
        if args.command in ("update", "run", "all")
        else None
    )
    output.configure(verbose=getattr(args, "verbose", False), log_path=log_path)

    dispatch = {
        "update": _cmd_update,
        "run": _cmd_run,
        "reconcile": _cmd_reconcile,
        "all": _cmd_all,
        "publish": _cmd_publish,
        "dash": _cmd_dash,
        "doctor": _cmd_doctor,
        "weekly-report": _cmd_weekly_report,
        "validate-tiers": _cmd_validate_tiers,
        "repair-tiers": _cmd_repair_tiers,
        "suggest-rebalance": _cmd_suggest_rebalance,
        "apply-rebalance": _cmd_apply_rebalance,
        "autobalance-wizard": _cmd_autobalance_wizard,
        "health-check": _cmd_health_check,
        "clear-cache": _cmd_clear_cache,
        "cache-stats": _cmd_cache_stats,
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

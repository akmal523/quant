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
from collections.abc import Sequence
from pathlib import Path


def _cmd_update(_args: argparse.Namespace) -> int:
    """Fetch market data + run the funnel (step 1)."""
    from quant.data.data_updater import main as updater_main

    updater_main()
    return 0


def _cmd_run(_args: argparse.Namespace) -> int:
    """Score, audit, and report (step 2)."""
    from quant.main import main as pipeline_main

    pipeline_main()
    return 0


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


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``quant`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="quant",
        description="Quant-AI systematic equity pipeline.",
    )
    parser.add_argument("--version", action="version", version="quant-ai 10.4.2")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    from quant import paths

    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "broker", None) is None:
        args.broker = str(paths.DATA_PORTFOLIO)

    dispatch = {
        "update": _cmd_update,
        "run": _cmd_run,
        "reconcile": _cmd_reconcile,
        "all": _cmd_all,
    }
    return dispatch[args.command](args)


__all__ = ["main", "build_parser"]

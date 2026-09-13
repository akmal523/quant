"""
reconcile_broker.py — Daily broker reconciliation CLI (v10.4.0, Phase 2).

Intent: one command to compare the DuckDB theoretical portfolio state against the
Trade Republic CSV export and alert on divergence. Run daily (cron) after the
broker export is refreshed.

Usage:
    python3 scripts/reconcile_broker.py [--broker path/to/tr_export.csv]

The broker export must contain Symbol and Value_EUR (or Current_Value_EUR).
Defaults to data/portfolio.csv (the broker-synced file).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from quant import paths  # noqa: E402
from quant.portfolio.portfolio import load_portfolio  # noqa: E402
from quant.execution.reconciliation import (  # noqa: E402
    build_theoretical_snapshot, reconcile, save_snapshot,
)


def _load_scan() -> pd.DataFrame:
    """Load the latest market scan (outputs/market_scan_v8.csv). Empty-safe."""
    scan_path = Path(paths.OUTPUTS_DIR) / "market_scan_v8.csv"
    if scan_path.exists():
        try:
            return pd.read_csv(scan_path)
        except Exception:
            pass
    return pd.DataFrame(columns=["Symbol", "Current_Price"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile theoretical vs broker portfolio.")
    parser.add_argument("--broker", default=str(paths.DATA_PORTFOLIO),
                        help="Broker CSV export (Symbol, Value_EUR).")
    args = parser.parse_args()

    today = dt.date.today().isoformat()
    portfolio_df = load_portfolio(str(paths.DATA_PORTFOLIO))
    if portfolio_df.empty:
        print("[!] portfolio.csv is empty or missing; nothing to reconcile.")
        return 1

    scan_df = _load_scan()
    theoretical = build_theoretical_snapshot(portfolio_df, scan_df, today)
    save_snapshot(theoretical)

    broker_df = pd.read_csv(args.broker, comment="#") if Path(args.broker).exists() else pd.DataFrame()
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


if __name__ == "__main__":
    raise SystemExit(main())

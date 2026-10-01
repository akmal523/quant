#!/usr/bin/env python3
"""
migrate_tiers.py — one-shot migration from the legacy 4-tier system to the
3-tier system (FORTRESS / ALPHA / SPECULATIVE).

Intent (v10.6.2, R-TIER-1): read the existing ``data/portfolio.csv``, classify
each symbol with the legacy ``classify_asset``, map it through
``LEGACY_TIER_MAPPING``, and write ``data/tiers.csv``. The user then edits
``data/tiers.csv`` freely; ``portfolio.csv`` is never modified.

Invariants:
  - Never overwrites a populated ``data/tiers.csv`` unless ``--force`` is given.
  - Never writes ``data/portfolio.csv``.
  - Prints the resulting tier distribution.

Usage:
    python scripts/migrate_tiers.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.portfolio.portfolio import load_portfolio
from quant.portfolio.tier_manager import (
    load_tiers,
    migrate_legacy_portfolio,
    save_tiers,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate portfolio to 3 tiers.")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing populated data/tiers.csv.",
    )
    args = parser.parse_args()

    existing = load_tiers()
    if not existing.empty and not args.force:
        print(f"data/tiers.csv already has {len(existing)} rows. "
              f"Re-run with --force to overwrite.")
        return 1

    portfolio_df = load_portfolio()
    if portfolio_df.empty:
        print("No positions found in data/portfolio.csv. Nothing to migrate.")
        return 0

    print(f"Found {len(portfolio_df)} positions. Migrating tiers...")
    tiers_df = migrate_legacy_portfolio(portfolio_df)
    path = save_tiers(tiers_df)

    print(f"Wrote {len(tiers_df)} rows to {path}.")
    print("Tier distribution:")
    for tier, count in tiers_df["tier"].value_counts().items():
        print(f"  {tier}: {count}")
    print("Edit data/tiers.csv to adjust assignments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

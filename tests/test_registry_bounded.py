"""
test_registry_bounded.py — broker_registry.csv bounded invariant (H3-fix).

Guards against the registry explosion (1090 rows) caused by feeding
universe_master into the registry. The ceiling is the routable set:
portfolio + CORE_ETFS + ACTIVE + a small buffer for curated rows.
"""
from __future__ import annotations

import csv
from pathlib import Path

from quant.config import CORE_ETFS

ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> int:
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def test_broker_registry_stays_bounded():
    registry = ROOT / "data" / "broker_registry.csv"
    portfolio = ROOT / "data" / "portfolio.csv"
    n_port = _rows(portfolio) if portfolio.exists() else 0
    bound = n_port + len(CORE_ETFS) + 20
    n_reg = _rows(registry)
    assert n_reg <= bound, (
        f"broker_registry.csv has {n_reg} rows, above the bounded ceiling {bound}; "
        "a write path likely used universe_master"
    )

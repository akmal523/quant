#!/usr/bin/env python3
"""
benchmark.py — Measure and report performance metrics (v10.6.5).

Runs standardized benchmarks on the current portfolio and prints results in a
format suitable for documentation. Read-only; no writes.

Usage:
    python scripts/benchmark.py
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.analytics.cache import get_cache_stats
from quant.analytics.scoring import batch_score_assets_optimized
from quant.data.database import batch_query_portfolio_data, load_prices_lazy
from quant.portfolio.portfolio import load_portfolio
from quant.portfolio.tier_manager import load_tiers


def benchmark_batch_query(symbols: list[str]) -> dict:
    """Benchmark the single-query batch load."""
    start = time.time()
    tracemalloc.start()
    data = batch_query_portfolio_data(symbols)
    elapsed = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "num_symbols": len(symbols),
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak / (1024 * 1024),
        "prices_loaded": sum(len(df) for df in data["prices"].values()),
    }


def benchmark_batch_scoring(symbols: list[str], tiers: dict) -> dict:
    """Benchmark parallel batch scoring."""
    start = time.time()
    tracemalloc.start()
    results = batch_score_assets_optimized(symbols, tiers, max_workers=4)
    elapsed = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "num_symbols": len(symbols),
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak / (1024 * 1024),
        "results_count": len(results),
    }


def benchmark_lazy_loading(symbol: str) -> dict:
    """Benchmark chunked lazy loading."""
    start = time.time()
    tracemalloc.start()
    total_rows = 0
    for chunk in load_prices_lazy(symbol, chunk_size=5000):
        total_rows += len(chunk)
    elapsed = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "symbol": symbol,
        "total_rows": total_rows,
        "elapsed_seconds": elapsed,
        "peak_memory_mb": peak / (1024 * 1024),
    }


def main() -> int:
    """Run all benchmarks and print results."""
    print("Loading portfolio...")
    portfolio_df = load_portfolio()
    tiers_df = load_tiers()
    symbols = portfolio_df["Symbol"].tolist() if not portfolio_df.empty else []
    tiers = dict(zip(tiers_df["symbol"], tiers_df["tier"])) if not tiers_df.empty else {}
    print(f"Portfolio has {len(symbols)} assets.\n")

    print("=" * 60)
    print("Benchmark 1: Batch query")
    print("=" * 60)
    r = benchmark_batch_query(symbols)
    print(f"  Symbols: {r['num_symbols']}")
    print(f"  Elapsed: {r['elapsed_seconds']:.3f} seconds")
    print(f"  Peak memory: {r['peak_memory_mb']:.2f} MB")
    print(f"  Price rows loaded: {r['prices_loaded']}\n")

    print("=" * 60)
    print("Benchmark 2: Batch scoring")
    print("=" * 60)
    r = benchmark_batch_scoring(symbols, tiers)
    print(f"  Symbols: {r['num_symbols']}")
    print(f"  Elapsed: {r['elapsed_seconds']:.3f} seconds")
    print(f"  Peak memory: {r['peak_memory_mb']:.2f} MB")
    print(f"  Results: {r['results_count']}\n")

    print("=" * 60)
    print("Benchmark 3: Lazy loading")
    print("=" * 60)
    if symbols:
        r = benchmark_lazy_loading(symbols[0])
        print(f"  Symbol: {r['symbol']}")
        print(f"  Total rows: {r['total_rows']}")
        print(f"  Elapsed: {r['elapsed_seconds']:.3f} seconds")
        print(f"  Peak memory: {r['peak_memory_mb']:.2f} MB\n")

    print("=" * 60)
    print("Benchmark 4: Cache statistics")
    print("=" * 60)
    stats = get_cache_stats()
    print(f"  Entries: {stats['num_entries']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB\n")

    print("Benchmark complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

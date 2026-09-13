"""
golden_util.py — Deterministic backtest fixture for golden-file snapshot tests.

Intent: produce a byte-stable backtest output so a refactor of scoring/backtest
logic that changes results by >0.01% fails CI. The synthetic series is seeded so
it is identical on every machine and every run (hermetic, no network).

Invariants:
  - build_synthetic_hist() is deterministic (fixed seed).
  - compute_snapshot() returns a JSON-serializable dict of backtest metrics.
  - Pure computation (no I/O).

Dependencies: numpy, pandas, quant.strategy.backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_synthetic_hist(n_days: int = 600, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic OHLC + Signal series for backtest snapshots.

    Intent: a fixed random walk with a simple SMA crossover signal. Seeded so the
    golden file is reproducible. Invariants: returns a DataFrame with Open, High,
    Low, Close, Volume, Signal; pure function (no I/O).
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rets = rng.normal(0.0004, 0.012, n_days)
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0.0, 0.002, n_days))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, n_days)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, n_days)))
    volume = rng.randint(1_000_000, 5_000_000, n_days).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    sma20 = df["Close"].rolling(20).mean()
    df["Signal"] = np.where(df["Close"] > sma20, "BUY", "SELL")
    return df


def compute_snapshot() -> dict:
    """Run the deterministic backtests and return their metrics.

    Intent: the single source of truth for the golden file. Both the test and the
    regeneration script call this. Invariants: returns a JSON-serializable dict.
    """
    from quant.strategy.backtest import (
        run_cost_aware_backtest, walk_forward_optimization, run_macro_backtest,
    )

    hist = build_synthetic_hist()
    return {
        "cost_aware": run_cost_aware_backtest(hist),
        "wfo": {
            k: v for k, v in walk_forward_optimization(hist).items()
            if k != "survivorship_bias_warning"
        },
        "macro": run_macro_backtest(hist),
    }

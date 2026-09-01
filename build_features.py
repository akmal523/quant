"""
build_features.py — Vectorized cross-sectional feature engine (Polars/DuckDB).

Intent: replace per-symbol Python loops with a single vectorized pass over the
whole market. All features computed in Rust (Polars) or pushed into DuckDB SQL.
Invariants: output DataFrame has one row per (Symbol, Date); all features are
computed using only past data (no lookahead — rolling/shift windows).
Dependencies: polars, duckdb (via database.get_connection).
"""
from __future__ import annotations

import polars as pl

from database import get_connection


def load_market_pl(conn=None) -> pl.DataFrame:
    """Load market_history into Polars natively (no pandas intermediate)."""
    conn = conn or get_connection()
    df = conn.execute(
        "SELECT Symbol, Date, Open, High, Low, Close, Volume, Sector "
        "FROM market_history ORDER BY Symbol ASC, Date ASC"
    ).pl()
    if "Date" in df.columns:
        df = df.with_columns(pl.col("Date").str.to_datetime())
    return df


def build_features(conn=None) -> pl.DataFrame:
    """Compute all cross-sectional features vectorized across the universe.

    Features:
      ret_1d, ret_1m, ret_6m, ret_12m  — momentum returns
      sma_20, sma_200                  — moving averages
      vol_20d, vol_60d                 — realized volatility
      adv_20                           — average daily volume (liquidity)
      uptrend, trend_strength          — trend flags
      max_drawdown_60d                 — drawdown (low-risk factor)
    """
    df = load_market_pl(conn)

    df = (
        df
        .sort(["Symbol", "Date"])
        .with_columns([
            pl.col("Close").pct_change().over("Symbol").alias("ret_1d"),
            pl.col("Close").pct_change().shift(20).over("Symbol").alias("ret_1m"),
            pl.col("Close").pct_change().shift(126).over("Symbol").alias("ret_6m"),
            pl.col("Close").pct_change().shift(252).over("Symbol").alias("ret_12m"),
            pl.col("Close").rolling_mean(20).over("Symbol").alias("sma_20"),
            pl.col("Close").rolling_mean(200).over("Symbol").alias("sma_200"),
            pl.col("Close").pct_change().rolling_std(20).over("Symbol").alias("vol_20d"),
            pl.col("Close").pct_change().rolling_std(60).over("Symbol").alias("vol_60d"),
            pl.col("Volume").rolling_mean(20).over("Symbol").alias("adv_20"),
        ])
        .with_columns([
            (pl.col("Close") > pl.col("sma_200")).alias("uptrend"),
            (pl.col("Close") / pl.col("sma_200") - 1).alias("trend_strength"),
        ])
    )

    # Max drawdown over trailing 60d (low-risk factor).
    df = df.with_columns(
        (pl.col("Close") / pl.col("Close").rolling_max(60).over("Symbol") - 1)
        .alias("max_drawdown_60d")
    )

    return df


def latest_features(conn=None) -> pl.DataFrame:
    """Return only the most recent feature row per symbol (for live scoring)."""
    df = build_features(conn)
    return (
        df
        .sort(["Symbol", "Date"])
        .group_by("Symbol")
        .last()
    )
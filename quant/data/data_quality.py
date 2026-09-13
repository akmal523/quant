"""
data_quality.py — Data Quality Gate (Part 3, Gap #1).

Intent: validate incoming market data BEFORE it enters DuckDB. Yahoo Finance
can return garbage (NaN, duplicate dates, split-adjustment errors, stale
prices). Without a gate, bad data silently corrupts the scoring pipeline.
Runs in data_updater.py before appending.

Invariants:
  - validate_batch returns (is_valid, issues); is_valid True iff no issues.
  - auto_repair removes duplicates/NaN and interpolates small gaps.
  - Pure computation; no I/O.

Dependencies: pandas.
"""
from __future__ import annotations

import pandas as pd


class DataQualityValidator:
    """Validates incoming market data before it enters DuckDB."""

    THRESHOLDS = {
        "max_daily_return": 0.25,   # 25% daily move without news = suspicious
        "min_price": 0.01,          # below 1 cent = error
        "max_price": 100000.0,      # above EUR 100k for ETFs = error
        "max_stale_days": 5,        # older than 5 days = stale
        "min_history_days": 60,     # need at least 60 days for scoring
    }

    def validate_batch(self, df: pd.DataFrame, symbol: str,
                       check_min_history: bool = True,
                       check_extreme_moves: bool = True) -> tuple[bool, list[str]]:
        """Validate a batch of market data. Returns (is_valid, issues).

        Intent: check_min_history=False and check_extreme_moves=False for
        incremental slices (only ~5 days fetched). The 60-day minimum and the
        >25% daily-move check are FULL-history invariants, not data-integrity
        checks for the append slice. A single >25% move on a real trading day
        (earnings/news) is legitimate and must not block the incremental update;
        the full history was already validated on the initial 5y fetch.
        """
        issues = []
        if df is None or df.empty:
            return False, ["Empty dataframe"]

        if "Close" not in df.columns:
            return False, ["Missing Close column"]

        close = df["Close"]

        nan_count = int(close.isna().sum())
        if nan_count > 0:
            issues.append(f"{nan_count} NaN values in Close price")

        if (close < 0).any():
            issues.append("Negative price detected")

        if (close > self.THRESHOLDS["max_price"]).any():
            issues.append("Price above max threshold")

        if check_extreme_moves:
            daily_return = close.pct_change()
            extreme = daily_return[daily_return.abs() > self.THRESHOLDS["max_daily_return"]]
            if len(extreme) > 0:
                issues.append(
                    f"{len(extreme)} days with >{self.THRESHOLDS['max_daily_return']*100:.0f}% moves"
                )

        if "Date" in df.columns:
            dup_count = int(df["Date"].duplicated().sum())
            if dup_count > 0:
                issues.append(f"{dup_count} duplicate dates")

            latest_date = pd.to_datetime(df["Date"]).max()
            days_stale = (pd.Timestamp.now() - latest_date).days
            if days_stale > self.THRESHOLDS["max_stale_days"]:
                issues.append(f"Data is {days_stale} days stale")

        if check_min_history and len(close) < self.THRESHOLDS["min_history_days"]:
            issues.append(
                f"Only {len(close)} days history (< {self.THRESHOLDS['min_history_days']})"
            )

        if close.iloc[-1] < self.THRESHOLDS["min_price"]:
            issues.append("Price below minimum threshold")

        return len(issues) == 0, issues

    def auto_repair(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Attempt automatic repairs for common issues."""
        if df is None or df.empty:
            return df

        if "Date" in df.columns:
            df = df.drop_duplicates(subset="Date", keep="last")
            df = df.sort_values("Date")

        df = df.dropna(subset=["Close"])

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            # time-weighted interpolation requires a DatetimeIndex.
            df = df.set_index("Date")
            df["Close"] = df["Close"].interpolate(method="time", limit=2)
            df = df.reset_index()

        df = df.dropna(subset=["Close"])
        return df


def repair_isolated_glitches(
    df: pd.DataFrame,
    max_jump: float = 0.5,
    revert_tol: float = 0.25,
) -> pd.DataFrame:
    """Repair isolated bad ticks (spike-and-revert) from the data source.

    Intent: Yahoo occasionally returns a single corrupt row (e.g. DFEN
    2024-06-03 Close=8.15 between ~23 neighbours, reverting the next day). Such
    a spike is neither a real move nor a split, but left in place it (a) makes
    detect_split see a false ratio jump and mangle the series, then (b) trips the
    hard drop assertion and aborts the whole run. Replace the bad row's OHLC with
    the mean of its neighbours.
    Invariants: only single-row, self-reverting spikes are touched; a sustained
    move (real split/crash) is never modified; pure function (no I/O).
    """
    if df is None or df.empty or "Close" not in df.columns or len(df) < 3:
        return df
    out = df.copy()
    close = out["Close"].astype(float).to_numpy()
    n = len(close)
    for i in range(1, n - 1):
        prev, cur, nxt = close[i - 1], close[i], close[i + 1]
        if prev <= 0 or nxt <= 0:
            continue
        jump = abs(cur / prev - 1.0)
        revert = abs(nxt / prev - 1.0)
        # Spike away from prev, then snap back to prev the next day => bad tick.
        if jump > max_jump and revert < revert_tol:
            for col in ("Open", "High", "Low", "Close"):
                if col in out.columns:
                    out.iloc[i, out.columns.get_loc(col)] = (
                        out[col].iloc[i - 1] + out[col].iloc[i + 1]
                    ) / 2.0
    return out
"""
corporate_actions.py — Automated Corporate Actions Engine (v10.4.0, Phase 1).

Intent: unadjusted data in a backtest guarantees false returns. When a stock
splits (e.g. 2:1), the raw Close series shows a ~50% overnight drop that is not
a real loss. This module detects splits from the price/volume discontinuity and
adjusts BOTH the historical prices and the portfolio cost basis so the return
series stays continuous.

Invariants:
  - detect_split is pure: (close, volume) -> list[SplitEvent].
  - adjust_for_split divides pre-split prices by ratio, multiplies volume by ratio.
  - adjust_cost_basis multiplies the entry price by 1/ratio (shares up, price down).
  - No I/O. pandas only.

State Transition: raw disconnected series -> detect -> adjust -> continuous series.

Dependencies: pandas, numpy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Common forward-split ratios (new shares per old share). A stock split almost
# always lands on one of these; using a discrete set avoids false positives from
# ordinary large moves.
SPLIT_RATIOS = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0)
# Reverse splits are the reciprocal of the same ratios.
REVERSE_RATIOS = tuple(1.0 / r for r in SPLIT_RATIOS)

# Relative tolerance when matching an observed price/volume jump to a known ratio.
_RATIO_TOL = 0.12


@dataclass(frozen=True)
class SplitEvent:
    """A detected split: date is the first post-split trading day."""

    date: pd.Timestamp
    ratio: float  # forward ratio, e.g. 2.0 for 2:1; 0.5 for a 1:2 reverse split


def _nearest_ratio(observed: float, candidates: tuple[float, ...]) -> float | None:
    """Return the candidate ratio within tolerance of observed, else None."""
    if observed <= 0:
        return None
    for r in candidates:
        if abs(observed - r) / r <= _RATIO_TOL:
            return r
    return None


def detect_split(close: pd.Series, volume: pd.Series | None = None) -> list[SplitEvent]:
    """Detect stock splits from a price discontinuity (optionally confirmed by volume).

    Intent: a split shows up as a large overnight Close jump that lands on a known
    ratio. When volume is supplied, require the same ratio on the volume jump as a
    confirmation, which suppresses false positives from genuine large moves.

    Invariants: returns events in ascending date order; pure function (no I/O).
    """
    events: list[SplitEvent] = []
    if close is None or len(close) < 2:
        return events

    close = close.astype(float)
    price_ratio = close.shift(1) / close  # >1 => price fell => forward split

    vol_ratio = None
    if volume is not None and len(volume) == len(close):
        vol = volume.astype(float).replace(0, np.nan)
        vol_ratio = vol / vol.shift(1)

    for i in range(1, len(close)):
        pr = price_ratio.iloc[i]
        if not np.isfinite(pr):
            continue

        ratio = _nearest_ratio(pr, SPLIT_RATIOS) or _nearest_ratio(pr, REVERSE_RATIOS)
        if ratio is None:
            continue

        # Volume confirmation: a forward split (ratio>1) should raise share volume
        # by ~ratio; a reverse split should lower it.
        if vol_ratio is not None:
            vr = vol_ratio.iloc[i]
            if np.isfinite(vr):
                expected = ratio if ratio >= 1 else 1.0 / ratio
                if abs(vr - expected) / expected > 0.50:
                    continue

        events.append(SplitEvent(date=close.index[i], ratio=float(ratio)))

    return events


def adjust_for_split(df: pd.DataFrame, event: SplitEvent) -> pd.DataFrame:
    """Return a copy of df with all rows BEFORE event.date split-adjusted.

    Intent: restate pre-split prices onto the post-split scale so the series is
    continuous. Prices divide by ratio; Volume multiplies by ratio.
    Invariants: rows at/after event.date are unchanged; pure function (no I/O).
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    dates = pd.to_datetime(out.index) if not pd.api.types.is_datetime64_any_dtype(out.index) else out.index
    mask = dates < pd.Timestamp(event.date)
    if not mask.any():
        return out

    for col in ("Open", "High", "Low", "Close"):
        if col in out.columns:
            out.loc[mask, col] = out.loc[mask, col] / event.ratio
    if "Volume" in out.columns:
        out.loc[mask, "Volume"] = out.loc[mask, "Volume"] * event.ratio
    return out


def adjust_cost_basis(avg_entry_price: float, ratio: float) -> float:
    """Adjust a portfolio cost basis for a forward split ratio.

    Intent: after a 2:1 split, shares double and price halves; the total cost is
    unchanged so the per-share entry price must be divided by the ratio.
    Invariants: returns >= 0; pure function (no I/O).
    """
    if avg_entry_price is None or ratio is None or ratio <= 0:
        return avg_entry_price
    return float(avg_entry_price) / float(ratio)


def apply_corporate_actions(
    df: pd.DataFrame,
    events: list[SplitEvent] | None = None,
) -> tuple[pd.DataFrame, list[SplitEvent]]:
    """Detect and apply all splits in one pass over a symbol's history.

    Intent: called by data_updater before features/scoring so the whole pipeline
    sees an adjusted series.
    Invariants: returns (adjusted_df, events); original df is not mutated.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return df, []

    if events is None:
        events = detect_split(df["Close"], df.get("Volume"))

    out = df
    # Apply oldest-first so ratio composition is correct across multiple splits.
    for ev in sorted(events, key=lambda e: e.date):
        out = adjust_for_split(out, ev)
    return out, list(events)
"""
tier_manager.py — 3-tier classification (FORTRESS / ALPHA / SPECULATIVE).

Intent (v10.6.2, R-TIER-1): the legacy 4-tier system (CORE / SATELLITE /
ACTIVE / SECTOR) is replaced by the 3-tier system. Tier assignments live in a
user-editable ``data/tiers.csv`` (``symbol,tier,last_updated,notes``);
``data/portfolio.csv`` stays broker-synced and untouched. An unlisted symbol
defaults to ALPHA (the safest active tier).

Invariants:
  - ``load_tiers`` never raises; returns an empty frame on any failure.
  - ``get_asset_tier`` always returns a member of ``VALID_TIERS``.
  - ``save_tiers`` writes atomically (temp file + ``os.replace``).
  - ``validate_tier_constraints`` returns human-readable warnings, never raises.
  - Pure I/O boundary: reads/writes only ``data/tiers.csv``.

Dependencies: pandas, quant.paths, quant.config.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from quant import paths
from quant.config import (
    DEFAULT_TIER,
    TIER_CONSTRAINTS,
    VALID_TIERS,
)

_COLUMNS = ["symbol", "tier", "last_updated", "notes"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw tiers frame to the canonical schema (never raises)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=_COLUMNS)
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    for col in _COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[_COLUMNS]
    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["tier"] = out["tier"].astype(str).str.strip().str.upper()
    out["last_updated"] = out["last_updated"].astype(str).str.strip()
    out["notes"] = out["notes"].fillna("").astype(str)
    out = out[out["symbol"] != ""]
    return out.reset_index(drop=True)


def load_tiers(filepath: str | None = None) -> pd.DataFrame:
    """Load ``data/tiers.csv`` with the canonical schema.

    Intent: the single read path for tier assignments. Missing file, bad
    columns, or a parse error all yield an empty frame (the caller then falls
    back to ``DEFAULT_TIER`` per symbol). Invariants: never raises; returns a
    frame with columns ``symbol,tier,last_updated,notes``.
    """
    path = filepath or paths.DATA_TIERS
    if not os.path.exists(path):
        return pd.DataFrame(columns=_COLUMNS)
    try:
        df = pd.read_csv(path, comment="#").dropna(how="all")
        return _normalize(df)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=_COLUMNS)


def save_tiers(df: pd.DataFrame, filepath: str | None = None) -> str:
    """Write tier assignments atomically. Returns the written path.

    Intent: the UI and the migration script both persist through this one
    function. Atomic write (temp file in the same dir + ``os.replace``) so a
    crash never leaves a half-written tiers file. Invariants: writes the
    canonical schema; never raises on a valid frame.
    """
    path = filepath or paths.DATA_TIERS
    out = _normalize(df)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            out.to_csv(fh, index=False)
        os.replace(tmp, target)
    except Exception:  # noqa: BLE001
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return str(target)


def tier_map(tiers_df: pd.DataFrame | None = None) -> dict[str, str]:
    """Return ``{symbol: tier}`` for the whole file (invalid tiers dropped)."""
    if tiers_df is None:
        tiers_df = load_tiers()
    if tiers_df is None or tiers_df.empty:
        return {}
    return {
        str(r["symbol"]): str(r["tier"]).upper()
        for _, r in tiers_df.iterrows()
        if str(r["tier"]).upper() in VALID_TIERS
    }


# ── v10.6.3: Unclassified detection, validation, repair ───────────────────────


def validate_tiers_csv(
    tiers_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """Validate tiers.csv and return (is_valid, errors).

    Intent (v10.6.3): catch missing columns, invalid tiers, duplicate symbols,
    orphan symbols (in tiers but not in the portfolio), and tier allocations
    that exceed their hard cap. Invariants: returns a list of plain sentences;
    never raises; empty list when valid.
    """
    errors: list[str] = []
    if tiers_df is None:
        return False, ["tiers.csv is missing or unreadable"]

    required = {"symbol", "tier", "last_updated", "notes"}
    missing = required - set(tiers_df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
        return False, errors

    invalid = {str(t).upper() for t in tiers_df["tier"]} - set(VALID_TIERS)
    invalid.discard("")
    if invalid:
        errors.append(
            f"Invalid tiers: {sorted(invalid)}. Must be one of {list(VALID_TIERS)}"
        )

    dups = tiers_df[tiers_df.duplicated(subset=["symbol"], keep=False)]
    if not dups.empty:
        errors.append(f"Duplicate symbols: {sorted(str(s) for s in dups['symbol'].unique())}")

    if portfolio_df is not None and not portfolio_df.empty and "Symbol" in portfolio_df.columns:
        portfolio_symbols = {str(s) for s in portfolio_df["Symbol"].astype(str)}
        tier_symbols = {str(s) for s in tiers_df["symbol"].astype(str)}
        orphans = tier_symbols - portfolio_symbols
        if orphans:
            errors.append(
                f"Symbols in tiers.csv but not in portfolio.csv: {sorted(orphans)}"
            )

        if "Current_Value_EUR" in portfolio_df.columns:
            total_value = float(pd.to_numeric(
                portfolio_df["Current_Value_EUR"], errors="coerce").fillna(0.0).sum())
            if total_value > 0:
                tmap = {str(r["symbol"]): str(r["tier"]).upper()
                        for _, r in tiers_df.iterrows()}
                for tier, constraints in TIER_CONSTRAINTS.items():
                    max_alloc = constraints.get("max_allocation")
                    if max_alloc is None:
                        continue
                    symbols = [s for s, t in tmap.items() if t == tier]
                    if not symbols:
                        continue
                    tier_value = float(pd.to_numeric(
                        portfolio_df.loc[portfolio_df["Symbol"].isin(symbols),
                                         "Current_Value_EUR"],
                        errors="coerce").fillna(0.0).sum())
                    pct = tier_value / total_value
                    if pct > max_alloc:
                        errors.append(
                            f"{tier} allocation is {pct:.1%}, exceeds max {max_alloc:.1%}"
                        )

    return (len(errors) == 0), errors


def repair_tiers_csv(
    tiers_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    """Repair common tiers.csv issues.

    Intent (v10.6.3): drop duplicate symbols (keep first), drop orphan symbols
    (not in the portfolio), and default invalid tiers to ALPHA. Invariants:
    returns a canonical tiers frame; never raises.
    """
    out = _normalize(tiers_df)
    out = out.drop_duplicates(subset=["symbol"], keep="first")
    if portfolio_df is not None and not portfolio_df.empty and "Symbol" in portfolio_df.columns:
        portfolio_symbols = {str(s) for s in portfolio_df["Symbol"].astype(str)}
        out = out[out["symbol"].isin(portfolio_symbols)]
    out["tier"] = out["tier"].apply(lambda t: t if t in VALID_TIERS else DEFAULT_TIER)
    return out.reset_index(drop=True)


# ── v10.6.4: Safe load and corruption repair ──────────────────────────────────

def load_tiers_safe() -> tuple[pd.DataFrame, list[str]]:
    """Load tiers.csv with comprehensive error handling.

    Intent (v10.6.4): never crash on a missing, empty, or corrupted tiers file.
    Returns ``(tiers_df, warnings)``; a missing/empty file yields an empty
    canonical frame; validation issues are returned as warnings. Invariants:
    never raises.
    """
    warnings: list[str] = []
    try:
        tiers_df = load_tiers()
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Failed to load tiers.csv: {e}. Using empty tiers.")
        return pd.DataFrame(columns=_COLUMNS), warnings

    if tiers_df is None or tiers_df.empty:
        return pd.DataFrame(columns=_COLUMNS), warnings

    try:
        from quant.portfolio.portfolio import load_portfolio
        portfolio_df = load_portfolio()
        is_valid, errors = validate_tiers_csv(tiers_df, portfolio_df)
        if not is_valid:
            warnings.extend(errors)
            warnings.append("Run 'quant repair-tiers' to attempt automatic repair.")
    except Exception as e:  # noqa: BLE001
        warnings.append(f"Validation skipped: {e}")

    return tiers_df, warnings

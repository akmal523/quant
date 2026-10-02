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
    LEGACY_TIER_MAPPING,
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


def get_asset_tier(symbol: str, tiers_df: pd.DataFrame | None = None) -> str:
    """Return the tier for a symbol, defaulting to ``DEFAULT_TIER``.

    Invariants: always returns a member of ``VALID_TIERS``; an unknown or
    invalid tier falls back to ``DEFAULT_TIER``. Pure (no I/O when a frame is
    supplied).
    """
    if tiers_df is None:
        tiers_df = load_tiers()
    if tiers_df is None or tiers_df.empty:
        return DEFAULT_TIER
    match = tiers_df[tiers_df["symbol"] == str(symbol).strip()]
    if match.empty:
        return DEFAULT_TIER
    tier = str(match.iloc[0]["tier"]).strip().upper()
    return tier if tier in VALID_TIERS else DEFAULT_TIER


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


def target_weights_invested(tiers_df: pd.DataFrame | None = None) -> dict[str, float]:
    """Per-symbol invested-pool targets (v10.7.3, Part 5.1).

    Seeded from ``config.TARGET_WEIGHTS_INVESTED``; overridable per symbol by an
    optional ``target_pct`` column in tiers.csv (a value > 1 is read as a
    percent, otherwise as a fraction). Never raises.
    """
    from quant.config import TARGET_WEIGHTS_INVESTED

    out = dict(TARGET_WEIGHTS_INVESTED)
    if tiers_df is None:
        try:
            tiers_df, _ = load_tiers_safe()
        except Exception:  # noqa: BLE001
            tiers_df = None
    if tiers_df is None or getattr(tiers_df, "empty", True):
        return out
    if "target_pct" not in tiers_df.columns:
        return out
    for _, r in tiers_df.iterrows():
        sym = str(r.get("symbol", "")).strip()
        try:
            pct = float(r.get("target_pct"))
        except (TypeError, ValueError):
            continue
        if sym and pct > 0:
            out[sym] = (pct / 100.0) if pct > 1 else pct
    return out


def validate_tier_constraints(
    tiers_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
) -> list[str]:
    """Return warnings where a tier's allocation exceeds its hard cap.

    Intent: SPECULATIVE is capped at 2 percent and ALPHA at 50 percent of the
    portfolio value. FORTRESS is uncapped. Invariants: returns a list of plain
    sentences; never raises; empty list when within limits or no data.
    """
    warnings: list[str] = []
    if portfolio_df is None or portfolio_df.empty:
        return warnings
    if "Symbol" not in portfolio_df.columns or "Current_Value_EUR" not in portfolio_df.columns:
        return warnings

    total_value = float(pd.to_numeric(
        portfolio_df["Current_Value_EUR"], errors="coerce").fillna(0.0).sum())
    if total_value <= 0:
        return warnings

    tmap = tier_map(tiers_df)
    for tier, constraints in TIER_CONSTRAINTS.items():
        max_alloc = constraints.get("max_allocation")
        if max_alloc is None:
            continue
        symbols = [s for s, t in tmap.items() if t == tier]
        if not symbols:
            continue
        tier_value = float(pd.to_numeric(
            portfolio_df.loc[portfolio_df["Symbol"].isin(symbols), "Current_Value_EUR"],
            errors="coerce").fillna(0.0).sum())
        tier_pct = tier_value / total_value
        if tier_pct > max_alloc:
            warnings.append(
                f"{tier} allocation is {tier_pct:.1%}, exceeds the "
                f"{max_alloc:.1%} cap. Move assets to another tier."
            )
    return warnings


def migrate_legacy_portfolio(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """Map an existing portfolio from the legacy 4-tier to the 3-tier system.

    Intent: one-shot migration. Each symbol is classified with the legacy
    ``classify_asset`` (CORE / SATELLITE / ACTIVE / SECTOR) and mapped through
    ``LEGACY_TIER_MAPPING``. Invariants: returns a canonical tiers frame; never
    raises; an empty portfolio yields an empty frame.
    """
    if portfolio_df is None or portfolio_df.empty:
        return pd.DataFrame(columns=_COLUMNS)

    # Lazy import avoids a circular import (portfolio.py imports tier_manager).
    from quant.portfolio.portfolio import classify_asset

    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    rows = []
    for _, row in portfolio_df.iterrows():
        symbol = str(row.get("Symbol", "")).strip()
        if not symbol:
            continue
        legacy_tier = classify_asset(symbol)
        new_tier = LEGACY_TIER_MAPPING.get(legacy_tier, DEFAULT_TIER)
        rows.append({
            "symbol": symbol,
            "tier": new_tier,
            "last_updated": today,
            "notes": f"Migrated from legacy {legacy_tier}",
        })
    return pd.DataFrame(rows, columns=_COLUMNS)


# ── v10.6.3: Unclassified detection, validation, repair ───────────────────────

def detect_unclassified_assets(
    portfolio_df: pd.DataFrame,
    tiers_df: pd.DataFrame,
) -> list[dict]:
    """Detect portfolio symbols with no tier assignment.

    Intent (v10.6.3): a symbol added to portfolio.csv but not to tiers.csv
    silently defaults to ALPHA, which may be wrong. This returns a recommended
    tier per unclassified symbol: ETF/CASH to FORTRESS, EQUITY to ALPHA, else
    SPECULATIVE. Invariants: returns a list of dicts
    ``{symbol, recommended_tier, reason}``; never raises; pure.
    """
    if portfolio_df is None or portfolio_df.empty or "Symbol" not in portfolio_df.columns:
        return []
    portfolio_symbols = {str(s) for s in portfolio_df["Symbol"].astype(str)}
    tier_symbols = (
        {str(s) for s in tiers_df["symbol"].astype(str)}
        if tiers_df is not None and not tiers_df.empty and "symbol" in tiers_df.columns
        else set()
    )
    unclassified = sorted(portfolio_symbols - tier_symbols)

    from quant.execution.taxonomy import get_instrument_class

    recommendations: list[dict] = []
    for symbol in unclassified:
        try:
            asset_class = get_instrument_class(symbol)
        except Exception:  # noqa: BLE001
            asset_class = "EQUITY"
        if asset_class in ("ETF", "CASH"):
            tier = "FORTRESS"
            reason = "ETF detected, recommended for long-term Sparplan"
        elif asset_class == "EQUITY":
            tier = "ALPHA"
            reason = "Stock detected, recommended for active trading"
        else:
            tier = "SPECULATIVE"
            reason = "Unknown asset class, recommended for the speculative bucket"
        recommendations.append({
            "symbol": symbol, "recommended_tier": tier, "reason": reason,
        })
    return recommendations


def auto_assign_tiers(
    recommendations: list[dict],
    tiers_df: pd.DataFrame,
) -> pd.DataFrame:
    """Append recommended tiers for unclassified assets.

    Intent (v10.6.3): persist the recommendations from
    ``detect_unclassified_assets``. Invariants: returns a canonical tiers frame;
    never raises; an empty recommendation list returns the input unchanged.
    """
    base = _normalize(tiers_df)
    if not recommendations:
        return base
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    new_rows = [
        {
            "symbol": rec["symbol"],
            "tier": rec["recommended_tier"],
            "last_updated": today,
            "notes": f"Auto-assigned: {rec['reason']}",
        }
        for rec in recommendations
    ]
    combined = pd.concat([base, pd.DataFrame(new_rows)], ignore_index=True)
    return _normalize(combined)


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


def _repair_corrupted_csv(filepath: str | None = None) -> pd.DataFrame:
    """Attempt to repair a corrupted tiers.csv by parsing raw text.

    Intent (v10.6.4): recover the valid rows from a file with stray lines or a
    broken header. Invariants: returns a canonical frame; raises ValueError when
    no valid data is found.
    """
    path = filepath or paths.DATA_TIERS
    if not os.path.exists(path):
        return pd.DataFrame(columns=_COLUMNS)
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()

    valid_lines: list[str] = []
    header_found = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "symbol" in line.lower() and "tier" in line.lower():
            header_found = True
            valid_lines.append(line)
        elif header_found and "," in line:
            if len(line.split(",")) >= 2:
                valid_lines.append(line)

    if not valid_lines:
        raise ValueError("No valid CSV data found in tiers.csv")

    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(valid_lines)))
    return _normalize(df)

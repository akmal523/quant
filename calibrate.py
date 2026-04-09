"""
calibrate.py — empirical weight calibration for scoring.py.

Builds a panel of monthly entry-point observations, extracts point-in-time
price features at each cutoff, computes forward returns as labels, then fits
three models to derive empirically-validated weights for deep_score().

Usage
-----
    python calibrate.py                     # all three methods; best written to JSON
    python calibrate.py --method logreg     # L2 logistic regression only
    python calibrate.py --method nelder     # L-BFGS-B rank-IC optimisation only
    python calibrate.py --method grid       # bucket-cap grid search only
    python calibrate.py --horizon 63        # forward window in trading days (default 63 ≈ 3 months)
    python calibrate.py --lookback 24       # entry-point history in months (default 24)
    python calibrate.py --topk 7            # assets per time-slice for portfolio ranking
    python calibrate.py --save-dataset      # also write calibration_dataset.csv for inspection

Outputs
-------
    calibrated_weights.json   — loaded automatically by scoring.py at import time

DATA CAVEATS (flagged inline in printed output)
-----------------------------------------------
  [C1] PE, PEG, ROE use today's yfinance values for all historical entry points.
       Lookahead bias is present in all three fundamental features.
  [C2] AI sentiment is set to neutral (0.5) for all historical observations.
       The sentiment weight cannot be calibrated without a sentiment archive.
  [C3] Analyst-target upside uses today's target applied to historical prices.
       Partial lookahead: the target level may have shifted over the window.
  [C4] ~540 observations with 12 monthly slices yields high-variance Sharpe/IC
       estimates.  L2 logistic regression is the most reliable method at this
       sample size; Nelder-Mead is exploratory.
  [C5] Monthly entry points with a 63-day forward window produce overlapping
       labels (~42-day overlap between adjacent months).  This inflates apparent
       cross-validation performance.  Use the OOS Sharpe column with caution.
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from scipy.optimize import minimize
except ImportError as exc:
    sys.exit(
        "calibrate.py requires scikit-learn and scipy.\n"
        "Install with:  pip install scikit-learn scipy\n"
        f"Error: {exc}"
    )

from indicators import ema, rsi as _rsi, safe_float
from scoring import geo_score_fallback
from tickers import TICKER_MAP

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_FILE = os.path.join(os.path.dirname(__file__), "calibrated_weights.json")

DEFAULT_WEIGHTS: dict[str, float] = {
    "pe": 15.0, "peg": 15.0, "roe": 10.0,   # fundamental bucket (cap 40)
    "rsi": 15.0, "upside": 15.0,              # technical bucket   (cap 30)
    "geo": 15.0, "sentiment": 15.0,           # geosent bucket     (cap 30)
    "vol_penalty": 3.0,                        # subtracted from geosent
}
DEFAULT_CAPS: dict[str, int] = {
    "fundamental": 40, "technical": 30, "geosent": 30,
}

# Ordered — order must be consistent throughout this file.
FEATURES = [
    "pe_norm", "peg_norm", "roe_norm",       # fundamental
    "rsi_norm", "upside_norm",                # technical
    "geo_norm", "sent_norm", "vol_pen",       # geosent (vol_pen is a penalty)
]

FEAT_TO_KEY = {
    "pe_norm": "pe", "peg_norm": "peg", "roe_norm": "roe",
    "rsi_norm": "rsi", "upside_norm": "upside",
    "geo_norm": "geo", "sent_norm": "sentiment", "vol_pen": "vol_penalty",
}

BUCKET_FEATURES = {
    "fundamental": ["pe_norm", "peg_norm", "roe_norm"],
    "technical":   ["rsi_norm", "upside_norm"],
    "geosent":     ["geo_norm", "sent_norm", "vol_pen"],
}

_MIN_OBS_WARN  = 200
_MIN_OBS_ABORT = 80


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE NORMALIZATION
# Converts raw inputs to [0, 1] sub-scores matching scoring.py step functions.
# When weights == DEFAULT_WEIGHTS the score produced is identical to the
# original hardcoded scoring.py output.
# ─────────────────────────────────────────────────────────────────────────────

def normalize_features(
    pe, peg, roe, rsi_val, upside, geo, sentiment, vol
) -> dict:
    """Return dict of normalized score features, each in [0, 1] or NaN."""
    nf: dict[str, float] = {}

    # PE tiers: 15 pts / 10 pts / 5 pts → norms 1.0 / 0.667 / 0.333
    if pe is not None:
        pv = float(pe)
        nf["pe_norm"] = 1.0 if 0 < pv < 20 else (10 / 15 if pv < 30 else 5 / 15)
    else:
        nf["pe_norm"] = float("nan")

    # PEG tiers: 15 / 8 / 3 → norms 1.0 / 0.533 / 0.200
    if peg is not None:
        pg = float(peg)
        nf["peg_norm"] = 1.0 if pg < 1 else (8 / 15 if pg < 2 else 3 / 15)
    else:
        nf["peg_norm"] = float("nan")

    # ROE tiers: 10 / 6 / 3 / 0 → norms 1.0 / 0.6 / 0.3 / 0.0
    if roe is not None:
        rv = float(roe)
        nf["roe_norm"] = (
            1.0 if rv > 0.20 else (0.6 if rv > 0.10 else (0.3 if rv > 0 else 0.0))
        )
    else:
        nf["roe_norm"] = float("nan")

    # RSI tiers: 15 (30–60) / 8 (60–70) / 3 (<30 or >70) → 1.0 / 0.533 / 0.200
    if rsi_val is not None:
        rv2 = float(rsi_val)
        nf["rsi_norm"] = 1.0 if 30 <= rv2 <= 60 else (8 / 15 if rv2 <= 70 else 3 / 15)
    else:
        nf["rsi_norm"] = float("nan")

    # Upside tiers: 15 (>25%) / 10 (>10%) / 5 (>0%) / 0 → 1.0 / 0.667 / 0.333 / 0.0
    if upside is not None:
        up = float(upside)
        nf["upside_norm"] = (
            1.0 if up > 25 else (10 / 15 if up > 10 else (5 / 15 if up > 0 else 0.0))
        )
    else:
        nf["upside_norm"] = float("nan")

    # Geo tiers: 15 (≤3) / 10 (≤6) / 3 (>6) → 1.0 / 0.667 / 0.200
    if geo is not None:
        nf["geo_norm"] = 1.0 if geo <= 3 else (10 / 15 if geo <= 6 else 3 / 15)
    else:
        nf["geo_norm"] = float("nan")

    # Sentiment: linear (−100, +100) → (0, 1).  Absent → 0.5 (neutral).
    nf["sent_norm"] = (float(sentiment) + 100) / 200 if sentiment is not None else 0.5

    # Vol penalty flag: 1 if annualised vol > 60%, else 0
    nf["vol_pen"] = 1.0 if vol is not None and float(vol) > 60 else 0.0

    return nf


# ─────────────────────────────────────────────────────────────────────────────
# PRICE FEATURES AT CUTOFF
# All computed from the series sliced to the entry date — no lookahead.
# ─────────────────────────────────────────────────────────────────────────────

def price_features_at_cutoff(
    close: pd.Series, high: pd.Series, low: pd.Series
) -> dict:
    """
    Return price-derived features using only the supplied history slice.
    Caller is responsible for slicing to the cutoff date before calling.

    Returns dict with keys: rsi, vol_ann, ema50_rel, bollinger_b, macd_hist.
    """
    pf: dict = {
        "rsi": None, "vol_ann": None, "ema50_rel": None,
        "bollinger_b": None, "macd_hist": None,
    }

    if len(close) < 30:
        return pf

    # RSI (14)
    pf["rsi"] = _rsi(close)

    # Annualised volatility — 252-bar window
    ret = close.pct_change().dropna().tail(252)
    if len(ret) >= 20:
        pf["vol_ann"] = float(ret.std() * np.sqrt(252) * 100)

    # EMA50-relative: (price / EMA50 − 1) × 100  (positive = above EMA)
    if len(close) >= 50:
        e50 = float(ema(close, 50).iloc[-1])
        price = float(close.iloc[-1])
        if e50 > 0:
            pf["ema50_rel"] = (price / e50 - 1) * 100

    # Bollinger %B (20-bar, 2σ)
    if len(close) >= 20:
        bb_mid = float(ema(close, 20).iloc[-1])
        bb_std = float(close.tail(20).std())
        price  = float(close.iloc[-1])
        if bb_std > 0:
            bb_up = bb_mid + 2 * bb_std
            bb_lo = bb_mid - 2 * bb_std
            pf["bollinger_b"] = (price - bb_lo) / (bb_up - bb_lo)

    # MACD histogram (EMA12 − EMA26 − signal(9))
    if len(close) >= 35:
        macd_line = ema(close, 12) - ema(close, 26)
        signal    = ema(macd_line, 9)
        pf["macd_hist"] = float((macd_line - signal).iloc[-1])

    return pf


# ─────────────────────────────────────────────────────────────────────────────
# DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    ticker_map: dict | None = None,
    lookback_months: int = 24,
    forward_days: int = 63,
    min_history_bars: int = 252,
) -> pd.DataFrame:
    """
    Build the calibration panel.

    For each ticker, roll across `lookback_months` monthly entry points.
    At each entry date, compute features using only data up to that date.
    Label is the forward return over `forward_days` trading days.

    Caveats printed to stdout (C1–C5 from module docstring).
    """
    if ticker_map is None:
        ticker_map = TICKER_MAP

    today = pd.Timestamp(datetime.now().date())
    # Latest valid entry date: must have at least forward_days bars of future data.
    latest_entry = today - pd.Timedelta(days=int(forward_days * 1.5))

    rows: list[dict] = []
    print(
        f"\n{'─'*70}\n"
        f"[calibrate] Building dataset\n"
        f"  Tickers        : {len(ticker_map)}\n"
        f"  Lookback       : {lookback_months} months\n"
        f"  Forward window : {forward_days} trading days\n"
        f"{'─'*70}\n"
        f"  [C1] PE/PEG/ROE use today's values (lookahead bias in fundamentals)\n"
        f"  [C2] Sentiment fixed at neutral (no historical archive)\n"
        f"  [C3] Upside uses today's analyst target vs historical price\n"
        f"{'─'*70}\n"
    )

    for name, symbol in ticker_map.items():
        try:
            tk   = yf.Ticker(symbol)
            info = tk.info or {}
            if not info or info.get("quoteType") is None:
                print(f"  SKIP  {symbol:<14} — no yfinance metadata")
                continue

            qt       = info.get("quoteType", "")
            is_etf   = qt in ("ETF", "MUTUALFUND")
            raw_ccy  = info.get("currency") or info.get("financialCurrency") or "USD"
            is_pence = raw_ccy in ("GBp", "GBX", "GBx")

            # Current fundamentals — [C1]
            pe  = safe_float(info.get("trailingPE"))
            peg = safe_float(info.get("pegRatio"))
            roe = safe_float(info.get("returnOnEquity"))
            tgt = safe_float(info.get("targetMeanPrice"))
            if is_pence and tgt:
                tgt /= 100

            # Geo score — static (country + sector only, no news bump) — no lookahead
            geo = geo_score_fallback(info, [], symbol)

            # Full price history
            hist = yf.download(
                symbol, start="2018-01-01", auto_adjust=True, progress=False
            )
            if hist.empty or "Close" not in hist.columns:
                print(f"  SKIP  {symbol:<14} — no price history")
                continue

            close_raw = hist["Close"].squeeze()
            high_raw  = hist["High"].squeeze() if "High"  in hist.columns else close_raw.copy()
            low_raw   = hist["Low"].squeeze()  if "Low"   in hist.columns else close_raw.copy()

            # Normalise pence tickers to pounds
            if is_pence:
                close_raw = close_raw / 100
                high_raw  = high_raw  / 100
                low_raw   = low_raw   / 100

            # Handle tz-aware index from newer yfinance versions
            if close_raw.index.tz is not None:
                close_raw.index = close_raw.index.tz_localize(None)
                high_raw.index  = high_raw.index.tz_localize(None)
                low_raw.index   = low_raw.index.tz_localize(None)

            n_obs = 0
            for month_offset in range(1, lookback_months + 1):
                cutoff_ts = today - pd.DateOffset(months=month_offset)

                # Skip if no room for a full forward window
                if cutoff_ts > latest_entry:
                    continue

                exit_window_end = cutoff_ts + pd.Timedelta(days=int(forward_days * 1.6))

                past_mask   = close_raw.index <= cutoff_ts
                future_mask = (close_raw.index > cutoff_ts) & \
                              (close_raw.index <= exit_window_end)

                c_past = close_raw[past_mask]
                h_past = high_raw[past_mask]
                l_past = low_raw[past_mask]
                c_fut  = close_raw[future_mask]

                if len(c_past) < min_history_bars or len(c_fut) < forward_days:
                    continue

                entry_price = float(c_past.iloc[-1])
                if entry_price <= 0:
                    continue

                exit_price  = float(c_fut.iloc[forward_days - 1])
                fwd_pnl     = (exit_price - entry_price) / entry_price * 100
                win         = int(fwd_pnl > 0)

                # Price-derived features (no lookahead) ─────────────────────
                pf = price_features_at_cutoff(c_past, h_past, l_past)

                # Upside at entry using today's analyst target — [C3]
                upside_at_t = (
                    (tgt - entry_price) / entry_price * 100
                    if tgt is not None and entry_price > 0
                    else None
                )

                nf = normalize_features(
                    pe, peg, roe,
                    pf["rsi"],
                    upside_at_t,
                    geo,
                    sentiment=None,      # [C2] no historical sentiment
                    vol=pf["vol_ann"],
                )

                rows.append({
                    "ticker":      symbol,
                    "name":        name,
                    "entry_date":  pd.Timestamp(cutoff_ts),
                    "entry_price": round(entry_price, 4),
                    "exit_price":  round(exit_price, 4),
                    "forward_pnl": round(fwd_pnl, 3),
                    "win":         win,
                    "is_etf":      is_etf,
                    **nf,
                    "ema50_rel":   pf["ema50_rel"],
                    "bollinger_b": pf["bollinger_b"],
                    "macd_hist":   pf["macd_hist"],
                })
                n_obs += 1

            status = "ok" if n_obs > 0 else "WARN: 0 observations"
            print(f"  {symbol:<14}  {n_obs:>3} obs   {status}")

        except Exception as exc:
            print(f"  ERROR {symbol:<14}: {exc}")
            continue

    if not rows:
        raise RuntimeError(
            "Dataset is empty.  Check that yfinance can reach the network "
            "and that TICKER_MAP contains valid symbols."
        )

    df = pd.DataFrame(rows)
    df["entry_date"] = pd.to_datetime(df["entry_date"])

    win_rate  = df["win"].mean()
    n_tickers = df["ticker"].nunique()
    print(
        f"\n{'─'*70}\n"
        f"[calibrate] Dataset complete\n"
        f"  Rows           : {len(df)}\n"
        f"  Tickers        : {n_tickers}\n"
        f"  Win rate       : {win_rate:.1%}\n"
        f"{'─'*70}\n"
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RELIABILITY WARNINGS
# ─────────────────────────────────────────────────────────────────────────────

def reliability_report(df: pd.DataFrame) -> list[str]:
    """
    Return a list of human-readable warning strings for the dataset.
    Warnings do not abort the run but should be reported prominently.
    """
    warns = []
    n = len(df)

    if n < _MIN_OBS_ABORT:
        raise RuntimeError(
            f"Only {n} observations — minimum {_MIN_OBS_ABORT} required.  "
            "Extend --lookback or add more tickers."
        )
    if n < _MIN_OBS_WARN:
        warns.append(
            f"SMALL SAMPLE: {n} observations (< {_MIN_OBS_WARN}).  "
            "Logistic regression weights have high variance.  "
            "Use L2 regularisation (already applied) and treat results as directional."
        )

    # Check for overlapping forward windows
    slices = df["entry_date"].dt.to_period("M").nunique()
    warns.append(
        f"[C5] {slices} monthly time-slices.  "
        "Adjacent entry points share ~two-thirds of their holding period.  "
        "Cross-validation accuracy is optimistically biased."
    )

    # Feature NaN rates
    for feat in FEATURES:
        nan_rate = df[feat].isna().mean()
        if nan_rate > 0.3:
            warns.append(
                f"Feature '{feat}' is missing in {nan_rate:.0%} of rows "
                "(imputed with column median — interpret its weight cautiously)."
            )
        if nan_rate == 1.0:
            warns.append(
                f"Feature '{feat}' is missing in all rows — weight will be set to 0."
            )

    # Class balance
    win_rate = df["win"].mean()
    if not 0.35 <= win_rate <= 0.65:
        warns.append(
            f"Class imbalance: win rate = {win_rate:.1%}.  "
            "Logistic regression accuracy metric may be misleading; "
            "inspect AUC-ROC instead."
        )

    warns += [
        "[C1] PE/PEG/ROE features use today's values — lookahead bias present.",
        "[C2] AI sentiment fixed at 0.5 (neutral) — cannot be calibrated from price history.",
        "[C3] Upside uses today's analyst target applied to historical prices.",
    ]

    return warns


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE HELPER
# Computes the composite score for a row using the given weight/cap dicts.
# Mirrors the logic in deep_score() — keep in sync with scoring.py changes.
# ─────────────────────────────────────────────────────────────────────────────

def _score_row(row: pd.Series, weights: dict, caps: dict) -> float:
    W  = weights
    NF = FEATURES

    def _get(col: str) -> float | None:
        v = row.get(col)
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

    # Fundamental bucket
    f = 0.0
    for feat in BUCKET_FEATURES["fundamental"]:
        v = _get(feat)
        if v is not None:
            f += W[FEAT_TO_KEY[feat]] * v
    f = min(f, caps["fundamental"])

    # Technical bucket
    t = 0.0
    for feat in BUCKET_FEATURES["technical"]:
        v = _get(feat)
        if v is not None:
            t += W[FEAT_TO_KEY[feat]] * v
    t = min(t, caps["technical"])

    # GeoSentiment bucket
    g = 0.0
    for feat in ["geo_norm", "sent_norm"]:
        v = _get(feat)
        if v is not None:
            g += W[FEAT_TO_KEY[feat]] * v
    vol_p = _get("vol_pen") or 0.0
    g -= W["vol_penalty"] * vol_p
    g = min(max(g, 0.0), caps["geosent"])

    return min(f + t + g, 100.0)


def _rank_ic(df: pd.DataFrame, weights: dict, caps: dict) -> float:
    """
    Mean Spearman rank IC across monthly time-slices.
    Returns the mean IC (0.0 if too few slices).
    """
    df = df.copy()
    df["score"] = df.apply(lambda r: _score_row(r, weights, caps), axis=1)
    df["ym"]    = df["entry_date"].dt.to_period("M")

    ics = []
    for _, grp in df.groupby("ym"):
        if len(grp) < 5:
            continue
        ic, _ = spearmanr(grp["score"], grp["forward_pnl"])
        if not np.isnan(ic):
            ics.append(ic)

    return float(np.mean(ics)) if ics else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: L2 LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def run_logistic(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Fit L2-regularised logistic regression on binary win/loss labels.
    Uses 5-fold stratified CV.  Features are median-imputed then standardised.

    Returns
    -------
    (weights, caps, meta)
      weights : dict matching DEFAULT_WEIGHTS key structure
      caps    : dict matching DEFAULT_CAPS (unchanged — logistic regression
                does not constrain to bucket caps)
      meta    : diagnostics dict for the calibration report
    """
    print("[calibrate] Method 1: L2 logistic regression")

    X_raw = df[FEATURES].values.astype(float)
    y     = df["win"].values.astype(int)

    # Median imputation for missing fundamental data
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)

    # Standardise so coefficient magnitudes are comparable
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_imp)

    # 5-fold stratified CV — class-balanced to handle win-rate skew
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model   = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")

    cv_acc  = cross_val_score(model, X_sc, y, cv=cv, scoring="accuracy")
    cv_auc  = cross_val_score(model, X_sc, y, cv=cv, scoring="roc_auc")

    # Fit on full dataset for coefficient extraction
    model.fit(X_sc, y)
    coef = model.coef_[0]          # shape (n_features,)

    print(f"  CV accuracy : {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  CV AUC-ROC  : {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    # Map standardised coefficients → scoring weights
    # 1. Exponentiate to get relative importances (all positive)
    # 2. vol_pen coefficient sign is inverted (it's a penalty, not a reward)
    signed_coef = coef.copy()
    vol_pen_idx = FEATURES.index("vol_pen")
    signed_coef[vol_pen_idx] = -coef[vol_pen_idx]   # invert: negative LR coef → positive penalty

    # Clip at zero: a negative coef after sign-flip means the feature is
    # counterproductive; its weight is forced to zero.
    pos_coef = np.clip(signed_coef, 0, None)

    # Warn if any coefficient was clipped
    for i, feat in enumerate(FEATURES):
        if signed_coef[i] < 0:
            print(
                f"  WARNING: '{feat}' has negative predictive direction "
                f"(coef={coef[i]:.3f}).  Weight forced to 0; "
                "consider removing it from the score."
            )

    # Normalise within each bucket to its cap
    weights = dict(DEFAULT_WEIGHTS)  # start from defaults
    for bucket, feats in BUCKET_FEATURES.items():
        cap     = DEFAULT_CAPS[bucket]
        indices = [FEATURES.index(f) for f in feats]
        bucket_coefs = pos_coef[indices]
        total_coef   = bucket_coefs.sum()
        if total_coef > 0:
            for feat, bc in zip(feats, bucket_coefs):
                key = FEAT_TO_KEY[feat]
                if key == "vol_penalty":
                    weights[key] = float(bc / total_coef * cap)
                else:
                    weights[key] = float(bc / total_coef * cap)
        # If all bucket coefs are zero, preserve defaults.

    # vol_penalty is special: it's not normalised with the bucket positive weights
    # but derived from the (sign-inverted) vol_pen coefficient proportion.
    # Re-derive it scaled to a sensible range [0, 10].
    raw_vol_pen_coef = pos_coef[vol_pen_idx]
    total_geosent_coef_excl_pen = sum(
        pos_coef[FEATURES.index(f)] for f in ["geo_norm", "sent_norm"]
    ) + 1e-9
    weights["vol_penalty"] = float(
        np.clip(raw_vol_pen_coef / total_geosent_coef_excl_pen * DEFAULT_CAPS["geosent"], 0, 10)
    )

    # Compute in-sample rank IC with calibrated weights
    ic = _rank_ic(df, weights, DEFAULT_CAPS)
    print(f"  In-sample rank IC : {ic:.4f}")
    print(f"  Calibrated weights: {_fmt_weights(weights)}\n")

    meta = {
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std":  round(float(cv_acc.std()),  4),
        "cv_auc_mean":      round(float(cv_auc.mean()), 4),
        "cv_auc_std":       round(float(cv_auc.std()),  4),
        "in_sample_rank_ic": round(ic, 4),
        "n_observations":   int(len(df)),
        "raw_coefs":        {FEATURES[i]: round(float(coef[i]), 4) for i in range(len(FEATURES))},
    }
    return weights, dict(DEFAULT_CAPS), meta


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: L-BFGS-B RANK-IC MAXIMISATION (Nelder-Mead equivalent, bounded)
# ─────────────────────────────────────────────────────────────────────────────

def _neg_ic_objective(params: np.ndarray, df: pd.DataFrame) -> float:
    """
    Negative mean rank-IC for scipy.minimize (minimise ⟹ maximise IC).
    `params` is a flat array of 8 weights in FEATURES order.
    """
    W = {FEAT_TO_KEY[f]: float(params[i]) for i, f in enumerate(FEATURES)}
    return -_rank_ic(df, W, DEFAULT_CAPS)


def run_nelder_mead(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    L-BFGS-B optimisation maximising mean rank-IC across monthly time-slices.

    Note: with ~12 time-slices the IC estimate has high variance.
    Result should be considered exploratory.  [C4]

    Returns (weights, caps, meta).
    """
    print("[calibrate] Method 2: L-BFGS-B rank-IC optimisation  [C4 — exploratory]")

    # Bounds: each weight ≥ 0; penalty capped at 10.
    # Upper bounds reflect original max values with 2× headroom.
    bounds = [
        (0,  30),   # pe
        (0,  30),   # peg
        (0,  20),   # roe
        (0,  30),   # rsi
        (0,  30),   # upside
        (0,  30),   # geo
        (0,  30),   # sentiment
        (0,  10),   # vol_penalty
    ]

    # Use default weights as starting point
    x0 = np.array([DEFAULT_WEIGHTS[FEAT_TO_KEY[f]] for f in FEATURES], dtype=float)

    # Multiple restarts with random perturbations for robustness
    best_result = None
    best_val    = np.inf
    rng         = np.random.default_rng(seed=42)

    for trial in range(8):
        if trial == 0:
            x_init = x0.copy()
        else:
            x_init = x0 * rng.uniform(0.5, 1.5, size=len(x0))
            x_init = np.clip(x_init, [b[0] for b in bounds], [b[1] for b in bounds])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                _neg_ic_objective, x_init, args=(df,),
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-9},
            )

        if res.fun < best_val:
            best_val    = res.fun
            best_result = res

    weights = {
        FEAT_TO_KEY[f]: float(best_result.x[i]) for i, f in enumerate(FEATURES)
    }

    ic = -best_val
    print(f"  Optimised rank IC : {ic:.4f}")
    print(f"  Calibrated weights: {_fmt_weights(weights)}\n")

    meta = {
        "optimised_rank_ic": round(ic, 4),
        "n_observations":    int(len(df)),
        "n_restarts":        8,
        "converged":         bool(best_result.success),
        "warning":           (
            "IC estimate is based on few time-slices — high variance.  "
            "Treat as directional only.  [C4]"
        ),
    }
    return weights, dict(DEFAULT_CAPS), meta


# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: BUCKET-CAP GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Grid search over bucket cap combinations.

    Searches:
        F_cap ∈ {20, 25, 30, 35, 40, 45, 50}
        T_cap ∈ {15, 20, 25, 30, 35, 40}
        G_cap = 100 − F_cap − T_cap   (valid if 10 ≤ G_cap ≤ 45)

    Within each candidate cap set, individual weights are scaled
    proportionally from defaults (so relative within-bucket importance
    is preserved).  The winning cap set is selected by 5-fold CV win rate.

    Returns (weights, caps, meta).
    """
    print("[calibrate] Method 3: Bucket-cap grid search")

    X_raw = df[FEATURES].values.astype(float)
    y     = df["win"].values.astype(int)

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_raw)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f_range = [20, 25, 30, 35, 40, 45, 50]
    t_range = [15, 20, 25, 30, 35, 40]

    results = []

    for f_cap in f_range:
        for t_cap in t_range:
            g_cap = 100 - f_cap - t_cap
            if not (10 <= g_cap <= 45):
                continue

            candidate_caps = {
                "fundamental": f_cap,
                "technical":   t_cap,
                "geosent":     g_cap,
            }
            # Scale within-bucket weights proportionally to new caps
            candidate_weights = {}
            for bucket, feats in BUCKET_FEATURES.items():
                orig_cap      = DEFAULT_CAPS[bucket]
                new_cap       = candidate_caps[bucket]
                scale         = new_cap / orig_cap
                for feat in feats:
                    key = FEAT_TO_KEY[feat]
                    candidate_weights[key] = DEFAULT_WEIGHTS[key] * scale

            # Compute composite score for all rows under this candidate
            scores = df.apply(
                lambda r: _score_row(r, candidate_weights, candidate_caps), axis=1
            ).values

            # 5-fold CV win rate: top-decile observations
            fold_win_rates = []
            for train_idx, test_idx in cv.split(X_imp, y):
                test_scores = scores[test_idx]
                test_wins   = y[test_idx]
                if len(test_scores) == 0:
                    continue
                thresh = np.percentile(test_scores, 75)
                high_score_mask = test_scores >= thresh
                if high_score_mask.sum() == 0:
                    continue
                fold_win_rates.append(test_wins[high_score_mask].mean())

            if not fold_win_rates:
                continue

            cv_win  = float(np.mean(fold_win_rates))
            rank_ic = _rank_ic(df, candidate_weights, candidate_caps)

            results.append({
                "f_cap":    f_cap,
                "t_cap":    t_cap,
                "g_cap":    g_cap,
                "cv_win":   cv_win,
                "rank_ic":  rank_ic,
                "weights":  candidate_weights,
                "caps":     candidate_caps,
            })

    if not results:
        print("  No valid grid combinations found.  Returning defaults.\n")
        return dict(DEFAULT_WEIGHTS), dict(DEFAULT_CAPS), {}

    # Rank by CV win rate (primary), rank IC (secondary)
    results.sort(key=lambda r: (r["cv_win"], r["rank_ic"]), reverse=True)
    best  = results[0]
    top5  = results[:5]

    print(
        f"  Best caps  : F={best['f_cap']} / T={best['t_cap']} / G={best['g_cap']}\n"
        f"  CV win rate: {best['cv_win']:.3f}\n"
        f"  Rank IC    : {best['rank_ic']:.4f}\n"
    )
    print("  Top-5 combinations:")
    for r in top5:
        print(
            f"    F={r['f_cap']:2d} T={r['t_cap']:2d} G={r['g_cap']:2d}  "
            f"CV-win={r['cv_win']:.3f}  IC={r['rank_ic']:.4f}"
        )
    print(f"  Calibrated weights: {_fmt_weights(best['weights'])}\n")

    meta = {
        "best_caps":         best["caps"],
        "cv_win_rate":       round(best["cv_win"], 4),
        "rank_ic":           round(best["rank_ic"], 4),
        "n_combinations_tested": len(results),
        "top_5": [
            {"f": r["f_cap"], "t": r["t_cap"], "g": r["g_cap"],
             "cv_win": round(r["cv_win"], 4), "rank_ic": round(r["rank_ic"], 4)}
            for r in top5
        ],
    }
    return best["weights"], best["caps"], meta


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_weights(
    weights: dict,
    caps: dict,
    method: str,
    meta: dict,
    caveats: list[str],
) -> None:
    """Write calibrated_weights.json.  Loaded automatically by scoring.py."""
    payload = {
        "method":        method,
        "generated_at":  datetime.now().isoformat(timespec="seconds"),
        "weights":       {k: round(float(v), 4) for k, v in weights.items()},
        "caps":          {k: int(v) for k, v in caps.items()},
        "meta":          meta,
        "caveats":       caveats,
    }
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[calibrate] Weights written → {WEIGHTS_FILE}\n")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_weights(w: dict) -> str:
    return "  ".join(f"{k}={v:.1f}" for k, v in w.items())


def _choose_best(results: list[tuple]) -> tuple:
    """
    Pick the method with the highest in-sample rank IC.
    results: list of (method_name, weights, caps, meta).
    """
    def _ic(meta):
        return max(
            meta.get("in_sample_rank_ic", 0.0),
            meta.get("optimised_rank_ic", 0.0),
            meta.get("rank_ic",           0.0),
        )
    return max(results, key=lambda r: _ic(r[3]))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate scoring.py weights from historical backtest data."
    )
    parser.add_argument(
        "--method",
        choices=["logreg", "nelder", "grid", "all"],
        default="all",
        help="Optimisation method (default: all three, best written to JSON).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=63,
        help="Forward return window in trading days (default: 63 ≈ 3 months).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=24,
        help="Months of history to sample entry points from (default: 24).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=7,
        help="Assets per time-slice used in portfolio Sharpe (default: 7).",
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="Write calibration_dataset.csv alongside the weights JSON.",
    )
    args = parser.parse_args()

    # ── Build dataset ────────────────────────────────────────────────────────
    df = build_dataset(
        lookback_months=args.lookback,
        forward_days=args.horizon,
    )

    # ── Reliability warnings ─────────────────────────────────────────────────
    warns = reliability_report(df)
    print(f"\n{'─'*70}\nRELIABILITY WARNINGS\n{'─'*70}")
    for w in warns:
        print(f"  ! {w}")
    print()

    if args.save_dataset:
        ds_path = os.path.join(os.path.dirname(WEIGHTS_FILE), "calibration_dataset.csv")
        df.to_csv(ds_path, index=False)
        print(f"[calibrate] Dataset saved → {ds_path}\n")

    # ── Run requested methods ────────────────────────────────────────────────
    all_results: list[tuple] = []

    if args.method in ("logreg", "all"):
        w, c, m = run_logistic(df)
        all_results.append(("logreg", w, c, m))

    if args.method in ("nelder", "all"):
        w, c, m = run_nelder_mead(df)
        all_results.append(("nelder", w, c, m))

    if args.method in ("grid", "all"):
        w, c, m = run_grid_search(df)
        all_results.append(("grid", w, c, m))

    # ── Select winner and write JSON ─────────────────────────────────────────
    if len(all_results) == 1:
        method_name, best_weights, best_caps, best_meta = all_results[0]
    else:
        method_name, best_weights, best_caps, best_meta = _choose_best(all_results)

    print(
        f"{'─'*70}\n"
        f"WINNING METHOD : {method_name}\n"
        f"{'─'*70}"
    )
    print("Final weights:")
    for key, val in best_weights.items():
        default = DEFAULT_WEIGHTS.get(key, "—")
        delta   = val - float(default) if isinstance(default, (int, float)) else 0.0
        bar     = "▲" if delta > 0.5 else ("▼" if delta < -0.5 else "─")
        print(f"  {key:<14} {val:6.2f}   (default {default:5.1f})  {bar}")
    print(f"\nBucket caps: {best_caps}")

    save_weights(
        weights=best_weights,
        caps=best_caps,
        method=method_name,
        meta=best_meta,
        caveats=warns,
    )

    print(
        "Run `python main.py` — scoring.py will pick up the new weights automatically.\n"
        "Re-run `python calibrate.py` whenever the watchlist or regime changes.\n"
    )


def run_calibration(
    method: str = "logreg",
    lookback_months: int = 24,
    forward_days: int = 63,
    topk: int = 7,
    save_dataset: bool = False,
    force: bool = False,
) -> dict:
    """
    Programmatic entry point called by main.py before the analysis pipeline.

    Runs the calibration, writes calibrated_weights.json (consumed by
    scoring.py at import time), and returns a config dict of indicator-period
    overrides for analyze().

    Parameters
    ----------
    method          : "logreg" | "nelder" | "grid" | "all"  (default "logreg")
    lookback_months : Entry-point history in months (default 24).
    forward_days    : Forward return window in trading days (default 63 ≈ 3 months).
    topk            : Assets per time-slice for portfolio ranking (default 7).
    save_dataset    : Also write calibration_dataset.csv when True.
    force           : Re-run even if a fresh calibrated_weights.json exists.

    Returns
    -------
    dict with keys understood by analyze():
        rsi_period, ema_fast, ema_slow, macd_signal
    These are derived from the winning bucket weights; currently they map the
    calibrated RSI/upside weights back to a practical period range.
    """
    import warnings as _warnings
    _warnings.filterwarnings("ignore")

    # Skip re-running if weights are fresh (< 12 hours old) and force=False.
    if not force and os.path.exists(WEIGHTS_FILE):
        age_h = (
            datetime.now().timestamp() - os.path.getmtime(WEIGHTS_FILE)
        ) / 3600
        if age_h < 12:
            print(
                f"[calibrate] Skipping — calibrated_weights.json is "
                f"{age_h:.1f}h old (< 12h).  Pass force=True to re-run.\n"
            )
            return _load_config_from_weights()

    # ── Build dataset ────────────────────────────────────────────────────────
    df = build_dataset(lookback_months=lookback_months, forward_days=forward_days)

    # ── Reliability warnings ─────────────────────────────────────────────────
    warns = reliability_report(df)
    print(f"\n{'─'*70}\nRELIABILITY WARNINGS\n{'─'*70}")
    for w in warns:
        print(f"  ! {w}")
    print()

    if save_dataset:
        ds_path = os.path.join(os.path.dirname(WEIGHTS_FILE), "calibration_dataset.csv")
        df.to_csv(ds_path, index=False)
        print(f"[calibrate] Dataset saved → {ds_path}\n")

    # ── Run requested methods ────────────────────────────────────────────────
    all_results: list[tuple] = []

    if method in ("logreg", "all"):
        w, c, m = run_logistic(df)
        all_results.append(("logreg", w, c, m))

    if method in ("nelder", "all"):
        w, c, m = run_nelder_mead(df)
        all_results.append(("nelder", w, c, m))

    if method in ("grid", "all"):
        w, c, m = run_grid_search(df)
        all_results.append(("grid", w, c, m))

    # ── Select winner and write JSON ─────────────────────────────────────────
    if len(all_results) == 1:
        method_name, best_weights, best_caps, best_meta = all_results[0]
    else:
        method_name, best_weights, best_caps, best_meta = _choose_best(all_results)

    print(
        f"{'─'*70}\n"
        f"WINNING METHOD : {method_name}\n"
        f"{'─'*70}"
    )
    print("Final weights:")
    for key, val in best_weights.items():
        default = DEFAULT_WEIGHTS.get(key, "—")
        delta   = val - float(default) if isinstance(default, (int, float)) else 0.0
        bar     = "▲" if delta > 0.5 else ("▼" if delta < -0.5 else "─")
        print(f"  {key:<14} {val:6.2f}   (default {default:5.1f})  {bar}")
    print(f"\nBucket caps: {best_caps}")

    save_weights(
        weights=best_weights,
        caps=best_caps,
        method=method_name,
        meta=best_meta,
        caveats=warns,
    )

    return _load_config_from_weights()


def _load_config_from_weights() -> dict:
    """
    Load calibrated_weights.json and derive indicator period overrides.

    The RSI period is mapped linearly from the calibrated RSI weight:
      weight ∈ [0, 30] → period ∈ [21, 9]  (higher weight → shorter, more reactive period)
    MACD fast/slow/signal remain at standard values unless overridden.
    """
    if not os.path.exists(WEIGHTS_FILE):
        return {}

    try:
        with open(WEIGHTS_FILE) as f:
            payload = json.load(f)
        weights = payload.get("weights", {})
        rsi_w   = float(weights.get("rsi", DEFAULT_WEIGHTS["rsi"]))
        # Linear interpolation: weight 0→period 21, weight 30→period 9
        rsi_period = int(round(21 - (rsi_w / 30) * 12))
        rsi_period = max(9, min(21, rsi_period))  # clamp to [9, 21]
        return {
            "rsi_period":  rsi_period,
            "ema_fast":    12,   # standard; extend here if grid search covers MACD
            "ema_slow":    26,
            "macd_signal":  9,
        }
    except Exception:
        return {}


if __name__ == "__main__":
    main()

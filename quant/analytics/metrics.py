"""
metrics.py — Advanced Alpha Metrics (v10.4.0, Phase 5).

Intent: Sharpe is easily manipulated. Institutions use robust, deflated metrics.
This module provides:
  - Deflated Sharpe Ratio (Bailey & Lopez de Prado): penalizes the Sharpe for the
    number of strategies tested and the skew/kurtosis of returns, proving the edge
    is not just data snooping.
  - Alpha Decay Curves: Information Coefficient (IC) of a signal at T+1/T+5/T+21.
    Smooth decay validates the exit logic.
  - Probabilistic Turnover: mean AND variance of turnover. High variance means
    unpredictable transaction costs.

Invariants: all functions are pure (no I/O); empty inputs return neutral values.
Dependencies: numpy, pandas, scipy.stats.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Euler-Mascheroni constant, used by the expected-maximum-Sharpe term.
_EULER_GAMMA = 0.5772156649015329


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int = 1,
    sr_variance: float | None = None,
    periods_per_year: int = 252,
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    Intent: the probability that the observed Sharpe is genuinely positive after
    accounting for (a) how many strategies were tried and (b) non-normal returns.
    A DSR near 1.0 means the edge survives multiple-testing scrutiny.

    Formula:
        SR0 = sqrt(V) * ((1-g)*Z^-1(1-1/N) + g*Z^-1(1-1/(N*e)))
        DSR = Z( (SR - SR0) * sqrt(T-1) /
                 sqrt(1 - skew*SR + (kurt-1)/4 * SR^2) )
    where V is the variance of the trial Sharpe estimates, g is Euler-Mascheroni,
    T is the number of observations, and kurt is the non-excess kurtosis.

    Invariants: returns a probability in [0, 1]; pure function (no I/O).
    """
    from scipy.stats import norm

    clean = returns.dropna() if returns is not None else pd.Series(dtype=float)
    T = len(clean)
    if T < 3 or clean.std() == 0:
        return 0.0

    sr = float(clean.mean() / clean.std())  # per-period Sharpe
    skew = float(clean.skew())
    kurt = float(clean.kurtosis()) + 3.0  # pandas kurtosis is excess

    if sr_variance is None:
        # Variance of the Sharpe estimator under the null.
        sr_variance = (1.0 + 0.5 * sr ** 2) / T
    sr_variance = max(float(sr_variance), 1e-12)

    n = max(int(n_trials), 1)
    if n == 1:
        e_max = 0.0
    else:
        e_max = math.sqrt(sr_variance) * (
            (1.0 - _EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n)
            + _EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * math.e))
        )

    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    z = (sr - e_max) * math.sqrt(T - 1) / denom
    return float(norm.cdf(z))


def information_coefficient(signal: pd.Series, forward_returns: pd.Series) -> float:
    """Spearman rank IC between a signal and forward returns.

    Intent: measure predictive power without assuming linearity. Invariants:
    returns a correlation in [-1, 1]; 0.0 if insufficient overlap; pure function.
    """
    if signal is None or forward_returns is None:
        return 0.0
    common = signal.dropna().index.intersection(forward_returns.dropna().index)
    if len(common) < 3:
        return 0.0
    ic = signal[common].corr(forward_returns[common], method="spearman")
    return float(ic) if pd.notna(ic) else 0.0


def alpha_decay_curve(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 21),
) -> pd.DataFrame:
    """IC of a signal at multiple forward horizons (alpha decay).

    Intent: a healthy signal decays smoothly. A jagged curve means the exit logic
    is misaligned with the signal's actual predictive horizon.
    Args:
        signals: wide DataFrame (dates x symbols) of signal values.
        prices: wide DataFrame (dates x symbols) of prices.
    Invariants: returns columns horizon, mean_ic, ic_std, n; pure function.
    """
    rows = []
    if signals is None or prices is None or signals.empty or prices.empty:
        return pd.DataFrame(columns=["horizon", "mean_ic", "ic_std", "n"])

    for h in horizons:
        fwd = prices.shift(-h) / prices - 1.0
        ics = []
        for d in signals.index:
            if d not in fwd.index:
                continue
            s = signals.loc[d]
            r = fwd.loc[d]
            common = s.dropna().index.intersection(r.dropna().index)
            if len(common) >= 3:
                ic = s[common].corr(r[common], method="spearman")
                if pd.notna(ic):
                    ics.append(float(ic))
        if ics:
            rows.append({
                "horizon": h,
                "mean_ic": round(float(np.mean(ics)), 4),
                "ic_std": round(float(np.std(ics)), 4),
                "n": len(ics),
            })
        else:
            rows.append({"horizon": h, "mean_ic": 0.0, "ic_std": 0.0, "n": 0})
    return pd.DataFrame(rows)


def turnover_stats(weights: pd.DataFrame) -> dict:
    """Mean and variance of portfolio turnover.

    Intent: expected turnover is not enough; its variance drives unpredictable
    transaction costs. High variance => size trades conservatively.
    Args:
        weights: wide DataFrame (dates x assets) of portfolio weights.
    Invariants: returns dict with mean_turnover, turnover_variance, turnover_std;
    pure function (no I/O).
    """
    empty = {"mean_turnover": 0.0, "turnover_variance": 0.0, "turnover_std": 0.0}
    if weights is None or weights.empty or len(weights) < 2:
        return empty
    turnover = weights.diff().abs().sum(axis=1).dropna()
    if turnover.empty:
        return empty
    return {
        "mean_turnover": round(float(turnover.mean()), 4),
        "turnover_variance": round(float(turnover.var()), 6),
        "turnover_std": round(float(turnover.std()), 4),
    }

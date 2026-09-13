"""
portfolio_context.py — Risk-Aware Portfolio Context Layer (Part 2, Upgrade #1).

Intent: evaluate each asset in the context of the whole portfolio, not in
isolation. Computes marginal risk contribution (MRC), factor exposure via PCA,
and a concentration penalty that modulates individual asset scores. This
catches hidden concentration risk (e.g. SXRV.DE + AMZN are ~80% correlated)
that per-asset scoring cannot detect.

Invariants:
  - concentration_penalty returns a multiplier in [0.5, 1.1].
  - compute_risk_contribution returns percentages summing to ~100.
  - Pure computation; no I/O.

Dependencies: numpy, pandas, sklearn.decomposition.PCA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class PortfolioContext:
    """Computes portfolio-level metrics that modulate individual asset scores."""

    def __init__(self, holdings_df: pd.DataFrame, returns_matrix: pd.DataFrame):
        self.holdings = holdings_df
        self.returns = returns_matrix  # Daily returns, columns = symbols
        self.weights = self._compute_current_weights()

    def _compute_current_weights(self) -> pd.Series:
        """Current portfolio weights based on market value."""
        total = self.holdings["Amount_EUR"].sum()
        if total <= 0:
            return pd.Series(dtype=float)
        return self.holdings.set_index("Symbol")["Amount_EUR"] / total

    def compute_factor_exposure(self) -> pd.DataFrame:
        """Run PCA on universe returns to identify hidden factor concentrations.

        Returns: DataFrame of factor loadings per asset (n_components=5).
        """
        from sklearn.decomposition import PCA

        clean = self.returns.dropna(axis=1, how="all").dropna()
        if clean.shape[0] < 5 or clean.shape[1] < 2:
            return pd.DataFrame()

        n_comp = min(5, clean.shape[1], clean.shape[0])
        pca = PCA(n_components=n_comp)
        pca.fit(clean)
        return pd.DataFrame(
            pca.components_.T,
            index=clean.columns,
            columns=[f"Factor_{i}" for i in range(n_comp)],
        )

    def compute_risk_contribution(self) -> pd.Series:
        """Marginal Risk Contribution (MRC) — % of portfolio vol per asset.

        Formula: MRC_i = w_i * (Sigma * w)_i / (w^T * Sigma * w).
        Returns percentages summing to ~100.
        """
        clean = self.returns.dropna(axis=1, how="all").dropna()
        if clean.shape[0] < 2 or clean.shape[1] < 2:
            return pd.Series(dtype=float)

        cov_matrix = clean.cov() * 252  # annualized
        w = self.weights.reindex(cov_matrix.index).fillna(0.0)
        if w.sum() <= 0:
            return pd.Series(dtype=float)

        portfolio_var = float(np.sqrt(w.T @ cov_matrix @ w))
        if portfolio_var <= 0:
            return pd.Series(dtype=float)

        marginal_risk = (cov_matrix @ w) / portfolio_var
        risk_contribution = w * marginal_risk
        total = risk_contribution.sum()
        if total == 0:
            return pd.Series(dtype=float)
        return (risk_contribution / total) * 100.0

    def concentration_penalty(self, symbol: str) -> float:
        """Penalize assets overweight in risk contribution vs capital allocation.

        Returns: multiplier [0.5, 1.1] — lower = more penalty.
        """
        risk_contribution = self.compute_risk_contribution()
        if symbol not in risk_contribution.index:
            return 1.0

        rc = risk_contribution[symbol]
        cw = float(self.weights.get(symbol, 0.0)) * 100.0
        risk_ratio = rc / max(cw, 1.0)

        if risk_ratio > 2.0:
            return 0.6   # 40% penalty
        if risk_ratio > 1.5:
            return 0.75  # 25% penalty
        if risk_ratio < 0.5:
            return 1.1   # 10% bonus (diversifier)
        return 1.0
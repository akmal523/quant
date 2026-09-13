"""
attribution.py — Brinson-Fachler P&L Attribution (Part 2, Upgrade #6).

Intent: decompose portfolio P&L into Allocation, Selection, and Interaction
effects vs a benchmark. Answers "did I outperform by overweighting winning
sectors (allocation) or by picking the best asset in each sector (selection)?"
Without attribution you cannot know what to improve.

Invariants:
  - compute_attribution returns a DataFrame with one row per sector.
  - Total_Effect = Allocation + Selection + Interaction.
  - Pure computation; no I/O.

Dependencies: pandas.
"""
from __future__ import annotations

import pandas as pd


class BrinsonFachlerAttribution:
    """Decomposes portfolio P&L into allocation/selection/interaction effects."""

    def __init__(self, portfolio_df, benchmark_weights, returns_by_asset,
                 returns_by_sector):
        self.portfolio = portfolio_df
        self.benchmark_weights = benchmark_weights  # sector -> weight
        self.returns_asset = returns_by_asset
        self.returns_sector = returns_by_sector

    def _benchmark_total_return(self, period_days: int) -> float:
        """Benchmark total return = sum of sector weights * sector returns."""
        total = 0.0
        for sector, wb in self.benchmark_weights.items():
            rb = self._benchmark_sector_return(sector, period_days)
            total += wb * rb
        return total

    def _benchmark_sector_return(self, sector: str, period_days: int) -> float:
        """Benchmark sector return for the period."""
        col = f"return_{period_days}d"
        if sector in self.returns_sector.index and col in self.returns_sector.columns:
            return float(self.returns_sector.loc[sector, col])
        return 0.0

    def compute_attribution(self, period_days: int = 30) -> pd.DataFrame:
        """Compute Brinson-Fachler attribution per sector."""
        total_value = self.portfolio["Amount_EUR"].sum()
        if total_value <= 0:
            return pd.DataFrame()

        rb_total = self._benchmark_total_return(period_days)
        results = []

        for sector in self.returns_sector.index:
            # Portfolio sector weight.
            sector_mask = self.portfolio["Sector"] == sector
            wp = (self.portfolio.loc[sector_mask, "Amount_EUR"].sum() / total_value
                  if sector_mask.any() else 0.0)
            # Benchmark sector weight.
            wb = self.benchmark_weights.get(sector, 0.0)

            rp_sector = float(self.returns_sector.loc[sector, f"return_{period_days}d"])
            rb_sector = self._benchmark_sector_return(sector, period_days)

            allocation = (wp - wb) * (rb_sector - rb_total)
            selection = wp * (rp_sector - rb_sector)
            interaction = (wp - wb) * (rp_sector - rb_sector)

            results.append({
                "Sector": sector,
                "Portfolio_Weight": round(wp * 100, 1),
                "Benchmark_Weight": round(wb * 100, 1),
                "Allocation_Effect": round(allocation * 100, 2),
                "Selection_Effect": round(selection * 100, 2),
                "Interaction": round(interaction * 100, 2),
                "Total_Effect": round((allocation + selection + interaction) * 100, 2),
            })

        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results).sort_values("Total_Effect", ascending=False)
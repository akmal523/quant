"""
tax_optimizer.py — German Tax-Loss Harvesting (Part 2, Upgrade #3).

Intent: make the bot tax-aware. In a taxable German account, realized losses
offset realized gains within the same year, and the first EUR 1,000 of gains
are tax-free (Sparerpauschbetrag). Selling losers to offset gains, then
repurchasing a correlated replacement, preserves market exposure while
reducing tax drag. Germany has no strict wash-sale rule (immediate repurchase
is allowed).

Key rules (Germany 2026):
  - Abgeltungsteuer 25% + 5.5% Soli = 26.375% on gains.
  - EUR 1,000 annual tax-free allowance.
  - Losses offset gains within the same year.

Invariants:
  - compute_tax_position returns a dict with net_taxable >= 0.
  - harvest_opportunities only proposes non-CORE losers with net benefit > 0.
  - Pure computation; no I/O.

Dependencies: pandas.
"""
from __future__ import annotations

import pandas as pd


class TaxOptimizer:
    """German tax-aware portfolio optimization."""

    def __init__(self, portfolio_df: pd.DataFrame, tax_free_allowance: float = 1000.0):
        self.portfolio = portfolio_df
        self.allowance = tax_free_allowance
        self.tax_rate = 0.26375

    def compute_tax_position(self) -> dict:
        """Current year tax position from unrealized + realized P&L."""
        pnl = self.portfolio.get("PnL_EUR", pd.Series(dtype=float))
        gains = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
        losses = float(abs(pnl[pnl < 0].sum())) if not pnl.empty else 0.0

        realized_gains = self._get_realized_gains_ytd()
        realized_losses = self._get_realized_losses_ytd()

        net_gains = max(0.0, realized_gains - realized_losses - self.allowance)
        estimated_tax = net_gains * self.tax_rate

        return {
            "unrealized_gains": round(gains, 2),
            "unrealized_losses": round(losses, 2),
            "realized_gains_ytd": round(realized_gains, 2),
            "realized_losses_ytd": round(realized_losses, 2),
            "net_taxable": round(net_gains, 2),
            "estimated_tax": round(estimated_tax, 2),
        }

    def harvest_opportunities(self) -> pd.DataFrame:
        """Identify positions to sell for tax benefit.

        Logic: sell losers (non-CORE) to offset realized gains, then repurchase
        a correlated replacement to maintain market exposure. Only propose when
        tax savings clear the 2 EUR round-trip fee by a 3x buffer.
        """
        tax_position = self.compute_tax_position()
        if tax_position["net_taxable"] <= 0:
            return pd.DataFrame()

        candidates = []
        for _, row in self.portfolio.iterrows():
            pnl = row.get("PnL_EUR", 0.0)
            tier = row.get("Tier", "ACTIVE")
            if pnl >= 0 or tier == "CORE":
                continue  # only losers, never harvest core buy-and-hold

            replacement = self._find_correlated_replacement(row["Symbol"])
            tax_savings = abs(float(pnl)) * self.tax_rate
            fee_cost = 2.0
            if tax_savings > fee_cost * 3:  # 3x buffer
                candidates.append({
                    "Sell_Symbol": row["Symbol"],
                    "Sell_Loss": round(float(pnl), 2),
                    "Tax_Savings": round(tax_savings, 2),
                    "Replacement": replacement,
                    "Net_Benefit": round(tax_savings - fee_cost, 2),
                })

        if not candidates:
            return pd.DataFrame()
        return pd.DataFrame(candidates).sort_values("Net_Benefit", ascending=False)

    def _get_realized_gains_ytd(self) -> float:
        """Realized gains YTD. Placeholder — wire to a trade log if available."""
        return 0.0

    def _get_realized_losses_ytd(self) -> float:
        """Realized losses YTD. Placeholder — wire to a trade log if available."""
        return 0.0

    def _find_correlated_replacement(self, symbol: str) -> str:
        """Find a similar ETF to maintain exposure after a tax-loss sale."""
        replacements = {
            "5J50.DE": "ITA",      # US defense ETF alternative
            "EUNL.DE": "IWDA.AS",  # Same underlying, different listing
            "SXRV.DE": "CNDX.L",   # Nasdaq alternative
        }
        return replacements.get(symbol, "CASH")
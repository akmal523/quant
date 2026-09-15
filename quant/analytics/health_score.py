"""
health_score.py — Portfolio Health Score (Part 3, Addition #3).

Intent: collapse many metrics (volatility, Sharpe, drawdown, concentration)
into a single 0-100 health score with component breakdown and recommendations.

Components:
  - Diversification (25%)
  - Risk-Adjusted Return (25%)
  - Drawdown Management (20%)
  - Cost Efficiency (15%)
  - Liquidity (15%)

Invariants:
  - compute returns a dict with total_score in [0, 100] and a grade.
  - Each component score is in [0, 100].
  - Pure computation; no I/O.

Dependencies: pandas.
"""
from __future__ import annotations



class PortfolioHealthScore:
    """Computes a 0-100 health score for the portfolio."""

    WEIGHTS = {
        "diversification": 0.25,
        "risk_adjusted_return": 0.25,
        "drawdown_management": 0.20,
        "cost_efficiency": 0.15,
        "liquidity": 0.15,
    }

    def compute(self, portfolio_data: dict) -> dict:
        """Compute the composite health score."""
        scores = {
            "diversification": self._diversification_score(portfolio_data),
            "risk_adjusted_return": self._risk_adjusted_return_score(portfolio_data),
            "drawdown_management": self._drawdown_management_score(portfolio_data),
            "cost_efficiency": self._cost_efficiency_score(portfolio_data),
            "liquidity": self._liquidity_score(portfolio_data),
        }

        total = sum(scores[k] * self.WEIGHTS[k] for k in scores)
        return {
            "total_score": round(total, 1),
            "components": {k: round(v, 1) for k, v in scores.items()},
            "grade": self._score_to_grade(total),
            "recommendations": self._generate_recommendations(scores),
        }

    def _diversification_score(self, data: dict) -> float:
        n_positions = len(data.get("holdings", []))
        hhi = self._compute_hhi(data.get("weights", {}))
        corr = data.get("correlation_to_spx", 0.8)

        position_score = min(n_positions / 10, 1) * 100
        hhi_score = max(0.0, 100 - hhi * 10000)
        corr_score = max(0.0, 100 - corr * 100)
        return (position_score + hhi_score + corr_score) / 3

    def _risk_adjusted_return_score(self, data: dict) -> float:
        sharpe = data.get("sharpe", 0.0)
        # Map Sharpe to 0-100: 0 -> 50, 1 -> 80, 2 -> 100.
        return float(max(0.0, min(100.0, 50 + sharpe * 30)))

    def _drawdown_management_score(self, data: dict) -> float:
        dd = data.get("max_drawdown", 0.0)
        if dd >= -0.05:
            return 90.0
        if dd >= -0.10:
            return 70.0
        if dd >= -0.20:
            return 50.0
        return 30.0

    def _cost_efficiency_score(self, data: dict) -> float:
        fee_drag = data.get("fee_drag_pct", 0.0)
        # Lower fee drag = higher score. 0% -> 100, 3% -> 40.
        return float(max(0.0, min(100.0, 100 - fee_drag * 20)))

    def _liquidity_score(self, data: dict) -> float:
        liquid_pct = data.get("liquid_pct", 1.0)
        return float(liquid_pct * 100)

    def _compute_hhi(self, weights: dict) -> float:
        """Herfindahl-Hirschman Index (sum of squared weights)."""
        if not weights:
            return 0.0
        return float(sum(w * w for w in weights.values()))

    def _score_to_grade(self, score: float) -> str:
        if score >= 90:
            return "A+"
        if score >= 80:
            return "A"
        if score >= 70:
            return "B"
        if score >= 60:
            return "C"
        if score >= 50:
            return "D"
        return "F"

    def _generate_recommendations(self, scores: dict) -> list[str]:
        recs = []
        if scores["diversification"] < 60:
            recs.append("Reduce concentration (add bonds/gold)")
        if scores["drawdown_management"] < 60:
            recs.append("Implement stop-losses to limit drawdowns")
        if scores["cost_efficiency"] < 60:
            recs.append("Reduce trade frequency to cut fee drag")
        if scores["liquidity"] < 60:
            recs.append("Add more liquid positions")
        return recs
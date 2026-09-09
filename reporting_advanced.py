"""
reporting_advanced.py — Unified Portfolio Manager Briefing (Part 2, Upgrade #10).

Intent: assemble the outputs of all Part 2 modules (risk monitor, portfolio
context, strategy engine, cash manager, tax optimizer, attribution, guardrails)
into a single coherent daily briefing. Replaces fragmented text output with a
structured, sectioned report.

Invariants:
  - build_briefing returns a multi-line string.
  - Pure formatting; no I/O.

Dependencies: all Part 2 modules.
"""
from __future__ import annotations


def _fmt_pct(x: float) -> str:
    return f"{x:+.1%}" if x is not None else "N/A"


def build_briefing(
    date_str: str,
    portfolio_value: float,
    pnl_eur: float,
    pnl_pct: float,
    cash_eur: float,
    cash_pct: float,
    risk_status: dict,
    risk_contrib: dict,
    concentration_penalties: dict,
    regime: str,
    strategy_weights: dict,
    ensemble_scores: dict,
    cash_target: float,
    dip_alerts: list,
    tax_position: dict,
    harvest_opportunities,
    attribution_df,
    guardrail_blocks: list,
) -> str:
    """Assemble the unified daily briefing string."""
    w = 100
    lines = []
    lines.append("=" * w)
    lines.append(f" PORTFOLIO MANAGER DAILY BRIEFING — {date_str}")
    lines.append("=" * w)

    # ── Portfolio Snapshot ──────────────────────────────────────────────────
    lines.append("\n[PORTFOLIO SNAPSHOT]")
    lines.append(f"  Value: EUR {portfolio_value:,.2f}  |  P&L: {pnl_eur:+,.2f} "
                 f"({pnl_pct:+.1%})  |  Cash: EUR {cash_eur:,.2f} ({cash_pct:.1%})")
    lines.append(f"  Risk: {risk_status.get('status', 'N/A')}  |  Regime: {regime}  |  "
                 f"VaR(95%): {_fmt_pct(risk_status.get('var_95', 0))}")

    # ── Risk & Diversification ─────────────────────────────────────────────
    lines.append("\n[RISK & DIVERSIFICATION]")
    if risk_contrib:
        top = sorted(risk_contrib.items(), key=lambda kv: -kv[1])[:4]
        rc_str = " | ".join(f"{s} {v:.0f}%" for s, v in top)
        lines.append(f"  Risk Contribution: {rc_str}")
    if concentration_penalties:
        pen_str = " | ".join(f"{s} x{p:.2f}" for s, p in concentration_penalties.items())
        lines.append(f"  Concentration Penalties: {pen_str}")

    # ── Strategy Performance ───────────────────────────────────────────────
    lines.append("\n[STRATEGY ENSEMBLE]")
    if strategy_weights:
        w_str = " | ".join(f"{k} {v:.0%}" for k, v in strategy_weights.items())
        lines.append(f"  Regime weights: {w_str}")
    if ensemble_scores:
        for sym, score in sorted(ensemble_scores.items(), key=lambda kv: -kv[1])[:5]:
            lines.append(f"  {sym}: ensemble {score:+.2f}")

    # ── Cash Strategy ──────────────────────────────────────────────────────
    lines.append("\n[CASH STRATEGY]")
    lines.append(f"  Target cash: {cash_target:.1%}")
    if dip_alerts:
        for sym, amt in dip_alerts:
            lines.append(f"  DIP BUY: {sym} -> EUR {amt:,.2f}")

    # ── Tax Optimization ───────────────────────────────────────────────────
    lines.append("\n[TAX OPTIMIZATION]")
    if tax_position:
        lines.append(f"  YTD Realized: +EUR {tax_position.get('realized_gains_ytd', 0):,.2f} "
                     f"| Tax Owed: EUR {tax_position.get('estimated_tax', 0):,.2f}")
    if harvest_opportunities is not None and not harvest_opportunities.empty:
        for _, r in harvest_opportunities.iterrows():
            lines.append(f"  HARVEST: Sell {r['Sell_Symbol']} (loss {r['Sell_Loss']:+.2f}) "
                         f"-> {r['Replacement']}, net {r['Net_Benefit']:+.2f}")

    # ── Attribution ────────────────────────────────────────────────────────
    lines.append("\n[P&L ATTRIBUTION]")
    if attribution_df is not None and not attribution_df.empty:
        for _, r in attribution_df.head(5).iterrows():
            lines.append(f"  {r['Sector']:<16} alloc {r['Allocation_Effect']:+.2f}%  "
                         f"sel {r['Selection_Effect']:+.2f}%  "
                         f"int {r['Interaction']:+.2f}%  total {r['Total_Effect']:+.2f}%")

    # ── Guardrails ─────────────────────────────────────────────────────────
    if guardrail_blocks:
        lines.append("\n[GUARDRAIL BLOCKS]")
        for b in guardrail_blocks:
            lines.append(f"  BLOCKED: {b}")

    lines.append("\n" + "=" * w)
    return "\n".join(lines)
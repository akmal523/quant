"""
briefing.py — Briefing document builder (v10.5.0, spec 3.4).

Intent: assemble briefing.md from the same single-sourced facts as the CLI:
Actions, Portfolio, Holdings, Risk, Evidence summary, Data health (blockers
only), Methodology. No banner blocks, no decorative separators, no metric
without a source line.

Invariants:
  - Every number carries label, unit, source, as-of (R3).
  - Data health lists blockers only; non-blocking issues stay in the run log.
  - Pure function: returns a markdown string; no I/O.

Dependencies: quant.reporting.actions.
"""
from __future__ import annotations

import pandas as pd

from quant.reporting.actions import build_actions


def build_briefing_md(
    *,
    as_of: str,
    version: str,
    regime_label: str,
    regime_prob: float,
    regime_source: str,
    audit_df: pd.DataFrame | None,
    account,
    total_value: float,
    pnl_eur: float,
    pnl_pct: float,
    with_news: int,
    without_news: int,
    latest_bar: str,
) -> str:
    """Return the briefing markdown (spec 3.4)."""
    actions = build_actions(audit_df)
    lines: list[str] = []

    lines.append(f"# Quant-AI Briefing {version}")
    lines.append("")
    lines.append(f"As-of {as_of}. Latest bar {latest_bar}.")
    lines.append("")

    # ── Actions ──────────────────────────────────────────────────────────────
    lines.append("## Actions")
    lines.append("")
    if actions:
        lines.append("| Symbol | Action | Amount EUR | Reason |")
        lines.append("|---|---|---:|---|")
        for a in actions:
            if a["blocked"]:
                lines.append(f"| {a['symbol']} | BLOCKED | | {a['reason']} |")
            else:
                sign = "+" if a["action"] == "BUY MORE" else "-"
                lines.append(
                    f"| {a['symbol']} | {a['action']} | {sign}{a['amount_eur']:.0f} | "
                    f"{a['reason']} |"
                )
    else:
        lines.append("No actions required today.")
    lines.append("")

    # ── Portfolio ────────────────────────────────────────────────────────────
    lines.append("## Portfolio")
    lines.append("")
    cash = f"{account.cash_eur:.2f} EUR (cash, manual input)" if account.cash_is_set \
        else "Cash not set. Set it in Portfolio to enable cash-aware recommendations."
    lines.append(f"- Value: {total_value:.2f} EUR (source: portfolio.csv)")
    lines.append(f"- Broker PnL: {pnl_eur:+.2f} EUR ({pnl_pct:+.2f}%) (source: broker)")
    lines.append(f"- Cash: {cash}")
    lines.append(f"- Risk profile: {account.risk_profile} (source: account.yaml)")
    lines.append(
        f"- Regime: {regime_label}, p={regime_prob:.2f} "
        f"(fit {regime_source}, as-of {as_of})"
    )
    lines.append("")

    # ── Holdings ─────────────────────────────────────────────────────────────
    lines.append("## Holdings")
    lines.append("")
    if audit_df is not None and not audit_df.empty:
        lines.append("| Symbol | Tier | Weight | Target | Drift | Status |")
        lines.append("|---|---|---:|---:|---:|---|")
        for _, r in audit_df.iterrows():
            lines.append(
                f"| {r.get('Symbol', '')} | {r.get('Tier', '')} | "
                f"{r.get('Current_Weight', '')} | {r.get('Target_Weight', '')} | "
                f"{r.get('Drift', '')} | {r.get('Signal', '')} |"
            )
    else:
        lines.append("No holdings.")
    lines.append("")

    # ── Risk ─────────────────────────────────────────────────────────────────
    lines.append("## Risk")
    lines.append("")
    limits = account.risk_limits()
    lines.append(
        f"- Profile limits: safety >= {limits[0]:.0%}, core >= {limits[1]:.0%}, "
        f"alpha <= {limits[2]:.0%}, max position {limits[3]:.0%}, "
        f"cash floor {limits[4]:.0%} (source: quant/config.py)"
    )
    lines.append("")

    # ── Evidence summary ─────────────────────────────────────────────────────
    lines.append("## Evidence summary")
    lines.append("")
    lines.append(
        f"{with_news} symbols scored with news evidence; {without_news} without "
        f"(sentiment neutral, confidence low)."
    )
    lines.append("")

    # ── Data health (blockers only) ──────────────────────────────────────────
    lines.append("## Data health")
    lines.append("")
    blockers = [a for a in actions if a["blocked"]]
    if blockers:
        for a in blockers:
            lines.append(f"- {a['remedy']}")
    else:
        lines.append("No blockers.")
    lines.append("")

    # ── Methodology ──────────────────────────────────────────────────────────
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Scores combine structural quality, tactical timing, and news sentiment. "
        "Portfolio PnL is copied from the broker, never price-guessed. "
        "Actions cite the drift threshold from quant/config.py that triggered them."
    )
    lines.append("")
    return "\n".join(lines)

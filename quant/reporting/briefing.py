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

from quant.engine.advice import build_advice
from quant.reporting.actions import _holdings_from_audit, _tier, build_actions
from quant.ui import copy as ui_copy


def _display_name(symbol: str) -> str:
    """Company name for a symbol (v10.7.3, Part 2.2). Never raises."""
    try:
        from quant.data.names import display_name

        return display_name(symbol)
    except Exception:  # noqa: BLE001
        return str(symbol)

_ACTION_WORD = {
    "sell_part": ui_copy.ADVICE_SELL_PART,
    "buy": ui_copy.ADVICE_BUY,
    "change_savings_plan": ui_copy.ADVICE_TOP_UP,
    "to_cash": ui_copy.ADVICE_TO_CASH,
}
_VERDICT = {
    "sell_part": ui_copy.ADVICE_SELL_PART,
    "buy": ui_copy.ADVICE_BUY,
    "change_savings_plan": ui_copy.ADVICE_TOP_UP,
    "keep": ui_copy.ADVICE_KEEP,
    "to_cash": ui_copy.ADVICE_TO_CASH,
}


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
    alerts: list[dict] | None = None,
    steps: list[dict] | None = None,
    money: dict | None = None,
) -> str:
    """Return the briefing markdown (spec 3.4)."""
    holdings = (_holdings_from_audit(audit_df)
                if (audit_df is not None and not audit_df.empty) else [])
    advice, _rejected = build_advice(holdings)
    lines: list[str] = []

    lines.append(f"# Quant-AI Briefing {version}")
    lines.append("")
    lines.append(f"As-of {as_of}. Latest bar {latest_bar}.")
    lines.append("")

    # ── Alerts (v10.7.0: the briefing starts with Alerts) ────────────────────
    lines.append("## Alerts")
    lines.append("")
    if alerts:
        for alert in alerts:
            lines.append(f"- {alert.get('message', '')}")
    else:
        lines.append("No open actions.")
    lines.append("")

    # ── Your steps this week (v10.7.0, Section 10.7) ─────────────────────────
    lines.append("## Your steps this week")
    lines.append("")
    if steps:
        for i, step in enumerate(steps, 1):
            amount = step.get("amount_eur")
            suffix = f" ({amount:.0f} EUR)" if amount else ""
            lines.append(f"{i}. {step.get('what', '')}{suffix}")
    else:
        lines.append("Nothing to do this week.")
    lines.append("")

    # ── Your money (v10.7.0, Section 10.7) ───────────────────────────────────
    if money:
        lines.append("## Your money")
        lines.append("")
        lines.append(f"- Invested: {money.get('invested_eur', 0):.2f} EUR")
        lines.append(f"- Operational cash: {money.get('cash_eur', 0):.2f} EUR")
        lines.append("")

    # ── Actions (v10.7.1: from the ONE advice pipeline) ──────────────────────
    lines.append("## Actions")
    lines.append("")
    actionable = [a for a in advice
                  if a["kind"] in ("sell_part", "buy", "change_savings_plan", "to_cash")]
    if actionable:
        lines.append("| Name | Action | Amount EUR | Reason |")
        lines.append("|---|---|---:|---|")
        for a in actionable:
            amount = f"{a['eur']:.0f}" if a.get("eur") else ""
            lines.append(
                f"| {a['company_name']} | {_ACTION_WORD.get(a['kind'], a['kind'])} | "
                f"{amount} | {a['why']} |"
            )
    else:
        lines.append("No actions required today.")
    lines.append("")

    # ── Portfolio ────────────────────────────────────────────────────────────
    lines.append("## Portfolio")
    lines.append("")
    cash = f"{account.cash_eur:.2f} EUR (cash, manual input)" if account.cash_is_set \
        else "Cash not set. Set it in Portfolio to enable cash-aware recommendations."
    lines.append(f"- Value: {total_value:.2f} EUR")
    lines.append(f"- Total profit or loss: {pnl_eur:+.2f} EUR ({pnl_pct:+.2f}%)")
    lines.append(f"- Cash: {cash}")
    lines.append(f"- Risk profile: {account.risk_profile}")
    lines.append(f"- Market regime: {regime_label}, confidence {regime_prob:.2f}")
    lines.append("")

    # ── Holdings (v10.7.1: dictionary tier words + plain verdicts) ───────────
    lines.append("## Holdings")
    lines.append("")
    if audit_df is not None and not audit_df.empty:
        verdict_by_symbol = {a["symbol"]: a["kind"] for a in advice if a.get("symbol")}
        lines.append("| Name | Tier | Weight | Target | Drift | Verdict |")
        lines.append("|---|---|---:|---:|---:|---|")
        for _, r in audit_df.iterrows():
            symbol = str(r.get("Symbol", ""))
            kind = verdict_by_symbol.get(symbol, "keep")
            lines.append(
                f"| {_display_name(symbol)} | {ui_copy.tier_word(_tier(r.get('Tier')))} | "
                f"{r.get('Current_Weight', '')} | {r.get('Target_Weight', '')} | "
                f"{r.get('Drift', '')} | {_VERDICT.get(kind, ui_copy.ADVICE_KEEP)} |"
            )
    else:
        lines.append("No holdings.")
    lines.append("")

    # ── Risk ─────────────────────────────────────────────────────────────────
    lines.append("## Risk")
    lines.append("")
    limits = account.risk_limits()
    lines.append(
        f"- Profile limits (of invested): long-term >= {limits[0]:.0%}, "
        f"active <= {limits[1]:.0%}, max single position {limits[2]:.0%} "
        f"(source: quant/config.py)"
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
    blockers = [a for a in build_actions(audit_df) if a["blocked"]]
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
    # v10.7.2 (Part 2.3): the honest news-pillar line when it is absent.
    try:
        from quant.engine import news_pillar

        if news_pillar.is_absent():
            lines.append(ui_copy.NEWS_PILLAR_ABSENT)
            lines.append("")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)

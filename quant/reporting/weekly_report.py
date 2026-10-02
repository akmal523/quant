"""
weekly_report.py — Weekly Friday Report (Markdown + self-contained HTML).

Intent (v10.6.2, R-PDF-1): generate the weekly Alpha report as Markdown (for
quick reading, git-friendly) and a self-contained HTML file with print CSS. The
user prints to PDF from the browser; no PDF library is added.

Invariants:
  - ``build_weekly_report`` returns (markdown, html); never raises.
  - ``save_weekly_report`` writes both files under ``outputs/reports/``.
  - Every number carries its unit; no emoji, no decorative separators.

Dependencies: pandas, markdown, jinja2, quant.paths, quant.portfolio.*.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from quant import __version__, paths

_TEMPLATE = Path(__file__).resolve().parent / "templates" / "weekly_report.html"
_DATA_SOURCES = "yfinance, SEC EDGAR, FinBERT"


def _display_name(symbol) -> str:
    """Company name for a symbol (v10.7.3, Part 2.2). Never raises."""
    try:
        from quant.data.names import display_name

        return display_name(str(symbol))
    except Exception:  # noqa: BLE001
        return str(symbol)


def _fmt_eur(value) -> str:
    try:
        return f"{float(value):,.2f} EUR"
    except (TypeError, ValueError):
        return "n/a"


def _build_empty_portfolio_report(as_of: str) -> str:
    """Report for an empty portfolio with first-step guidance (v10.6.3)."""
    lines = [
        "# Quant-AI Weekly Report",
        "",
        f"**As-of:** {as_of} | **Version:** {__version__}",
        "",
        "## No assets in the portfolio",
        "",
        "Your portfolio is currently empty. To get started:",
        "",
        "1. Add assets to data/portfolio.csv (broker-synced).",
        "2. Assign tiers in data/tiers.csv (or run the migration script).",
        "3. Run the review to generate signals.",
        "",
        "### Recommended first steps",
        "",
        "- FORTRESS: start with broad ETFs (URTH, SPY) via a savings plan.",
        "- ALPHA: add 3-5 tactical stocks you want to actively trade.",
        "- SPECULATIVE: at most 2 percent in high-risk bets (optional).",
        "",
        "See README.md for detailed setup instructions.",
    ]
    return "\n".join(lines)


def _tier_rows(portfolio_df: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Return the rows of portfolio_df in the given tier (empty-safe)."""
    if portfolio_df is None or portfolio_df.empty or "Tier" not in portfolio_df.columns:
        return pd.DataFrame()
    return portfolio_df[portfolio_df["Tier"] == tier]


def _tier_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Render a Markdown table for the given columns. Empty -> a note line."""
    if df is None or df.empty:
        return ["*No assets in this tier.*", ""]
    # v10.7.3 (Part 2.2): show the company name next to the symbol.
    df = df.copy()
    if "Symbol" in df.columns and "Name" not in df.columns:
        df["Name"] = df["Symbol"].map(_display_name)
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in columns:
            v = r.get(c)
            if c in ("Current_Value_EUR", "Value_EUR"):
                cells.append(_fmt_eur(v))
            elif isinstance(v, float):
                cells.append(f"{v:.1f}")
            else:
                cells.append(str(v) if v is not None else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def build_weekly_report(
    as_of: str,
    portfolio_df: pd.DataFrame | None = None,
    audit_df: pd.DataFrame | None = None,
    emergency_amount: float | None = None,
) -> tuple[str, str]:
    """Build the weekly report as (markdown, html).

    Intent: one report covering the three tiers, the emergency sell order, and
    this week's signals. Invariants: returns two strings; never raises; missing
    data renders an honest note, not an error.
    """
    if portfolio_df is None:
        try:
            from quant.portfolio.portfolio import load_portfolio_with_tiers
            portfolio_df = load_portfolio_with_tiers()
        except Exception:  # noqa: BLE001
            portfolio_df = pd.DataFrame()

    if audit_df is None:
        audit_df = pd.DataFrame()

    if portfolio_df is None or portfolio_df.empty:
        markdown_text = _build_empty_portfolio_report(as_of)
        return markdown_text, _markdown_to_html(markdown_text, as_of)

    md: list[str] = []
    md.append("# Quant-AI Weekly Report")
    md.append("")
    md.append(f"**As-of:** {as_of} | **Version:** {__version__}")
    md.append("")
    # v10.7.5 (Part 3.4): say so when the optimizer fell back to equal weight.
    try:
        from quant.portfolio.optimizer import optimizer_fallback_line

        _fallback = optimizer_fallback_line()
        if _fallback:
            md.append(f"**{_fallback}**")
            md.append("")
    except Exception:  # noqa: BLE001
        pass

    # Fortress
    md.append("## Fortress (eternal holdings)")
    md.append("")
    md.append("Never sold, to avoid capital gains tax. The monthly decision "
              "adjusts savings-plan amounts only.")
    md.append("")
    fortress = _tier_rows(portfolio_df, "FORTRESS")
    md += _tier_table(fortress, ["Name", "Symbol", "Current_Value_EUR", "Tier"])

    # Alpha
    md.append("## Alpha (active accumulation)")
    md.append("")
    md.append("Weekly rebalancing on Fridays. Sell when cash is needed.")
    md.append("")
    alpha = _tier_rows(portfolio_df, "ALPHA")
    md += _tier_table(alpha, ["Name", "Symbol", "Current_Value_EUR", "Tier"])

    # Speculative
    md.append("## Speculative (high-risk bets)")
    md.append("")
    md.append("Hard cap 2 percent of the portfolio. Stop-loss -50 percent, "
              "take-profit +100 percent.")
    md.append("")
    spec = _tier_rows(portfolio_df, "SPECULATIVE")
    md += _tier_table(spec, ["Name", "Symbol", "Current_Value_EUR", "Tier"])

    # If you need cash now (v10.7.0 dictionary)
    md.append("## If you need cash now")
    md.append("")
    if emergency_amount and not portfolio_df.empty:
        from quant.portfolio.risk import emergency_sell_plan
        plan = emergency_sell_plan(float(emergency_amount), portfolio_df)
        if plan["fortress_warning"]:
            md.append(f"Warning: {plan['fortress_warning']}")
            md.append("")
        if plan["recommendations"]:
            md.append(f"If you need {_fmt_eur(emergency_amount)}, sell in this order:")
            md.append("")
            for i, h in enumerate(plan["recommendations"], 1):
                pnl = float(h.get("pnl_eur", 0) or 0)
                tax = pnl * 0.26375
                if pnl < 0:
                    note = "loss, tax-loss harvest"
                elif pnl > 0:
                    note = "profit, taxable"
                else:
                    note = "no gain or loss"
                md.append(
                    f"{i}. {_display_name(h['symbol'])} ({_fmt_eur(h['value_eur'])}) - "
                    f"tax {_fmt_eur(tax)} ({note})"
                )
        else:
            md.append("*No holdings available for an emergency sale.*")
    else:
        md.append("Provide an amount to see the recommended sell order "
                  "(liquidity first, tax-loss harvest second).")
    md.append("")

    # Signals
    md.append("## This week's signals")
    md.append("")
    if audit_df is not None and not audit_df.empty and "Signal" in audit_df.columns:
        buys = audit_df[audit_df["Signal"].isin(["BUY", "BUY_SPECULATIVE", "INCREASE_SPARPLAN"])]
        if buys.empty:
            md.append("*No buy signals this week.*")
        else:
            for _, r in buys.iterrows():
                md.append(f"- {_display_name(r.get('Symbol'))}: {r.get('Signal')} "
                          f"({r.get('Tier')})")
    else:
        md.append("*No signals available.*")
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"*Data sources: {_DATA_SOURCES}*")

    markdown_text = "\n".join(md)
    html_text = _markdown_to_html(markdown_text, as_of)
    return markdown_text, html_text


def _markdown_to_html(markdown_text: str, as_of: str) -> str:
    """Convert the report Markdown to self-contained HTML with print CSS."""
    try:
        import markdown as _md
        from jinja2 import Template
        body = _md.markdown(markdown_text, extensions=["tables", "fenced_code"])
        template = Template(_TEMPLATE.read_text(encoding="utf-8"))
        return template.render(
            as_of=as_of,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            version=__version__,
            content=body,
            data_sources=_DATA_SOURCES,
        )
    except Exception:  # noqa: BLE001
        # Fallback: a minimal HTML wrapper so the report is never lost.
        return (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>Quant-AI Weekly Report - {as_of}</title></head>"
            f"<body><pre>{markdown_text}</pre></body></html>"
        )


def save_weekly_report(
    as_of: str,
    portfolio_df: pd.DataFrame | None = None,
    audit_df: pd.DataFrame | None = None,
    emergency_amount: float | None = None,
) -> tuple[Path, Path]:
    """Generate and save the weekly report. Returns (markdown_path, html_path).

    Invariants: writes under ``outputs/reports/``; creates the dir if missing;
    never raises on a valid report.
    """
    md_text, html_text = build_weekly_report(
        as_of, portfolio_df=portfolio_df, audit_df=audit_df,
        emergency_amount=emergency_amount,
    )
    reports_dir = Path(paths.OUTPUTS_DIR) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    date_str = str(as_of).replace("-", "")
    md_path = reports_dir / f"weekly_{date_str}.md"
    html_path = reports_dir / f"weekly_{date_str}.html"
    md_path.write_text(md_text, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    return md_path, html_path

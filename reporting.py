"""
reporting.py — Terminal output, conditional-colour Excel workbook, CSV export.
"""
from __future__ import annotations
import os
import pandas as pd


# ── Colour maps for Excel conditional formatting ──────────────────────────────
_HORIZON_FILL = {
    "RETIREMENT (20yr+)":         "C6EFCE",   # green
    "BUSINESS COLLATERAL (10yr)": "FFEB9C",   # amber
    "SPECULATIVE (short-term)":   "FFC7CE",   # red tint
}
_SIGNAL_FILL = {
    "BUY":  "C6EFCE",
    "HOLD": "FFEB9C",
    "SELL": "FFC7CE",
}
_AUDIT_FILL = {
    "URGENT SELL":       "FFC7CE",
    "BUY MORE (DCA OK)": "C6EFCE",
    "HOLD":              "FFEB9C",
    "NOT SCANNED":       "D9D9D9",
}


# ─── Terminal ─────────────────────────────────────────────────────────────────

def print_terminal_report(scan_df: pd.DataFrame, audit_df: pd.DataFrame | None = None) -> None:
    w = 165
    print("\n" + "=" * w)
    print("  QUANT-AI SECTOR SCANNER  v5  —  FinBERT Edition")
    print("=" * w)

    # Global Top-3
    print("\n  [ GLOBAL TOP 3 ]")
    _hdr()
    for _, r in scan_df.nlargest(3, "Total_Score").iterrows():
        _row(r)

    # Per-sector Top-3
    print(f"\n  [ SECTOR LEADERS  —  Top {3} per Sector ]")
    for sector, grp in scan_df.groupby("Sector"):
        if sector == "Unknown":
            continue
        print(f"\n  ─── {sector.upper()} " + "─" * max(0, w - 8 - len(sector)))
        _hdr()
        for _, r in grp.nlargest(3, "Total_Score").iterrows():
            _row(r)

    # Portfolio audit
    if audit_df is not None and not audit_df.empty:
        from portfolio import print_audit_report
        print_audit_report(audit_df)


def _hdr() -> None:
    print(
        f"  {'Symbol':<8}  {'Name':<28}  {'Score':>5}  {'RSI':>5}  "
        f"{'FB':>5}  {'Signal':<6}  {'Horizon':<28}  Reasoning"
    )
    print("  " + "-" * 130)


def _row(r: pd.Series) -> None:
    rsi = r.get("RSI")
    fb  = r.get("FinBERT_Score")
    print(
        f"  {str(r.get('Symbol',''))[:8]:<8}  "
        f"{str(r.get('Name',''))[:28]:<28}  "
        f"{int(r.get('Total_Score', 0)):>5}  "
        f"{rsi if rsi is None else f'{rsi:.0f}':>5}  "
        f"{fb  if fb  is None else f'{fb:+.0f}':>5}  "
        f"{str(r.get('Signal','')):<6}  "
        f"{str(r.get('Horizon',''))[:28]:<28}"
        f"{str(r.get('Reasoning',''))}"
    )


# ─── Excel Export ─────────────────────────────────────────────────────────────

def export_excel(
    scan_df: pd.DataFrame,
    audit_df: pd.DataFrame | None,
    out_dir: str = "outputs",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "market_scan.xlsx")

    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("[Report] openpyxl not installed — skipping Excel export.")
        return

    header_fill = PatternFill("solid", fgColor="1F3864")
    header_font = Font(bold=True, color="FFFFFF")

    def _style_sheet(ws: "openpyxl.worksheet.worksheet.Worksheet",
                     col_fill_map: dict[str, dict[str, str]]) -> None:
        """Apply header styling, frozen row, conditional row fill, auto-width."""
        # Header
        for cell in ws[1]:
            cell.fill      = header_fill
            cell.font      = header_font
            cell.alignment = Alignment(horizontal="center")
        ws.freeze_panes = "A2"

        # Identify target columns by header name
        col_indices: dict[str, int] = {}
        for i, cell in enumerate(ws[1], 1):
            if cell.value in col_fill_map:
                col_indices[cell.value] = i

        # Row colours
        for row in ws.iter_rows(min_row=2):
            color: str | None = None
            for col_name, col_idx in col_indices.items():
                val = row[col_idx - 1].value
                if val in col_fill_map[col_name]:
                    color = col_fill_map[col_name][val]
                    break
            if color:
                fill = PatternFill("solid", fgColor=color)
                for cell in row:
                    cell.fill = fill

        # Auto-width
        for col in ws.columns:
            max_len = max((len(str(c.value or "")) for c in col), default=8)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 2, 45)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        scan_df.to_excel(writer, sheet_name="Market Scan", index=False)
        _style_sheet(
            writer.sheets["Market Scan"],
            {"Horizon": _HORIZON_FILL, "Signal": _SIGNAL_FILL},
        )

        if audit_df is not None and not audit_df.empty:
            audit_df.to_excel(writer, sheet_name="Portfolio Audit", index=False)
            _style_sheet(
                writer.sheets["Portfolio Audit"],
                {"Audit_Decision": _AUDIT_FILL},
            )

    print(f"  [Report] Excel → {path}")


# ─── CSV Export ───────────────────────────────────────────────────────────────

def export_csv(
    scan_df: pd.DataFrame,
    audit_df: pd.DataFrame | None,
    out_dir: str = "outputs",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    scan_path = os.path.join(out_dir, "market_scan.csv")
    scan_df.to_csv(scan_path, index=False)
    print(f"  [Report] CSV    → {scan_path}")

    if audit_df is not None and not audit_df.empty:
        audit_path = os.path.join(out_dir, "portfolio_audit.csv")
        audit_df.to_csv(audit_path, index=False)
        print(f"  [Report] Audit  → {audit_path}")

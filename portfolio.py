"""
portfolio.py — Portfolio audit engine.

Reads portfolio.csv (Symbol, Buy_Price, Amount_EUR), cross-references
the live scanner results, and issues a structured decision per holding.

Decision hierarchy:
  URGENT SELL       — Signal == SELL  OR FinBERT_Score < −60  OR RSI > 75
  BUY MORE (DCA OK) — Signal == BUY   AND unrealised PnL < 20% (room to average)
  HOLD              — all other cases
  NOT SCANNED       — symbol not found in the scanned universe
"""
from __future__ import annotations
import os
import pandas as pd


def load_portfolio(filepath: str = "portfolio.csv") -> pd.DataFrame:
    """
    Load portfolio CSV.
    Required columns: Symbol, Buy_Price
    Optional column:  Amount_EUR
    Returns empty DataFrame (with correct columns) if file is missing.
    """
    required_cols = ["Symbol", "Buy_Price", "Amount_EUR"]
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=required_cols)

    try:
        df = pd.read_csv(filepath).dropna(how="all")
        df.columns = df.columns.str.strip()

        if "Amount_EUR" not in df.columns:
            df["Amount_EUR"] = 0.0

        df["Symbol"]     = df["Symbol"].astype(str).str.strip()
        df["Buy_Price"]  = pd.to_numeric(df["Buy_Price"],  errors="coerce")
        df["Amount_EUR"] = pd.to_numeric(df["Amount_EUR"], errors="coerce").fillna(0.0)

        return df.dropna(subset=["Symbol", "Buy_Price"])

    except Exception as exc:
        print(f"[Portfolio] Load error: {exc}")
        return pd.DataFrame(columns=required_cols)


def audit(portfolio_df: pd.DataFrame, scan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-reference portfolio holdings against scanner output.

    Parameters
    ----------
    portfolio_df : DataFrame from load_portfolio()
    scan_df      : main scanner results DataFrame

    Returns
    -------
    audit DataFrame with one row per holding and an Audit_Decision column.
    """
    if portfolio_df.empty or scan_df.empty:
        return pd.DataFrame()

    scan_idx = scan_df.set_index("Symbol")
    rows: list[dict] = []

    for _, prow in portfolio_df.iterrows():
        sym = str(prow["Symbol"])

        if sym not in scan_idx.index:
            rows.append({
                "Symbol":            sym,
                "Buy_Price":         prow["Buy_Price"],
                "Amount_EUR":        prow.get("Amount_EUR", 0.0),
                "Current_Price_EUR": None,
                "PnL_pct":           None,
                "Signal":            "N/A",
                "FinBERT_Score":     0.0,
                "RSI":               None,
                "Total_Score":       None,
                "Horizon":           "N/A",
                "Audit_Decision":    "NOT SCANNED",
                "Reasoning":         "Symbol not in scanned universe",
            })
            continue

        s             = scan_idx.loc[sym]
        buy_price     = float(prow["Buy_Price"])
        # Prefer EUR-normalised close; fall back to raw close
        current_price = float(s.get("Close_EUR") or s.get("Close") or buy_price)
        pnl_pct       = (current_price - buy_price) / buy_price * 100 if buy_price else 0.0

        signal        = str(s.get("Signal", "HOLD"))
        fb_score      = float(s.get("FinBERT_Score") or 0.0)
        rsi_val       = s.get("RSI")
        rsi           = float(rsi_val) if rsi_val is not None else 50.0
        horizon       = str(s.get("Horizon", ""))
        total_score   = s.get("Total_Score", 0)
        risk_drivers  = str(s.get("Risk_Drivers", ""))

        # ── Decision ─────────────────────────────────────────────────────────
        if signal == "SELL" or fb_score < -60 or rsi > 75:
            decision  = "URGENT SELL"
            reasoning = f"Signal={signal} | FinBERT={fb_score:+.0f} | RSI={rsi:.0f}"
            if risk_drivers:
                reasoning += f" | {risk_drivers[:80]}"

        elif signal == "BUY" and pnl_pct <= 20.0:
            decision  = "BUY MORE (DCA OK)"
            reasoning = f"Score={total_score} | PnL={pnl_pct:+.1f}% | {horizon}"

        else:
            decision  = "HOLD"
            reasoning = f"Score={total_score} | PnL={pnl_pct:+.1f}% | {horizon}"

        rows.append({
            "Symbol":            sym,
            "Buy_Price":         round(buy_price, 4),
            "Amount_EUR":        round(float(prow.get("Amount_EUR", 0.0)), 2),
            "Current_Price_EUR": round(current_price, 4),
            "PnL_pct":           round(pnl_pct, 2),
            "Signal":            signal,
            "FinBERT_Score":     round(fb_score, 1),
            "RSI":               round(rsi, 1),
            "Total_Score":       total_score,
            "Horizon":           horizon,
            "Audit_Decision":    decision,
            "Reasoning":         reasoning,
        })

    return pd.DataFrame(rows)


def print_audit_report(audit_df: pd.DataFrame) -> None:
    if audit_df.empty:
        print("[Portfolio Audit] No holdings to audit.")
        return

    w = 150
    print("\n" + "=" * w)
    print("  PORTFOLIO AUDIT — DECISION ENGINE")
    print("=" * w)

    # URGENT SELL first
    urgent = audit_df[audit_df["Audit_Decision"] == "URGENT SELL"]
    if not urgent.empty:
        print("\n  !! URGENT SELL ALERTS !!")
        for _, r in urgent.iterrows():
            pnl = r["PnL_pct"]
            pnl_str = f"{pnl:+.1f}%" if pnl is not None else "N/A"
            print(f"  {r['Symbol']:<8} | PnL: {pnl_str:>7} | RSI: {r['RSI']:.0f} | {r['Reasoning']}")

    for decision in ["BUY MORE (DCA OK)", "HOLD", "NOT SCANNED"]:
        subset = audit_df[audit_df["Audit_Decision"] == decision]
        if subset.empty:
            continue
        print(f"\n  [ {decision} ]")
        for _, r in subset.iterrows():
            pnl = r["PnL_pct"]
            pnl_str = f"{pnl:+.1f}%" if pnl is not None else "N/A"
            print(f"  {r['Symbol']:<8} | PnL: {pnl_str:>7} | {r['Reasoning']}")

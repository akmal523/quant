"""
portfolio.py — Portfolio audit engine v6.
Integrates Stewardship and Relative-Value triggers for trade decisions.
"""
from __future__ import annotations
import os
import pandas as pd

def load_portfolio(filepath: str = "portfolio.csv") -> pd.DataFrame:
    """
    Load portfolio CSV. Required columns: Symbol, Buy_Price, Amount_EUR
    """
    required_cols = ["Symbol", "Buy_Price", "Amount_EUR"]
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=required_cols)

    try:
        df = pd.read_csv(filepath).dropna(how="all")
        df.columns = df.columns.str.strip()
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 if col != "Symbol" else "UNKNOWN"
        
        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        df["Buy_Price"] = pd.to_numeric(df["Buy_Price"], errors="coerce")
        df["Amount_EUR"] = pd.to_numeric(df["Amount_EUR"], errors="coerce")
        return df.dropna(subset=["Symbol"])
    except Exception:
        return pd.DataFrame(columns=required_cols)

def audit_portfolio(portfolio_df: pd.DataFrame, scan_df: pd.DataFrame) -> pd.DataFrame:
    """
    Decision hierarchy:
    1. URGENT SELL: Signal is SELL OR (Stewardship < 5 AND PnL < 0)
    2. BUY MORE: Signal is BUY AND PnL < 15% (preventing chasing at peaks)
    3. HOLD: All other cases
    """
    rows = []
    scan_map = scan_df.set_index("Symbol").to_dict("index")

    for _, p_row in portfolio_df.iterrows():
        symbol = p_row["Symbol"]
        buy_price = p_row["Buy_Price"]
        
        if symbol not in scan_map:
            rows.append({**p_row, "Audit_Decision": "NOT SCANNED", "Reasoning": "Asset not in current universe", "Total_Score": 0, "Signal": "N/A"})
            continue

        s = scan_map[symbol]
        curr_price = s["Price_EUR"]
        pnl_pct = ((curr_price - buy_price) / buy_price * 100) if buy_price else 0
        
        # Decision Logic
        decision = "HOLD"
        reasoning = "Maintain position"
        
        if s["Signal"] == "SELL":
            decision = "URGENT SELL"
            reasoning = "Scoring model indicates exit"
        elif s["Stewardship"] < 5 and pnl_pct < 0:
            decision = "URGENT SELL"
            reasoning = "Fundamental quality floor breached (Low Stewardship)"
        elif s["Signal"] == "BUY" and pnl_pct < 15:
            decision = "BUY MORE (DCA OK)"
            reasoning = "High quality setup with room for position expansion"
        elif s["RSI"] > 80:
            decision = "URGENT SELL"
            reasoning = "Extreme overbought exhaustion"
        
        rows.append({
            "Symbol": symbol,
            "PnL_pct": round(pnl_pct, 2),
            "Audit_Decision": decision,
            "Reasoning": f"{reasoning} | News: {s.get('Reasoning', 'N/A')}",
            "Total_Score": s["Total_Score"],
            "Stewardship": s["Stewardship"],
            "Signal": s["Signal"]
        })

    return pd.DataFrame(rows)

def print_audit_report(audit_df: pd.DataFrame) -> None:
    """Print the formatted portfolio audit to the terminal."""
    w = 165
    print("\n" + "=" * w)
    print("  PORTFOLIO AUDIT REPORT")
    print("=" * w)
    
    print(f"  {'Symbol':<10} {'Decision':<20} {'PnL %':>10} {'Score':>8} {'Signal':<10} {'Reasoning'}")
    print("  " + "-" * 130)
    
    for _, row in audit_df.iterrows():
        pnl = row.get("PnL_pct", 0)
        pnl_str = f"{pnl:+.1f}%" if pd.notnull(pnl) else "N/A"
        
        print(f"  {str(row.get('Symbol', '')):<10} "
              f"{str(row.get('Audit_Decision', '')):<20} "
              f"{pnl_str:>10} "
              f"{int(row.get('Total_Score', 0)):>8} "
              f"{str(row.get('Signal', '')):<10} "
              f"{str(row.get('Reasoning', ''))}")
    print("\n")

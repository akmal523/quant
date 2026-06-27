"""
portfolio.py — Portfolio audit engine v9.
Schema: Symbol,Buy_Price,Amount_EUR,Original_Amount.
Original_Amount tracks original cost basis for PnL tracking.
"""
from __future__ import annotations
import os
import pandas as pd

def load_portfolio(filepath: str = "portfolio.csv") -> pd.DataFrame:
    required_cols = ["Symbol", "Buy_Price", "Amount_EUR"]
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=required_cols)

    try:
        # comment='#' strips inline comments (e.g. "#Global Aerospace, Defence")
        # BEFORE CSV parsing, preventing stray commas from creating extra fields.
        df = pd.read_csv(filepath, comment="#").dropna(how="all")
        df.columns = df.columns.str.strip()
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 if col != "Symbol" else "UNKNOWN"
        
        # Optional Original_Amount column — defaults to Amount_EUR if absent
        if "Original_Amount" not in df.columns:
            df["Original_Amount"] = df["Amount_EUR"]
        
        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        df["Buy_Price"] = pd.to_numeric(df["Buy_Price"].astype(str).str.strip(), errors="coerce")
        df["Amount_EUR"] = pd.to_numeric(df["Amount_EUR"].astype(str).str.strip(), errors="coerce")
        df["Original_Amount"] = pd.to_numeric(df["Original_Amount"].astype(str).str.strip(), errors="coerce")
        return df.dropna(subset=["Symbol"]).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=required_cols)

def audit_portfolio(portfolio_df: pd.DataFrame, scan_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scan_map = scan_df.set_index("Symbol").to_dict("index")

    for _, p_row in portfolio_df.iterrows():
        symbol = p_row["Symbol"]
        buy_price = p_row["Buy_Price"]
        orig_amount = p_row.get("Original_Amount", p_row["Amount_EUR"])
        
        if symbol not in scan_map:
            rows.append({**p_row, "Audit_Decision": "NOT SCANNED", "Reasoning": "Asset not in current universe", "Active_Score": 0, "Signal": "N/A"})
            continue

        s = scan_map[symbol]
        curr_price = s.get("Current_Price", 0)
        no_price_data = (curr_price is None or curr_price == 0 or
                         (isinstance(curr_price, float) and pd.isna(curr_price)))
        
        decision = "HOLD"
        reasoning = "Maintain position"
        
        if no_price_data:
            pnl_pct = float('nan')
            pnl_eur = float('nan')
            current_value = float('nan')
            decision = "NO DATA"
            reasoning = f"No valid price data to compute PnL"
        else:
            pnl_pct = ((curr_price - buy_price) / buy_price * 100) if buy_price and buy_price > 0 else 0
            shares = orig_amount / buy_price if buy_price > 0 else 0.0
            current_value = curr_price * shares
            pnl_eur = current_value - orig_amount
            
            if s["Signal"] == "SELL":
                decision = "URGENT SELL"
                reasoning = "Scoring model indicates exit"
            elif s["Stewardship"] < 5 and pnl_pct < 0:
                decision = "URGENT SELL"
                reasoning = "Fundamental quality floor breached (Low Stewardship)"
            elif s["Signal"] == "BUY" and pnl_pct < 15:
                decision = "BUY MORE (DCA OK)"
                reasoning = "High quality setup with room for position expansion"
        
        rows.append({
            "Symbol": symbol,
            "PnL_pct": round(pnl_pct, 2),
            "PnL_EUR": round(pnl_eur, 2),
            "Current_Price": curr_price,
            "Original_Amount": round(orig_amount, 2),
            "Current_Value": round(current_value, 2),
            "Audit_Decision": decision,
            "Reasoning": f"{reasoning} | NLP: {s.get('NLP_Reasoning', 'N/A')}",
            "Active_Score": s.get("Active_Score", 0),
            "Stewardship": s.get("Stewardship", 0),
            "Signal": s.get("Signal", "HOLD")
        })

    return pd.DataFrame(rows)

def print_audit_report(audit_df: pd.DataFrame) -> None:
    w = 180
    print("\n" + "=" * w)
    print("  PORTFOLIO AUDIT REPORT")
    print("=" * w)
    
    print(f"  {'Symbol':<10} {'Decision':<20} {'PnL %':>8} {'PnL €':>10} {'Score':>6} {'Signal':<8} {'Reasoning'}")
    print("  " + "-" * 140)
    
    for _, row in audit_df.iterrows():
        pnl = row.get("PnL_pct", 0)
        pnl_str = f"{pnl:+.1f}%" if pd.notnull(pnl) else "N/A"
        pnl_eur = row.get("PnL_EUR", 0)
        pnl_eur_str = f"€{pnl_eur:+.2f}" if pd.notnull(pnl_eur) else "N/A"
        
        print(f"  {str(row.get('Symbol', '')):<10} "
              f"{str(row.get('Audit_Decision', '')):<20} "
              f"{pnl_str:>8} "
              f"{pnl_eur_str:>10} "
              f"{float(row.get('Active_Score', 0)):>6.1f} "
              f"{str(row.get('Signal', '')):<8} "
              f"{str(row.get('Reasoning', ''))}")
    print("\n")


def account_effectiveness(audit_df: pd.DataFrame, portfolio_df: pd.DataFrame) -> dict:
    """
    Calculate overall account effectiveness metrics from the portfolio audit.
    
    CSV schema: Buy_Price = avg cost per share, Amount_EUR = current market value,
                Original_Amount = original cost basis.
    PnL per position = current_value - cost_basis.
    Shares derived from Original_Amount / Buy_Price.
    
    Returns a dict with:
      - total_invested: sum of Original_Amount (cost basis)
      - total_value: sum of position current values
      - total_pnl_eur: total_value - total_invested
      - total_pnl_pct: weighted PnL percentage
      - weighted_score: value-weighted average Active_Score
    """
    # Build lookup: {"Symbol": {"Buy_Price": float, "Original_Amount": float}}
    port_map = {}
    if not portfolio_df.empty:
        for _, r in portfolio_df.iterrows():
            buy = pd.to_numeric(r.get("Buy_Price", 0), errors="coerce")
            orig = pd.to_numeric(r.get("Original_Amount", r.get("Amount_EUR", 0)), errors="coerce")
            port_map[r["Symbol"]] = {
                "Buy_Price": float(buy) if pd.notna(buy) and buy > 0 else 0.0,
                "Original_Amount": float(orig) if pd.notna(orig) and orig > 0 else 0.0,
            }
    
    total_invested = 0.0
    total_value = 0.0
    weighted_score_sum = 0.0
    position_count = 0
    active_count = 0
    
    for _, row in audit_df.iterrows():
        sym = row["Symbol"]
        p_info = port_map.get(sym, {"Buy_Price": 0, "Original_Amount": 0})
        cost_basis = p_info["Original_Amount"]
        buy_price = p_info["Buy_Price"]
        
        # Skip unscanned or zero-invested positions
        if cost_basis <= 0 or buy_price <= 0:
            position_count += 1
            continue
        curr_price = row.get("Current_Price", 0)
        if pd.isna(curr_price) or curr_price is None or curr_price == 0:
            position_count += 1
            continue
        
        curr_price = float(curr_price)
        buy_price = float(buy_price)
        
        # Derive shares from Original_Amount cost basis
        shares = cost_basis / buy_price if buy_price > 0 else 0.0
        value = curr_price * shares
        
        total_invested += cost_basis
        total_value += value
        score = row.get("Active_Score", 0)
        if pd.notna(score):
            weighted_score_sum += float(score) * value
        position_count += 1
        active_count += 1
    
    total_pnl_eur = total_value - total_invested
    total_pnl_pct = (total_pnl_eur / total_invested * 100) if total_invested > 0 else 0.0
    weighted_score = (weighted_score_sum / total_value) if total_value > 0 else 0.0
    
    return {
        "total_invested": round(total_invested, 2),
        "total_value": round(total_value, 2),
        "total_pnl_eur": round(total_pnl_eur, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "weighted_score": round(weighted_score, 1),
        "position_count": position_count,
        "active_count": active_count,
    }


def print_effectiveness_report(eff: dict) -> None:
    """Print the account effectiveness summary."""
    w = 70
    print("\n" + "=" * w)
    print("  ACCOUNT EFFECTIVENESS")
    print("=" * w)
    print(f"  Positions held:      {eff['position_count']}  ({eff['active_count']} with data)")
    print(f"  Total invested:      €{eff['total_invested']:>10,.2f}")
    print(f"  Current value:       €{eff['total_value']:>10,.2f}")
    print(f"  Total PnL:           €{eff['total_pnl_eur']:>+10,.2f}  ({eff['total_pnl_pct']:+.2f}%)")
    print(f"  Portfolio avg score:  {eff['weighted_score']:>5.1f} / 100")
    
    # Qualitative rating
    if eff['total_pnl_pct'] > 20:
        rating = "EXCELLENT"
    elif eff['total_pnl_pct'] > 10:
        rating = "GOOD"
    elif eff['total_pnl_pct'] > 0:
        rating = "POSITIVE"
    elif eff['total_pnl_pct'] > -10:
        rating = "SLIGHTLY NEGATIVE — REDUCE RISK"
    else:
        rating = "DRAWDOWN — REVIEW ALL POSITIONS"
    print(f"  Rating:              {rating}")
    print("=" * w)

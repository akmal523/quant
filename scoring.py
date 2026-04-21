import numpy as np
import pandas as pd

def calculate_z_score(current_val, historical_series):
    """Measures how many standard deviations the current value is from the mean."""
    if historical_series is None or historical_series.empty:
        return 0
    mean = historical_series.mean()
    std = historical_series.std()
    return (current_val - mean) / std if std > 0 else 0

def technical_score_v2(rsi_val, price, sma50, hist_close, sector_rsi_median) -> float:
    """
    Precision Technical Scoring:
    - Distance from SMA50: Measured in Z-scores (Target: -1 to +1 Std Dev).
    - RSI Context: Measured relative to Sector Median.
    """
    t_score = 0.0
    
    # 1. Distance from Mean (Z-Score) - Max 15 pts
    # We want price to be 'reasonably' near the trend, not overextended.
    if price and sma50:
        z = calculate_z_score(price, hist_close.tail(50))
        # Reward prices between -0.5 and +1.5 standard deviations from SMA50
        if -0.5 <= z <= 1.5:
            t_score += 15
        elif -1.0 <= z <= 2.0:
            t_score += 7
            
    # 2. Sector-Relative RSI - Max 15 pts
    # If Sector Median RSI is 60 (bull move), an RSI of 50 is 'relatively' cheap.
    if rsi_val and sector_rsi_median:
        relative_rsi = rsi_val - sector_rsi_median
        # Reward assets that are 5-15 points 'cooler' than their peers
        if -15 <= relative_rsi <= -5:
            t_score += 15
        elif -20 <= relative_rsi <= 5:
            t_score += 8
    elif rsi_val:
        # Fallback to static if sector data is missing
        if 30 <= rsi_val <= 60:
            t_score += 15
            
    return t_score

def trade_signal_v2(adj_score, rsi_val, sector_rsi_median, upside):
    """
    Adaptive Trade Signals.
    """
    # Dynamic RSI Overbought: Sector Median + 15 points, capped at 85
    sell_threshold = min(85, (sector_rsi_median + 15)) if sector_rsi_median else 72
    
    if adj_score >= 65 and (upside is None or float(upside) > 5):
        return "BUY"
    if adj_score >= 50:
        return "HOLD"
    if rsi_val > sell_threshold or (upside is not None and float(upside) < -10):
        return "SELL"
    return "HOLD"


def stewardship_score(debt_to_equity, payout_ratio, dividend_yield) -> float:
    """
    Evaluates intrinsic financial health and management of resources.
    Max: 20 pts.
    """
    s_score = 0.0
    
    # 1. Solvency (Avoidance of excessive leverage)
    de = float(debt_to_equity) if debt_to_equity is not None else 2.0
    if de < 0.5: s_score += 10    # Pristine balance sheet
    elif de < 1.0: s_score += 5   # Manageable debt
    
    # 2. Dividend Stewardship (Sustainability)
    payout = float(payout_ratio) if payout_ratio is not None else 1.0
    div_y = float(dividend_yield) if dividend_yield else 0.0
    
    if div_y > 0:
        # Reward sustainable payouts (30-70% range). 
        # Over 90% suggests capital exhaustion; under 20% suggests lack of distribution.
        if 0.3 <= payout <= 0.7: s_score += 10
        elif payout < 0.9: s_score += 5
    else:
        # For non-dividend payers, reward high cash retention for growth (Low Payout)
        if payout < 0.2: s_score += 5
            
    return s_score

def composite_score_v3(pe, peg, roe, stewardship_val, tech_val, geo_sent_val, vol) -> tuple[float, dict]:
    """
    Rebalanced Model (100 pts):
    - Intrinsic Fundamentals (PE/PEG/ROE): 30 pts
    - Stewardship (Solvency/Payout): 20 pts
    - Technical Stability (Z-Score): 25 pts
    - GeoSentiment (News/Risk): 25 pts
    """
    # 1. Fundamental (30)
    f_score = 0
    if pe and float(pe) < 20: f_score += 10
    if peg and float(peg) < 1.2: f_score += 10
    if roe and float(roe) > 0.15: f_score += 10
    
    # 2. Rebalanced Total
    total = f_score + stewardship_val + tech_val + geo_sent_val
    
    # 3. Volatility Penalty (Unchanged)
    if vol and float(vol) > 0.50:
        total -= 5
        
    return max(0, min(100, total))

def stewardship_trade_signal(adj_score, stewardship_val, upside):
    """
    Strict Signal: Requires a 'Quality Floor'.
    """
    # Quality Floor: If stewardship/solvency score is < 5, force HOLD/SELL
    # preventing speculation on 'junk' stocks regardless of news.
    if stewardship_val < 5:
        return "SELL" if adj_score < 40 else "HOLD"

    if adj_score >= 70 and (upside is None or float(upside) > 5):
        return "BUY"
    if adj_score >= 50:
        return "HOLD"
    return "SELL"


def classify_horizon(div_yield, vol, roe, pe) -> str:
    """
    Classifies asset into investment horizon buckets based on structural stability.
    """
    div = float(div_yield) if div_yield else 0.0
    v   = float(vol)       if vol       else 1.0
    r   = float(roe)       if roe       else 0.0
    p   = float(pe)        if pe        else 999.0

    if div >= 0.02 and v < 0.25 and r > 0.10 and 0 < p < 30:
        return "RETIREMENT (20yr+)"
    if v < 0.35 and r > 0.05 and 0 < p < 40:
        return "BUSINESS COLLATERAL (10yr)"
    return "SPECULATIVE (short-term)"

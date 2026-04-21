"""
risk.py — Covariance and correlation analysis module.
Prevents portfolio concentration by penalizing highly correlated assets.
"""
import pandas as pd

def build_correlation_matrix(history_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Builds a Pearson correlation matrix from a dictionary of historical prices.
    
    Args:
        history_dict: Dictionary mapping ticker symbols to their historical DataFrames.
                      Requires a 'Close' column.
    Returns:
        A pandas DataFrame representing the correlation matrix of daily returns.
    """
    close_prices = {}
    
    # Extract closing prices for all valid historical data
    for symbol, df in history_dict.items():
        if df is not None and not df.empty and "Close" in df.columns:
            close_prices[symbol] = df["Close"]
    
    if not close_prices:
        return pd.DataFrame()
        
    # Combine individual series into a single DataFrame
    price_df = pd.DataFrame(close_prices)
    
    # Calculate daily percentage returns, dropping initial NaN rows
    returns_df = price_df.pct_change().dropna(how="all")
    
    # Compute the Pearson correlation matrix
    corr_matrix = returns_df.corr(method="pearson")
    
    return corr_matrix

def calculate_correlation_penalty(
    symbol: str, 
    current_portfolio: list[str], 
    corr_matrix: pd.DataFrame, 
    threshold: float = 0.70
) -> float:
    """
    Calculates a scoring penalty if the candidate asset is highly correlated 
    with assets already present in the portfolio.
    
    Args:
        symbol: The ticker symbol being evaluated.
        current_portfolio: List of symbols currently held in the portfolio.
        corr_matrix: Pre-calculated correlation matrix.
        threshold: Correlation level above which a penalty is applied.
        
    Returns:
        Float representing the penalty points to deduct from the asset's composite score.
    """
    if corr_matrix.empty or symbol not in corr_matrix.columns:
        return 0.0
        
    penalty = 0.0
    
    # Evaluate correlation against every existing holding
    for holding in current_portfolio:
        if holding in corr_matrix.columns and holding != symbol:
            corr_value = corr_matrix.loc[symbol, holding]
            
            # Apply mathematical penalty for strongly positive correlation
            if corr_value > threshold:
                # Scale penalty exponentially based on deviation above the threshold
                excess_corr = corr_value - threshold
                # Maximum penalty per correlated asset is 10.0 points
                penalty += (excess_corr / (1.0 - threshold)) * 10.0
                
    # Cap the absolute maximum penalty at 25 points to prevent negative score inversion
    return min(penalty, 25.0)

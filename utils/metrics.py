"""
Performance metrics for evaluating trading strategies.
"""
import numpy as np
from typing import List, Dict, Union

def compute_returns(portfolio_values: List[float]) -> np.ndarray:
    """
    Compute returns from portfolio values.
    
    Args:
        portfolio_values: List of portfolio values over time
        
    Returns:
        Array of returns
    """
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    return returns

def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
        
    # Annualize returns and volatility (assuming daily data)
    annualized_return = np.mean(returns) * 252
    annualized_vol = np.std(returns) * np.sqrt(252)
    
    if annualized_vol == 0:
        return 0.0
        
    sharpe = (annualized_return - risk_free_rate) / annualized_vol
    return sharpe

def compute_max_drawdown(portfolio_values: List[float]) -> float:
    """
    Compute maximum drawdown.
    
    Args:
        portfolio_values: List of portfolio values over time
        
    Returns:
        Maximum drawdown as a percentage
    """
    values = np.array(portfolio_values)
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return np.max(drawdown)

def compute_win_rate(trades: List[Dict[str, Union[float, int]]]) -> float:
    """
    Compute win rate from list of trades.
    
    Args:
        trades: List of trade dictionaries with 'profit' key
        
    Returns:
        Win rate as a percentage
    """
    if not trades:
        return 0.0
        
    winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
    return winning_trades / len(trades)

def compute_profit_factor(trades: List[Dict[str, Union[float, int]]]) -> float:
    """
    Compute profit factor (gross profit / gross loss).
    
    Args:
        trades: List of trade dictionaries with 'profit' key
        
    Returns:
        Profit factor
    """
    if not trades:
        return 0.0
        
    gross_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
    gross_loss = abs(sum(trade['profit'] for trade in trades if trade['profit'] < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
        
    return gross_profit / gross_loss

def compute_metrics(portfolio_values: List[float], trades: List[Dict[str, Union[float, int]]]) -> Dict[str, float]:
    """
    Compute all performance metrics.
    
    Args:
        portfolio_values: List of portfolio values over time
        trades: List of trade dictionaries
        
    Returns:
        Dictionary of metrics
    """
    returns = compute_returns(portfolio_values)
    
    metrics = {
        'total_return': (portfolio_values[-1] / portfolio_values[0]) - 1,
        'sharpe_ratio': compute_sharpe_ratio(returns),
        'max_drawdown': compute_max_drawdown(portfolio_values),
        'win_rate': compute_win_rate(trades),
        'profit_factor': compute_profit_factor(trades),
        'num_trades': len(trades)
    }
    
    return metrics 
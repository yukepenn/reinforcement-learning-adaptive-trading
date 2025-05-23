"""
Performance and risk metrics for trading strategies.

This module provides functions for calculating:
1. Return-based metrics (total return, annualized return)
2. Risk-adjusted metrics (Sharpe ratio, Sortino ratio, Calmar ratio)
3. Drawdown metrics (maximum drawdown, average drawdown)
4. Trade statistics (win rate, profit factor, average trade)
5. Risk metrics (volatility, downside deviation)

The module handles edge cases (e.g., no trades) and provides
detailed documentation for each metric's calculation.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from scipy import stats

logger = logging.getLogger(__name__)

def calculate_returns(
    portfolio_values: np.ndarray,
    initial_value: float
) -> Dict[str, float]:
    """
    Calculate return-based metrics.
    
    Args:
        portfolio_values: Array of portfolio values over time
        initial_value: Initial portfolio value
        
    Returns:
        Dictionary of return metrics
    """
    try:
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        total_return = (portfolio_values[-1] - initial_value) / initial_value
        
        # Annualized return (assuming daily data)
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'mean_return': np.mean(returns),
            'std_return': np.std(returns)
        }
        
    except Exception as e:
        logger.error(f"Error calculating returns: {str(e)}")
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'mean_return': 0.0,
            'std_return': 0.0
        }

def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily)
        
    Returns:
        Sharpe ratio
    """
    try:
        if len(returns) < 2:
            return 0.0
            
        # Calculate excess returns
        excess_returns = returns - risk_free_rate / periods_per_year
        
        # Calculate Sharpe ratio
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)
        
        if std_excess_return == 0:
            return 0.0
            
        sharpe = np.sqrt(periods_per_year) * mean_excess_return / std_excess_return
        return sharpe
        
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {str(e)}")
        return 0.0

def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily)
        
    Returns:
        Sortino ratio
    """
    try:
        if len(returns) < 2:
            return 0.0
            
        # Calculate excess returns
        excess_returns = returns - risk_free_rate / periods_per_year
        
        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0.0
            
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
            
        # Calculate Sortino ratio
        sortino = np.sqrt(periods_per_year) * np.mean(excess_returns) / downside_std
        return sortino
        
    except Exception as e:
        logger.error(f"Error calculating Sortino ratio: {str(e)}")
        return 0.0

def calculate_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Calmar ratio.
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year (default: 252 for daily)
        
    Returns:
        Calmar ratio
    """
    try:
        if len(returns) < 2:
            return 0.0
            
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate maximum drawdown
        max_dd = calculate_max_drawdown(cum_returns)
        if max_dd == 0:
            return 0.0
            
        # Calculate Calmar ratio
        annualized_return = (cum_returns[-1] ** (periods_per_year / len(returns))) - 1
        calmar = annualized_return / max_dd
        return calmar
        
    except Exception as e:
        logger.error(f"Error calculating Calmar ratio: {str(e)}")
        return 0.0

def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        portfolio_values: Array of portfolio values
        
    Returns:
        Maximum drawdown as a decimal
    """
    try:
        if len(portfolio_values) < 2:
            return 0.0
            
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdowns
        drawdowns = (running_max - portfolio_values) / running_max
        
        # Get maximum drawdown
        max_dd = np.max(drawdowns)
        return max_dd
        
    except Exception as e:
        logger.error(f"Error calculating maximum drawdown: {str(e)}")
        return 0.0

def calculate_trade_metrics(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate trade-based metrics.
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary of trade metrics
    """
    try:
        if not trades:
            return {
                'num_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
            
        # Extract trade profits
        profits = [trade['profit'] for trade in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        # Calculate metrics
        num_trades = len(trades)
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0.0
        
        total_profit = sum(winning_trades) if winning_trades else 0.0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_trade = np.mean(profits) if profits else 0.0
        avg_win = np.mean(winning_trades) if winning_trades else 0.0
        avg_loss = np.mean(losing_trades) if losing_trades else 0.0
        
        largest_win = max(winning_trades) if winning_trades else 0.0
        largest_loss = min(losing_trades) if losing_trades else 0.0
        
        return {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }
        
    except Exception as e:
        logger.error(f"Error calculating trade metrics: {str(e)}")
        return {
            'num_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }

def calculate_risk_metrics(
    returns: np.ndarray,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate risk metrics.
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year (default: 252 for daily)
        
    Returns:
        Dictionary of risk metrics
    """
    try:
        if len(returns) < 2:
            return {
                'volatility': 0.0,
                'downside_deviation': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
            
        # Calculate volatility
        volatility = np.std(returns) * np.sqrt(periods_per_year)
        
        # Calculate downside deviation
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0.0
        
        # Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        # Calculate higher moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
        
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {str(e)}")
        return {
            'volatility': 0.0,
            'downside_deviation': 0.0,
            'var_95': 0.0,
            'cvar_95': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }

def compute_all_metrics(
    portfolio_values: np.ndarray,
    trades: List[Dict[str, Any]],
    initial_value: float,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, Any]:
    """
    Compute all performance and risk metrics.
    
    Args:
        portfolio_values: Array of portfolio values
        trades: List of trade dictionaries
        initial_value: Initial portfolio value
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 for daily)
        
    Returns:
        Dictionary containing all metrics
    """
    try:
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Compute all metrics
        return_metrics = calculate_returns(portfolio_values, initial_value)
        trade_metrics = calculate_trade_metrics(trades)
        risk_metrics = calculate_risk_metrics(returns, periods_per_year)
        
        # Calculate risk-adjusted returns
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
        calmar = calculate_calmar_ratio(returns, periods_per_year)
        
        # Combine all metrics
        metrics = {
            'returns': return_metrics,
            'trades': trade_metrics,
            'risk': risk_metrics,
            'risk_adjusted': {
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        return {
            'returns': {},
            'trades': {},
            'risk': {},
            'risk_adjusted': {}
        }

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test metrics calculation
    portfolio_values = np.array([100000, 101000, 102000, 101500, 103000])
    trades = [
        {'profit': 1000, 'timestamp': '2024-01-01'},
        {'profit': -500, 'timestamp': '2024-01-02'},
        {'profit': 1500, 'timestamp': '2024-01-03'}
    ]
    
    metrics = compute_all_metrics(
        portfolio_values=portfolio_values,
        trades=trades,
        initial_value=100000
    )
    
    print("\nPerformance Metrics:")
    print(json.dumps(metrics, indent=2)) 
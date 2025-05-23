"""
Evaluation script for trained PPO trading agent.

This script evaluates a trained trading agent on unseen test data:
1. Loads the trained model and environment
2. Runs evaluation with deterministic actions
3. Records detailed trade information
4. Computes comprehensive performance metrics
5. Generates visualization plots
"""
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils import config_utils, logging_utils, data_utils, metrics
from environment.trading_env import TradingEnv

def load_model_and_env(config: dict, test_data: bool = True) -> tuple:
    """
    Load trained model and environment.
    
    Args:
        config: Configuration dictionary
        test_data: Whether to use test data (True) or training data (False)
        
    Returns:
        Tuple of (model, env)
    """
    logger = logging.getLogger(__name__)
    
    # Load data
    if test_data:
        logger.info("Loading test data...")
        _, _, prices, features, _ = data_utils.prepare_training_data(config)
    else:
        logger.info("Loading training data...")
        prices, features, _, _, _ = data_utils.prepare_training_data(config)
    
    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    env = TradingEnv(
        prices=prices,
        features=features,
        initial_cash=config['environment']['initial_cash'],
        transaction_cost=config['environment']['transaction_cost'],
        position_limit=config['environment']['position_limit'],
        window_size=config['environment']['window_size'],
        reward_scaling=config['environment']['reward_scaling']
    )
    
    # Wrap environment
    env = DummyVecEnv([lambda: env])
    
    # Load normalization parameters
    vec_normalize_path = os.path.join(config['logging']['log_dir'], 'vec_normalize.pkl')
    if os.path.exists(vec_normalize_path):
        logger.info("Loading environment normalization parameters...")
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False  # Set to evaluation mode
    else:
        logger.warning("No normalization parameters found. Using unnormalized environment.")
    
    # Load trained model
    logger.info("Loading trained model...")
    model_path = os.path.join(config['logging']['log_dir'], 'best_model', 'best_model.zip')
    if not os.path.exists(model_path):
        model_path = os.path.join(config['logging']['log_dir'], 'final_model.zip')
    model = PPO.load(model_path)
    
    return model, env

def run_evaluation(
    model: PPO,
    env: VecNormalize,
    config: dict
) -> tuple:
    """
    Run evaluation with deterministic actions.
    
    Args:
        model: Trained PPO model
        env: Vectorized environment
        config: Configuration dictionary
        
    Returns:
        Tuple of (portfolio_values, trades, actions, timestamps)
    """
    logger = logging.getLogger(__name__)
    
    # Initialize tracking variables
    portfolio_values = []
    trades = []
    actions = []
    timestamps = []
    current_position = 0
    
    # Run evaluation
    logger.info("Running evaluation...")
    obs = env.reset()
    done = False
    
    while not done:
        # Get deterministic action
        action, _ = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Record information
        portfolio_values.append(info[0]['portfolio_value'])
        actions.append(action[0])
        timestamps.append(info[0].get('timestamp', len(timestamps)))
        
        # Record trade if position changed
        if info[0].get('position', 0) != current_position:
            trade = {
                'timestamp': info[0].get('timestamp', len(timestamps)),
                'action': action[0],
                'price': info[0].get('price', 0),
                'position': info[0].get('position', 0),
                'portfolio_value': info[0]['portfolio_value'],
                'profit': info[0].get('trade_profit', 0),
                'cumulative_profit': info[0].get('cumulative_profit', 0)
            }
            trades.append(trade)
            current_position = info[0].get('position', 0)
    
    return portfolio_values, trades, actions, timestamps

def plot_results(
    portfolio_values: list,
    trades: list,
    prices: np.ndarray,
    timestamps: list,
    save_dir: str
) -> None:
    """
    Generate and save visualization plots.
    
    Args:
        portfolio_values: List of portfolio values
        trades: List of trade dictionaries
        prices: Array of price data
        timestamps: List of timestamps
        save_dir: Directory to save plots
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating plots...")
    
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot portfolio value
    ax1.plot(timestamps, portfolio_values, label='Portfolio Value', color='blue')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value ($)')
    ax1.grid(True)
    
    # Plot price and trades
    ax2.plot(timestamps, prices, label='Price', color='gray', alpha=0.5)
    
    # Plot buy/sell points
    for trade in trades:
        if trade['action'] == 1:  # Buy
            ax2.scatter(trade['timestamp'], trade['price'], color='green', marker='^', s=100)
        elif trade['action'] == 2:  # Sell
            ax2.scatter(trade['timestamp'], trade['price'], color='red', marker='v', s=100)
    
    ax2.set_title('Price and Trades')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'))
    plt.close()
    
    logger.info("Plots saved successfully!")

def main():
    """Main evaluation function."""
    # Load configuration
    config = config_utils.load_config("config/config.yaml")
    
    # Setup logging
    logging_utils.setup_logging(config['logging']['log_dir'])
    logger = logging.getLogger(__name__)
    
    try:
        # Create results directory
        results_dir = os.path.join(config['logging']['log_dir'], 'evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        # Load model and environment
        model, env = load_model_and_env(config, test_data=True)
        
        # Run evaluation
        portfolio_values, trades, actions, timestamps = run_evaluation(model, env, config)
        
        # Get price data for plotting
        prices = env.envs[0].prices
        
        # Compute metrics
        logger.info("Computing evaluation metrics...")
        metrics_dict = metrics.compute_all_metrics(
            portfolio_values=np.array(portfolio_values),
            trades=trades,
            initial_value=config['environment']['initial_cash'],
            risk_free_rate=config['training'].get('risk_free_rate', 0.02)
        )
        
        # Save results
        logger.info("Saving evaluation results...")
        
        # Save portfolio values
        pd.DataFrame({
            'timestamp': timestamps,
            'portfolio_value': portfolio_values,
            'action': actions
        }).to_csv(os.path.join(results_dir, 'portfolio_values.csv'), index=False)
        
        # Save trades
        if trades:
            pd.DataFrame(trades).to_csv(os.path.join(results_dir, 'trades.csv'), index=False)
        else:
            logger.warning("No trades were made during evaluation.")
        
        # Save metrics
        pd.DataFrame([metrics_dict]).to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
        
        # Generate plots
        plot_results(portfolio_values, trades, prices, timestamps, results_dir)
        
        # Log summary
        logger.info("\nEvaluation Summary:")
        logger.info(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
        logger.info(f"Total return: {metrics_dict['returns']['total_return']:.2%}")
        logger.info(f"Sharpe ratio: {metrics_dict['risk_adjusted']['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics_dict['risk']['max_drawdown']:.2%}")
        logger.info(f"Number of trades: {metrics_dict['trades']['num_trades']}")
        logger.info(f"Win rate: {metrics_dict['trades']['win_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 
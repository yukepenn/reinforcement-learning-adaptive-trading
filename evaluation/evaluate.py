"""
Evaluation script for trained PPO trading agent.
"""
import os
import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils.config_utils import get_config
from utils.logging_utils import setup_logging, log_metrics
from utils.data_utils import load_data, load_processed_data
from utils.metrics import compute_metrics
from environment.trading_env import TradingEnv

def main():
    """Main evaluation function."""
    # Load configuration
    config = get_config("config/config.yaml")
    
    # Setup logging
    setup_logging(config['logging']['log_dir'])
    logger = logging.getLogger(__name__)
    
    try:
        # Load processed data
        logger.info("Loading processed data...")
        features, prices, scaler = load_processed_data(config['data']['processed_data_path'])
        
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
        env = VecNormalize.load(
            os.path.join(config['logging']['log_dir'], 'vec_normalize.pkl'),
            env
        )
        
        # Load trained model
        logger.info("Loading trained model...")
        model = PPO.load(os.path.join(config['logging']['log_dir'], 'final_model'))
        
        # Run evaluation
        logger.info("Running evaluation...")
        obs = env.reset()
        done = False
        portfolio_values = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            portfolio_values.append(info[0]['portfolio_value'])
        
        # Extract trades from environment
        trades = env.envs[0].trades if hasattr(env.envs[0], 'trades') else []
        
        # Compute profit for each trade if not present
        for i, trade in enumerate(trades):
            if 'profit' not in trade:
                # Profit is realized on closing a position; estimate as change in portfolio value since last trade
                if i == 0:
                    trade['profit'] = 0.0
                else:
                    trade['profit'] = trades[i]['portfolio_value'] - trades[i-1]['portfolio_value']
        
        if not trades:
            logger.warning("No trades were made during evaluation.")
        
        # Compute metrics
        logger.info("Computing evaluation metrics...")
        metrics = compute_metrics(portfolio_values, trades)
        
        # Save results
        logger.info("Saving evaluation results...")
        results_dir = os.path.join(config['logging']['log_dir'], 'evaluation')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save portfolio values
        pd.DataFrame({
            'portfolio_value': portfolio_values
        }).to_csv(os.path.join(results_dir, 'portfolio_values.csv'))
        
        # Save trades
        pd.DataFrame(trades).to_csv(os.path.join(results_dir, 'trades.csv'))
        
        # Save metrics
        pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, 'metrics.csv'))
        
        logger.info("Evaluation complete!")
        logger.info(f"Final portfolio value: ${portfolio_values[-1]:,.2f}")
        logger.info(f"Total return: {metrics['total_return']:.2%}")
        logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {metrics['max_drawdown']:.2%}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
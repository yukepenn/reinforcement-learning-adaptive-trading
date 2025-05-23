"""
Evaluation script for trained PPO trading agent.
"""
import os
import sys
from pathlib import Path

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
    # Load configuration
    config = get_config("config/config.yaml")
    
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'], name='evaluation')
    
    # Load processed data
    logger.info("Loading processed data...")
    features, prices, scaler = load_processed_data(config['data']['processed_path'])
    
    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    eval_env = TradingEnv(
        prices=prices,
        features=features,
        **config['environment']
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize.load(
        os.path.join(config['logging']['log_dir'], "vec_normalize.pkl"),
        eval_env
    )
    
    # Load trained model
    logger.info("Loading trained model...")
    model = PPO.load(
        os.path.join(config['logging']['log_dir'], "best_model"),
        env=eval_env
    )
    
    # Run evaluation
    logger.info("Running evaluation...")
    obs = eval_env.reset()
    done = False
    portfolio_values = []
    trades = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, info = eval_env.step(action)
        
        # Record portfolio value
        portfolio_values.append(info[0]['portfolio_value'])
        
        # Record trades
        if info[0]['trades']:
            trades.extend(info[0]['trades'])
    
    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(portfolio_values, trades)
    log_metrics(logger, metrics, prefix="Evaluation")
    
    # Save results
    logger.info("Saving results...")
    results_dir = os.path.join(config['logging']['log_dir'], "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save portfolio values
    pd.DataFrame({
        'portfolio_value': portfolio_values
    }).to_csv(os.path.join(results_dir, "portfolio_values.csv"))
    
    # Save trades
    if trades:
        pd.DataFrame(trades).to_csv(os.path.join(results_dir, "trades.csv"))
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(results_dir, "metrics.csv"))
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main() 
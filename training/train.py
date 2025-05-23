"""
Training script for PPO trading agent.
"""
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, config_utils, logging_utils
from features import feature_engineering
from environment.trading_env import TradingEnv

def main():
    # Load configuration
    config = config_utils.load_config("config/config.yaml")
    
    # Setup logging
    logging_utils.setup_logging(config['logging']['log_dir'])
    
    # Create directories if they don't exist
    os.makedirs(config['data']['processed_data_path'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
    
    # Load and prepare data
    train_prices, train_features, test_prices, test_features = data_utils.prepare_training_data(config)
    
    # Create training environment
    train_env = TradingEnv(
        prices=train_prices,
        features=train_features,
        initial_cash=config['environment']['initial_cash'],
        transaction_cost=config['environment']['transaction_cost'],
        position_limit=config['environment']['position_limit'],
        window_size=config['environment']['window_size'],
        reward_scaling=config['environment']['reward_scaling']
    )
    
    # Wrap environment
    train_env = Monitor(train_env)
    train_env = DummyVecEnv([lambda: train_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = TradingEnv(
        prices=test_prices,
        features=test_features,
        initial_cash=config['environment']['initial_cash'],
        transaction_cost=config['environment']['transaction_cost'],
        position_limit=config['environment']['position_limit'],
        window_size=config['environment']['window_size'],
        reward_scaling=config['environment']['reward_scaling']
    )
    
    # Wrap evaluation environment
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config['logging']['log_dir'], 'best_model'),
        log_path=os.path.join(config['logging']['log_dir'], 'eval_results'),
        eval_freq=config['logging']['eval_freq'],
        n_eval_episodes=config['logging']['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['logging']['save_freq'],
        save_path=os.path.join(config['logging']['log_dir'], 'checkpoints'),
        name_prefix='ppo_model'
    )
    
    # Create and train model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=config['logging']['tensorboard_dir'],
        **config['training']['ppo_params']
    )
    
    # Train the model
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Save final model and environment
    model.save(os.path.join(config['logging']['log_dir'], 'final_model'))
    train_env.save(os.path.join(config['logging']['log_dir'], 'vec_normalize.pkl'))

if __name__ == "__main__":
    main() 
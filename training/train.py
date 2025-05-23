"""
Training script for PPO trading agent.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from utils.config_utils import get_config
from utils.logging_utils import setup_logging, log_config
from utils.data_utils import load_data, split_data, prepare_data, save_processed_data
from features.feature_engineering import create_feature_matrix, normalize_features
from environment.trading_env import TradingEnv

def main():
    # Load configuration
    config = get_config("config/config.yaml")
    
    # Setup logging
    logger = setup_logging(config['logging']['log_dir'], name='training')
    log_config(logger, config)
    
    # Create directories
    os.makedirs(config['data']['processed_path'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    df = load_data(config['data']['raw_path'])
    train_df, test_df = split_data(df, config['data']['train_ratio'])
    
    # Create features
    logger.info("Creating features...")
    train_features = create_feature_matrix(
        train_df,
        config['environment']['window_size'],
        config['features']
    )
    test_features = create_feature_matrix(
        test_df,
        config['environment']['window_size'],
        config['features']
    )
    
    # Normalize features
    train_features, scaler = normalize_features(train_features)
    test_features = scaler.transform(test_features)
    
    # Save processed data
    save_processed_data(
        train_features,
        train_df['close'].values[config['environment']['window_size']-1:],
        scaler,
        config['data']['processed_path']
    )
    
    # Create training environment
    logger.info("Creating training environment...")
    train_env = TradingEnv(
        prices=train_df['close'].values[config['environment']['window_size']-1:],
        features=train_features,
        **config['environment']
    )
    train_env = DummyVecEnv([lambda: train_env])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )
    
    # Create evaluation environment
    eval_env = TradingEnv(
        prices=test_df['close'].values[config['environment']['window_size']-1:],
        features=test_features,
        **config['environment']
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        training=False
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config['logging']['log_dir'],
        log_path=config['logging']['log_dir'],
        eval_freq=config['logging']['eval_freq'],
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['logging']['save_freq'],
        save_path=config['logging']['log_dir'],
        name_prefix="ppo_trading"
    )
    
    # Create and train model
    logger.info("Creating and training PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        tensorboard_log=config['logging']['tensorboard_dir'],
        verbose=1,
        **config['training']['ppo_params']
    )
    
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Save final model and environment
    logger.info("Saving final model and environment...")
    model.save(os.path.join(config['logging']['log_dir'], "final_model"))
    train_env.save(os.path.join(config['logging']['log_dir'], "vec_normalize.pkl"))
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main() 
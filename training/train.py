"""
Training script for PPO trading agent.

This script orchestrates the training process for the trading agent:
1. Loads configuration and sets up logging
2. Prepares and processes training data
3. Creates and configures training and evaluation environments
4. Sets up model and training callbacks
5. Trains the model with proper monitoring and checkpointing
"""
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_utils, config_utils, logging_utils
from features import feature_engineering
from environment.trading_env import TradingEnv

def setup_directories(config: dict) -> None:
    """Create necessary directories for training."""
    os.makedirs(config['data']['processed_data_path'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['tensorboard_dir'], exist_ok=True)
    os.makedirs(os.path.join(config['logging']['log_dir'], 'best_model'), exist_ok=True)
    os.makedirs(os.path.join(config['logging']['log_dir'], 'checkpoints'), exist_ok=True)

def create_env(
    prices: np.ndarray,
    features: np.ndarray,
    config: dict,
    is_training: bool = True
) -> VecNormalize:
    """
    Create and configure a trading environment.
    
    Args:
        prices: Price data array
        features: Feature data array
        config: Configuration dictionary
        is_training: Whether this is a training environment
        
    Returns:
        Vectorized and normalized environment
    """
    # Create base environment
    env = TradingEnv(
        prices=prices,
        features=features,
        initial_cash=config['environment']['initial_cash'],
        transaction_cost=config['environment']['transaction_cost'],
        position_limit=config['environment']['position_limit'],
        window_size=config['environment']['window_size'],
        reward_scaling=config['environment']['reward_scaling']
    )
    
    # Set random seed for reproducibility
    env.reset(seed=config['training']['seed'])
    
    # Wrap with Monitor
    env = Monitor(env)
    
    # Create vectorized environment
    if config['training'].get('num_envs', 1) > 1 and is_training:
        env = SubprocVecEnv([
            lambda: TradingEnv(
                prices=prices,
                features=features,
                initial_cash=config['environment']['initial_cash'],
                transaction_cost=config['environment']['transaction_cost'],
                position_limit=config['environment']['position_limit'],
                window_size=config['environment']['window_size'],
                reward_scaling=config['environment']['reward_scaling']
            ) for _ in range(config['training']['num_envs'])
        ])
    else:
        env = DummyVecEnv([lambda: env])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        training=is_training
    )
    
    return env

def setup_callbacks(
    eval_env: VecNormalize,
    config: dict
) -> list:
    """
    Set up training callbacks.
    
    Args:
        eval_env: Evaluation environment
        config: Configuration dictionary
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(config['logging']['log_dir'], 'best_model'),
        log_path=os.path.join(config['logging']['log_dir'], 'eval_results'),
        eval_freq=config['logging']['eval_freq'],
        n_eval_episodes=config['logging']['n_eval_episodes'],
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config['logging']['save_freq'],
        save_path=os.path.join(config['logging']['log_dir'], 'checkpoints'),
        name_prefix='ppo_model'
    )
    callbacks.append(checkpoint_callback)
    
    return callbacks

def train_model(
    train_env: VecNormalize,
    eval_env: VecNormalize,
    config: dict
) -> PPO:
    """
    Create and train the PPO model.
    
    Args:
        train_env: Training environment
        eval_env: Evaluation environment
        config: Configuration dictionary
        
    Returns:
        Trained PPO model
    """
    # Set random seed for reproducibility
    set_random_seed(config['training']['seed'])
    
    # Create model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=config['logging']['tensorboard_dir'],
        **config['training']['ppo_params']
    )
    
    # Setup callbacks
    callbacks = setup_callbacks(eval_env, config)
    
    # Train model
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=callbacks
    )
    
    return model

def main():
    """Main training function."""
    # Load configuration
    config = config_utils.load_config("config/config.yaml")
    
    # Setup logging
    logging_utils.setup_logging(config['logging']['log_dir'])
    logger = logging_utils.get_logger(__name__)
    
    try:
        # Setup directories
        setup_directories(config)
        
        # Load and prepare data
        logger.info("Loading and preparing training data...")
        train_prices, train_features, test_prices, test_features, scaler = data_utils.prepare_training_data(config)
        
        # Save processed data
        logger.info("Saving processed data...")
        data_utils.save_processed_data(
            features=train_features,
            prices=train_prices,
            scaler=scaler,
            save_dir=config['data']['processed_data_path']
        )
        
        # Create environments
        logger.info("Creating training and evaluation environments...")
        train_env = create_env(train_prices, train_features, config, is_training=True)
        eval_env = create_env(test_prices, test_features, config, is_training=False)
        
        # Train model
        logger.info("Starting model training...")
        model = train_model(train_env, eval_env, config)
        
        # Save final model and environment
        logger.info("Saving final model and environment...")
        model.save(os.path.join(config['logging']['log_dir'], 'final_model'))
        train_env.save(os.path.join(config['logging']['log_dir'], 'vec_normalize.pkl'))
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 
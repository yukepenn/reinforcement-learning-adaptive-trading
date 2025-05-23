"""
Configuration utilities for loading and validating YAML config files.
"""
import os
import yaml
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")
            
    return config

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Required sections
    required_sections = ['data', 'environment', 'features', 'training', 'logging']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}'")
    
    # Data section
    required_data_fields = ['symbol', 'start_date', 'end_date', 'train_ratio', 'raw_data_path', 'processed_data_path']
    for field in required_data_fields:
        if field not in config['data']:
            raise ValueError(f"Missing required field '{field}' in data section")
    
    # Environment section
    required_env_fields = ['initial_cash', 'transaction_cost', 'position_limit', 'window_size', 'reward_scaling']
    for field in required_env_fields:
        if field not in config['environment']:
            raise ValueError(f"Missing required field '{field}' in environment section")
    
    # Features section
    if 'technical_indicators' not in config['features']:
        raise ValueError("Missing 'technical_indicators' in features section")
    if 'lookback_periods' not in config['features']:
        raise ValueError("Missing 'lookback_periods' in features section")
    
    # Training section
    required_training_fields = ['total_timesteps', 'seed', 'ppo_params']
    for field in required_training_fields:
        if field not in config['training']:
            raise ValueError(f"Missing required field '{field}' in training section")
    
    # PPO parameters
    required_ppo_fields = [
        'learning_rate', 'n_steps', 'batch_size', 'n_epochs',
        'gamma', 'gae_lambda', 'clip_range', 'ent_coef',
        'vf_coef', 'max_grad_norm'
    ]
    for field in required_ppo_fields:
        if field not in config['training']['ppo_params']:
            raise ValueError(f"Missing required field '{field}' in PPO parameters")
    
    # Logging section
    required_logging_fields = ['log_dir', 'tensorboard_dir', 'eval_freq', 'n_eval_episodes', 'save_freq']
    for field in required_logging_fields:
        if field not in config['logging']:
            raise ValueError(f"Missing required field '{field}' in logging section")

def get_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Validated configuration dictionary
    """
    config = load_config(config_path)
    validate_config(config)
    return config 
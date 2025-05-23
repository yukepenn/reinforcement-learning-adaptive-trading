"""
Configuration management utilities for the trading environment.

This module provides functions for:
1. Loading and validating configuration from YAML files
2. Type checking and validation of config values
3. Setting default values for missing parameters
4. Converting config values to appropriate types

The module ensures that all required parameters are present and valid
before they are used in training or evaluation.
"""
import yaml
from typing import Dict, Any, Optional, List, Union
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    'data': {
        'file_path': 'data/raw/prices.csv',
        'processed_dir': 'data/processed',
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'start_date': None,
        'end_date': None,
        'augment': False,
        'augment_methods': ['noise'],
        'noise_level': 0.001
    },
    'features': {
        'window_sizes': [5, 10, 20, 50, 200],
        'use_cache': True
    },
    'environment': {
        'initial_cash': 100000.0,
        'transaction_cost': 0.001,
        'position_limit': 1,
        'window_size': 20,
        'reward_scaling': 1.0,
        'baseline_penalty': 0.001,
        'hold_penalty': 0.0005,
        'stop_loss_pct': 0.2
    },
    'training': {
        'total_timesteps': 100000,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': None,
        'tensorboard_log': 'logs/tensorboard',
        'policy_kwargs': {
            'net_arch': [64, 64]
        }
    },
    'evaluation': {
        'n_eval_episodes': 10,
        'eval_freq': 10000,
        'deterministic': True
    },
    'logging': {
        'level': 'INFO',
        'log_dir': 'logs',
        'save_freq': 10000,
        'save_path': 'models'
    }
}

# Required configuration sections
REQUIRED_SECTIONS = [
    'data',
    'features',
    'environment',
    'training',
    'evaluation',
    'logging'
]

# Type definitions for config values
CONFIG_TYPES = {
    'data': {
        'file_path': str,
        'processed_dir': str,
        'train_ratio': float,
        'val_ratio': float,
        'start_date': (str, type(None)),
        'end_date': (str, type(None)),
        'augment': bool,
        'augment_methods': list,
        'noise_level': float
    },
    'features': {
        'window_sizes': list,
        'use_cache': bool
    },
    'environment': {
        'initial_cash': float,
        'transaction_cost': float,
        'position_limit': int,
        'window_size': int,
        'reward_scaling': float,
        'baseline_penalty': float,
        'hold_penalty': float,
        'stop_loss_pct': float
    },
    'training': {
        'total_timesteps': int,
        'learning_rate': float,
        'n_steps': int,
        'batch_size': int,
        'n_epochs': int,
        'gamma': float,
        'gae_lambda': float,
        'clip_range': float,
        'ent_coef': float,
        'vf_coef': float,
        'max_grad_norm': float,
        'use_sde': bool,
        'sde_sample_freq': int,
        'target_kl': (float, type(None)),
        'tensorboard_log': str,
        'policy_kwargs': dict
    },
    'evaluation': {
        'n_eval_episodes': int,
        'eval_freq': int,
        'deterministic': bool
    },
    'logging': {
        'level': str,
        'log_dir': str,
        'save_freq': int,
        'save_path': str
    }
}

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    try:
        # Check if config file exists
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return DEFAULT_CONFIG.copy()
            
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate config
        validate_config(config)
        
        # Merge with defaults
        config = merge_with_defaults(config)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If config is invalid
    """
    # Check required sections
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing_sections:
        raise ValueError(f"Missing required sections: {missing_sections}")
        
    # Check types and values
    for section, params in CONFIG_TYPES.items():
        if section not in config:
            continue
            
        for param, expected_type in params.items():
            if param not in config[section]:
                continue
                
            value = config[section][param]
            
            # Check type
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Invalid type for {section}.{param}: "
                    f"expected {expected_type}, got {type(value)}"
                )
                
            # Check value ranges
            if param == 'train_ratio' and not 0 < value < 1:
                raise ValueError(f"train_ratio must be between 0 and 1, got {value}")
            elif param == 'val_ratio' and not 0 < value < 1:
                raise ValueError(f"val_ratio must be between 0 and 1, got {value}")
            elif param == 'learning_rate' and value <= 0:
                raise ValueError(f"learning_rate must be positive, got {value}")
            elif param == 'gamma' and not 0 < value < 1:
                raise ValueError(f"gamma must be between 0 and 1, got {value}")
            elif param == 'stop_loss_pct' and not 0 < value < 1:
                raise ValueError(f"stop_loss_pct must be between 0 and 1, got {value}")

def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration with default values.
    
    Args:
        config: User configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = DEFAULT_CONFIG.copy()
    
    def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> None:
        for key, value in d2.items():
            if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
                merge_dicts(d1[key], value)
            else:
                d1[key] = value
                
    merge_dicts(merged, config)
    return merged

def save_config(config: Dict[str, Any], config_path: str = "config/config.yaml") -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info(f"Saved configuration to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")
        raise

def get_training_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training parameters from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of training parameters
    """
    return {
        'total_timesteps': config['training']['total_timesteps'],
        'learning_rate': config['training']['learning_rate'],
        'n_steps': config['training']['n_steps'],
        'batch_size': config['training']['batch_size'],
        'n_epochs': config['training']['n_epochs'],
        'gamma': config['training']['gamma'],
        'gae_lambda': config['training']['gae_lambda'],
        'clip_range': config['training']['clip_range'],
        'ent_coef': config['training']['ent_coef'],
        'vf_coef': config['training']['vf_coef'],
        'max_grad_norm': config['training']['max_grad_norm'],
        'use_sde': config['training']['use_sde'],
        'sde_sample_freq': config['training']['sde_sample_freq'],
        'target_kl': config['training']['target_kl'],
        'tensorboard_log': config['training']['tensorboard_log'],
        'policy_kwargs': config['training']['policy_kwargs']
    }

def get_env_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract environment parameters from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of environment parameters
    """
    return {
        'initial_cash': config['environment']['initial_cash'],
        'transaction_cost': config['environment']['transaction_cost'],
        'position_limit': config['environment']['position_limit'],
        'window_size': config['environment']['window_size'],
        'reward_scaling': config['environment']['reward_scaling'],
        'baseline_penalty': config['environment']['baseline_penalty'],
        'hold_penalty': config['environment']['hold_penalty'],
        'stop_loss_pct': config['environment']['stop_loss_pct']
    }

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test config loading and validation
    config = load_config()
    print("Configuration loaded successfully!")
    
    # Test parameter extraction
    training_params = get_training_params(config)
    env_params = get_env_params(config)
    print("\nTraining parameters:", training_params)
    print("\nEnvironment parameters:", env_params) 
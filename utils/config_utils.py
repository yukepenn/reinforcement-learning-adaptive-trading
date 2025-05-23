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

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary has required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['data', 'environment', 'features', 'training', 'logging']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")
            
    # Validate data section
    data_required = ['raw_path', 'processed_path', 'train_ratio', 'test_ratio']
    for field in data_required:
        if field not in config['data']:
            raise ValueError(f"Missing required field '{field}' in data section")
            
    # Validate environment section
    env_required = ['initial_cash', 'transaction_cost', 'window_size', 'position_limit']
    for field in env_required:
        if field not in config['environment']:
            raise ValueError(f"Missing required field '{field}' in environment section")
            
    # Validate training section
    if 'ppo_params' not in config['training']:
        raise ValueError("Missing 'ppo_params' in training section")
        
    return True

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
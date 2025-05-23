"""
Logging utilities for setting up and managing logs.
"""
import os
import logging
from datetime import datetime
from typing import Optional

def setup_logging(log_dir: str, name: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration with both file and console handlers.
    
    Args:
        log_dir: Directory to store log files
        name: Optional name for the logger (defaults to timestamp)
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"{name}_{timestamp}" if name else timestamp
    log_file = os.path.join(log_dir, f"{log_name}.log")
    
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_config(logger: logging.Logger, config: dict) -> None:
    """
    Log configuration parameters.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
    """
    logger.info("Configuration:")
    for section, params in config.items():
        logger.info(f"\n{section.upper()}:")
        for key, value in params.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for k, v in value.items():
                    logger.info(f"    {k}: {v}")
            else:
                logger.info(f"  {key}: {value}")

def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = "") -> None:
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics to log
        prefix: Optional prefix for log messages
    """
    logger.info(f"\n{prefix}Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}") 
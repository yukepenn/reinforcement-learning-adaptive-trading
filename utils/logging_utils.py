"""
Logging utilities for the trading environment.

This module provides functions for:
1. Setting up Python logging with consistent formatting
2. Configuring TensorBoard logging
3. Logging training metrics and environment statistics
4. Managing log files and directories

The module ensures consistent logging across the project and provides
visualization capabilities through TensorBoard.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from datetime import datetime
import json
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd

# Default logging format
DEFAULT_LOG_FORMAT = (
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Default logging configuration
DEFAULT_LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': DEFAULT_LOG_FORMAT
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'standard',
            'filename': 'logs/trading.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        config: Optional custom logging configuration
    """
    try:
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Use default config if none provided
        if config is None:
            config = DEFAULT_LOGGING_CONFIG.copy()
            
        # Update log file path
        config['handlers']['file']['filename'] = os.path.join(
            log_dir,
            f"trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        # Update log level
        config['handlers']['console']['level'] = log_level
        config['handlers']['file']['level'] = log_level
        config['loggers']['']['level'] = log_level
        
        # Configure logging
        logging.config.dictConfig(config)
        
        logging.info(f"Logging configured with level {log_level}")
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def log_training_metrics(
    metrics: Dict[str, float],
    step: int,
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log training metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Current training step
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger(__name__)
        
    # Log metrics
    for name, value in metrics.items():
        logger.info(f"Step {step} - {name}: {value:.4f}")
        
    # Save metrics to file
    metrics_file = os.path.join("logs", "training_metrics.json")
    try:
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
            
        history.append({
            'step': step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        with open(metrics_file, 'w') as f:
            json.dump(history, f, indent=2)
            
    except Exception as e:
        logger.error(f"Error saving metrics: {str(e)}")

def log_environment_stats(
    stats: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log environment statistics.
    
    Args:
        stats: Dictionary of environment statistics
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger(__name__)
        
    # Log statistics
    logger.info("Environment Statistics:")
    for name, value in stats.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info(f"  {name}: {value}")

def load_tensorboard_data(log_dir: str) -> pd.DataFrame:
    """
    Load training metrics from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        
    Returns:
        DataFrame containing training metrics
    """
    try:
        # Initialize event accumulator
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={
                event_accumulator.SCALARS: 0,
            }
        )
        ea.Reload()
        
        # Get all scalar events
        metrics = {}
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            metrics[tag] = [event.value for event in events]
            
        # Convert to DataFrame
        df = pd.DataFrame(metrics)
        return df
        
    except Exception as e:
        logging.error(f"Error loading TensorBoard data: {str(e)}")
        return pd.DataFrame()

def plot_training_metrics(
    metrics_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot training metrics.
    
    Args:
        metrics_df: DataFrame containing training metrics
        save_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics')
        
        # Plot metrics
        if 'ep_rew_mean' in metrics_df.columns:
            axes[0, 0].plot(metrics_df['ep_rew_mean'])
            axes[0, 0].set_title('Mean Episode Reward')
            
        if 'loss' in metrics_df.columns:
            axes[0, 1].plot(metrics_df['loss'])
            axes[0, 1].set_title('Loss')
            
        if 'explained_variance' in metrics_df.columns:
            axes[1, 0].plot(metrics_df['explained_variance'])
            axes[1, 0].set_title('Explained Variance')
            
        if 'entropy_loss' in metrics_df.columns:
            axes[1, 1].plot(metrics_df['entropy_loss'])
            axes[1, 1].set_title('Entropy Loss')
            
        # Save plot
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Saved training metrics plot to {save_path}")
            
        plt.close()
        
    except Exception as e:
        logging.error(f"Error plotting training metrics: {str(e)}")

def log_trade_execution(
    trade: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log trade execution details.
    
    Args:
        trade: Dictionary containing trade details
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger(__name__)
        
    # Log trade details
    logger.info(
        f"Trade executed:\n"
        f"  Action: {trade['action']}\n"
        f"  Price: {trade['price']:.2f}\n"
        f"  Quantity: {trade['quantity']}\n"
        f"  Cost: {trade['cost']:.2f}\n"
        f"  Portfolio Value: {trade['portfolio_value']:.2f}\n"
        f"  Timestamp: {trade['timestamp']}"
    )

def log_evaluation_results(
    results: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> None:
    """
    Log evaluation results.
    
    Args:
        results: Dictionary containing evaluation results
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger(__name__)
        
    # Log results
    logger.info("Evaluation Results:")
    for metric, value in results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Test logging
    logger = get_logger(__name__)
    logger.info("Testing logging setup")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test metrics logging
    metrics = {
        'ep_rew_mean': 100.0,
        'loss': 0.5,
        'explained_variance': 0.8
    }
    log_training_metrics(metrics, step=1)
    
    # Test environment stats logging
    stats = {
        'portfolio_value': 100000.0,
        'position': 1,
        'cash': 50000.0
    }
    log_environment_stats(stats) 
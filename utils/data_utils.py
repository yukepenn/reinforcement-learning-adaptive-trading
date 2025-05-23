"""
Data utilities for loading, preprocessing, and augmenting trading data.

This module provides functions for:
1. Loading and validating raw price data
2. Splitting data into training and testing sets
3. Data augmentation and preprocessing
4. Feature engineering and scaling
5. Caching processed data for faster loading

The module ensures temporal consistency in data splits and provides
various augmentation techniques for training data.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import logging
from features import feature_engineering
from pathlib import Path
import yaml
import joblib
import random

from features.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def load_data(
    file_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    required_columns: List[str] = ['Open', 'High', 'Low', 'Close', 'Volume']
) -> pd.DataFrame:
    """
    Load and validate price data from CSV file.
    
    Args:
        file_path: Path to CSV file
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        required_columns: List of required columns
        
    Returns:
        DataFrame with price data
    """
    try:
        # Load data
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.set_index('Date', inplace=True)
        
        # Validate columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Filter by date range
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
            
        # Log data statistics
        logger.info(
            f"Loaded {len(df)} timesteps of price data from {df.index.min()} to {df.index.max()}\n"
            f"Columns: {', '.join(df.columns)}\n"
            f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB"
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into training, validation, and testing sets.
    
    Args:
        df: DataFrame with price data
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        shuffle: Whether to shuffle data before splitting
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if shuffle:
        if seed is not None:
            random.seed(seed)
        indices = list(range(len(df)))
        random.shuffle(indices)
        df = df.iloc[indices]
    
    # Calculate split indices
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    
    # Split data
    train_data = df[:train_size]
    val_data = df[train_size:train_size + val_size]
    test_data = df[train_size + val_size:]
    
    # Log split statistics
    logger.info(
        f"Split data into:\n"
        f"- Training: {len(train_data)} samples ({train_ratio:.1%})\n"
        f"- Validation: {len(val_data)} samples ({val_ratio:.1%})\n"
        f"- Testing: {len(test_data)} samples ({(1 - train_ratio - val_ratio):.1%})"
    )
    
    return train_data, val_data, test_data

def augment_data(
    df: pd.DataFrame,
    methods: List[str] = ['noise', 'window'],
    noise_level: float = 0.001,
    window_size: int = 20,
    overlap: float = 0.5
) -> pd.DataFrame:
    """
    Augment training data using various methods.
    
    Args:
        df: DataFrame with price data
        methods: List of augmentation methods to apply
        noise_level: Standard deviation of Gaussian noise
        window_size: Size of sliding window for window augmentation
        overlap: Overlap ratio between windows
        
    Returns:
        Augmented DataFrame
    """
    augmented_data = []
    
    for method in methods:
        if method == 'noise':
            # Add Gaussian noise to prices
            noisy_data = df.copy()
            for col in ['Open', 'High', 'Low', 'Close']:
                noise = np.random.normal(0, noise_level * df[col].mean(), size=len(df))
                noisy_data[col] = df[col] * (1 + noise)
            augmented_data.append(noisy_data)
            
        elif method == 'window':
            # Create overlapping windows
            step_size = int(window_size * (1 - overlap))
            for i in range(0, len(df) - window_size + 1, step_size):
                window = df.iloc[i:i + window_size].copy()
                window.index = pd.date_range(
                    start=df.index[0],
                    periods=window_size,
                    freq=df.index.freq
                )
                augmented_data.append(window)
                
    # Combine original and augmented data
    if augmented_data:
        result = pd.concat([df] + augmented_data)
        result = result.sort_index()
        
        logger.info(
            f"Augmented data using methods: {methods}\n"
            f"Original samples: {len(df)}\n"
            f"Augmented samples: {len(result)}"
        )
        
        return result
    
    return df

def prepare_training_data(
    config: Dict[str, Any],
    use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare training and testing data with features.
    
    Args:
        config: Configuration dictionary
        use_cache: Whether to use cached processed data
        
    Returns:
        Tuple of (train_prices, train_features, test_prices, test_features, scaler)
    """
    # Load data
    df = load_data(
        config['data']['file_path'],
        config['data'].get('start_date'),
        config['data'].get('end_date')
    )
    
    # Split data
    train_data, val_data, test_data = split_data(
        df,
        config['data']['train_ratio'],
        config['data'].get('val_ratio', 0.1)
    )
    
    # Augment training data
    if config['data'].get('augment', False):
        train_data = augment_data(
            train_data,
            methods=config['data'].get('augment_methods', ['noise']),
            noise_level=config['data'].get('noise_level', 0.001)
        )
    
    # Initialize feature engineer
    engineer = FeatureEngineer(
        window_sizes=config['features']['window_sizes'],
        cache_dir=config['data']['processed_dir']
    )
    
    # Compute features
    train_features, feature_names = engineer.create_feature_matrix(train_data, use_cache)
    test_features, _ = engineer.create_feature_matrix(test_data, use_cache)
    
    # Scale features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)
    
    # Extract prices
    train_prices = train_data['Close'].values
    test_prices = test_data['Close'].values
    
    # Save processed data
    if use_cache:
        save_processed_data(
            config['data']['processed_dir'],
            train_features,
            test_features,
            train_prices,
            test_prices,
            scaler
        )
    
    return train_prices, train_features, test_prices, test_features, scaler

def save_processed_data(
    directory: str,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_prices: np.ndarray,
    test_prices: np.ndarray,
    scaler: StandardScaler
) -> None:
    """
    Save processed data to disk.
    
    Args:
        directory: Directory to save data
        train_features: Training feature matrix
        test_features: Testing feature matrix
        train_prices: Training price array
        test_prices: Testing price array
        scaler: Fitted StandardScaler
    """
    try:
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save data
        np.save(f"{directory}/train_features.npy", train_features)
        np.save(f"{directory}/test_features.npy", test_features)
        np.save(f"{directory}/train_prices.npy", train_prices)
        np.save(f"{directory}/test_prices.npy", test_prices)
        joblib.dump(scaler, f"{directory}/scaler.joblib")
        
        logger.info(f"Saved processed data to {directory}")
        
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise

def load_processed_data(directory: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Load processed data from disk.
    
    Args:
        directory: Directory containing processed data
        
    Returns:
        Tuple of (train_features, test_features, train_prices, test_prices, scaler)
    """
    try:
        # Load data
        train_features = np.load(f"{directory}/train_features.npy")
        test_features = np.load(f"{directory}/test_features.npy")
        train_prices = np.load(f"{directory}/train_prices.npy")
        test_prices = np.load(f"{directory}/test_prices.npy")
        scaler = joblib.load(f"{directory}/scaler.joblib")
        
        logger.info(f"Loaded processed data from {directory}")
        
        return train_features, test_features, train_prices, test_prices, scaler
        
    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        raise

def download_yahoo_data(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Download historical price data from Yahoo Finance.
    
    Args:
        symbol (str): Stock symbol (e.g., 'ZN=F' for 10-year Treasury futures)
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format. Defaults to today.
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    try:
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data downloaded for {symbol}")
            
        # Ensure all required columns are present
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns in downloaded data")
            
        # Sort by date and reset index
        data = data.sort_index()
        data.index.name = 'Date'
        
        logger.info(f"Successfully downloaded {len(data)} days of data")
        return data
        
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath (str): Path to the data file
    
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        # Try to load with Date as index first
        try:
            data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        except ValueError:
            # If Date column not found, try loading with default index
            data = pd.read_csv(filepath)
            # Convert the first column to datetime if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
        
        # Convert numeric columns to float
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        logger.info(f"Loaded data from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_data(data: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data and split into training and testing sets.
    
    Args:
        data (pd.DataFrame): Raw price data
        train_ratio (float): Ratio of data to use for training (default: 0.8)
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Training and testing DataFrames
    """
    try:
        # Handle missing values
        data = data.ffill()  # Forward fill
        data = data.bfill()  # Backward fill any remaining NaNs
        
        # Calculate daily returns
        data['Returns'] = data['Close'].pct_change()
        
        # Split data
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        logger.info(f"Split data into {len(train_data)} training and {len(test_data)} testing samples")
        
        return train_data, test_data
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def save_data(data: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        data (pd.DataFrame): Data to save
        filepath (str): Path to save the data
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data: {str(e)}")
        raise

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config = load_config()
    
    # Prepare data
    train_prices, train_features, test_prices, test_features, scaler = prepare_training_data(config) 
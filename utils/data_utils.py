"""
Data loading and preprocessing utilities.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import logging
from features import feature_engineering

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load price data from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with price data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    df = pd.read_csv(file_path)
    
    # Validate required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
        
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Sort by timestamp
    df.sort_index(inplace=True)
    
    return df

def split_data(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
        df: DataFrame to split
        train_ratio: Ratio of data to use for training
        
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df

def prepare_data(
    df: pd.DataFrame,
    window_size: int,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare data for the trading environment.
    
    Args:
        df: DataFrame with price data
        window_size: Number of time steps to include in observation
        scaler: Optional pre-fitted scaler
        
    Returns:
        Tuple of (features, prices, scaler)
    """
    # Extract price data
    prices = df['close'].values
    
    # Create features
    features = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size]
        feature_vector = []
        
        # Price features
        feature_vector.extend([
            window['close'].values[-1] / window['open'].values[0] - 1,  # Return
            window['high'].max() / window['low'].min() - 1,  # Range
            window['volume'].mean()  # Average volume
        ])
        
        # Technical indicators
        feature_vector.extend([
            window['close'].pct_change().mean(),  # Mean return
            window['close'].pct_change().std(),   # Return volatility
            window['close'].rolling(5).mean().iloc[-1] / window['close'].iloc[-1] - 1,  # MA5
            window['close'].rolling(10).mean().iloc[-1] / window['close'].iloc[-1] - 1,  # MA10
            window['close'].rolling(20).mean().iloc[-1] / window['close'].iloc[-1] - 1,  # MA20
        ])
        
        features.append(feature_vector)
    
    features = np.array(features)
    prices = prices[window_size - 1:]
    
    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    
    return features, prices, scaler

def save_processed_data(
    features: np.ndarray,
    prices: np.ndarray,
    scaler: StandardScaler,
    save_dir: str
) -> None:
    """
    Save processed data and scaler.
    
    Args:
        features: Feature array
        prices: Price array
        scaler: Fitted scaler
        save_dir: Directory to save files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'features.npy'), features)
    np.save(os.path.join(save_dir, 'prices.npy'), prices)
    
    import joblib
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))

def load_processed_data(save_dir: str) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Load processed data and scaler.
    
    Args:
        save_dir: Directory containing saved files
        
    Returns:
        Tuple of (features, prices, scaler)
    """
    features = np.load(os.path.join(save_dir, 'features.npy'))
    prices = np.load(os.path.join(save_dir, 'prices.npy'))
    
    import joblib
    scaler = joblib.load(os.path.join(save_dir, 'scaler.joblib'))
    
    return features, prices, scaler

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

def prepare_training_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare data for training and testing.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]: 
            train_prices, train_features, test_prices, test_features, scaler
    """
    try:
        # Get data paths from config
        raw_data_path = config['data']['raw_data_path']
        processed_data_path = config['data']['processed_data_path']
        
        # Download data if not exists
        if not os.path.exists(raw_data_path):
            data = download_yahoo_data(
                symbol=config['data']['symbol'],
                start_date=config['data']['start_date'],
                end_date=config['data']['end_date']
            )
            save_data(data, raw_data_path)
        else:
            data = load_data(raw_data_path)
        
        # Preprocess and split data
        train_data, test_data = preprocess_data(
            data, 
            train_ratio=config['data']['train_ratio']
        )
        
        # Save processed data
        save_data(train_data, os.path.join(processed_data_path, 'train.csv'))
        save_data(test_data, os.path.join(processed_data_path, 'test.csv'))
        
        # Create features
        train_features = feature_engineering.create_feature_matrix(train_data, config)
        test_features = feature_engineering.create_feature_matrix(test_data, config)
        
        # Scale features
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        
        # Convert prices to numpy arrays
        train_prices = train_data['Close'].values
        test_prices = test_data['Close'].values
        
        logger.info(f"Prepared training data with shapes: prices={train_prices.shape}, features={train_features.shape}")
        logger.info(f"Prepared testing data with shapes: prices={test_prices.shape}, features={test_features.shape}")
        
        return train_prices, train_features, test_prices, test_features, scaler
        
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
        raise 
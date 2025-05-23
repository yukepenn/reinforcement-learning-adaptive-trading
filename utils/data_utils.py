"""
Data loading and preprocessing utilities.
"""
import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler

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
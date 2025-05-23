"""
Feature engineering for trading data.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

def compute_technical_indicators(
    df: pd.DataFrame,
    indicators: List[str],
    lookback_periods: List[int]
) -> pd.DataFrame:
    """
    Compute technical indicators for the given DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        indicators: List of indicator names to compute
        lookback_periods: List of lookback periods for each indicator
        
    Returns:
        DataFrame with added technical indicators
    """
    result_df = df.copy()
    
    for period in lookback_periods:
        # Moving Averages
        if 'sma' in indicators:
            sma = SMAIndicator(close=df['close'], window=period)
            result_df[f'sma_{period}'] = sma.sma_indicator()
            
        if 'ema' in indicators:
            ema = EMAIndicator(close=df['close'], window=period)
            result_df[f'ema_{period}'] = ema.ema_indicator()
            
        # RSI
        if 'rsi' in indicators:
            rsi = RSIIndicator(close=df['close'], window=period)
            result_df[f'rsi_{period}'] = rsi.rsi()
            
        # MACD
        if 'macd' in indicators:
            macd = MACD(
                close=df['close'],
                window_slow=period,
                window_fast=period // 2,
                window_sign=period // 4
            )
            result_df[f'macd_{period}'] = macd.macd()
            result_df[f'macd_signal_{period}'] = macd.macd_signal()
            result_df[f'macd_diff_{period}'] = macd.macd_diff()
            
        # Bollinger Bands
        if 'bollinger' in indicators:
            bb = BollingerBands(
                close=df['close'],
                window=period,
                window_dev=2
            )
            result_df[f'bb_high_{period}'] = bb.bollinger_hband()
            result_df[f'bb_low_{period}'] = bb.bollinger_lband()
            result_df[f'bb_mid_{period}'] = bb.bollinger_mavg()
            
    return result_df

def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic price-based features.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added price features
    """
    result_df = df.copy()
    
    # Returns
    result_df['returns'] = df['close'].pct_change()
    result_df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    result_df['volatility'] = result_df['returns'].rolling(window=20).std()
    
    # Price ranges
    result_df['daily_range'] = (df['high'] - df['low']) / df['close']
    result_df['daily_range_pct'] = (df['high'] - df['low']) / df['open']
    
    # Volume features
    result_df['volume_ma'] = df['volume'].rolling(window=20).mean()
    result_df['volume_std'] = df['volume'].rolling(window=20).std()
    result_df['volume_ratio'] = df['volume'] / result_df['volume_ma']
    
    return result_df

def compute_sma(data: pd.DataFrame, window: int) -> pd.Series:
    """Compute Simple Moving Average."""
    return data['Close'].rolling(window=window).mean()

def compute_ema(data: pd.DataFrame, window: int) -> pd.Series:
    """Compute Exponential Moving Average."""
    return data['Close'].ewm(span=window, adjust=False).mean()

def compute_rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD, Signal line, and MACD histogram."""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def compute_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands."""
    sma = compute_sma(data, window)
    std = data['Close'].rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def compute_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute rolling volatility."""
    returns = data['Close'].pct_change()
    return returns.rolling(window=window).std()

def compute_drawdown(data: pd.DataFrame, window: int = 252) -> pd.Series:
    """Compute rolling drawdown."""
    rolling_max = data['Close'].rolling(window=window, min_periods=1).max()
    drawdown = (data['Close'] - rolling_max) / rolling_max
    return drawdown

def create_feature_matrix(
    df: pd.DataFrame,
    window_size: int,
    feature_config: Dict[str, Any]
) -> np.ndarray:
    """
    Create feature matrix for the trading environment.
    
    Args:
        df: DataFrame with price data and indicators
        window_size: Number of time steps to include in observation
        feature_config: Configuration for feature computation
        
    Returns:
        Feature matrix of shape (n_samples, n_features)
    """
    # Compute technical indicators
    df = compute_technical_indicators(
        df,
        feature_config['technical_indicators'],
        feature_config['lookback_periods']
    )
    
    # Compute price features
    df = compute_price_features(df)
    
    # Drop NaN values
    df = df.dropna()
    
    # Create feature matrix
    features = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size]
        feature_vector = []
        
        # Add all features from the last timestep
        feature_vector.extend(window.iloc[-1].values)
        
        # Add statistics over the window
        for col in df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                feature_vector.extend([
                    window[col].mean(),
                    window[col].std(),
                    window[col].min(),
                    window[col].max()
                ])
        
        features.append(feature_vector)
    
    return np.array(features)

def normalize_features(
    features: np.ndarray,
    scaler: Any = None
) -> Tuple[np.ndarray, Any]:
    """
    Normalize features using StandardScaler.
    
    Args:
        features: Feature matrix
        scaler: Optional pre-fitted scaler
        
    Returns:
        Tuple of (normalized_features, scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
    else:
        normalized_features = scaler.transform(features)
    
    return normalized_features, scaler

def create_feature_matrix(data: pd.DataFrame, config: Dict[str, Any]) -> np.ndarray:
    """
    Create a feature matrix from the price data.
    
    Args:
        data (pd.DataFrame): Price data
        config (Dict[str, Any]): Configuration dictionary
    
    Returns:
        np.ndarray: Feature matrix of shape (n_samples, n_features)
    """
    try:
        features = []
        feature_names = []
        
        # Add price and returns
        features.append(data['Close'].values)
        feature_names.append('close')
        features.append(data['Returns'].values)
        feature_names.append('returns')
        
        # Add SMAs
        for window in config['features']['lookback_periods']:
            sma = compute_sma(data, window)
            features.append(sma.values)
            feature_names.append(f'sma_{window}')
        
        # Add EMAs
        for window in config['features']['lookback_periods']:
            ema = compute_ema(data, window)
            features.append(ema.values)
            feature_names.append(f'ema_{window}')
        
        # Add RSI
        rsi = compute_rsi(data)
        features.append(rsi.values)
        feature_names.append('rsi')
        
        # Add MACD
        macd, signal, hist = compute_macd(data)
        features.extend([macd.values, signal.values, hist.values])
        feature_names.extend(['macd', 'macd_signal', 'macd_hist'])
        
        # Add Bollinger Bands
        upper, middle, lower = compute_bollinger_bands(data)
        features.extend([upper.values, middle.values, lower.values])
        feature_names.extend(['bb_upper', 'bb_middle', 'bb_lower'])
        
        # Add volatility
        vol = compute_volatility(data)
        features.append(vol.values)
        feature_names.append('volatility')
        
        # Add drawdown
        dd = compute_drawdown(data)
        features.append(dd.values)
        feature_names.append('drawdown')
        
        # Stack features
        feature_matrix = np.column_stack(features)
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        # Normalize if specified
        if config['data']['normalize']:
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
        
        logger.info(f"Created feature matrix with shape {feature_matrix.shape}")
        logger.info(f"Feature names: {feature_names}")
        
        return feature_matrix
        
    except Exception as e:
        logger.error(f"Error creating feature matrix: {str(e)}")
        raise

def prepare_features(train_data: pd.DataFrame, test_data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare features for both training and testing data.
    
    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Testing data
        config (Dict[str, Any]): Configuration dictionary
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Training and testing feature matrices
    """
    try:
        # Create feature matrices
        train_features = create_feature_matrix(train_data, config)
        test_features = create_feature_matrix(test_data, config)
        
        return train_features, test_features
        
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        raise 
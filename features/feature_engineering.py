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
    from sklearn.preprocessing import StandardScaler
    
    if scaler is None:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
    else:
        normalized_features = scaler.transform(features)
    
    return normalized_features, scaler 
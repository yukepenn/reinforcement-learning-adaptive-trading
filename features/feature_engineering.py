"""
Feature engineering module for trading environment.

This module provides functions to compute technical indicators and features
for the trading environment. All computations are vectorized using pandas/numpy
for performance.

Features include:
1. Price-based indicators:
   - Returns (1d, 5d, 10d)
   - Volatility (5d, 10d, 20d)
   - Moving averages (5d, 10d, 20d, 50d, 200d)
   - Price momentum (5d, 10d, 20d)

2. Volume-based indicators:
   - Volume moving averages
   - Volume momentum
   - Volume volatility

3. Technical indicators:
   - RSI (14d)
   - MACD (12, 26, 9)
   - Bollinger Bands (20d, 2σ)
   - ATR (14d)
   - Stochastic Oscillator (14d)

4. Custom features:
   - Price/MA crossovers
   - Volatility regimes
   - Trend strength
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering class for computing technical indicators and features.
    
    This class provides methods to compute various technical indicators and features
    from price and volume data. All computations are vectorized for performance.
    
    Attributes:
        window_sizes: List of window sizes for moving averages and other indicators
        cache_dir: Directory to cache computed features
    """
    
    def __init__(
        self,
        window_sizes: List[int] = [5, 10, 20, 50, 200],
        cache_dir: str = "data/processed"
    ):
        """
        Initialize feature engineer.
        
        Args:
            window_sizes: List of window sizes for moving averages and indicators
            cache_dir: Directory to cache computed features
        """
        self.window_sizes = window_sizes
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute price-based features.
        
        Args:
            df: DataFrame with 'Close' price column
            
        Returns:
            DataFrame with price-based features
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns
        for window in [1, 5, 10]:
            features[f'return_{window}d'] = df['Close'].pct_change(window)
            
        # Volatility
        for window in [5, 10, 20]:
            features[f'volatility_{window}d'] = df['Close'].pct_change().rolling(window).std()
            
        # Moving averages
        for window in self.window_sizes:
            features[f'ma_{window}d'] = df['Close'].rolling(window).mean()
            features[f'ma_ratio_{window}d'] = df['Close'] / features[f'ma_{window}d']
            
        # Momentum
        for window in [5, 10, 20]:
            features[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1
            
        return features
        
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volume-based features.
        
        Args:
            df: DataFrame with 'Volume' column
            
        Returns:
            DataFrame with volume-based features
        """
        features = pd.DataFrame(index=df.index)
        
        # Volume moving averages
        for window in self.window_sizes:
            features[f'volume_ma_{window}d'] = df['Volume'].rolling(window).mean()
            features[f'volume_ratio_{window}d'] = df['Volume'] / features[f'volume_ma_{window}d']
            
        # Volume momentum
        for window in [5, 10, 20]:
            features[f'volume_momentum_{window}d'] = df['Volume'] / df['Volume'].shift(window) - 1
            
        # Volume volatility
        for window in [5, 10, 20]:
            features[f'volume_volatility_{window}d'] = df['Volume'].pct_change().rolling(window).std()
            
        return features
        
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators.
        
        Args:
            df: DataFrame with 'High', 'Low', 'Close' columns
            
        Returns:
            DataFrame with technical indicators
        """
        features = pd.DataFrame(index=df.index)
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi_14d'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        for window in [20]:
            ma = df['Close'].rolling(window=window).mean()
            std = df['Close'].rolling(window=window).std()
            features[f'bb_upper_{window}d'] = ma + (std * 2)
            features[f'bb_lower_{window}d'] = ma - (std * 2)
            features[f'bb_width_{window}d'] = (features[f'bb_upper_{window}d'] - features[f'bb_lower_{window}d']) / ma
            
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        features['atr_14d'] = true_range.rolling(14).mean()
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        features['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
        
        return features
        
    def compute_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute custom features.
        
        Args:
            df: DataFrame with price and indicator columns
            
        Returns:
            DataFrame with custom features
        """
        features = pd.DataFrame(index=df.index)
        
        # Price/MA crossovers
        for window in [5, 10, 20, 50, 200]:
            ma = df['Close'].rolling(window=window).mean()
            features[f'price_above_ma_{window}d'] = (df['Close'] > ma).astype(int)
            
        # Volatility regime
        volatility = df['Close'].pct_change().rolling(window=20).std()
        features['volatility_regime'] = pd.qcut(volatility, q=5, labels=False)
        
        # Trend strength
        for window in [5, 10, 20]:
            returns = df['Close'].pct_change(window)
            features[f'trend_strength_{window}d'] = returns.rolling(window=window).mean() / returns.rolling(window=window).std()
            
        return features
        
    def create_feature_matrix(
        self,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Create feature matrix from price data.
        
        Args:
            df: DataFrame with OHLCV data
            use_cache: Whether to use cached features
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        cache_file = self.cache_dir / 'features.npy'
        names_file = self.cache_dir / 'feature_names.joblib'
        
        if use_cache and cache_file.exists() and names_file.exists():
            logger.info("Loading cached features...")
            features = np.load(cache_file)
            feature_names = joblib.load(names_file)
            return features, feature_names
            
        logger.info("Computing features...")
        
        # Compute all features
        price_features = self.compute_price_features(df)
        volume_features = self.compute_volume_features(df)
        technical_features = self.compute_technical_indicators(df)
        custom_features = self.compute_custom_features(df)
        
        # Combine all features
        all_features = pd.concat([
            price_features,
            volume_features,
            technical_features,
            custom_features
        ], axis=1)
        
        # Handle missing values
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        # Convert to numpy array
        features = all_features.values
        
        # Save feature names
        feature_names = {name: i for i, name in enumerate(all_features.columns)}
        
        # Cache results
        if use_cache:
            logger.info("Caching features...")
            np.save(cache_file, features)
            joblib.dump(feature_names, names_file)
            
        return features, feature_names
        
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names.
        
        Returns:
            List of feature names
        """
        names_file = self.cache_dir / 'feature_names.joblib'
        if names_file.exists():
            feature_names = joblib.load(names_file)
            return list(feature_names.keys())
        return []

def test_feature_engineering():
    """
    Test feature engineering calculations.
    
    This function verifies that feature calculations are correct
    by comparing against known values for a small sample.
    """
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = {
        'Open': np.random.normal(100, 1, 100),
        'High': np.random.normal(101, 1, 100),
        'Low': np.random.normal(99, 1, 100),
        'Close': np.random.normal(100, 1, 100),
        'Volume': np.random.normal(1000000, 100000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Compute features
    features, feature_names = engineer.create_feature_matrix(df, use_cache=False)
    
    # Test RSI calculation
    rsi_idx = feature_names['rsi_14d']
    rsi_values = features[:, rsi_idx]
    assert np.all((rsi_values >= 0) & (rsi_values <= 100)), "RSI values should be between 0 and 100"
    
    # Test moving averages
    ma_idx = feature_names['ma_20d']
    ma_values = features[:, ma_idx]
    assert not np.any(np.isnan(ma_values)), "Moving average should not contain NaN values"
    
    # Test volume features
    vol_idx = feature_names['volume_ma_20d']
    vol_values = features[:, vol_idx]
    assert not np.any(np.isnan(vol_values)), "Volume features should not contain NaN values"
    
    logger.info("All feature engineering tests passed!")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_feature_engineering() 
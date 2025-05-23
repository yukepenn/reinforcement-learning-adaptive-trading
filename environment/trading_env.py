"""
Custom Gymnasium environment for trading.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any

class TradingEnv(gym.Env):
    """
    Custom trading environment that follows gymnasium interface.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_cash: float = 100000.0,
        transaction_cost: float = 0.001,
        position_limit: int = 1,
        window_size: int = 20,
        reward_scaling: float = 1.0
    ):
        """
        Initialize trading environment.
        
        Args:
            prices: Array of price data
            features: Array of feature data
            initial_cash: Starting cash balance
            transaction_cost: Cost per trade as a fraction
            position_limit: Maximum position size
            window_size: Number of time steps in observation
            reward_scaling: Scaling factor for rewards
        """
        super(TradingEnv, self).__init__()
        
        self.prices = prices
        self.features = features
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.position_limit = position_limit
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        
        # Validate inputs
        assert len(prices) == len(features), "Price and feature arrays must have same length"
        assert window_size > 0, "Window size must be positive"
        
        # Define action space (0: sell, 1: hold, 2: buy)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(features.shape[1],),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset state variables
        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.position = 0
        self.trades = []
        self.portfolio_value = self.initial_cash
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Trading action (0: sell, 1: hold, 2: buy)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get current price
        current_price = self.prices[self.current_step]
        
        # Execute trade
        new_position = action - 1  # Convert action to position (-1, 0, 1)
        position_change = new_position - self.position
        
        # Calculate transaction cost
        cost = abs(position_change) * current_price * self.transaction_cost
        
        # Update position and cash
        self.position = new_position
        self.cash -= cost
        
        # Calculate portfolio value
        self.portfolio_value = self.cash + self.position * current_price
        
        # Calculate reward (change in portfolio value)
        reward = (self.portfolio_value - self.initial_cash) * self.reward_scaling
        
        # Record trade if position changed
        if position_change != 0:
            self.trades.append({
                'step': self.current_step,
                'price': current_price,
                'position': self.position,
                'cost': cost,
                'portfolio_value': self.portfolio_value
            })
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.prices) - 1
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, False, info
        
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Feature vector for current time step
        """
        return self.features[self.current_step].astype(np.float32)
        
    def _get_info(self) -> Dict[str, Any]:
        """
        Get current environment info.
        
        Returns:
            Dictionary of environment information
        """
        return {
            'step': self.current_step,
            'cash': self.cash,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'trades': self.trades
        }
        
    def render(self, mode: str = 'human') -> None:
        """
        Render environment (not implemented).
        """
        pass
        
    def close(self) -> None:
        """
        Clean up environment (not implemented).
        """
        pass 
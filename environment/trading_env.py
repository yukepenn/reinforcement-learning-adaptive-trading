"""
Custom Gymnasium environment for trading.

State Space:
    - Technical indicators (passed as features array)
    - Current position (-1 to 1)
    - Current cash balance
    - Current portfolio value

Action Space:
    0: Sell/Short (target position = -position_limit)
    1: Hold (maintain current position)
    2: Buy/Long (target position = position_limit)

Reward Structure:
    Base Components:
    - Portfolio return (percentage change in portfolio value)
    - Transaction costs (scaled by portfolio value)
    - Position penalties (for holding cash or underperforming baseline)
    - Stop loss penalty (if triggered)

    Additional Factors:
    - Baseline comparison (relative to buy-and-hold strategy)
    - Risk-adjusted returns (through position management)
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import logging

from utils.logging_utils import setup_logging

class TradingEnv(gym.Env):
    """
    Custom trading environment that follows gymnasium interface.
    
    This environment simulates a trading scenario where an agent can:
    - Take long or short positions up to a specified limit
    - Hold positions across multiple time steps
    - Pay transaction costs for position changes
    - Face penalties for poor performance or excessive risk
    
    The environment maintains internal state including:
    - Current position (-position_limit to position_limit)
    - Cash balance
    - Portfolio value
    - Trade history
    
    The agent receives rewards based on:
    - Portfolio performance relative to baseline
    - Risk management (position sizing, stop losses)
    - Transaction cost efficiency
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
        reward_scaling: float = 1.0,
        baseline_penalty: float = 0.001,  # Penalty for not taking positions
        hold_penalty: float = 0.0005,    # Penalty for holding cash
        stop_loss_pct: float = 0.2,      # Stop loss threshold (20% loss)
        log_level: str = 'INFO'
    ):
        """
        Initialize trading environment.
        
        Args:
            prices: Array of price data (numpy array)
            features: Array of feature data (numpy array)
            initial_cash: Starting cash balance
            transaction_cost: Cost per trade as a fraction
            position_limit: Maximum position size
            window_size: Number of time steps in observation
            reward_scaling: Scaling factor for rewards
            baseline_penalty: Penalty for underperforming buy-and-hold
            hold_penalty: Penalty for holding cash instead of positions
            stop_loss_pct: Stop loss threshold as percentage of initial capital
            log_level: Logging level for environment events
        """
        super(TradingEnv, self).__init__()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Store input data
        self.prices = prices
        self.features = features
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.position_limit = position_limit
        self.window_size = window_size
        self.reward_scaling = reward_scaling
        self.baseline_penalty = baseline_penalty
        self.hold_penalty = hold_penalty
        self.stop_loss_pct = stop_loss_pct
        
        # Calculate stop loss threshold
        self.stop_loss_threshold = initial_cash * (1 - stop_loss_pct)
        
        # Calculate buy-and-hold baseline performance
        self.baseline_returns = np.diff(prices) / prices[:-1]
        self.baseline_cumulative = np.cumprod(1 + self.baseline_returns)
        
        # Validate inputs
        assert len(prices) == len(features), "Price and feature arrays must have same length"
        assert window_size > 0, "Window size must be positive"
        assert 0 < stop_loss_pct < 1, "Stop loss percentage must be between 0 and 1"
        
        # Define action space (0: sell, 1: hold, 2: buy)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space (features + position)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(features.shape[1] + 1,),  # +1 for position
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        self.logger.info(
            f"Trading environment initialized with:\n"
            f"- Initial cash: ${initial_cash:,.2f}\n"
            f"- Position limit: {position_limit}\n"
            f"- Transaction cost: {transaction_cost:.2%}\n"
            f"- Stop loss: {stop_loss_pct:.1%}"
        )
        
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
        
        self.logger.debug(f"Environment reset at step {self.current_step}")
        
        return observation, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: Trading action (0: sell, 1: hold, 2: buy)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
            terminated: True if episode ends due to stop loss or end of data
            truncated: True if episode is truncated (not used in this implementation)
        """
        # Get current price and previous portfolio value
        current_price = self.prices[self.current_step]
        previous_portfolio_value = self.portfolio_value
        
        # Execute trade
        if action == 0:  # Sell
            target_position = -self.position_limit
        elif action == 2:  # Buy
            target_position = self.position_limit
        else:  # Hold
            target_position = self.position
            
        # Calculate position change
        position_change = target_position - self.position
        
        # Calculate transaction cost
        cost = abs(position_change) * current_price * self.transaction_cost
        
        # Update position and cash
        self.position = target_position
        self.cash -= cost
        
        # Calculate portfolio value
        self.portfolio_value = self.cash + self.position * current_price
        
        # Check for stop loss
        stop_loss_triggered = self.portfolio_value <= self.stop_loss_threshold
        
        # Calculate reward components
        # 1. Portfolio value change
        portfolio_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        
        # 2. Baseline comparison (buy-and-hold performance)
        baseline_return = self.baseline_returns[self.current_step - 1] if self.current_step > 0 else 0
        baseline_comparison = portfolio_return - baseline_return
        
        # 3. Position penalties
        position_penalty = 0.0
        if self.position == 0:  # No position
            position_penalty = self.hold_penalty
        elif portfolio_return < baseline_return:  # Underperforming baseline
            position_penalty = self.baseline_penalty
        
        # Combine reward components
        reward = (
            portfolio_return -  # Base return
            position_penalty -  # Position and baseline penalties
            (cost / previous_portfolio_value)  # Transaction cost impact
        ) * self.reward_scaling
        
        # Apply additional penalty for stop loss
        if stop_loss_triggered:
            reward -= 1.0  # Significant penalty for hitting stop loss
        
        # Record trade if position changed
        if position_change != 0:
            trade_profit = (current_price - self.prices[self.current_step - 1]) * self.position
            trade_info = {
                'step': self.current_step,
                'price': current_price,
                'position': self.position,
                'cost': cost,
                'portfolio_value': self.portfolio_value,
                'profit': trade_profit,
                'action': action,
                'portfolio_return': portfolio_return,
                'baseline_return': baseline_return,
                'reward': reward,
                'stop_loss_triggered': stop_loss_triggered
            }
            self.trades.append(trade_info)
            
            self.logger.debug(
                f"Trade executed at step {self.current_step}:\n"
                f"- Action: {action} ({'Sell' if action == 0 else 'Hold' if action == 1 else 'Buy'})\n"
                f"- Price: ${current_price:.2f}\n"
                f"- Position: {self.position}\n"
                f"- Cost: ${cost:.2f}\n"
                f"- Profit: ${trade_profit:.2f}"
            )
        
        # Move to next step
        self.current_step += 1
        
        # Check termination conditions
        done = (
            self.current_step >= len(self.prices) - 1 or  # End of data
            stop_loss_triggered  # Stop loss triggered
        )
        
        if done:
            self.logger.info(
                f"Episode ended at step {self.current_step}:\n"
                f"- Final portfolio value: ${self.portfolio_value:.2f}\n"
                f"- Total trades: {len(self.trades)}\n"
                f"- Stop loss triggered: {stop_loss_triggered}"
            )
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add stop loss information to info
        info['stop_loss_triggered'] = stop_loss_triggered
        info['stop_loss_threshold'] = self.stop_loss_threshold
        
        return observation, reward, done, False, info
        
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Feature vector for current time step with position appended
        """
        # Combine features with current position
        features = self.features[self.current_step].astype(np.float32)
        position = np.array([self.position], dtype=np.float32)
        return np.concatenate([features, position])
        
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
            'trades': self.trades,
            'trade': self.position
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
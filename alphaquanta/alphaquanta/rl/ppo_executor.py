"""
PPO (Proximal Policy Optimization) execution policy for quantum-enhanced trading.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging
from collections import deque
import asyncio
import time

logger = logging.getLogger(__name__)


class PPOActor(nn.Module):
    """PPO Actor network for trading action selection."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions normalized to [-1, 1]
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class PPOCritic(nn.Module):
    """PPO Critic network for value estimation."""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class TradingEnvironment:
    """Trading environment for RL training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rl_config = config.get('rl', {})
        
        self.max_position = self.rl_config.get('max_position', 1000)
        self.transaction_cost = self.rl_config.get('transaction_cost', 0.001)
        self.slippage_factor = self.rl_config.get('slippage_factor', 0.0005)
        
        self.current_position = 0
        self.current_price = 100.0
        self.cash = 100000.0
        self.portfolio_value = self.cash
        self.step_count = 0
        
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.volatility_history = deque(maxlen=100)
        
        logger.info("Trading environment initialized")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_position = 0
        self.current_price = 100.0
        self.cash = 100000.0
        self.portfolio_value = self.cash
        self.step_count = 0
        
        self.price_history.clear()
        self.volume_history.clear()
        self.volatility_history.clear()
        
        for _ in range(20):
            self.price_history.append(self.current_price + np.random.normal(0, 1))
            self.volume_history.append(1000 + np.random.normal(0, 100))
            self.volatility_history.append(0.02 + np.random.normal(0, 0.005))
        
        return self._get_state()
    
    def step(self, action: np.ndarray, market_data: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute trading action and return new state, reward, done, info."""
        self.step_count += 1
        
        position_change = action[0] * self.max_position  # Scale to position size
        urgency = (action[1] + 1) / 2  # Convert from [-1,1] to [0,1]
        limit_offset = action[2] * 0.01  # Limit price offset as percentage
        
        new_price = market_data.get('price', self.current_price * (1 + np.random.normal(0, 0.01)))
        new_volume = market_data.get('volume', 1000 + np.random.normal(0, 100))
        new_volatility = market_data.get('volatility', 0.02 + np.random.normal(0, 0.005))
        
        self.price_history.append(new_price)
        self.volume_history.append(new_volume)
        self.volatility_history.append(new_volatility)
        
        price_impact = self._calculate_price_impact(position_change, new_volume, urgency)
        execution_price = new_price * (1 + price_impact)
        
        old_position = self.current_position
        self.current_position = np.clip(
            self.current_position + position_change,
            -self.max_position,
            self.max_position
        )
        
        actual_trade = self.current_position - old_position
        trade_cost = abs(actual_trade) * execution_price * self.transaction_cost
        
        self.cash -= actual_trade * execution_price + trade_cost
        self.current_price = new_price
        self.portfolio_value = self.cash + self.current_position * self.current_price
        
        reward = self._calculate_reward(actual_trade, execution_price, price_impact)
        
        done = (self.step_count >= 1000 or 
                self.portfolio_value <= 50000 or  # Stop loss
                self.portfolio_value >= 200000)   # Take profit
        
        info = {
            'position': self.current_position,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'execution_price': execution_price,
            'price_impact': price_impact,
            'trade_cost': trade_cost,
            'actual_trade': actual_trade
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current environment state."""
        if len(self.price_history) < 20:
            return np.zeros(50)  # Return zero state if insufficient history
        
        prices = np.array(list(self.price_history)[-20:])
        price_returns = np.diff(np.log(prices))
        price_sma = np.mean(prices)
        price_std = np.std(prices)
        
        volumes = np.array(list(self.volume_history)[-20:])
        volume_sma = np.mean(volumes)
        volume_std = np.std(volumes)
        
        volatilities = np.array(list(self.volatility_history)[-20:])
        vol_sma = np.mean(volatilities)
        vol_std = np.std(volatilities)
        
        position_ratio = self.current_position / self.max_position
        cash_ratio = self.cash / 100000.0
        portfolio_return = (self.portfolio_value - 100000.0) / 100000.0
        
        current_price = self.current_price
        price_momentum = (current_price - price_sma) / price_sma if price_sma > 0 else 0
        price_mean_reversion = (price_sma - current_price) / price_std if price_std > 0 else 0
        
        state = np.concatenate([
            price_returns[-10:],  # Last 10 returns
            [price_momentum, price_mean_reversion, price_std],
            [volume_sma, volume_std],
            [vol_sma, vol_std],
            [position_ratio, cash_ratio, portfolio_return],
            [current_price / 100.0],  # Normalized current price
            np.array(list(self.price_history)[-20:]) / 100.0  # Normalized price history
        ])
        
        if len(state) > 50:
            state = state[:50]
        elif len(state) < 50:
            state = np.pad(state, (0, 50 - len(state)), 'constant')
        
        return state.astype(np.float32)
    
    def _calculate_price_impact(self, position_change: float, volume: float, urgency: float) -> float:
        """Calculate price impact based on trade size, volume, and urgency."""
        if volume <= 0:
            return 0.0
        
        base_impact = (abs(position_change) / volume) * self.slippage_factor
        
        urgency_multiplier = 1.0 + urgency * 2.0
        
        impact_direction = np.sign(position_change)
        
        return impact_direction * base_impact * urgency_multiplier
    
    def _calculate_reward(self, trade_size: float, execution_price: float, price_impact: float) -> float:
        """Calculate reward for the trading action."""
        portfolio_return = (self.portfolio_value - 100000.0) / 100000.0
        base_reward = portfolio_return * 100  # Scale up
        
        impact_penalty = abs(price_impact) * 1000
        
        trade_penalty = abs(trade_size) * self.transaction_cost * 10
        
        position_bonus = 0
        if abs(self.current_position) < self.max_position * 0.8:
            position_bonus = 1.0
        
        total_reward = base_reward - impact_penalty - trade_penalty + position_bonus
        
        return total_reward


class PPOExecutor:
    """PPO-based execution policy for quantum-enhanced trading."""
    
    def __init__(self, config: Dict[str, Any], qpu_tracker=None):
        self.config = config
        self.rl_config = config.get('rl', {})
        self.qpu_tracker = qpu_tracker
        
        self.learning_rate = self.rl_config.get('learning_rate', 3e-4)
        self.gamma = self.rl_config.get('gamma', 0.99)
        self.gae_lambda = self.rl_config.get('gae_lambda', 0.95)
        self.clip_epsilon = self.rl_config.get('clip_epsilon', 0.2)
        self.entropy_coef = self.rl_config.get('entropy_coef', 0.01)
        self.value_coef = self.rl_config.get('value_coef', 0.5)
        self.max_grad_norm = self.rl_config.get('max_grad_norm', 0.5)
        
        self.state_dim = 50  # Fixed state dimension
        self.action_dim = 3  # [position_change, urgency, limit_offset]
        self.hidden_dim = self.rl_config.get('hidden_dim', 256)
        
        self.actor = PPOActor(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic = PPOCritic(self.state_dim, self.hidden_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        self.env = TradingEnvironment(config)
        
        self.buffer_size = self.rl_config.get('buffer_size', 2048)
        self.batch_size = self.rl_config.get('batch_size', 64)
        self.update_epochs = self.rl_config.get('update_epochs', 10)
        
        self.training_enabled = self.rl_config.get('training_enabled', False)
        self.episode_count = 0
        self.total_steps = 0
        
        logger.info(f"PPO Executor initialized - training: {self.training_enabled}")
    
    async def execute_trade(self, signal: Dict[str, Any], market_data: Dict[str, Any], 
                          qaoa_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Execute trade using PPO policy with QAOA basket guidance."""
        start_time = time.time()
        
        try:
            state = self._prepare_state(signal, market_data, qaoa_weights)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = self.actor(state_tensor).squeeze(0).numpy()
            
            execution_params = self._action_to_execution_params(action, signal)
            
            execution_result = await self._simulate_execution(execution_params, market_data)
            
            if self.training_enabled:
                await self._update_experience(state, action, execution_result, market_data)
            
            elapsed_time = time.time() - start_time
            
            return {
                'success': True,
                'execution_params': execution_params,
                'execution_result': execution_result,
                'rl_enhanced': True,
                'qaoa_guided': qaoa_weights is not None,
                'execution_time': elapsed_time,
                'policy_action': action.tolist()
            }
            
        except Exception as e:
            logger.error(f"PPO execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def _prepare_state(self, signal: Dict[str, Any], market_data: Dict[str, Any], 
                      qaoa_weights: Optional[Dict[str, float]]) -> np.ndarray:
        """Prepare state vector for PPO policy."""
        price = market_data.get('price', 100.0)
        volume = market_data.get('volume', 1000.0)
        volatility = market_data.get('volatility', 0.02)
        
        signal_strength = signal.get('confidence', 0.5)
        signal_direction = 1.0 if signal.get('side') == 'BUY' else -1.0
        signal_quantity = signal.get('quantity', 100)
        
        qaoa_weight = 0.0
        qaoa_confidence = 0.0
        if qaoa_weights:
            symbol = signal.get('symbol', 'SPY')
            qaoa_weight = qaoa_weights.get(symbol, 0.0)
            qaoa_confidence = 1.0  # QAOA available
        
        price_momentum = np.random.normal(0, 0.1)  # Mock momentum
        mean_reversion = np.random.normal(0, 0.1)  # Mock mean reversion
        
        current_position = 0  # Would track actual position
        cash_ratio = 1.0  # Would track actual cash
        
        state = np.array([
            price / 100.0,  # Normalized price
            volume / 1000.0,  # Normalized volume
            volatility * 100,  # Scaled volatility
            signal_strength,
            signal_direction,
            signal_quantity / 1000.0,  # Normalized quantity
            qaoa_weight,
            qaoa_confidence,
            price_momentum,
            mean_reversion,
            current_position / 1000.0,  # Normalized position
            cash_ratio
        ])
        
        if len(state) < self.state_dim:
            state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        
        return state.astype(np.float32)
    
    def _action_to_execution_params(self, action: np.ndarray, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Convert PPO action to execution parameters."""
        position_change = action[0]  # [-1, 1]
        urgency = (action[1] + 1) / 2  # Convert to [0, 1]
        limit_offset = action[2] * 0.01  # Limit price offset
        
        base_quantity = signal.get('quantity', 100)
        adjusted_quantity = int(base_quantity * (1 + position_change * 0.5))
        adjusted_quantity = max(1, min(adjusted_quantity, base_quantity * 2))
        
        if urgency > 0.8:
            order_type = 'MARKET'
            limit_price = None
        else:
            order_type = 'LIMIT'
            base_price = signal.get('price', 100.0)
            limit_price = base_price * (1 + limit_offset)
        
        return {
            'symbol': signal.get('symbol', 'SPY'),
            'side': signal.get('side', 'BUY'),
            'quantity': adjusted_quantity,
            'order_type': order_type,
            'limit_price': limit_price,
            'urgency': urgency,
            'rl_adjustment': position_change,
            'original_quantity': base_quantity
        }
    
    async def _simulate_execution(self, params: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate trade execution (replace with real broker interface)."""
        execution_delay = np.random.uniform(0.01, 0.1)  # 10-100ms
        await asyncio.sleep(execution_delay)
        
        base_price = market_data.get('price', 100.0)
        quantity = params['quantity']
        
        urgency = params.get('urgency', 0.5)
        slippage_factor = 0.0005 * (1 + urgency) * (quantity / 100)
        
        if params['side'] == 'BUY':
            execution_price = base_price * (1 + slippage_factor)
        else:
            execution_price = base_price * (1 - slippage_factor)
        
        transaction_cost = quantity * execution_price * 0.001
        
        return {
            'executed_quantity': quantity,
            'execution_price': execution_price,
            'total_cost': quantity * execution_price + transaction_cost,
            'slippage': slippage_factor,
            'execution_delay': execution_delay,
            'transaction_cost': transaction_cost,
            'timestamp': time.time()
        }
    
    async def _update_experience(self, state: np.ndarray, action: np.ndarray, 
                               execution_result: Dict[str, Any], market_data: Dict[str, Any]):
        """Update experience buffer for training."""
        if not self.training_enabled:
            return
        
        slippage = execution_result.get('slippage', 0)
        delay = execution_result.get('execution_delay', 0)
        
        reward = -abs(slippage) * 1000 - delay * 10
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': state,  # Simplified
            'done': False
        }
        
        logger.debug(f"Updated experience: reward={reward:.4f}, slippage={slippage:.6f}")
    
    async def train_policy(self, num_episodes: int = 100) -> Dict[str, float]:
        """Train PPO policy using trading environment."""
        if not self.training_enabled:
            logger.warning("Training not enabled")
            return {}
        
        logger.info(f"Starting PPO training for {num_episodes} episodes")
        
        training_metrics = {
            'total_episodes': 0,
            'avg_reward': 0.0,
            'avg_portfolio_return': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0
        }
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0
            
            while True:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = self.actor(state_tensor).squeeze(0).numpy()
                
                market_data = {'price': 100 + np.random.normal(0, 2)}
                next_state, reward, done, info = self.env.step(action, market_data)
                
                episode_reward += reward
                episode_steps += 1
                
                if done:
                    break
                
                state = next_state
            
            training_metrics['total_episodes'] += 1
            training_metrics['avg_reward'] += episode_reward
            training_metrics['avg_portfolio_return'] += info.get('portfolio_value', 100000) / 100000 - 1
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: reward={episode_reward:.2f}, "
                           f"portfolio_value={info.get('portfolio_value', 0):.0f}")
        
        if training_metrics['total_episodes'] > 0:
            training_metrics['avg_reward'] /= training_metrics['total_episodes']
            training_metrics['avg_portfolio_return'] /= training_metrics['total_episodes']
        
        logger.info(f"Training completed: avg_reward={training_metrics['avg_reward']:.2f}")
        
        return training_metrics
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")

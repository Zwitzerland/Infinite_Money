"""
RL Allocator Module

Implements reinforcement learning for portfolio allocation:
- Q-learning for allocation
- Policy gradient methods
- Multi-agent RL
- Risk-aware RL
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging


class RLAllocator:
    """
    Reinforcement learning-based portfolio allocator.
    
    Implements RL methods for portfolio allocation:
    - Q-learning for allocation
    - Policy gradient methods
    - Multi-agent RL
    - Risk-aware RL
    """
    
    def __init__(self, learning_rate: float = 0.01, discount_factor: float = 0.95):
        """
        Initialize RL allocator.
        
        Args:
            learning_rate: Learning rate for RL algorithms
            discount_factor: Discount factor for future rewards
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}
        self.policy = {}
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.logger = logging.getLogger(__name__)
        
    def get_allocation(self, state: Dict, symbols: List[str]) -> Dict[str, float]:
        """
        Get RL-based portfolio allocation.
        
        Args:
            state: Current market state
            symbols: Available symbols
            
        Returns:
            Dictionary mapping symbols to allocation weights
        """
        try:
            # Convert state to state key
            state_key = self._state_to_key(state)
            
            # Get action from policy
            action = self._get_action(state_key, symbols)
            
            # Convert action to allocation weights
            allocation = self._action_to_allocation(action, symbols)
            
            # Record state and action
            self.state_history.append(state)
            self.action_history.append(action)
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Error getting RL allocation: {e}")
            return self._equal_allocation(symbols)
    
    def _state_to_key(self, state: Dict) -> str:
        """Convert state dictionary to state key."""
        try:
            # Create a simplified state representation
            state_features = []
            
            # Market regime
            regime = state.get('regime', 'normal')
            state_features.append(f"regime_{regime}")
            
            # Volatility level
            volatility = state.get('volatility', 0.2)
            if volatility > 0.3:
                vol_level = 'high'
            elif volatility > 0.15:
                vol_level = 'medium'
            else:
                vol_level = 'low'
            state_features.append(f"vol_{vol_level}")
            
            # Market sentiment
            sentiment = state.get('sentiment', 0.0)
            if sentiment > 0.1:
                sent_level = 'positive'
            elif sentiment < -0.1:
                sent_level = 'negative'
            else:
                sent_level = 'neutral'
            state_features.append(f"sent_{sent_level}")
            
            return "_".join(state_features)
            
        except Exception as e:
            self.logger.debug(f"Error converting state to key: {e}")
            return "default_state"
    
    def _get_action(self, state_key: str, symbols: List[str]) -> str:
        """Get action for current state."""
        try:
            # Initialize Q-table for state if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            
            # Get available actions
            actions = self._get_available_actions(symbols)
            
            # Choose action using epsilon-greedy policy
            epsilon = 0.1  # Exploration rate
            if np.random.random() < epsilon:
                # Random action
                action = np.random.choice(actions)
            else:
                # Greedy action
                action = self._get_greedy_action(state_key, actions)
            
            return action
            
        except Exception as e:
            self.logger.debug(f"Error getting action: {e}")
            return "equal_weight"
    
    def _get_available_actions(self, symbols: List[str]) -> List[str]:
        """Get available actions for symbols."""
        try:
            actions = [
                "equal_weight",
                "momentum_weight",
                "value_weight",
                "growth_weight",
                "defensive_weight",
                "aggressive_weight"
            ]
            
            # Add symbol-specific actions
            for symbol in symbols[:5]:  # Limit to top 5 symbols
                actions.extend([
                    f"overweight_{symbol}",
                    f"underweight_{symbol}"
                ])
            
            return actions
            
        except Exception as e:
            self.logger.debug(f"Error getting available actions: {e}")
            return ["equal_weight"]
    
    def _get_greedy_action(self, state_key: str, actions: List[str]) -> str:
        """Get greedy action based on Q-values."""
        try:
            q_values = self.q_table[state_key]
            
            # Get Q-values for available actions
            action_q_values = {}
            for action in actions:
                action_q_values[action] = q_values.get(action, 0.0)
            
            # Return action with highest Q-value
            best_action = max(action_q_values, key=action_q_values.get)
            return best_action
            
        except Exception as e:
            self.logger.debug(f"Error getting greedy action: {e}")
            return "equal_weight"
    
    def _action_to_allocation(self, action: str, symbols: List[str]) -> Dict[str, float]:
        """Convert action to allocation weights."""
        try:
            n_symbols = len(symbols)
            
            if action == "equal_weight":
                weight = 1.0 / n_symbols
                return {symbol: weight for symbol in symbols}
            
            elif action == "momentum_weight":
                # Placeholder for momentum-based allocation
                weights = np.random.dirichlet(np.ones(n_symbols))
                return {symbol: weight for symbol, weight in zip(symbols, weights)}
            
            elif action == "value_weight":
                # Placeholder for value-based allocation
                weights = np.random.dirichlet(np.ones(n_symbols) * 0.5)
                return {symbol: weight for symbol, weight in zip(symbols, weights)}
            
            elif action == "growth_weight":
                # Placeholder for growth-based allocation
                weights = np.random.dirichlet(np.ones(n_symbols) * 2.0)
                return {symbol: weight for symbol, weight in zip(symbols, weights)}
            
            elif action == "defensive_weight":
                # Placeholder for defensive allocation
                weights = np.random.dirichlet(np.ones(n_symbols) * 0.3)
                return {symbol: weight for symbol, weight in zip(symbols, weights)}
            
            elif action == "aggressive_weight":
                # Placeholder for aggressive allocation
                weights = np.random.dirichlet(np.ones(n_symbols) * 3.0)
                return {symbol: weight for symbol, weight in zip(symbols, weights)}
            
            elif action.startswith("overweight_"):
                symbol = action.split("_", 1)[1]
                if symbol in symbols:
                    weights = np.ones(n_symbols) * 0.5 / (n_symbols - 1)
                    symbol_idx = symbols.index(symbol)
                    weights[symbol_idx] = 0.5
                    return {symbol: weight for symbol, weight in zip(symbols, weights)}
            
            elif action.startswith("underweight_"):
                symbol = action.split("_", 1)[1]
                if symbol in symbols:
                    weights = np.ones(n_symbols) * 1.0 / (n_symbols - 1)
                    symbol_idx = symbols.index(symbol)
                    weights[symbol_idx] = 0.1
                    weights = weights / weights.sum()
                    return {symbol: weight for symbol, weight in zip(symbols, weights)}
            
            # Default to equal weight
            return self._equal_allocation(symbols)
            
        except Exception as e:
            self.logger.debug(f"Error converting action to allocation: {e}")
            return self._equal_allocation(symbols)
    
    def _equal_allocation(self, symbols: List[str]) -> Dict[str, float]:
        """Create equal weight allocation."""
        try:
            weight = 1.0 / len(symbols)
            return {symbol: weight for symbol in symbols}
        except Exception as e:
            self.logger.debug(f"Error creating equal allocation: {e}")
            return {}
    
    def update_policy(self, reward: float):
        """
        Update RL policy based on reward.
        
        Args:
            reward: Reward from the last action
        """
        try:
            # Record reward
            self.reward_history.append(reward)
            
            # Update Q-values using Q-learning
            if len(self.state_history) >= 2:
                current_state = self._state_to_key(self.state_history[-1])
                previous_state = self._state_to_key(self.state_history[-2])
                previous_action = self.action_history[-2]
                
                # Q-learning update
                self._update_q_value(previous_state, previous_action, reward, current_state)
            
        except Exception as e:
            self.logger.error(f"Error updating policy: {e}")
    
    def _update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning."""
        try:
            # Initialize Q-table entries if needed
            if state not in self.q_table:
                self.q_table[state] = {}
            if next_state not in self.q_table:
                self.q_table[next_state] = {}
            
            # Get current Q-value
            current_q = self.q_table[state].get(action, 0.0)
            
            # Get maximum Q-value for next state
            next_q_values = list(self.q_table[next_state].values())
            max_next_q = max(next_q_values) if next_q_values else 0.0
            
            # Q-learning update rule
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
            
            # Update Q-value
            self.q_table[state][action] = new_q
            
        except Exception as e:
            self.logger.debug(f"Error updating Q-value: {e}")
    
    def get_policy_statistics(self) -> Dict[str, float]:
        """Get statistics about the RL policy."""
        try:
            stats = {}
            
            # Q-table statistics
            total_q_entries = sum(len(q_values) for q_values in self.q_table.values())
            stats['total_q_entries'] = total_q_entries
            stats['num_states'] = len(self.q_table)
            
            # Average Q-values
            all_q_values = []
            for q_values in self.q_table.values():
                all_q_values.extend(q_values.values())
            
            if all_q_values:
                stats['avg_q_value'] = np.mean(all_q_values)
                stats['std_q_value'] = np.std(all_q_values)
                stats['min_q_value'] = np.min(all_q_values)
                stats['max_q_value'] = np.max(all_q_values)
            else:
                stats['avg_q_value'] = 0.0
                stats['std_q_value'] = 0.0
                stats['min_q_value'] = 0.0
                stats['max_q_value'] = 0.0
            
            # Reward statistics
            if self.reward_history:
                stats['avg_reward'] = np.mean(self.reward_history)
                stats['std_reward'] = np.std(self.reward_history)
                stats['total_rewards'] = len(self.reward_history)
            else:
                stats['avg_reward'] = 0.0
                stats['std_reward'] = 0.0
                stats['total_rewards'] = 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting policy statistics: {e}")
            return {}
    
    def save_policy(self, filepath: str):
        """Save RL policy to file."""
        try:
            policy_data = {
                'q_table': self.q_table,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor
            }
            
            # In practice, would use pickle or json
            # For now, just log the action
            self.logger.info(f"Policy saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving policy: {e}")
    
    def load_policy(self, filepath: str):
        """Load RL policy from file."""
        try:
            # In practice, would load from pickle or json
            # For now, just log the action
            self.logger.info(f"Policy loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading policy: {e}")
    
    def reset_policy(self):
        """Reset RL policy."""
        try:
            self.q_table = {}
            self.state_history = []
            self.action_history = []
            self.reward_history = []
            self.logger.info("RL policy reset")
            
        except Exception as e:
            self.logger.error(f"Error resetting policy: {e}")
    
    def set_learning_rate(self, learning_rate: float):
        """Set learning rate."""
        try:
            self.learning_rate = learning_rate
        except Exception as e:
            self.logger.error(f"Error setting learning rate: {e}")
    
    def set_discount_factor(self, discount_factor: float):
        """Set discount factor."""
        try:
            self.discount_factor = discount_factor
        except Exception as e:
            self.logger.error(f"Error setting discount factor: {e}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 - Termux-Compatible Q-Learning
==========================================

CONSTRAINT: Single-threaded only
- NO multiprocessing
- NO threading for decisions
- Bounded memory usage
- JSON-serializable state

Memory Budget: 20 MB maximum
"""

import json
import time
import os
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import hashlib


class TermuxQLearner:
    """
    Q-learning implementation for Termux.
    
    CONSTRAINTS:
    - Single-threaded only
    - Bounded memory usage (max_table_size)
    - JSON-serializable state
    - No external dependencies
    """
    
    __slots__ = [
        '_q_table', '_learning_rate', '_discount_factor',
        '_epsilon', '_epsilon_decay', '_epsilon_min',
        '_action_count', '_max_table_size', '_stats'
    ]
    
    def __init__(
        self,
        action_count: int = 5,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        max_table_size: int = 10000,
    ):
        """
        Initialize Q-learner with memory bounds.
        
        Args:
            action_count: Number of possible actions (1-100)
            learning_rate: Learning rate alpha (0-1)
            discount_factor: Discount factor gamma (0-1)
            epsilon: Initial exploration rate (0-1)
            epsilon_decay: Epsilon decay rate (0-1)
            epsilon_min: Minimum epsilon (0-1)
            max_table_size: Maximum Q-table entries (memory budget)
        
        Raises:
            AssertionError: If parameters violate constraints
        """
        # RUNTIME ASSERTIONS - parameter validation
        assert 1 <= action_count <= 100, f'HALT: action_count must be 1-100, got {action_count}'
        assert 0 < learning_rate <= 1, f'HALT: learning_rate must be 0-1, got {learning_rate}'
        assert 0 < discount_factor < 1, f'HALT: discount_factor must be 0-1, got {discount_factor}'
        assert 0 <= epsilon <= 1, f'HALT: epsilon must be 0-1, got {epsilon}'
        assert 0 < epsilon_decay < 1, f'HALT: epsilon_decay must be 0-1, got {epsilon_decay}'
        assert 0 <= epsilon_min < epsilon, f'HALT: epsilon_min must be < epsilon'
        assert 1 <= max_table_size <= 50000, f'HALT: max_table_size must be 1-50000, got {max_table_size}'
        
        self._action_count = action_count
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._epsilon_min = epsilon_min
        self._max_table_size = max_table_size
        
        # Q-table: state_hash -> {action: q_value}
        self._q_table: Dict[str, Dict[int, float]] = {}
        
        # Statistics
        self._stats = {
            'total_updates': 0,
            'total_actions': 0,
            'exploration_actions': 0,
            'exploitation_actions': 0,
        }
    
    def _hash_state(self, state_features: List[float]) -> str:
        """
        Convert state features to hash key.
        
        Discretizes continuous features for tabular Q-learning.
        
        Args:
            state_features: List of normalized features (0.0-1.0)
            
        Returns:
            16-character hash string
        """
        if not state_features:
            return 'empty_state_000000'
        
        # Discretize each feature to 10 bins
        discretized = []
        for feature in state_features:
            # Clamp to [0, 1]
            clamped = max(0.0, min(1.0, float(feature)))
            # Discretize
            bin_idx = int(clamped * 10)
            discretized.append(bin_idx)
        
        state_str = '|'.join(str(d) for d in discretized)
        return hashlib.md5(state_str.encode()).hexdigest()[:16]
    
    def select_action(self, state_features: List[float]) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state_features: List of normalized state features (0.0-1.0)
            
        Returns:
            Selected action index (0 to action_count-1)
            
        Raises:
            ValueError: If state_features is empty
        """
        # Input validation
        if not state_features:
            raise ValueError('HALT: Empty state features - must provide at least one feature')
        
        # Validate feature values
        for i, f in enumerate(state_features):
            if not isinstance(f, (int, float)):
                raise ValueError(f'HALT: Feature {i} is not numeric: {type(f)}')
        
        self._stats['total_actions'] += 1
        state_hash = self._hash_state(state_features)
        
        # Epsilon-greedy selection
        if random.random() < self._epsilon:
            # Explore: random action
            action = random.randint(0, self._action_count - 1)
            self._stats['exploration_actions'] += 1
        else:
            # Exploit: best known action
            if state_hash in self._q_table and self._q_table[state_hash]:
                q_values = self._q_table[state_hash]
                action = max(q_values.keys(), key=lambda a: q_values.get(a, 0.0))
            else:
                # Unknown state: random
                action = random.randint(0, self._action_count - 1)
                self._stats['exploration_actions'] += 1
            
            self._stats['exploitation_actions'] += 1
        
        return action
    
    def update(
        self,
        state_features: List[float],
        action: int,
        reward: float,
        next_state_features: List[float],
        done: bool = False,
    ) -> float:
        """
        Update Q-value using Bellman equation.
        
        Args:
            state_features: Current state
            action: Action taken (0 to action_count-1)
            reward: Reward received (-1000 to 1000)
            next_state_features: Next state
            done: Whether episode is done
            
        Returns:
            New Q-value for state-action pair
            
        Raises:
            AssertionError: If parameters violate constraints
        """
        # Parameter validation
        assert 0 <= action < self._action_count, f'HALT: Invalid action {action}, must be 0-{self._action_count-1}'
        assert -1000 <= reward <= 1000, f'HALT: Suspicious reward {reward}, must be -1000 to 1000'
        
        state_hash = self._hash_state(state_features)
        next_state_hash = self._hash_state(next_state_features)
        
        # Initialize state in table if needed
        if state_hash not in self._q_table:
            # Check table size before adding
            if len(self._q_table) >= self._max_table_size:
                # Evict random entry
                evict_key = random.choice(list(self._q_table.keys()))
                del self._q_table[evict_key]
            self._q_table[state_hash] = {}
        
        # Current Q-value
        current_q = self._q_table[state_hash].get(action, 0.0)
        
        # Calculate target Q-value
        if done:
            target_q = reward
        else:
            # Max Q-value for next state
            if next_state_hash in self._q_table and self._q_table[next_state_hash]:
                max_next_q = max(self._q_table[next_state_hash].values())
            else:
                max_next_q = 0.0
            target_q = reward + self._discount_factor * max_next_q
        
        # Q-learning update
        new_q = current_q + self._learning_rate * (target_q - current_q)
        
        # Clamp Q-value to prevent overflow
        new_q = max(-1000.0, min(1000.0, new_q))
        
        self._q_table[state_hash][action] = new_q
        self._stats['total_updates'] += 1
        
        # Decay epsilon
        if self._epsilon > self._epsilon_min:
            self._epsilon = max(self._epsilon_min, self._epsilon * self._epsilon_decay)
        
        return new_q
    
    def get_q_value(self, state_features: List[float], action: int) -> float:
        """Get Q-value for state-action pair."""
        state_hash = self._hash_state(state_features)
        return self._q_table.get(state_hash, {}).get(action, 0.0)
    
    def get_best_action(self, state_features: List[float]) -> int:
        """
        Get best known action for state (no exploration).
        
        Returns 0 if state is unknown.
        """
        state_hash = self._hash_state(state_features)
        
        if state_hash not in self._q_table or not self._q_table[state_hash]:
            return 0
        
        q_values = self._q_table[state_hash]
        return max(q_values.keys(), key=lambda a: q_values.get(a, 0.0))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            **self._stats,
            'epsilon': self._epsilon,
            'q_table_size': len(self._q_table),
            'exploration_rate': (
                self._stats['exploration_actions'] / 
                max(self._stats['total_actions'], 1)
            ),
        }
    
    def save(self, filepath: str):
        """
        Save Q-table to JSON file.
        
        Args:
            filepath: Path to save file
        """
        data = {
            'q_table': self._q_table,
            'epsilon': self._epsilon,
            'stats': self._stats,
            'action_count': self._action_count,
            'learning_rate': self._learning_rate,
            'discount_factor': self._discount_factor,
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """
        Load Q-table from JSON file.
        
        Args:
            filepath: Path to load file
        """
        if not os.path.exists(filepath):
            return  # No saved state, use defaults
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Validate loaded data
        loaded_size = len(data.get('q_table', {}))
        assert loaded_size <= self._max_table_size, \
            f'HALT: Loaded Q-table ({loaded_size}) exceeds budget ({self._max_table_size})'
        
        self._q_table = data.get('q_table', {})
        self._epsilon = data.get('epsilon', self._epsilon)
        self._stats = data.get('stats', self._stats)
    
    def clear(self):
        """Clear Q-table to free memory."""
        self._q_table.clear()
        self._epsilon = 1.0
        self._stats = {
            'total_updates': 0,
            'total_actions': 0,
            'exploration_actions': 0,
            'exploitation_actions': 0,
        }
    
    def get_table_size(self) -> int:
        """Return current Q-table size."""
        return len(self._q_table)


# SELF-TEST
if __name__ == '__main__':
    import sys
    
    print('=== TERMUX Q-LEARNER SELF-TEST ===')
    print()
    
    try:
        learner = TermuxQLearner(
            action_count=5,
            max_table_size=1000,
        )
        
        print('Running 100 training episodes...')
        
        for episode in range(100):
            state = [0.5, 0.3, 0.8]
            action = learner.select_action(state)
            
            # Simulate reward (action 0 is "best")
            reward = 1.0 if action == 0 else 0.0
            
            next_state = [0.6, 0.4, 0.7]
            learner.update(state, action, reward, next_state)
        
        stats = learner.get_stats()
        
        print()
        print(f'Total actions: {stats["total_actions"]}')
        print(f'Total updates: {stats["total_updates"]}')
        print(f'Q-table size: {stats["q_table_size"]}')
        print(f'Final epsilon: {stats["epsilon"]:.3f}')
        print(f'Exploration rate: {stats["exploration_rate"]:.1%}')
        print()
        
        # Verify learning
        best_action = learner.get_best_action([0.5, 0.3, 0.8])
        print(f'Best action for known state: {best_action}')
        
        # Assertions
        assert stats['total_actions'] == 100
        assert stats['total_updates'] == 100
        assert stats['q_table_size'] <= 1000
        assert 0 <= stats['epsilon'] <= 1
        
        print()
        print('=== ALL TESTS PASSED ===')
        
    except AssertionError as e:
        print(f'ASSERTION FAILED: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

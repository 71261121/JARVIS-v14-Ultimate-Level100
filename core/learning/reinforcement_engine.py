#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Reinforcement Learning Engine
====================================================

Phase 3: Learn from decision outcomes using RL principles.

This module implements a simplified reinforcement learning system:
- State representation
- Action selection
- Reward calculation
- Policy improvement
- Experience replay

Key Features:
- No heavy ML frameworks (Termux compatible)
- Q-learning based approach
- Experience replay for learning
- Adaptive exploration

Author: JARVIS AI Project
Version: 3.0.0
Target Level: 60-70
"""

import time
import json
import logging
import threading
import math
import random
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class ActionType(Enum):
    """Types of actions the system can take"""
    CODE_GENERATE = auto()
    CODE_MODIFY = auto()
    CODE_REFACTOR = auto()
    BUG_FIX = auto()
    TEST_RUN = auto()
    BACKUP_CREATE = auto()
    ROLLBACK = auto()
    ANALYZE = auto()
    WAIT = auto()


class RewardType(Enum):
    """Types of rewards"""
    POSITIVE = auto()    # Good outcome
    NEGATIVE = auto()    # Bad outcome
    NEUTRAL = auto()     # No significant impact


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class State:
    """
    System state for decision making.
    
    Represents the current situation/context.
    """
    # Health metrics
    health_score: float = 100.0
    error_count: int = 0
    recent_failures: int = 0
    
    # Resource state
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    # Goal state
    active_goals: int = 0
    goal_progress: float = 0.0
    
    # Code state
    code_complexity: float = 0.0
    test_coverage: float = 0.0
    issues_found: int = 0
    
    # Time context
    time_since_last_modification: float = 0.0
    
    def to_feature_vector(self) -> List[float]:
        """Convert state to feature vector for learning"""
        return [
            self.health_score / 100.0,
            min(self.error_count / 10.0, 1.0),
            min(self.recent_failures / 5.0, 1.0),
            self.memory_usage / 100.0,
            self.cpu_usage / 100.0,
            min(self.active_goals / 5.0, 1.0),
            self.goal_progress / 100.0,
            min(self.code_complexity / 50.0, 1.0),
            self.test_coverage / 100.0,
            min(self.issues_found / 10.0, 1.0),
            min(self.time_since_last_modification / 3600.0, 1.0),
        ]
    
    def get_hash(self) -> str:
        """Get hash of state for Q-table lookup"""
        features = self.to_feature_vector()
        # Discretize for hashing
        discrete = [int(f * 10) for f in features]
        return hashlib.md5(str(discrete).encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'health_score': self.health_score,
            'error_count': self.error_count,
            'recent_failures': self.recent_failures,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'active_goals': self.active_goals,
            'goal_progress': self.goal_progress,
            'code_complexity': self.code_complexity,
            'test_coverage': self.test_coverage,
            'issues_found': self.issues_found,
        }


@dataclass
class Action:
    """
    An action the system can take.
    """
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    # Context
    target_file: Optional[str] = None
    target_function: Optional[str] = None
    
    def get_id(self) -> str:
        """Get action identifier"""
        return f"{self.action_type.name}_{hashlib.md5(str(self.parameters).encode()).hexdigest()[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.action_type.name,
            'parameters': self.parameters,
            'description': self.description,
            'target_file': self.target_file,
        }


@dataclass
class Reward:
    """
    Reward signal from an action.
    """
    value: float = 0.0
    reward_type: RewardType = RewardType.NEUTRAL
    
    # Components
    goal_progress_component: float = 0.0
    health_component: float = 0.0
    error_reduction_component: float = 0.0
    quality_improvement_component: float = 0.0
    
    # Metadata
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    
    @property
    def is_positive(self) -> bool:
        return self.value > 0
    
    @property
    def is_negative(self) -> bool:
        return self.value < 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'value': self.value,
            'type': self.reward_type.name,
            'components': {
                'goal_progress': self.goal_progress_component,
                'health': self.health_component,
                'error_reduction': self.error_reduction_component,
                'quality_improvement': self.quality_improvement_component,
            },
            'description': self.description,
        }


@dataclass
class Experience:
    """
    A single learning experience.
    
    Records state, action, reward, and next state for learning.
    """
    id: str = field(default_factory=lambda: f"exp_{int(time.time() * 1000)}")
    
    # The experience tuple
    state: Optional[State] = None
    action: Optional[Action] = None
    reward: Optional[Reward] = None
    next_state: Optional[State] = None
    
    # Timing
    timestamp: float = field(default_factory=time.time)
    
    # Outcome
    success: bool = False
    outcome_description: str = ""
    
    # Learning
    used_for_learning: bool = False
    learning_weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'state': self.state.to_dict() if self.state else None,
            'action': self.action.to_dict() if self.action else None,
            'reward': self.reward.to_dict() if self.reward else None,
            'success': self.success,
            'outcome': self.outcome_description,
            'timestamp': self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REINFORCEMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ReinforcementEngine:
    """
    Reinforcement learning engine for JARVIS.
    
    Uses Q-learning approach:
    - Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
    
    Features:
    - Experience replay for stable learning
    - Adaptive exploration (epsilon-greedy)
    - State discretization for manageable Q-table
    - No heavy ML dependencies
    
    Usage:
        engine = ReinforcementEngine()
        
        # Record experience
        engine.record_experience(state, action, reward, next_state)
        
        # Get best action
        best_action = engine.select_action(current_state)
        
        # Learn from experiences
        engine.learn()
    """
    
    # Learning parameters
    DEFAULT_ALPHA = 0.1      # Learning rate
    DEFAULT_GAMMA = 0.95     # Discount factor
    DEFAULT_EPSILON = 0.2    # Exploration rate
    
    # Experience replay
    REPLAY_BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    
    def __init__(
        self,
        learning_rate: float = None,
        discount_factor: float = None,
        exploration_rate: float = None,
        storage_path: str = None,
    ):
        """
        Initialize reinforcement engine.
        
        Args:
            learning_rate: Alpha parameter
            discount_factor: Gamma parameter
            exploration_rate: Epsilon for exploration
            storage_path: Path to save/load Q-table
        """
        self._alpha = learning_rate or self.DEFAULT_ALPHA
        self._gamma = discount_factor or self.DEFAULT_GAMMA
        self._epsilon = exploration_rate or self.DEFAULT_EPSILON
        self._storage_path = storage_path
        
        # Q-table: state_hash -> {action_id: q_value}
        self._q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self._experience_buffer: deque = deque(maxlen=self.REPLAY_BUFFER_SIZE)
        
        # Action statistics
        self._action_counts: Dict[str, int] = defaultdict(int)
        self._action_success: Dict[str, int] = defaultdict(int)
        
        # State visit counts
        self._state_visits: Dict[str, int] = defaultdict(int)
        
        # Learning statistics
        self._stats = {
            'experiences_recorded': 0,
            'learning_iterations': 0,
            'total_reward': 0.0,
            'states_visited': 0,
            'actions_taken': 0,
        }
        
        self._lock = threading.RLock()
        
        # Load existing data
        if storage_path:
            self._load()
        
        logger.info("ReinforcementEngine initialized")
    
    def record_experience(
        self,
        state: State,
        action: Action,
        reward: Reward,
        next_state: State,
        success: bool = False,
        outcome: str = "",
    ) -> Experience:
        """
        Record an experience for learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            success: Whether action succeeded
            outcome: Description of outcome
            
        Returns:
            Recorded Experience
        """
        with self._lock:
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                success=success,
                outcome_description=outcome,
            )
            
            # Add to replay buffer
            self._experience_buffer.append(experience)
            
            # Update statistics
            self._stats['experiences_recorded'] += 1
            self._stats['total_reward'] += reward.value
            
            # Track action statistics
            action_id = action.get_id()
            self._action_counts[action_id] += 1
            if success:
                self._action_success[action_id] += 1
            
            # Track state visits
            state_hash = state.get_hash()
            self._state_visits[state_hash] += 1
            
            # Immediate Q-value update (online learning)
            self._update_q_value(experience)
            
            return experience
    
    def select_action(
        self,
        state: State,
        available_actions: List[Action] = None,
        explore: bool = True,
    ) -> Optional[Action]:
        """
        Select best action for state.
        
        Uses epsilon-greedy policy:
        - With probability epsilon: explore (random action)
        - Otherwise: exploit (best known action)
        
        Args:
            state: Current state
            available_actions: Actions to choose from
            explore: Whether to allow exploration
            
        Returns:
            Selected Action or None
        """
        if not available_actions:
            return None
        
        with self._lock:
            state_hash = state.get_hash()
            
            # Exploration
            if explore and random.random() < self._epsilon:
                action = random.choice(available_actions)
                logger.debug(f"Exploring: selected {action.action_type.name}")
                return action
            
            # Exploitation: select best action
            best_action = None
            best_q = float('-inf')
            
            for action in available_actions:
                action_id = action.get_id()
                q_value = self._q_table[state_hash][action_id]
                
                # Add exploration bonus for less-visited actions
                visits = self._action_counts[action_id]
                if visits > 0:
                    bonus = math.sqrt(2 * math.log(self._stats['actions_taken'] + 1) / visits)
                    q_value += bonus * 0.1
                
                if q_value > best_q:
                    best_q = q_value
                    best_action = action
            
            self._stats['actions_taken'] += 1
            
            if best_action:
                logger.debug(f"Exploiting: selected {best_action.action_type.name} (Q={best_q:.2f})")
            
            return best_action
    
    def learn(self, batch_size: int = None) -> float:
        """
        Learn from experience replay.
        
        Performs Q-learning updates on a batch of experiences.
        
        Args:
            batch_size: Number of experiences to learn from
            
        Returns:
            Average TD error
        """
        batch_size = batch_size or self.BATCH_SIZE
        
        with self._lock:
            if len(self._experience_buffer) < batch_size:
                return 0.0
            
            # Sample batch
            batch = random.sample(list(self._experience_buffer), batch_size)
            
            total_td_error = 0.0
            
            for experience in batch:
                td_error = self._update_q_value(experience)
                total_td_error += abs(td_error)
                experience.used_for_learning = True
            
            self._stats['learning_iterations'] += 1
            
            avg_td_error = total_td_error / batch_size
            logger.debug(f"Learning iteration: avg TD error = {avg_td_error:.4f}")
            
            return avg_td_error
    
    def _update_q_value(self, experience: Experience) -> float:
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        """
        if not experience.state or not experience.action or not experience.reward:
            return 0.0
        
        state_hash = experience.state.get_hash()
        action_id = experience.action.get_id()
        
        # Current Q-value
        current_q = self._q_table[state_hash][action_id]
        
        # Calculate target Q-value
        if experience.next_state:
            next_state_hash = experience.next_state.get_hash()
            # Max Q for next state
            max_next_q = max(self._q_table[next_state_hash].values()) if self._q_table[next_state_hash] else 0.0
            target_q = experience.reward.value + self._gamma * max_next_q
        else:
            target_q = experience.reward.value
        
        # TD error
        td_error = target_q - current_q
        
        # Update Q-value
        new_q = current_q + self._alpha * td_error
        self._q_table[state_hash][action_id] = new_q
        
        return td_error
    
    def get_q_value(self, state: State, action: Action) -> float:
        """Get Q-value for state-action pair"""
        state_hash = state.get_hash()
        action_id = action.get_id()
        return self._q_table[state_hash][action_id]
    
    def get_best_q_value(self, state: State) -> Tuple[str, float]:
        """Get best action and Q-value for state"""
        state_hash = state.get_hash()
        
        if not self._q_table[state_hash]:
            return ("", 0.0)
        
        best_action = max(self._q_table[state_hash].items(), key=lambda x: x[1])
        return best_action
    
    def get_action_success_rate(self, action_type: ActionType) -> float:
        """Get success rate for action type"""
        total = 0
        success = 0
        
        for action_id, count in self._action_counts.items():
            if action_id.startswith(action_type.name):
                total += count
                success += self._action_success.get(action_id, 0)
        
        if total == 0:
            return 0.5  # Default
        
        return success / total
    
    def decay_exploration(self, decay_rate: float = 0.995, min_epsilon: float = 0.05):
        """Decay exploration rate"""
        self._epsilon = max(min_epsilon, self._epsilon * decay_rate)
        logger.debug(f"Exploration rate: {self._epsilon:.3f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['q_table_size'] = len(self._q_table)
            stats['experience_buffer_size'] = len(self._experience_buffer)
            stats['epsilon'] = self._epsilon
            stats['unique_actions'] = len(self._action_counts)
            stats['unique_states'] = len(self._state_visits)
            return stats
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get summary of learned policy"""
        with self._lock:
            # Best action for each known state
            policy = {}
            for state_hash, actions in self._q_table.items():
                if actions:
                    best_action = max(actions.items(), key=lambda x: x[1])
                    policy[state_hash] = {
                        'best_action': best_action[0],
                        'q_value': best_action[1],
                    }
            
            return {
                'states_with_policy': len(policy),
                'policy': dict(list(policy.items())[:10]),  # First 10
            }
    
    def _save(self):
        """Save Q-table to storage"""
        if not self._storage_path:
            return
        
        try:
            data = {
                'q_table': {k: dict(v) for k, v in self._q_table.items()},
                'stats': self._stats,
                'action_counts': dict(self._action_counts),
                'action_success': dict(self._action_success),
                'epsilon': self._epsilon,
            }
            
            path = Path(self._storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save Q-table: {e}")
    
    def _load(self):
        """Load Q-table from storage"""
        if not self._storage_path:
            return
        
        try:
            path = Path(self._storage_path)
            if not path.exists():
                return
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            self._q_table = defaultdict(lambda: defaultdict(float))
            for state_hash, actions in data.get('q_table', {}).items():
                for action_id, q_value in actions.items():
                    self._q_table[state_hash][action_id] = q_value
            
            self._stats.update(data.get('stats', {}))
            self._action_counts.update(data.get('action_counts', {}))
            self._action_success.update(data.get('action_success', {}))
            self._epsilon = data.get('epsilon', self.DEFAULT_EPSILON)
            
            logger.info(f"Loaded Q-table with {len(self._q_table)} states")
            
        except Exception as e:
            logger.error(f"Failed to load Q-table: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[ReinforcementEngine] = None


def get_reinforcement_engine() -> ReinforcementEngine:
    """Get global reinforcement engine"""
    global _engine
    if _engine is None:
        _engine = ReinforcementEngine()
    return _engine


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Reinforcement Engine Test")
    print("="*60)
    
    engine = ReinforcementEngine()
    
    # Create experiences
    for i in range(10):
        state = State(
            health_score=80 + i * 2,
            error_count=max(0, 5 - i),
        )
        
        action = Action(
            action_type=ActionType.CODE_MODIFY,
            description=f"Fix bug {i}",
        )
        
        reward = Reward(
            value=1.0 if i > 5 else -0.5,
            reward_type=RewardType.POSITIVE if i > 5 else RewardType.NEGATIVE,
        )
        
        next_state = State(
            health_score=state.health_score + (5 if i > 5 else -5),
            error_count=max(0, state.error_count - 1),
        )
        
        engine.record_experience(state, action, reward, next_state, success=i > 5)
    
    print(f"\nRecorded {engine.get_stats()['experiences_recorded']} experiences")
    
    # Learn
    td_error = engine.learn()
    print(f"Learning TD error: {td_error:.4f}")
    
    # Select action
    test_state = State(health_score=90, error_count=1)
    available = [
        Action(ActionType.CODE_MODIFY, description="Fix bug"),
        Action(ActionType.CODE_REFACTOR, description="Refactor"),
    ]
    
    selected = engine.select_action(test_state, available)
    if selected:
        print(f"\nSelected action: {selected.action_type.name}")
        print(f"Q-value: {engine.get_q_value(test_state, selected):.3f}")
    
    # Stats
    stats = engine.get_stats()
    print(f"\nStats:")
    print(f"  Q-table size: {stats['q_table_size']}")
    print(f"  Epsilon: {stats['epsilon']:.3f}")
    print(f"  Total reward: {stats['total_reward']:.2f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

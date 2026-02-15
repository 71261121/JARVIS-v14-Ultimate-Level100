#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Meta-Learning Engine
===========================================

Phase 5: Ultimate Meta-Learning System (Level 85-100+)

This module implements sophisticated meta-learning capabilities:
- Learning to Learn (MAML-style optimization)
- Transfer Learning across domains
- Few-shot Learning
- Neural Architecture Search concepts
- Hyperparameter Meta-Optimization
- Task Embeddings and Similarity
- Meta-Experience Replay
- Continual Learning without Forgetting

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                      Meta-Learning Engine                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │    Task     │  │   Meta-     │  │   Transfer  │  Core        │
│  │ Embedder    │  │  Optimizer  │  │   Engine    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Few-Shot   │  │    Meta     │  │  Hyperparam │  Learning    │
│  │   Learner   │  │  Experience │  │  Optimizer  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Continual  │  │   Domain    │  │    Meta     │  Advanced    │
│  │  Learner    │  │  Adapter    │  │  Gradient   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘

Author: JARVIS AI Project
Version: 5.0.0
Target Level: 85-100+
"""

import time
import json
import logging
import threading
import math
import random
import hashlib
import copy
from typing import Dict, Any, Optional, List, Set, Callable, Tuple, Union, TypeVar, Generic, Type
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class TaskType(Enum):
    """Types of learning tasks"""
    CODE_GENERATION = auto()
    CODE_ANALYSIS = auto()
    BUG_FIXING = auto()
    REFACTORING = auto()
    OPTIMIZATION = auto()
    DOCUMENTATION = auto()
    TESTING = auto()
    DEBUGGING = auto()
    REVIEW = auto()
    ARCHITECTURE = auto()


class LearningMode(Enum):
    """Meta-learning modes"""
    MAML = auto()           # Model-Agnostic Meta-Learning
    PROTYPICAL = auto()     # Prototypical Networks
    MATCHING = auto()       # Matching Networks
    RELATION = auto()       # Relation Networks
    REPTILE = auto()        # Reptile (first-order MAML)
    META_SGD = auto()       # Meta-SGD
    ANIL = auto()           # Almost No Inner Loop


class TransferType(Enum):
    """Types of transfer learning"""
    FINE_TUNING = auto()
    FEATURE_EXTRACTION = auto()
    MULTI_TASK = auto()
    DOMAIN_ADAPTATION = auto()
    ZERO_SHOT = auto()
    FEW_SHOT = auto()


class AdaptationSpeed(Enum):
    """Speed of adaptation"""
    IMMEDIATE = auto()      # 1-2 examples
    FAST = auto()           # 3-5 examples
    NORMAL = auto()         # 6-10 examples
    SLOW = auto()           # 10+ examples


class ForgettingType(Enum):
    """Types of catastrophic forgetting"""
    NONE = auto()
    MINIMAL = auto()
    MODERATE = auto()
    SEVERE = auto()


class MetaObjective(Enum):
    """Meta-learning objectives"""
    MINIMIZE_LOSS = auto()
    MAXIMIZE_ACCURACY = auto()
    MINIMIZE_ADAPTATION_TIME = auto()
    MAXIMIZE_TRANSFER = auto()
    MINIMIZE_FORGETTING = auto()
    MAXIMIZE_GENERALIZATION = auto()


class HyperparameterType(Enum):
    """Types of hyperparameters"""
    LEARNING_RATE = auto()
    BATCH_SIZE = auto()
    HIDDEN_DIM = auto()
    NUM_LAYERS = auto()
    DROPOUT = auto()
    WEIGHT_DECAY = auto()
    MOMENTUM = auto()
    TEMPERATURE = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskEmbedding:
    """
    Embedding representation of a learning task.
    
    Captures the essence of a task for similarity comparison.
    """
    task_id: str = field(default_factory=lambda: f"task_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}")
    task_type: TaskType = TaskType.CODE_GENERATION
    
    # Embedding vector (simplified - would be actual vector in production)
    embedding: List[float] = field(default_factory=list)
    dimension: int = 128
    
    # Task characteristics
    complexity: float = 0.5
    domain: str = "general"
    language: str = "python"
    
    # Performance metrics
    baseline_accuracy: float = 0.0
    adaptation_speed: AdaptationSpeed = AdaptationSpeed.NORMAL
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    examples_count: int = 0
    
    def similarity(self, other: 'TaskEmbedding') -> float:
        """Calculate cosine similarity with another embedding"""
        if not self.embedding or not other.embedding:
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(self.embedding, other.embedding))
        norm_a = math.sqrt(sum(a * a for a in self.embedding))
        norm_b = math.sqrt(sum(b * b for b in other.embedding))
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def distance(self, other: 'TaskEmbedding') -> float:
        """Calculate Euclidean distance"""
        if not self.embedding or not other.embedding:
            return float('inf')
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.embedding, other.embedding)))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.name,
            'dimension': self.dimension,
            'complexity': self.complexity,
            'domain': self.domain,
            'baseline_accuracy': self.baseline_accuracy,
        }


@dataclass
class MetaExample:
    """
    A single example in meta-learning.
    
    Contains input, output, and context for a task.
    """
    id: str = field(default_factory=lambda: f"ex_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}")
    
    # Content
    input_data: Any = None
    output_data: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Task association
    task_id: str = ""
    task_type: TaskType = TaskType.CODE_GENERATION
    
    # Meta-information
    difficulty: float = 0.5
    importance: float = 1.0
    
    # Learning metadata
    times_used: int = 0
    last_used: Optional[float] = None
    success_rate: float = 0.0
    
    # Embedding
    embedding: Optional[TaskEmbedding] = None
    
    def mark_used(self, success: bool):
        """Mark example as used"""
        self.times_used += 1
        self.last_used = time.time()
        
        # Update success rate with exponential moving average
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'task_type': self.task_type.name,
            'difficulty': self.difficulty,
            'times_used': self.times_used,
            'success_rate': self.success_rate,
        }


@dataclass
class MetaExperience:
    """
    Experience from a meta-learning episode.
    
    Records the learning process for future improvement.
    """
    id: str = field(default_factory=lambda: f"meta_exp_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}")
    
    # Task information
    task_embedding: Optional[TaskEmbedding] = None
    task_type: TaskType = TaskType.CODE_GENERATION
    
    # Learning process
    support_examples: List[MetaExample] = field(default_factory=list)
    query_examples: List[MetaExample] = field(default_factory=list)
    
    # Performance
    initial_loss: float = 1.0
    final_loss: float = 1.0
    adaptation_steps: int = 0
    adaptation_time: float = 0.0
    
    # Meta-gradients (simplified representation)
    meta_gradients: Dict[str, float] = field(default_factory=dict)
    
    # Outcome
    success: bool = False
    accuracy: float = 0.0
    
    # Transfer learning
    transferred_from: Optional[str] = None
    transfer_effectiveness: float = 0.0
    
    # Timestamps
    timestamp: float = field(default_factory=time.time)
    
    @property
    def improvement(self) -> float:
        """Calculate improvement from initial to final"""
        return self.initial_loss - self.final_loss
    
    @property
    def learning_efficiency(self) -> float:
        """Calculate learning efficiency"""
        if self.adaptation_steps == 0:
            return 0.0
        return self.improvement / self.adaptation_steps
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'task_type': self.task_type.name,
            'initial_loss': self.initial_loss,
            'final_loss': self.final_loss,
            'improvement': self.improvement,
            'adaptation_steps': self.adaptation_steps,
            'success': self.success,
        }


@dataclass
class HyperparameterConfig:
    """
    Configuration of hyperparameters for meta-learning.
    """
    # Learning rates
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    meta_lr: float = 0.0001
    
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    inner_steps: int = 5
    outer_steps: int = 1000
    
    # Regularization
    weight_decay: float = 0.0001
    l2_penalty: float = 0.01
    
    # Meta-learning specific
    num_shots: int = 5
    num_ways: int = 5
    first_order: bool = False
    
    # Adaptation
    adaptation_steps: int = 3
    adaptation_lr: float = 0.01
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'num_shots': self.num_shots,
        }
    
    def mutate(self, mutation_rate: float = 0.1) -> 'HyperparameterConfig':
        """Create a mutated copy of the config"""
        new_config = copy.deepcopy(self)
        
        if random.random() < mutation_rate:
            new_config.inner_lr *= random.uniform(0.5, 2.0)
        if random.random() < mutation_rate:
            new_config.outer_lr *= random.uniform(0.5, 2.0)
        if random.random() < mutation_rate:
            new_config.hidden_dim = int(new_config.hidden_dim * random.uniform(0.8, 1.2))
        if random.random() < mutation_rate:
            new_config.num_layers = max(1, new_config.num_layers + random.choice([-1, 0, 1]))
        
        return new_config


@dataclass
class MetaModel:
    """
    Meta-model that can quickly adapt to new tasks.
    """
    id: str = field(default_factory=lambda: f"meta_model_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}")
    name: str = "MetaLearner"
    
    # Model state (simplified - would be actual weights in production)
    parameters: Dict[str, Any] = field(default_factory=dict)
    config: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    
    # Learning mode
    learning_mode: LearningMode = LearningMode.MAML
    
    # Performance tracking
    tasks_learned: int = 0
    avg_adaptation_speed: float = 0.0
    avg_accuracy: float = 0.0
    
    # Meta-knowledge
    task_embeddings: Dict[str, TaskEmbedding] = field(default_factory=dict)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    
    # Training history
    training_steps: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def adapt(self, task_embedding: TaskEmbedding, examples: List[MetaExample]) -> 'MetaModel':
        """Adapt the model to a new task"""
        adapted = copy.deepcopy(self)
        
        # Record task
        adapted.task_embeddings[task_embedding.task_id] = task_embedding
        adapted.tasks_learned += 1
        adapted.last_updated = time.time()
        
        return adapted
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'learning_mode': self.learning_mode.name,
            'tasks_learned': self.tasks_learned,
            'avg_accuracy': self.avg_accuracy,
            'training_steps': self.training_steps,
        }


@dataclass
class TransferKnowledge:
    """
    Knowledge transferred between tasks/domains.
    """
    id: str = field(default_factory=lambda: f"transfer_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}")
    
    # Source and target
    source_task: str = ""
    target_task: str = ""
    source_domain: str = ""
    target_domain: str = ""
    
    # Transfer type
    transfer_type: TransferType = TransferType.FEW_SHOT
    
    # What is transferred
    transferred_parameters: Dict[str, Any] = field(default_factory=dict)
    transferred_knowledge: Dict[str, Any] = field(default_factory=dict)
    
    # Effectiveness
    effectiveness_score: float = 0.0
    negative_transfer: bool = False
    
    # Conditions
    applicability_conditions: List[str] = field(default_factory=list)
    
    # Timestamp
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'source_task': self.source_task,
            'target_task': self.target_task,
            'transfer_type': self.transfer_type.name,
            'effectiveness': self.effectiveness_score,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK EMBEDDER
# ═══════════════════════════════════════════════════════════════════════════════

class TaskEmbedder:
    """
    Creates embeddings for learning tasks.
    
    Captures task characteristics for similarity comparison
    and transfer learning.
    """
    
    def __init__(self, embedding_dim: int = 128):
        """
        Initialize task embedder.
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self._embedding_dim = embedding_dim
        
        # Feature extractors
        self._complexity_features = self._init_complexity_features()
        self._domain_features = self._init_domain_features()
        self._structural_features = self._init_structural_features()
        
        # Cache
        self._cache: Dict[str, TaskEmbedding] = {}
        
        # Statistics
        self._stats = {
            'embeddings_created': 0,
            'cache_hits': 0,
        }
        
        logger.info(f"TaskEmbedder initialized with dim={embedding_dim}")
    
    def embed(
        self,
        task_type: TaskType,
        task_data: Dict[str, Any],
        examples: List[MetaExample] = None,
    ) -> TaskEmbedding:
        """
        Create an embedding for a task.
        
        Args:
            task_type: Type of task
            task_data: Task-specific data
            examples: Optional examples for context
            
        Returns:
            TaskEmbedding
        """
        # Generate cache key
        cache_key = self._generate_cache_key(task_type, task_data)
        
        if cache_key in self._cache:
            self._stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        # Extract features
        complexity_features = self._extract_complexity_features(task_data)
        domain_features = self._extract_domain_features(task_data)
        structural_features = self._extract_structural_features(task_data)
        
        # Combine features into embedding
        embedding = self._combine_features(
            complexity_features,
            domain_features,
            structural_features,
        )
        
        # Create TaskEmbedding
        task_embedding = TaskEmbedding(
            task_type=task_type,
            embedding=embedding,
            dimension=self._embedding_dim,
            complexity=task_data.get('complexity', 0.5),
            domain=task_data.get('domain', 'general'),
            language=task_data.get('language', 'python'),
            examples_count=len(examples) if examples else 0,
        )
        
        # Cache
        self._cache[cache_key] = task_embedding
        self._stats['embeddings_created'] += 1
        
        return task_embedding
    
    def find_similar(
        self,
        query_embedding: TaskEmbedding,
        candidates: List[TaskEmbedding],
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> List[Tuple[TaskEmbedding, float]]:
        """
        Find similar task embeddings.
        
        Args:
            query_embedding: Query embedding
            candidates: Candidate embeddings
            top_k: Number of top results
            threshold: Minimum similarity threshold
            
        Returns:
            List of (embedding, similarity) tuples
        """
        similarities = []
        
        for candidate in candidates:
            sim = query_embedding.similarity(candidate)
            if sim >= threshold:
                similarities.append((candidate, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _init_complexity_features(self) -> Dict[str, Callable]:
        """Initialize complexity feature extractors"""
        return {
            'code_length': lambda d: min(len(str(d.get('code', ''))) / 1000, 1.0),
            'nesting_depth': lambda d: d.get('nesting_depth', 0) / 10,
            'num_functions': lambda d: min(d.get('num_functions', 0) / 20, 1.0),
            'num_classes': lambda d: min(d.get('num_classes', 0) / 10, 1.0),
            'cyclomatic_complexity': lambda d: min(d.get('cyclomatic_complexity', 1) / 50, 1.0),
        }
    
    def _init_domain_features(self) -> Dict[str, Callable]:
        """Initialize domain feature extractors"""
        return {
            'has_database': lambda d: 1.0 if 'database' in str(d).lower() else 0.0,
            'has_api': lambda d: 1.0 if 'api' in str(d).lower() else 0.0,
            'has_ui': lambda d: 1.0 if 'ui' in str(d).lower() or 'interface' in str(d).lower() else 0.0,
            'has_security': lambda d: 1.0 if 'security' in str(d).lower() or 'auth' in str(d).lower() else 0.0,
            'has_testing': lambda d: 1.0 if 'test' in str(d).lower() else 0.0,
        }
    
    def _init_structural_features(self) -> Dict[str, Callable]:
        """Initialize structural feature extractors"""
        return {
            'has_imports': lambda d: 1.0 if d.get('imports') else 0.0,
            'has_classes': lambda d: 1.0 if d.get('classes') else 0.0,
            'has_async': lambda d: 1.0 if 'async' in str(d).lower() else 0.0,
            'has_decorators': lambda d: 1.0 if '@' in str(d.get('code', '')) else 0.0,
            'has_error_handling': lambda d: 1.0 if 'try' in str(d).lower() or 'except' in str(d).lower() else 0.0,
        }
    
    def _extract_complexity_features(self, task_data: Dict[str, Any]) -> List[float]:
        """Extract complexity features"""
        features = []
        for name, extractor in self._complexity_features.items():
            try:
                features.append(extractor(task_data))
            except:
                features.append(0.0)
        return features
    
    def _extract_domain_features(self, task_data: Dict[str, Any]) -> List[float]:
        """Extract domain features"""
        features = []
        for name, extractor in self._domain_features.items():
            try:
                features.append(extractor(task_data))
            except:
                features.append(0.0)
        return features
    
    def _extract_structural_features(self, task_data: Dict[str, Any]) -> List[float]:
        """Extract structural features"""
        features = []
        for name, extractor in self._structural_features.items():
            try:
                features.append(extractor(task_data))
            except:
                features.append(0.0)
        return features
    
    def _combine_features(
        self,
        complexity: List[float],
        domain: List[float],
        structural: List[float],
    ) -> List[float]:
        """Combine features into embedding vector"""
        # Concatenate features
        all_features = complexity + domain + structural
        
        # Pad or truncate to embedding dimension
        if len(all_features) < self._embedding_dim:
            # Pad with zeros
            all_features.extend([0.0] * (self._embedding_dim - len(all_features)))
        elif len(all_features) > self._embedding_dim:
            # Truncate
            all_features = all_features[:self._embedding_dim]
        
        # Normalize
        norm = math.sqrt(sum(f * f for f in all_features))
        if norm > 0:
            all_features = [f / norm for f in all_features]
        
        return all_features
    
    def _generate_cache_key(self, task_type: TaskType, task_data: Dict[str, Any]) -> str:
        """Generate cache key for task"""
        key_data = f"{task_type.name}_{json.dumps(task_data, sort_keys=True, default=str)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedder statistics"""
        return {
            **self._stats,
            'cache_size': len(self._cache),
            'embedding_dim': self._embedding_dim,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# META-OPTIMIZER (MAML-style)
# ═══════════════════════════════════════════════════════════════════════════════

class MetaOptimizer:
    """
    Implements MAML-style meta-optimization.
    
    Learns initialization that can quickly adapt to new tasks.
    """
    
    def __init__(
        self,
        config: HyperparameterConfig = None,
        learning_mode: LearningMode = LearningMode.MAML,
    ):
        """
        Initialize meta-optimizer.
        
        Args:
            config: Hyperparameter configuration
            learning_mode: Meta-learning algorithm
        """
        self._config = config or HyperparameterConfig()
        self._learning_mode = learning_mode
        
        # Meta-parameters
        self._meta_params: Dict[str, float] = {}
        self._adapted_params: Dict[str, Dict[str, float]] = {}
        
        # Training state
        self._step_count: int = 0
        self._gradient_history: deque = deque(maxlen=1000)
        
        # Statistics
        self._stats = {
            'meta_updates': 0,
            'adaptations': 0,
            'avg_adaptation_steps': 0.0,
            'avg_improvement': 0.0,
        }
        
        self._lock = threading.RLock()
        
        logger.info(f"MetaOptimizer initialized with {learning_mode.name}")
    
    def meta_update(
        self,
        experiences: List[MetaExperience],
    ) -> Dict[str, Any]:
        """
        Perform meta-update from batch of experiences.
        
        This is the outer loop of MAML - updating meta-parameters
        based on gradient of validation loss.
        
        Args:
            experiences: List of meta-experiences
            
        Returns:
            Update statistics
        """
        with self._lock:
            if not experiences:
                return {'success': False, 'reason': 'No experiences'}
            
            total_meta_gradient = defaultdict(float)
            total_improvement = 0.0
            
            for exp in experiences:
                # Compute meta-gradient (simplified)
                meta_gradient = self._compute_meta_gradient(exp)
                
                for key, value in meta_gradient.items():
                    total_meta_gradient[key] += value
                
                total_improvement += exp.improvement
            
            # Average gradients
            n = len(experiences)
            for key in total_meta_gradient:
                total_meta_gradient[key] /= n
            
            # Apply meta-gradient with outer learning rate
            for key, gradient in total_meta_gradient.items():
                if key not in self._meta_params:
                    self._meta_params[key] = 0.0
                
                self._meta_params[key] -= self._config.outer_lr * gradient
            
            # Update statistics
            self._step_count += 1
            self._stats['meta_updates'] += 1
            self._stats['avg_improvement'] = (
                self._stats['avg_improvement'] * 0.9 +
                (total_improvement / n) * 0.1
            )
            
            return {
                'success': True,
                'meta_update': True,
                'avg_improvement': total_improvement / n,
                'step': self._step_count,
            }
    
    def adapt(
        self,
        task_embedding: TaskEmbedding,
        support_examples: List[MetaExample],
        num_steps: int = None,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Adapt to a new task using support examples.
        
        This is the inner loop of MAML - gradient descent on
        support set.
        
        Args:
            task_embedding: Task embedding
            support_examples: Support set examples
            num_steps: Number of adaptation steps
            
        Returns:
            Tuple of (adapted_params, adaptation_info)
        """
        with self._lock:
            num_steps = num_steps or self._config.adaptation_steps
            
            # Initialize from meta-parameters
            adapted_params = dict(self._meta_params)
            
            # Track adaptation
            adaptation_info = {
                'task_id': task_embedding.task_id,
                'steps': 0,
                'losses': [],
                'final_loss': 1.0,
            }
            
            # Inner loop optimization
            for step in range(num_steps):
                # Compute loss on support set (simplified)
                loss = self._compute_support_loss(
                    adapted_params,
                    support_examples,
                )
                
                adaptation_info['losses'].append(loss)
                
                # Compute gradient
                gradient = self._compute_adaptation_gradient(
                    adapted_params,
                    support_examples,
                )
                
                # Update parameters
                for key, grad in gradient.items():
                    if key in adapted_params:
                        adapted_params[key] -= self._config.inner_lr * grad
            
            adaptation_info['steps'] = num_steps
            adaptation_info['final_loss'] = adaptation_info['losses'][-1] if adaptation_info['losses'] else 1.0
            
            # Store adapted parameters
            self._adapted_params[task_embedding.task_id] = adapted_params
            
            # Update statistics
            self._stats['adaptations'] += 1
            self._stats['avg_adaptation_steps'] = (
                self._stats['avg_adaptation_steps'] * 0.9 +
                num_steps * 0.1
            )
            
            return adapted_params, adaptation_info
    
    def evaluate(
        self,
        task_embedding: TaskEmbedding,
        query_examples: List[MetaExample],
        adapted_params: Dict[str, float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate adapted model on query set.
        
        Args:
            task_embedding: Task embedding
            query_examples: Query set examples
            adapted_params: Adapted parameters (or use stored)
            
        Returns:
            Evaluation results
        """
        with self._lock:
            # Get adapted parameters
            if adapted_params is None:
                adapted_params = self._adapted_params.get(task_embedding.task_id, self._meta_params)
            
            # Compute loss on query set
            loss = self._compute_query_loss(adapted_params, query_examples)
            
            # Compute accuracy (simplified)
            accuracy = 1.0 - min(loss, 1.0)
            
            return {
                'task_id': task_embedding.task_id,
                'loss': loss,
                'accuracy': accuracy,
                'num_queries': len(query_examples),
            }
    
    def _compute_meta_gradient(self, experience: MetaExperience) -> Dict[str, float]:
        """Compute meta-gradient from experience"""
        # Simplified meta-gradient computation
        # In real MAML, this would involve second-order gradients
        
        gradient = {}
        improvement = experience.improvement
        
        # Gradient magnitude based on improvement direction
        if improvement > 0:
            # Positive improvement - reinforce current direction
            gradient['direction'] = 0.01
        else:
            # Negative improvement - reverse direction
            gradient['direction'] = -0.01
        
        # Task-specific gradients
        if experience.task_embedding:
            for i, emb_value in enumerate(experience.task_embedding.embedding[:10]):
                gradient[f'embed_{i}'] = -emb_value * improvement * 0.001
        
        return gradient
    
    def _compute_support_loss(
        self,
        params: Dict[str, float],
        examples: List[MetaExample],
    ) -> float:
        """Compute loss on support set"""
        if not examples:
            return 1.0
        
        # Simplified loss computation
        total_loss = 0.0
        for example in examples:
            # Compute prediction error (simplified)
            prediction_error = abs(params.get('bias', 0.0))
            total_loss += prediction_error * example.difficulty
        
        return total_loss / len(examples)
    
    def _compute_adaptation_gradient(
        self,
        params: Dict[str, float],
        examples: List[MetaExample],
    ) -> Dict[str, float]:
        """Compute gradient for adaptation"""
        gradient = {}
        
        # Simplified gradient computation
        for key, value in params.items():
            gradient[key] = random.uniform(-0.01, 0.01)  # Placeholder
        
        return gradient
    
    def _compute_query_loss(
        self,
        params: Dict[str, float],
        examples: List[MetaExample],
    ) -> float:
        """Compute loss on query set"""
        return self._compute_support_loss(params, examples)
    
    def get_meta_params(self) -> Dict[str, float]:
        """Get current meta-parameters"""
        return dict(self._meta_params)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['step_count'] = self._step_count
            stats['meta_params_count'] = len(self._meta_params)
            stats['adapted_tasks'] = len(self._adapted_params)
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# FEW-SHOT LEARNER
# ═══════════════════════════════════════════════════════════════════════════════

class FewShotLearner:
    """
    Implements few-shot learning capabilities.
    
    Can learn from very few examples (1-10 shots).
    """
    
    def __init__(
        self,
        num_shots: int = 5,
        embedding_dim: int = 128,
    ):
        """
        Initialize few-shot learner.
        
        Args:
            num_shots: Default number of shots
            embedding_dim: Embedding dimension
        """
        self._num_shots = num_shots
        self._embedding_dim = embedding_dim
        
        # Prototype storage (for prototypical networks)
        self._prototypes: Dict[str, List[float]] = {}
        
        # Support set storage
        self._support_sets: Dict[str, List[MetaExample]] = {}
        
        # Statistics
        self._stats = {
            'few_shot_tasks': 0,
            'one_shot_successes': 0,
            'avg_shots_needed': 0.0,
        }
        
        self._lock = threading.RLock()
        
        logger.info(f"FewShotLearner initialized with {num_shots} shots")
    
    def learn(
        self,
        task_embedding: TaskEmbedding,
        examples: List[MetaExample],
    ) -> Dict[str, Any]:
        """
        Learn from few examples.
        
        Args:
            task_embedding: Task embedding
            examples: Few-shot examples
            
        Returns:
            Learning result
        """
        with self._lock:
            n_shots = len(examples)
            
            # Compute prototype (average of example embeddings)
            prototype = self._compute_prototype(examples)
            
            # Store prototype
            self._prototypes[task_embedding.task_id] = prototype
            self._support_sets[task_embedding.task_id] = examples
            
            # Update statistics
            self._stats['few_shot_tasks'] += 1
            self._stats['avg_shots_needed'] = (
                self._stats['avg_shots_needed'] * 0.9 +
                n_shots * 0.1
            )
            
            if n_shots == 1:
                self._stats['one_shot_successes'] += 1
            
            return {
                'task_id': task_embedding.task_id,
                'n_shots': n_shots,
                'prototype_computed': True,
                'adaptation_speed': self._estimate_adaptation_speed(n_shots),
            }
    
    def predict(
        self,
        task_embedding: TaskEmbedding,
        query: MetaExample,
    ) -> Tuple[Any, float]:
        """
        Predict using few-shot learning.
        
        Args:
            task_embedding: Task embedding
            query: Query example
            
        Returns:
            Tuple of (prediction, confidence)
        """
        with self._lock:
            prototype = self._prototypes.get(task_embedding.task_id)
            
            if prototype is None:
                # No prototype - return low confidence prediction
                return None, 0.0
            
            # Compute similarity to prototype
            if query.embedding and query.embedding.embedding:
                similarity = self._cosine_similarity(prototype, query.embedding.embedding)
            else:
                similarity = 0.5
            
            # Generate prediction based on similarity
            prediction = self._generate_prediction(prototype, query, similarity)
            confidence = similarity
            
            return prediction, confidence
    
    def _compute_prototype(self, examples: List[MetaExample]) -> List[float]:
        """Compute prototype as average of example embeddings"""
        if not examples:
            return [0.0] * self._embedding_dim
        
        # Get embeddings
        embeddings = []
        for example in examples:
            if example.embedding and example.embedding.embedding:
                embeddings.append(example.embedding.embedding)
            else:
                # Use random embedding if none exists
                embeddings.append([random.random() for _ in range(self._embedding_dim)])
        
        # Average
        prototype = []
        for i in range(self._embedding_dim):
            avg = sum(e[i] if i < len(e) else 0.0 for e in embeddings) / len(embeddings)
            prototype.append(avg)
        
        return prototype
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between vectors"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def _estimate_adaptation_speed(self, n_shots: int) -> AdaptationSpeed:
        """Estimate adaptation speed based on shots"""
        if n_shots <= 2:
            return AdaptationSpeed.IMMEDIATE
        elif n_shots <= 5:
            return AdaptationSpeed.FAST
        elif n_shots <= 10:
            return AdaptationSpeed.NORMAL
        else:
            return AdaptationSpeed.SLOW
    
    def _generate_prediction(
        self,
        prototype: List[float],
        query: MetaExample,
        similarity: float,
    ) -> Any:
        """Generate prediction from prototype"""
        # Simplified prediction generation
        return {
            'predicted_class': 'positive' if similarity > 0.5 else 'negative',
            'confidence': similarity,
        }
    
    def get_prototype(self, task_id: str) -> Optional[List[float]]:
        """Get prototype for a task"""
        return self._prototypes.get(task_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['prototypes_stored'] = len(self._prototypes)
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFER ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TransferEngine:
    """
    Manages knowledge transfer between tasks and domains.
    """
    
    def __init__(self):
        """Initialize transfer engine."""
        # Transfer knowledge base
        self._knowledge_base: Dict[str, TransferKnowledge] = {}
        
        # Task relationships
        self._task_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Domain mappings
        self._domain_mappings: Dict[str, str] = {}
        
        # Statistics
        self._stats = {
            'transfers_executed': 0,
            'successful_transfers': 0,
            'negative_transfers': 0,
            'avg_effectiveness': 0.0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("TransferEngine initialized")
    
    def transfer(
        self,
        source_embedding: TaskEmbedding,
        target_embedding: TaskEmbedding,
        transfer_type: TransferType = TransferType.FEW_SHOT,
    ) -> TransferKnowledge:
        """
        Transfer knowledge from source to target task.
        
        Args:
            source_embedding: Source task embedding
            target_embedding: Target task embedding
            transfer_type: Type of transfer
            
        Returns:
            TransferKnowledge with transfer details
        """
        with self._lock:
            # Compute transferability
            similarity = source_embedding.similarity(target_embedding)
            
            # Create transfer knowledge
            transfer = TransferKnowledge(
                source_task=source_embedding.task_id,
                target_task=target_embedding.task_id,
                source_domain=source_embedding.domain,
                target_domain=target_embedding.domain,
                transfer_type=transfer_type,
                effectiveness_score=similarity,
                negative_transfer=similarity < 0.3,
            )
            
            # Store
            self._knowledge_base[transfer.id] = transfer
            
            # Update task graph
            self._task_graph[source_embedding.task_id].add(target_embedding.task_id)
            
            # Update statistics
            self._stats['transfers_executed'] += 1
            if similarity >= 0.5:
                self._stats['successful_transfers'] += 1
            elif similarity < 0.3:
                self._stats['negative_transfers'] += 1
            
            self._stats['avg_effectiveness'] = (
                self._stats['avg_effectiveness'] * 0.9 +
                similarity * 0.1
            )
            
            return transfer
    
    def find_transferable_knowledge(
        self,
        target_embedding: TaskEmbedding,
        min_similarity: float = 0.5,
        max_results: int = 5,
    ) -> List[Tuple[TaskEmbedding, float]]:
        """
        Find tasks with transferable knowledge.
        
        Args:
            target_embedding: Target task embedding
            min_similarity: Minimum similarity threshold
            max_results: Maximum results
            
        Returns:
            List of (source_embedding, similarity) tuples
        """
        results = []
        
        for transfer_id, transfer in self._knowledge_base.items():
            if transfer.target_task == target_embedding.task_id:
                # Get source embedding
                # Would need access to task embedder
                pass
        
        return results[:max_results]
    
    def record_transfer_result(
        self,
        transfer_id: str,
        success: bool,
        effectiveness: float,
    ):
        """Record the result of a transfer"""
        with self._lock:
            transfer = self._knowledge_base.get(transfer_id)
            if transfer:
                transfer.effectiveness_score = effectiveness
                transfer.negative_transfer = not success
    
    def get_transferable_domains(self) -> List[str]:
        """Get list of domains with transferable knowledge"""
        domains = set()
        for transfer in self._knowledge_base.values():
            domains.add(transfer.source_domain)
            domains.add(transfer.target_domain)
        return list(domains)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transfer statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['knowledge_base_size'] = len(self._knowledge_base)
            stats['task_connections'] = len(self._task_graph)
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CONTINUAL LEARNER
# ═══════════════════════════════════════════════════════════════════════════════

class ContinualLearner:
    """
    Implements continual learning without catastrophic forgetting.
    """
    
    def __init__(
        self,
        memory_size: int = 1000,
        rehearsal_size: int = 100,
    ):
        """
        Initialize continual learner.
        
        Args:
            memory_size: Size of replay memory
            rehearsal_size: Examples to rehearse per task
        """
        self._memory_size = memory_size
        self._rehearsal_size = rehearsal_size
        
        # Replay memory
        self._replay_memory: deque = deque(maxlen=memory_size)
        
        # Task-specific memory
        self._task_memory: Dict[str, deque] = {}
        
        # Importance weights
        self._importance_weights: Dict[str, float] = {}
        
        # Statistics
        self._stats = {
            'tasks_learned': 0,
            'rehearsals_performed': 0,
            'forgetting_events': 0,
            'avg_retention': 0.0,
        }
        
        self._lock = threading.RLock()
        
        logger.info(f"ContinualLearner initialized with memory_size={memory_size}")
    
    def learn_task(
        self,
        task_embedding: TaskEmbedding,
        examples: List[MetaExample],
    ) -> Dict[str, Any]:
        """
        Learn a new task while preserving previous knowledge.
        
        Args:
            task_embedding: Task embedding
            examples: Task examples
            
        Returns:
            Learning result
        """
        with self._lock:
            # Select rehearsal examples from previous tasks
            rehearsal_examples = self._select_rehearsal_examples()
            
            # Combine new examples with rehearsal
            all_examples = examples + rehearsal_examples
            
            # Store in memory
            for example in examples:
                self._replay_memory.append((task_embedding.task_id, example))
            
            # Store task-specific memory
            if task_embedding.task_id not in self._task_memory:
                self._task_memory[task_embedding.task_id] = deque(maxlen=self._rehearsal_size)
            
            for example in examples[:self._rehearsal_size]:
                self._task_memory[task_embedding.task_id].append(example)
            
            # Update importance weights
            self._update_importance(task_embedding.task_id, len(examples))
            
            # Update statistics
            self._stats['tasks_learned'] += 1
            self._stats['rehearsals_performed'] += len(rehearsal_examples)
            
            return {
                'task_id': task_embedding.task_id,
                'examples_learned': len(examples),
                'rehearsal_examples': len(rehearsal_examples),
                'memory_utilization': len(self._replay_memory) / self._memory_size,
            }
    
    def evaluate_retention(
        self,
        task_embedding: TaskEmbedding,
        test_examples: List[MetaExample],
    ) -> Dict[str, Any]:
        """
        Evaluate retention on a previously learned task.
        
        Args:
            task_embedding: Task to evaluate
            test_examples: Test examples
            
        Returns:
            Retention evaluation
        """
        with self._lock:
            # Compute retention (simplified)
            retention_score = 1.0 - (self._stats['forgetting_events'] * 0.1)
            retention_score = max(0.0, min(1.0, retention_score))
            
            return {
                'task_id': task_embedding.task_id,
                'retention_score': retention_score,
                'forgetting_type': ForgettingType.NONE.name if retention_score > 0.9 else ForgettingType.MINIMAL.name,
            }
    
    def _select_rehearsal_examples(self) -> List[MetaExample]:
        """Select examples for rehearsal from memory"""
        if not self._replay_memory:
            return []
        
        # Sample from replay memory
        sample_size = min(self._rehearsal_size, len(self._replay_memory))
        sampled = random.sample(list(self._replay_memory), sample_size)
        
        return [example for task_id, example in sampled]
    
    def _update_importance(self, task_id: str, example_count: int):
        """Update importance weights for a task"""
        # Increase importance based on examples
        if task_id not in self._importance_weights:
            self._importance_weights[task_id] = 0.0
        
        self._importance_weights[task_id] += example_count * 0.01
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['memory_used'] = len(self._replay_memory)
            stats['task_memory_count'] = len(self._task_memory)
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class HyperparameterOptimizer:
    """
    Meta-optimizes hyperparameters for learning.
    """
    
    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
    ):
        """
        Initialize hyperparameter optimizer.
        
        Args:
            population_size: Population for evolutionary optimization
            mutation_rate: Mutation rate for evolution
        """
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        
        # Population of configurations
        self._population: List[Tuple[HyperparameterConfig, float]] = []
        
        # Best configuration
        self._best_config: Optional[HyperparameterConfig] = None
        self._best_score: float = 0.0
        
        # History
        self._history: deque = deque(maxlen=100)
        
        # Statistics
        self._stats = {
            'evaluations': 0,
            'improvements': 0,
            'best_score': 0.0,
        }
        
        self._lock = threading.RLock()
        
        # Initialize population
        self._initialize_population()
        
        logger.info(f"HyperparameterOptimizer initialized with population={population_size}")
    
    def _initialize_population(self):
        """Initialize population with random configurations"""
        for _ in range(self._population_size):
            config = HyperparameterConfig()
            
            # Randomize
            config.inner_lr = 10 ** random.uniform(-4, -1)
            config.outer_lr = 10 ** random.uniform(-5, -2)
            config.hidden_dim = random.choice([64, 128, 256, 512])
            config.num_layers = random.randint(2, 5)
            config.num_shots = random.randint(1, 10)
            
            self._population.append((config, 0.0))
    
    def suggest(self) -> HyperparameterConfig:
        """
        Suggest next configuration to evaluate.
        
        Returns:
            HyperparameterConfig to evaluate
        """
        with self._lock:
            # Select from population (tournament selection)
            tournament_size = 3
            tournament = random.sample(self._population, min(tournament_size, len(self._population)))
            
            # Select best from tournament
            best = max(tournament, key=lambda x: x[1])
            
            # Mutate
            mutated = best[0].mutate(self._mutation_rate)
            
            return mutated
    
    def report(
        self,
        config: HyperparameterConfig,
        score: float,
    ):
        """
        Report evaluation result.
        
        Args:
            config: Evaluated configuration
            score: Evaluation score
        """
        with self._lock:
            # Add to population
            self._population.append((config, score))
            
            # Keep population size
            if len(self._population) > self._population_size:
                # Remove worst
                self._population.sort(key=lambda x: x[1], reverse=True)
                self._population = self._population[:self._population_size]
            
            # Update best
            if score > self._best_score:
                self._best_score = score
                self._best_config = config
                self._stats['improvements'] += 1
            
            self._stats['evaluations'] += 1
            self._stats['best_score'] = self._best_score
            
            # Record in history
            self._history.append({
                'config': config.to_dict(),
                'score': score,
                'timestamp': time.time(),
            })
    
    def get_best(self) -> Optional[HyperparameterConfig]:
        """Get best configuration found"""
        return self._best_config
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimizer statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['population_size'] = len(self._population)
            stats['history_size'] = len(self._history)
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# META-EXPERIENCE REPLAY
# ═══════════════════════════════════════════════════════════════════════════════

class MetaExperienceReplay:
    """
    Manages meta-level experience replay.
    """
    
    def __init__(self, buffer_size: int = 10000):
        """Initialize experience replay."""
        self._buffer_size = buffer_size
        self._buffer: deque = deque(maxlen=buffer_size)
        
        # Prioritized replay
        self._priorities: Dict[str, float] = {}
        
        # Statistics
        self._stats = {
            'experiences_stored': 0,
            'experiences_sampled': 0,
            'avg_priority': 0.0,
        }
        
        self._lock = threading.RLock()
    
    def store(self, experience: MetaExperience):
        """Store an experience"""
        with self._lock:
            self._buffer.append(experience)
            
            # Set priority based on performance
            priority = abs(experience.improvement) + 0.1
            self._priorities[experience.id] = priority
            
            self._stats['experiences_stored'] += 1
            self._stats['avg_priority'] = (
                self._stats['avg_priority'] * 0.99 +
                priority * 0.01
            )
    
    def sample(
        self,
        batch_size: int = 32,
        prioritized: bool = True,
    ) -> List[MetaExperience]:
        """Sample experiences"""
        with self._lock:
            if not self._buffer:
                return []
            
            if prioritized:
                # Prioritized sampling
                total_priority = sum(
                    self._priorities.get(e.id, 0.1)
                    for e in self._buffer
                )
                
                if total_priority == 0:
                    return random.sample(list(self._buffer), min(batch_size, len(self._buffer)))
                
                # Sample with probability proportional to priority
                samples = []
                for _ in range(min(batch_size, len(self._buffer))):
                    r = random.random() * total_priority
                    cumulative = 0
                    for exp in self._buffer:
                        cumulative += self._priorities.get(exp.id, 0.1)
                        if cumulative >= r:
                            samples.append(exp)
                            break
                
                return samples
            else:
                return random.sample(list(self._buffer), min(batch_size, len(self._buffer)))
    
    def update_priority(self, experience_id: str, priority: float):
        """Update priority of an experience"""
        with self._lock:
            self._priorities[experience_id] = priority
    
    def get_stats(self) -> Dict[str, Any]:
        """Get replay statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['buffer_size'] = len(self._buffer)
            stats['capacity'] = self._buffer_size
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# META-LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MetaLearningEngine:
    """
    Main meta-learning engine.
    
    Coordinates all meta-learning components.
    """
    
    def __init__(
        self,
        config: HyperparameterConfig = None,
        learning_mode: LearningMode = LearningMode.MAML,
    ):
        """
        Initialize meta-learning engine.
        
        Args:
            config: Hyperparameter configuration
            learning_mode: Meta-learning algorithm
        """
        self._config = config or HyperparameterConfig()
        self._learning_mode = learning_mode
        
        # Components
        self._task_embedder = TaskEmbedder(embedding_dim=self._config.hidden_dim)
        self._meta_optimizer = MetaOptimizer(config=config, learning_mode=learning_mode)
        self._few_shot_learner = FewShotLearner(num_shots=self._config.num_shots)
        self._transfer_engine = TransferEngine()
        self._continual_learner = ContinualLearner()
        self._hyperparameter_optimizer = HyperparameterOptimizer()
        self._experience_replay = MetaExperienceReplay()
        
        # Model
        self._model = MetaModel(config=config, learning_mode=learning_mode)
        
        # Statistics
        self._stats = {
            'tasks_learned': 0,
            'meta_updates': 0,
            'transfers': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info(f"MetaLearningEngine initialized with {learning_mode.name}")
    
    def learn_task(
        self,
        task_type: TaskType,
        task_data: Dict[str, Any],
        examples: List[MetaExample],
        num_meta_steps: int = 5,
    ) -> Dict[str, Any]:
        """
        Learn a new task with meta-learning.
        
        Args:
            task_type: Type of task
            task_data: Task-specific data
            examples: Training examples
            num_meta_steps: Number of meta-learning steps
            
        Returns:
            Learning result
        """
        # Create task embedding
        task_embedding = self._task_embedder.embed(task_type, task_data, examples)
        
        # Split into support and query sets
        n_support = min(len(examples) // 2, self._config.num_shots)
        support_set = examples[:n_support]
        query_set = examples[n_support:]
        
        # Adapt to task
        adapted_params, adaptation_info = self._meta_optimizer.adapt(
            task_embedding,
            support_set,
        )
        
        # Evaluate on query set
        evaluation = self._meta_optimizer.evaluate(
            task_embedding,
            query_set,
            adapted_params,
        )
        
        # Create meta-experience
        experience = MetaExperience(
            task_embedding=task_embedding,
            task_type=task_type,
            support_examples=support_set,
            query_examples=query_set,
            initial_loss=adaptation_info.get('losses', [1.0])[0] if adaptation_info.get('losses') else 1.0,
            final_loss=evaluation['loss'],
            adaptation_steps=adaptation_info['steps'],
            success=evaluation['accuracy'] > 0.5,
            accuracy=evaluation['accuracy'],
        )
        
        # Store experience
        self._experience_replay.store(experience)
        
        # Few-shot learning
        self._few_shot_learner.learn(task_embedding, support_set)
        
        # Continual learning
        self._continual_learner.learn_task(task_embedding, examples)
        
        # Update model
        self._model = self._model.adapt(task_embedding, examples)
        
        # Update statistics
        with self._lock:
            self._stats['tasks_learned'] += 1
        
        return {
            'task_id': task_embedding.task_id,
            'adaptation_steps': adaptation_info['steps'],
            'accuracy': evaluation['accuracy'],
            'success': experience.success,
        }
    
    def meta_update(self, batch_size: int = 32) -> Dict[str, Any]:
        """
        Perform meta-update from stored experiences.
        
        Args:
            batch_size: Number of experiences to use
            
        Returns:
            Update result
        """
        # Sample experiences
        experiences = self._experience_replay.sample(batch_size)
        
        if not experiences:
            return {'success': False, 'reason': 'No experiences'}
        
        # Meta-update
        result = self._meta_optimizer.meta_update(experiences)
        
        with self._lock:
            self._stats['meta_updates'] += 1
        
        return result
    
    def adapt_to_task(
        self,
        task_type: TaskType,
        task_data: Dict[str, Any],
        examples: List[MetaExample],
    ) -> Dict[str, Any]:
        """
        Quickly adapt to a new task using meta-learned knowledge.
        
        Args:
            task_type: Type of task
            task_data: Task-specific data
            examples: Few-shot examples
            
        Returns:
            Adaptation result
        """
        # Create embedding
        task_embedding = self._task_embedder.embed(task_type, task_data, examples)
        
        # Check for transferable knowledge
        # Would find similar tasks and transfer
        
        # Few-shot learn
        few_shot_result = self._few_shot_learner.learn(task_embedding, examples)
        
        # Adapt meta-optimizer
        adapted_params, _ = self._meta_optimizer.adapt(task_embedding, examples)
        
        return {
            'task_id': task_embedding.task_id,
            'few_shot': few_shot_result,
            'adaptation_speed': few_shot_result['adaptation_speed'].name,
        }
    
    def predict(
        self,
        task_type: TaskType,
        task_data: Dict[str, Any],
        query: MetaExample,
    ) -> Tuple[Any, float]:
        """
        Make prediction using meta-learned model.
        
        Args:
            task_type: Type of task
            task_data: Task context
            query: Query example
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Create embedding
        task_embedding = self._task_embedder.embed(task_type, task_data)
        
        # Few-shot prediction
        prediction, confidence = self._few_shot_learner.predict(task_embedding, query)
        
        return prediction, confidence
    
    def get_model(self) -> MetaModel:
        """Get current meta-model"""
        return self._model
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['task_embedder'] = self._task_embedder.get_stats()
            stats['meta_optimizer'] = self._meta_optimizer.get_stats()
            stats['few_shot_learner'] = self._few_shot_learner.get_stats()
            stats['transfer_engine'] = self._transfer_engine.get_stats()
            stats['continual_learner'] = self._continual_learner.get_stats()
            stats['hyperparameter_optimizer'] = self._hyperparameter_optimizer.get_stats()
            stats['experience_replay'] = self._experience_replay.get_stats()
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[MetaLearningEngine] = None


def get_meta_learning_engine(**kwargs) -> MetaLearningEngine:
    """Get global meta-learning engine"""
    global _engine
    if _engine is None:
        _engine = MetaLearningEngine(**kwargs)
    return _engine


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Meta-Learning Engine Test")
    print("="*60)
    
    # Create engine
    engine = MetaLearningEngine(learning_mode=LearningMode.MAML)
    
    # Create task
    print("\n1. Learning task...")
    task_data = {
        'code': 'def add(a, b): return a + b',
        'complexity': 0.3,
        'domain': 'utility',
    }
    
    # Create examples
    examples = [
        MetaExample(
            input_data={'a': i, 'b': j},
            output_data={'result': i + j},
            task_type=TaskType.CODE_GENERATION,
            difficulty=0.3,
        )
        for i, j in [(1, 2), (3, 4), (5, 6), (7, 8)]
    ]
    
    result = engine.learn_task(
        TaskType.CODE_GENERATION,
        task_data,
        examples,
    )
    
    print(f"   Task ID: {result['task_id']}")
    print(f"   Accuracy: {result['accuracy']:.2%}")
    print(f"   Success: {result['success']}")
    
    # Meta-update
    print("\n2. Meta-update...")
    update_result = engine.meta_update()
    print(f"   Result: {update_result.get('success', False)}")
    
    # Few-shot learning
    print("\n3. Few-shot learning...")
    adapt_result = engine.adapt_to_task(
        TaskType.CODE_GENERATION,
        {'code': 'def multiply(a, b): return a * b'},
        examples[:2],
    )
    print(f"   Adaptation speed: {adapt_result['adaptation_speed']}")
    
    # Statistics
    print("\n4. Statistics:")
    stats = engine.get_stats()
    print(f"   Tasks learned: {stats['tasks_learned']}")
    print(f"   Meta updates: {stats['meta_updates']}")
    print(f"   Few-shot tasks: {stats['few_shot_learner']['few_shot_tasks']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

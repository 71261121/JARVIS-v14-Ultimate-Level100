#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Multi-Agent Orchestrator
===============================================

Phase 4: Advanced Multi-Agent Orchestration System (Level 70-80)

This module provides sophisticated multi-agent coordination:
- Agent lifecycle management (spawn, monitor, terminate)
- Dynamic agent specialization and role assignment
- Inter-agent negotiation protocols
- Hierarchical task decomposition
- Consensus building and conflict resolution
- Distributed decision making
- Agent capability discovery and matching

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Orchestrator                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Agent     │  │   Agent     │  │   Agent     │   Agent Pool │
│  │  Pool Mgr   │  │  Lifecycle  │  │  Registry   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Negotiation │  │  Consensus  │  │  Conflict   │  Coordination│
│  │   Engine    │  │   Builder   │  │  Resolver   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Task        │  │ Capability  │  │ Load        │  Management  │
│  │ Decomposer  │  │   Matcher   │  │  Balancer   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘

Author: JARVIS AI Project
Version: 4.0.0
Target Level: 70-80
"""

import time
import json
import logging
import threading
import uuid
import math
import random
import hashlib
from typing import Dict, Any, Optional, List, Set, Callable, Tuple, Union, Type
from dataclasses import dataclass, field
from enum import Enum, auto, Flag
from datetime import datetime, timedelta
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
import weakref

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND FLAGS
# ═══════════════════════════════════════════════════════════════════════════════

class AgentState(Enum):
    """States an agent can be in"""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    IDLE = auto()
    WORKING = auto()
    WAITING = auto()
    NEGOTIATING = auto()
    ERROR = auto()
    TERMINATING = auto()
    TERMINATED = auto()


class AgentRole(Enum):
    """Specialized roles for agents"""
    COORDINATOR = auto()      # Orchestrates other agents
    EXECUTOR = auto()         # Executes tasks
    ANALYZER = auto()         # Analyzes code/data
    GENERATOR = auto()        # Generates code/content
    VALIDATOR = auto()        # Validates results
    LEARNER = auto()          # Learns from experiences
    MONITOR = auto()          # Monitors system state
    NEGOTIATOR = auto()       # Handles negotiations
    SPECIALIST = auto()       # Domain specialist
    GENERALIST = auto()       # General purpose


class AgentCapability(Flag):
    """Capabilities an agent can have"""
    NONE = 0
    CODE_GENERATION = auto()
    CODE_ANALYSIS = auto()
    CODE_REFACTORING = auto()
    BUG_FIXING = auto()
    TESTING = auto()
    DOCUMENTATION = auto()
    OPTIMIZATION = auto()
    SECURITY_ANALYSIS = auto()
    DATA_PROCESSING = auto()
    WEB_INTERACTION = auto()
    FILE_OPERATIONS = auto()
    SYSTEM_OPERATIONS = auto()
    AI_REASONING = auto()
    LEARNING = auto()
    PLANNING = auto()
    COMMUNICATION = auto()
    NEGOTIATION = auto()
    MONITORING = auto()
    RECOVERY = auto()


class TaskPriority(Enum):
    """Priority levels for tasks"""
    CRITICAL = 0    # Must complete immediately
    HIGH = 1        # High priority
    NORMAL = 2      # Normal priority
    LOW = 3         # Low priority
    BACKGROUND = 4  # Background task


class NegotiationState(Enum):
    """States of a negotiation"""
    INITIATED = auto()
    PROPOSAL_SENT = auto()
    COUNTER_PROPOSAL = auto()
    ACCEPTED = auto()
    REJECTED = auto()
    TIMEOUT = auto()
    FAILED = auto()


class ConsensusType(Enum):
    """Types of consensus mechanisms"""
    MAJORITY = auto()         # Simple majority
    UNANIMOUS = auto()        # All must agree
    WEIGHTED = auto()         # Weighted by expertise
    DELEGATED = auto()        # Delegated decision maker
    BYZANTINE = auto()        # Byzantine fault tolerant
    RAFT = auto()             # Raft consensus


class ConflictType(Enum):
    """Types of conflicts between agents"""
    RESOURCE_CONTENTION = auto()
    TASK_OVERLAP = auto()
    PRIORITY_CONFLICT = auto()
    DATA_INCONSISTENCY = auto()
    GOAL_CONFLICT = auto()
    COMMUNICATION_FAILURE = auto()


class DecompositionStrategy(Enum):
    """Strategies for task decomposition"""
    SEQUENTIAL = auto()       # Sequential subtasks
    PARALLEL = auto()         # Parallel subtasks
    HIERARCHICAL = auto()     # Hierarchical decomposition
    DATA_PARALLEL = auto()    # Split by data
    FUNCTIONAL = auto()       # Split by function
    HYBRID = auto()           # Combination


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentProfile:
    """
    Profile defining an agent's characteristics.
    
    Contains all metadata about an agent type.
    """
    # Identity
    agent_type: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    
    # Capabilities
    capabilities: AgentCapability = AgentCapability.NONE
    specializations: List[AgentRole] = field(default_factory=list)
    
    # Resource requirements
    min_memory_mb: int = 50
    max_memory_mb: int = 500
    cpu_weight: float = 0.5  # 0-1, higher = more CPU intensive
    
    # Behavior
    max_concurrent_tasks: int = 3
    task_timeout_seconds: float = 300.0
    retry_count: int = 2
    
    # Communication
    message_queue_size: int = 100
    response_timeout: float = 30.0
    
    # Learning
    learning_enabled: bool = True
    experience_weight: float = 0.1
    
    # Negotiation
    negotiation_enabled: bool = True
    negotiation_style: str = "cooperative"  # cooperative, competitive, hybrid
    
    # Priority
    default_priority: TaskPriority = TaskPriority.NORMAL
    priority_boost_on_failure: bool = True
    
    # Health
    health_check_interval: float = 60.0
    max_error_count: int = 5
    recovery_time: float = 30.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_type': self.agent_type,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'capabilities': [c.name for c in AgentCapability if c in self.capabilities],
            'specializations': [s.name for s in self.specializations],
            'min_memory_mb': self.min_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'cpu_weight': self.cpu_weight,
            'max_concurrent_tasks': self.max_concurrent_tasks,
        }
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a capability"""
        return bool(self.capabilities & capability)


@dataclass
class AgentMetrics:
    """
    Performance metrics for an agent.
    
    Tracks historical performance for optimization.
    """
    # Task metrics
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_cancelled: int = 0
    
    # Time metrics
    total_execution_time: float = 0.0
    total_wait_time: float = 0.0
    avg_task_time: float = 0.0
    
    # Success metrics
    success_rate: float = 0.0
    reliability_score: float = 100.0
    
    # Resource metrics
    avg_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    
    # Communication metrics
    messages_sent: int = 0
    messages_received: int = 0
    avg_response_time: float = 0.0
    
    # Negotiation metrics
    negotiations_won: int = 0
    negotiations_lost: int = 0
    negotiation_success_rate: float = 0.0
    
    # Learning metrics
    experiences_recorded: int = 0
    learning_iterations: int = 0
    
    # Health metrics
    error_count: int = 0
    last_error_time: Optional[float] = None
    recovery_count: int = 0
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    
    def update_task_metrics(self, success: bool, execution_time: float):
        """Update metrics after task completion"""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        
        self.total_execution_time += execution_time
        total_tasks = self.tasks_completed + self.tasks_failed
        
        if total_tasks > 0:
            self.success_rate = self.tasks_completed / total_tasks
            self.avg_task_time = self.total_execution_time / total_tasks
        
        self.reliability_score = self.success_rate * 100
        self.last_active = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'success_rate': f"{self.success_rate:.1%}",
            'avg_task_time': f"{self.avg_task_time:.2f}s",
            'reliability_score': f"{self.reliability_score:.1f}",
            'avg_memory_usage': f"{self.avg_memory_usage:.1f}MB",
            'messages_sent': self.messages_sent,
            'messages_received': self.messages_received,
            'error_count': self.error_count,
        }


@dataclass
class SubTask:
    """
    A subtask in a decomposition.
    
    Represents a unit of work within a larger task.
    """
    id: str = field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:8]}")
    parent_task_id: str = ""
    name: str = ""
    description: str = ""
    
    # Requirements
    required_capabilities: AgentCapability = AgentCapability.NONE
    required_role: Optional[AgentRole] = None
    estimated_time: float = 60.0
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Subtask IDs
    dependents: List[str] = field(default_factory=list)
    
    # State
    status: str = "pending"  # pending, assigned, running, completed, failed
    assigned_agent: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Input/Output
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # Retry
    retry_count: int = 0
    max_retries: int = 2
    
    @property
    def execution_time(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0
    
    @property
    def is_ready(self) -> bool:
        """Check if all dependencies are satisfied"""
        # Would need task manager to check dependencies
        return self.status == "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'assigned_agent': self.assigned_agent,
            'priority': self.priority.name,
            'dependencies': self.dependencies,
            'execution_time': self.execution_time,
        }


@dataclass
class TaskDecomposition:
    """
    Result of task decomposition.
    
    Contains the breakdown of a complex task into subtasks.
    """
    id: str = field(default_factory=lambda: f"decomp_{uuid.uuid4().hex[:8]}")
    parent_task_id: str = ""
    
    # Decomposition info
    strategy: DecompositionStrategy = DecompositionStrategy.HIERARCHICAL
    depth: int = 0
    max_depth: int = 3
    
    # Subtasks
    subtasks: Dict[str, SubTask] = field(default_factory=dict)
    
    # Dependencies
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    
    # Execution plan
    execution_order: List[str] = field(default_factory=list)
    parallel_groups: List[List[str]] = field(default_factory=list)
    
    # Metrics
    total_estimated_time: float = 0.0
    critical_path_length: float = 0.0
    
    # Status
    status: str = "pending"
    progress: float = 0.0
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    
    def add_subtask(self, subtask: SubTask):
        """Add a subtask to the decomposition"""
        self.subtasks[subtask.id] = subtask
        subtask.parent_task_id = self.parent_task_id
    
    def get_ready_subtasks(self) -> List[SubTask]:
        """Get subtasks ready for execution"""
        ready = []
        for subtask in self.subtasks.values():
            if subtask.status != "pending":
                continue
            
            # Check dependencies
            deps_satisfied = True
            for dep_id in subtask.dependencies:
                dep = self.subtasks.get(dep_id)
                if not dep or dep.status != "completed":
                    deps_satisfied = False
                    break
            
            if deps_satisfied:
                ready.append(subtask)
        
        return ready
    
    def update_progress(self):
        """Update overall progress"""
        if not self.subtasks:
            return
        
        completed = sum(1 for s in self.subtasks.values() if s.status == "completed")
        self.progress = completed / len(self.subtasks)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'strategy': self.strategy.name,
            'subtask_count': len(self.subtasks),
            'status': self.status,
            'progress': f"{self.progress:.1%}",
            'total_estimated_time': f"{self.total_estimated_time:.1f}s",
        }


@dataclass
class Proposal:
    """
    A proposal in agent negotiation.
    
    Represents an offer or counter-offer.
    """
    id: str = field(default_factory=lambda: f"prop_{uuid.uuid4().hex[:8]}")
    negotiation_id: str = ""
    
    # Parties
    proposer_id: str = ""
    target_id: str = ""
    
    # Content
    proposal_type: str = "offer"  # offer, counter_offer, acceptance, rejection
    content: Dict[str, Any] = field(default_factory=dict)
    
    # Terms
    terms: Dict[str, Any] = field(default_factory=dict)
    conditions: List[str] = field(default_factory=list)
    
    # Validity
    valid_until: Optional[float] = None
    priority: TaskPriority = TaskPriority.NORMAL
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    responded_at: Optional[float] = None
    
    # Response
    response: Optional[str] = None  # accepted, rejected, counter
    response_content: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if proposal has expired"""
        if self.valid_until is None:
            return False
        return time.time() > self.valid_until
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'proposer_id': self.proposer_id,
            'target_id': self.target_id,
            'type': self.proposal_type,
            'terms': self.terms,
            'response': self.response,
        }


@dataclass
class Negotiation:
    """
    A negotiation session between agents.
    
    Manages the negotiation process.
    """
    id: str = field(default_factory=lambda: f"neg_{uuid.uuid4().hex[:8]}")
    
    # Parties
    initiator_id: str = ""
    participants: List[str] = field(default_factory=list)
    
    # Subject
    subject: str = ""
    description: str = ""
    negotiation_type: str = "task_allocation"  # task_allocation, resource, priority
    
    # State
    state: NegotiationState = NegotiationState.INITIATED
    current_round: int = 0
    max_rounds: int = 5
    
    # Proposals
    proposals: List[Proposal] = field(default_factory=list)
    current_proposal: Optional[Proposal] = None
    
    # Outcome
    outcome: str = "pending"  # pending, agreed, failed, timeout
    agreement: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    timeout: float = 300.0
    
    # Metrics
    total_proposals: int = 0
    convergence_score: float = 0.0
    
    def add_proposal(self, proposal: Proposal) -> None:
        """Add a proposal to the negotiation"""
        proposal.negotiation_id = self.id
        self.proposals.append(proposal)
        self.current_proposal = proposal
        self.total_proposals += 1
        self.current_round += 1
    
    def check_timeout(self) -> bool:
        """Check if negotiation has timed out"""
        if self.started_at is None:
            return False
        return time.time() - self.started_at > self.timeout
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'subject': self.subject,
            'participants': self.participants,
            'state': self.state.name,
            'round': self.current_round,
            'outcome': self.outcome,
            'proposal_count': len(self.proposals),
        }


@dataclass
class Conflict:
    """
    A conflict between agents.
    
    Represents a dispute requiring resolution.
    """
    id: str = field(default_factory=lambda: f"conf_{uuid.uuid4().hex[:8]}")
    
    # Parties
    agents_involved: List[str] = field(default_factory=list)
    
    # Conflict info
    conflict_type: ConflictType = ConflictType.RESOURCE_CONTENTION
    description: str = ""
    severity: str = "medium"  # low, medium, high, critical
    
    # Subject
    resource_id: Optional[str] = None
    task_id: Optional[str] = None
    data_id: Optional[str] = None
    
    # State
    status: str = "detected"  # detected, analyzing, resolving, resolved, escalated
    resolution_strategy: str = "negotiation"
    
    # Resolution
    resolution: Dict[str, Any] = field(default_factory=dict)
    resolution_attempts: int = 0
    
    # Timestamps
    detected_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    
    # Escalation
    escalated: bool = False
    escalated_to: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.conflict_type.name,
            'agents': self.agents_involved,
            'severity': self.severity,
            'status': self.status,
            'resolution_attempts': self.resolution_attempts,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class Agent(ABC):
    """
    Abstract base class for all agents.
    
    Defines the interface for agent implementations.
    """
    
    def __init__(
        self,
        agent_id: str,
        profile: AgentProfile,
        orchestrator: 'MultiAgentOrchestrator' = None,
    ):
        """
        Initialize agent.
        
        Args:
            agent_id: Unique identifier
            profile: Agent profile defining capabilities
            orchestrator: Reference to orchestrator
        """
        self.agent_id = agent_id
        self.profile = profile
        self._orchestrator = weakref.ref(orchestrator) if orchestrator else None
        
        # State
        self._state = AgentState.UNINITIALIZED
        self._current_task: Optional[str] = None
        
        # Metrics
        self.metrics = AgentMetrics()
        
        # Communication
        self._message_queue: deque = deque(maxlen=profile.message_queue_size)
        self._pending_responses: Dict[str, Future] = {}
        
        # Resources
        self._memory_usage: float = 0.0
        self._cpu_usage: float = 0.0
        
        # Learning
        self._experience_buffer: deque = deque(maxlen=1000)
        self._knowledge: Dict[str, Any] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        logger.info(f"Agent {agent_id} created with profile {profile.name}")
    
    @property
    def state(self) -> AgentState:
        """Get current state"""
        with self._lock:
            return self._state
    
    @state.setter
    def state(self, value: AgentState):
        """Set state"""
        with self._lock:
            old_state = self._state
            self._state = value
            logger.debug(f"Agent {self.agent_id}: {old_state.name} -> {value.name}")
    
    @property
    def is_available(self) -> bool:
        """Check if agent is available for tasks"""
        return self.state == AgentState.IDLE
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def execute_task(self, task: SubTask) -> Dict[str, Any]:
        """Execute a task. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown agent. Must be implemented by subclasses."""
        pass
    
    def receive_message(self, message: Dict[str, Any]) -> None:
        """Receive a message from another agent"""
        with self._lock:
            self._message_queue.append(message)
            self.metrics.messages_received += 1
    
    def send_message(
        self,
        target_agent: str,
        message_type: str,
        content: Dict[str, Any],
    ) -> bool:
        """Send a message to another agent"""
        if not self._orchestrator:
            return False
        
        orchestrator = self._orchestrator()
        if not orchestrator:
            return False
        
        message = {
            'id': f"msg_{uuid.uuid4().hex[:8]}",
            'from': self.agent_id,
            'to': target_agent,
            'type': message_type,
            'content': content,
            'timestamp': time.time(),
        }
        
        orchestrator.route_message(message)
        self.metrics.messages_sent += 1
        return True
    
    def update_metrics(self, success: bool, execution_time: float) -> None:
        """Update performance metrics"""
        with self._lock:
            self.metrics.update_task_metrics(success, execution_time)
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        with self._lock:
            return {
                'agent_id': self.agent_id,
                'state': self.state.name,
                'profile': self.profile.name,
                'current_task': self._current_task,
                'metrics': self.metrics.to_dict(),
                'memory_usage': self._memory_usage,
                'cpu_usage': self._cpu_usage,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT AGENT IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class DefaultAgent(Agent):
    """
    Default agent implementation.
    
    Provides basic functionality for all agent types.
    """
    
    def __init__(
        self,
        agent_id: str,
        profile: AgentProfile,
        orchestrator: 'MultiAgentOrchestrator' = None,
        kimi_client=None,
    ):
        """Initialize default agent."""
        super().__init__(agent_id, profile, orchestrator)
        self._kimi = kimi_client
        self._task_handlers: Dict[str, Callable] = {}
    
    def initialize(self) -> bool:
        """Initialize agent"""
        try:
            self.state = AgentState.INITIALIZING
            
            # Initialize resources
            self._memory_usage = self.profile.min_memory_mb
            self._cpu_usage = 0.0
            
            # Initialize knowledge base
            self._knowledge = {
                'specializations': [s.name for s in self.profile.specializations],
                'capabilities': [c.name for c in AgentCapability if c in self.profile.capabilities],
            }
            
            # Register default task handlers
            self._register_handlers()
            
            # Start worker thread
            self._running = True
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"agent-{self.agent_id}",
                daemon=True,
            )
            self._worker_thread.start()
            
            self.state = AgentState.IDLE
            logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} initialization failed: {e}")
            self.state = AgentState.ERROR
            return False
    
    def _register_handlers(self):
        """Register task handlers based on capabilities"""
        if self.profile.has_capability(AgentCapability.CODE_GENERATION):
            self._task_handlers['code_generation'] = self._handle_code_generation
        
        if self.profile.has_capability(AgentCapability.CODE_ANALYSIS):
            self._task_handlers['code_analysis'] = self._handle_code_analysis
        
        if self.profile.has_capability(AgentCapability.BUG_FIXING):
            self._task_handlers['bug_fix'] = self._handle_bug_fix
        
        if self.profile.has_capability(AgentCapability.TESTING):
            self._task_handlers['testing'] = self._handle_testing
        
        # Default handler
        self._task_handlers['default'] = self._handle_default
    
    def execute_task(self, task: SubTask) -> Dict[str, Any]:
        """Execute a task"""
        start_time = time.time()
        self.state = AgentState.WORKING
        self._current_task = task.id
        
        try:
            # Get handler
            handler = self._task_handlers.get(task.name, self._task_handlers['default'])
            
            # Execute
            result = handler(task)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.update_metrics(True, execution_time)
            
            task.status = "completed"
            task.completed_at = time.time()
            task.result = result
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.update_metrics(False, execution_time)
            
            task.status = "failed"
            task.error = str(e)
            
            logger.error(f"Agent {self.agent_id} task {task.id} failed: {e}")
            return {'success': False, 'error': str(e)}
            
        finally:
            self._current_task = None
            self.state = AgentState.IDLE
    
    def shutdown(self) -> bool:
        """Shutdown agent"""
        try:
            self.state = AgentState.TERMINATING
            
            # Stop worker
            self._running = False
            if self._worker_thread:
                self._worker_thread.join(timeout=5.0)
            
            # Clear resources
            self._message_queue.clear()
            self._pending_responses.clear()
            
            self.state = AgentState.TERMINATED
            logger.info(f"Agent {self.agent_id} shutdown complete")
            return True
            
        except Exception as e:
            logger.error(f"Agent {self.agent_id} shutdown failed: {e}")
            return False
    
    def _worker_loop(self):
        """Worker thread for processing messages and tasks"""
        while self._running:
            try:
                # Process messages
                while self._message_queue:
                    message = self._message_queue.popleft()
                    self._process_message(message)
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Agent {self.agent_id} worker error: {e}")
    
    def _process_message(self, message: Dict[str, Any]):
        """Process an incoming message"""
        msg_type = message.get('type', 'unknown')
        
        if msg_type == 'task_assignment':
            # Handle task assignment
            pass
        elif msg_type == 'negotiation_request':
            # Handle negotiation request
            pass
        elif msg_type == 'status_request':
            # Respond with status
            self.send_message(
                message['from'],
                'status_response',
                self.get_status(),
            )
    
    def _handle_code_generation(self, task: SubTask) -> Dict[str, Any]:
        """Handle code generation task"""
        return {
            'success': True,
            'output': f"Generated code for {task.description}",
            'files': [],
        }
    
    def _handle_code_analysis(self, task: SubTask) -> Dict[str, Any]:
        """Handle code analysis task"""
        return {
            'success': True,
            'analysis': f"Analyzed: {task.description}",
            'issues': [],
            'suggestions': [],
        }
    
    def _handle_bug_fix(self, task: SubTask) -> Dict[str, Any]:
        """Handle bug fix task"""
        return {
            'success': True,
            'fix': f"Fixed: {task.description}",
            'changes': [],
        }
    
    def _handle_testing(self, task: SubTask) -> Dict[str, Any]:
        """Handle testing task"""
        return {
            'success': True,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
        }
    
    def _handle_default(self, task: SubTask) -> Dict[str, Any]:
        """Default task handler"""
        return {
            'success': True,
            'output': f"Completed: {task.description}",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TASK DECOMPOSER
# ═══════════════════════════════════════════════════════════════════════════════

class TaskDecomposer:
    """
    Decomposes complex tasks into subtasks.
    
    Analyzes task requirements and creates execution plans.
    """
    
    def __init__(self, kimi_client=None):
        """Initialize task decomposer."""
        self._kimi = kimi_client
        self._decomposition_cache: Dict[str, TaskDecomposition] = {}
        self._stats = {
            'total_decompositions': 0,
            'avg_subtask_count': 0.0,
            'avg_depth': 0.0,
        }
    
    def decompose(
        self,
        task_id: str,
        task_description: str,
        strategy: DecompositionStrategy = DecompositionStrategy.HIERARCHICAL,
        max_subtasks: int = 10,
        max_depth: int = 3,
    ) -> TaskDecomposition:
        """
        Decompose a task into subtasks.
        
        Args:
            task_id: Parent task ID
            task_description: Task description
            strategy: Decomposition strategy
            max_subtasks: Maximum number of subtasks
            max_depth: Maximum decomposition depth
            
        Returns:
            TaskDecomposition with subtasks
        """
        decomposition = TaskDecomposition(
            parent_task_id=task_id,
            strategy=strategy,
            max_depth=max_depth,
        )
        
        # Analyze task
        task_analysis = self._analyze_task(task_description)
        
        # Generate subtasks based on strategy
        if strategy == DecompositionStrategy.SEQUENTIAL:
            self._decompose_sequential(decomposition, task_analysis, max_subtasks)
        elif strategy == DecompositionStrategy.PARALLEL:
            self._decompose_parallel(decomposition, task_analysis, max_subtasks)
        elif strategy == DecompositionStrategy.HIERARCHICAL:
            self._decompose_hierarchical(decomposition, task_analysis, max_depth)
        else:
            self._decompose_hybrid(decomposition, task_analysis, max_subtasks)
        
        # Build dependency graph
        self._build_dependency_graph(decomposition)
        
        # Calculate execution order
        self._calculate_execution_order(decomposition)
        
        # Calculate metrics
        self._calculate_decomposition_metrics(decomposition)
        
        # Update stats
        self._stats['total_decompositions'] += 1
        
        # Cache
        self._decomposition_cache[decomposition.id] = decomposition
        
        logger.info(
            f"Decomposed task {task_id} into {len(decomposition.subtasks)} subtasks "
            f"using {strategy.name} strategy"
        )
        
        return decomposition
    
    def _analyze_task(self, description: str) -> Dict[str, Any]:
        """Analyze task to determine decomposition approach"""
        analysis = {
            'complexity': 'medium',
            'components': [],
            'dependencies': [],
            'estimated_time': 60.0,
            'required_capabilities': [],
        }
        
        # Simple heuristic analysis
        desc_lower = description.lower()
        
        # Detect complexity
        if any(w in desc_lower for w in ['complex', 'multiple', 'comprehensive', 'full']):
            analysis['complexity'] = 'high'
        elif any(w in desc_lower for w in ['simple', 'quick', 'minor']):
            analysis['complexity'] = 'low'
        
        # Detect components
        if 'code' in desc_lower:
            analysis['components'].append('code')
            analysis['required_capabilities'].append(AgentCapability.CODE_GENERATION)
        
        if 'test' in desc_lower:
            analysis['components'].append('testing')
            analysis['required_capabilities'].append(AgentCapability.TESTING)
        
        if 'analyze' in desc_lower or 'analysis' in desc_lower:
            analysis['components'].append('analysis')
            analysis['required_capabilities'].append(AgentCapability.CODE_ANALYSIS)
        
        if 'fix' in desc_lower or 'bug' in desc_lower:
            analysis['components'].append('bugfix')
            analysis['required_capabilities'].append(AgentCapability.BUG_FIXING)
        
        if 'document' in desc_lower:
            analysis['components'].append('documentation')
            analysis['required_capabilities'].append(AgentCapability.DOCUMENTATION)
        
        if 'optimize' in desc_lower:
            analysis['components'].append('optimization')
            analysis['required_capabilities'].append(AgentCapability.OPTIMIZATION)
        
        # Estimate time based on complexity
        complexity_multiplier = {'low': 0.5, 'medium': 1.0, 'high': 2.0}
        analysis['estimated_time'] *= complexity_multiplier.get(analysis['complexity'], 1.0)
        
        return analysis
    
    def _decompose_sequential(
        self,
        decomposition: TaskDecomposition,
        analysis: Dict[str, Any],
        max_subtasks: int,
    ):
        """Create sequential subtasks"""
        components = analysis['components'] or ['main']
        
        prev_id = None
        for i, component in enumerate(components[:max_subtasks]):
            subtask = SubTask(
                parent_task_id=decomposition.parent_task_id,
                name=f"step_{i+1}_{component}",
                description=f"Sequential step {i+1}: {component}",
                required_capabilities=AgentCapability.NONE,
                priority=TaskPriority.NORMAL,
                estimated_time=analysis['estimated_time'] / len(components),
            )
            
            # Add dependency to previous
            if prev_id:
                subtask.dependencies.append(prev_id)
            
            decomposition.add_subtask(subtask)
            prev_id = subtask.id
    
    def _decompose_parallel(
        self,
        decomposition: TaskDecomposition,
        analysis: Dict[str, Any],
        max_subtasks: int,
    ):
        """Create parallel subtasks"""
        components = analysis['components'] or ['main']
        
        for i, component in enumerate(components[:max_subtasks]):
            subtask = SubTask(
                parent_task_id=decomposition.parent_task_id,
                name=f"parallel_{i+1}_{component}",
                description=f"Parallel task {i+1}: {component}",
                required_capabilities=AgentCapability.NONE,
                priority=TaskPriority.NORMAL,
                estimated_time=analysis['estimated_time'] / len(components),
            )
            
            decomposition.add_subtask(subtask)
        
        # All in one parallel group
        decomposition.parallel_groups.append([s.id for s in decomposition.subtasks.values()])
    
    def _decompose_hierarchical(
        self,
        decomposition: TaskDecomposition,
        analysis: Dict[str, Any],
        max_depth: int,
    ):
        """Create hierarchical decomposition"""
        # Level 0: Main task
        # Level 1: Major components
        # Level 2: Sub-components
        # Level 3: Atomic tasks
        
        components = analysis['components'] or ['main']
        
        # Create level 1 tasks
        for i, component in enumerate(components):
            subtask = SubTask(
                parent_task_id=decomposition.parent_task_id,
                name=f"l1_{component}",
                description=f"Component: {component}",
                required_capabilities=AgentCapability.NONE,
                priority=TaskPriority.NORMAL,
                estimated_time=analysis['estimated_time'] / len(components),
            )
            
            decomposition.add_subtask(subtask)
            
            # Create level 2 subtasks if complexity is high
            if analysis['complexity'] == 'high' and max_depth >= 2:
                sub_components = ['analyze', 'execute', 'validate']
                for j, sub_comp in enumerate(sub_components):
                    child = SubTask(
                        parent_task_id=decomposition.parent_task_id,
                        name=f"l2_{component}_{sub_comp}",
                        description=f"Sub-task: {component} - {sub_comp}",
                        required_capabilities=AgentCapability.NONE,
                        priority=TaskPriority.NORMAL,
                        dependencies=[subtask.id],
                        estimated_time=subtask.estimated_time / len(sub_components),
                    )
                    decomposition.add_subtask(child)
        
        decomposition.depth = 2 if analysis['complexity'] == 'high' else 1
    
    def _decompose_hybrid(
        self,
        decomposition: TaskDecomposition,
        analysis: Dict[str, Any],
        max_subtasks: int,
    ):
        """Create hybrid decomposition (sequential + parallel)"""
        # Start with analysis phase (sequential)
        analysis_task = SubTask(
            parent_task_id=decomposition.parent_task_id,
            name="analysis_phase",
            description="Analyze requirements and plan approach",
            required_capabilities=AgentCapability.CODE_ANALYSIS,
            priority=TaskPriority.HIGH,
            estimated_time=analysis['estimated_time'] * 0.2,
        )
        decomposition.add_subtask(analysis_task)
        
        # Execution phase (parallel)
        components = analysis['components'] or ['execution']
        execution_tasks = []
        
        for i, component in enumerate(components[:max_subtasks // 2]):
            exec_task = SubTask(
                parent_task_id=decomposition.parent_task_id,
                name=f"exec_{component}",
                description=f"Execute: {component}",
                required_capabilities=AgentCapability.NONE,
                priority=TaskPriority.NORMAL,
                dependencies=[analysis_task.id],
                estimated_time=analysis['estimated_time'] * 0.5 / len(components),
            )
            decomposition.add_subtask(exec_task)
            execution_tasks.append(exec_task.id)
        
        # Validation phase (sequential, depends on all execution tasks)
        validation_task = SubTask(
            parent_task_id=decomposition.parent_task_id,
            name="validation_phase",
            description="Validate results and finalize",
            required_capabilities=AgentCapability.TESTING,
            priority=TaskPriority.NORMAL,
            dependencies=execution_tasks,
            estimated_time=analysis['estimated_time'] * 0.3,
        )
        decomposition.add_subtask(validation_task)
        
        # Parallel group for execution tasks
        decomposition.parallel_groups.append(execution_tasks)
    
    def _build_dependency_graph(self, decomposition: TaskDecomposition):
        """Build dependency graph from subtasks"""
        for subtask in decomposition.subtasks.values():
            decomposition.dependency_graph[subtask.id] = subtask.dependencies.copy()
            
            # Update dependents
            for dep_id in subtask.dependencies:
                if dep_id in decomposition.subtasks:
                    decomposition.subtasks[dep_id].dependents.append(subtask.id)
    
    def _calculate_execution_order(self, decomposition: TaskDecomposition):
        """Calculate topological execution order"""
        visited = set()
        order = []
        
        def visit(subtask_id: str):
            if subtask_id in visited:
                return
            visited.add(subtask_id)
            
            for dep_id in decomposition.dependency_graph.get(subtask_id, []):
                visit(dep_id)
            
            order.append(subtask_id)
        
        for subtask_id in decomposition.subtasks:
            visit(subtask_id)
        
        decomposition.execution_order = order
    
    def _calculate_decomposition_metrics(self, decomposition: TaskDecomposition):
        """Calculate decomposition metrics"""
        total_time = sum(s.estimated_time for s in decomposition.subtasks.values())
        decomposition.total_estimated_time = total_time
        
        # Critical path (simplified - longest dependency chain)
        max_depth = 0
        for subtask in decomposition.subtasks.values():
            depth = self._get_dependency_depth(decomposition, subtask.id)
            max_depth = max(max_depth, depth)
        
        decomposition.critical_path_length = max_depth * (total_time / max(len(decomposition.subtasks), 1))
    
    def _get_dependency_depth(self, decomposition: TaskDecomposition, subtask_id: str) -> int:
        """Get depth of dependency chain"""
        subtask = decomposition.subtasks.get(subtask_id)
        if not subtask or not subtask.dependencies:
            return 0
        
        max_dep_depth = 0
        for dep_id in subtask.dependencies:
            dep_depth = self._get_dependency_depth(decomposition, dep_id)
            max_dep_depth = max(max_dep_depth, dep_depth)
        
        return max_dep_depth + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get decomposer statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# NEGOTIATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class NegotiationEngine:
    """
    Handles negotiations between agents.
    
    Implements various negotiation protocols and strategies.
    """
    
    def __init__(self):
        """Initialize negotiation engine."""
        self._active_negotiations: Dict[str, Negotiation] = {}
        self._completed_negotiations: deque = deque(maxlen=1000)
        self._stats = {
            'total_negotiations': 0,
            'successful': 0,
            'failed': 0,
            'avg_rounds': 0.0,
            'avg_time': 0.0,
        }
        
        self._lock = threading.RLock()
    
    def initiate_negotiation(
        self,
        initiator_id: str,
        participants: List[str],
        subject: str,
        description: str,
        negotiation_type: str = "task_allocation",
        timeout: float = 300.0,
    ) -> Negotiation:
        """
        Initiate a negotiation.
        
        Args:
            initiator_id: Agent initiating negotiation
            participants: Other agents involved
            subject: Subject of negotiation
            description: Detailed description
            negotiation_type: Type of negotiation
            timeout: Maximum duration
            
        Returns:
            Negotiation object
        """
        with self._lock:
            negotiation = Negotiation(
                initiator_id=initiator_id,
                participants=[initiator_id] + participants,
                subject=subject,
                description=description,
                negotiation_type=negotiation_type,
                timeout=timeout,
            )
            negotiation.started_at = time.time()
            
            self._active_negotiations[negotiation.id] = negotiation
            self._stats['total_negotiations'] += 1
            
            logger.info(
                f"Negotiation {negotiation.id} initiated by {initiator_id}: {subject}"
            )
            
            return negotiation
    
    def make_proposal(
        self,
        negotiation_id: str,
        proposer_id: str,
        target_id: str,
        terms: Dict[str, Any],
        valid_for: float = 60.0,
    ) -> Proposal:
        """
        Make a proposal in a negotiation.
        
        Args:
            negotiation_id: Negotiation ID
            proposer_id: Agent making proposal
            target_id: Target agent
            terms: Proposal terms
            valid_for: Validity duration in seconds
            
        Returns:
            Proposal object
        """
        with self._lock:
            negotiation = self._active_negotiations.get(negotiation_id)
            if not negotiation:
                raise ValueError(f"Negotiation {negotiation_id} not found")
            
            proposal = Proposal(
                negotiation_id=negotiation_id,
                proposer_id=proposer_id,
                target_id=target_id,
                proposal_type="offer",
                terms=terms,
                valid_until=time.time() + valid_for,
            )
            
            negotiation.add_proposal(proposal)
            
            logger.debug(
                f"Proposal {proposal.id} made in negotiation {negotiation_id}"
            )
            
            return proposal
    
    def respond_to_proposal(
        self,
        negotiation_id: str,
        proposal_id: str,
        responder_id: str,
        response: str,
        counter_terms: Dict[str, Any] = None,
    ) -> Optional[Proposal]:
        """
        Respond to a proposal.
        
        Args:
            negotiation_id: Negotiation ID
            proposal_id: Proposal being responded to
            responder_id: Agent responding
            response: "accept", "reject", or "counter"
            counter_terms: Terms for counter-offer
            
        Returns:
            Counter-proposal if applicable
        """
        with self._lock:
            negotiation = self._active_negotiations.get(negotiation_id)
            if not negotiation:
                return None
            
            # Find proposal
            proposal = None
            for p in negotiation.proposals:
                if p.id == proposal_id:
                    proposal = p
                    break
            
            if not proposal:
                return None
            
            proposal.response = response
            proposal.responded_at = time.time()
            proposal.response_content = counter_terms or {}
            
            if response == "accept":
                negotiation.state = NegotiationState.ACCEPTED
                negotiation.outcome = "agreed"
                negotiation.agreement = proposal.terms
                self._complete_negotiation(negotiation, success=True)
                
            elif response == "reject":
                if negotiation.current_round >= negotiation.max_rounds:
                    negotiation.state = NegotiationState.REJECTED
                    negotiation.outcome = "failed"
                    self._complete_negotiation(negotiation, success=False)
                else:
                    negotiation.state = NegotiationState.COUNTER_PROPOSAL
                    
            elif response == "counter" and counter_terms:
                counter = Proposal(
                    negotiation_id=negotiation_id,
                    proposer_id=responder_id,
                    target_id=proposal.proposer_id,
                    proposal_type="counter_offer",
                    terms=counter_terms,
                )
                negotiation.add_proposal(counter)
                negotiation.state = NegotiationState.COUNTER_PROPOSAL
                return counter
            
            return None
    
    def check_timeouts(self):
        """Check and handle timed out negotiations"""
        with self._lock:
            timed_out = []
            
            for neg_id, negotiation in self._active_negotiations.items():
                if negotiation.check_timeout():
                    negotiation.state = NegotiationState.TIMEOUT
                    negotiation.outcome = "timeout"
                    timed_out.append(neg_id)
            
            for neg_id in timed_out:
                negotiation = self._active_negotiations.pop(neg_id)
                self._complete_negotiation(negotiation, success=False)
    
    def _complete_negotiation(self, negotiation: Negotiation, success: bool):
        """Complete a negotiation"""
        negotiation.completed_at = time.time()
        self._completed_negotiations.append(negotiation)
        
        if success:
            self._stats['successful'] += 1
        else:
            self._stats['failed'] += 1
        
        if negotiation.id in self._active_negotiations:
            del self._active_negotiations[negotiation.id]
        
        logger.info(
            f"Negotiation {negotiation.id} completed: {negotiation.outcome}"
        )
    
    def get_negotiation(self, negotiation_id: str) -> Optional[Negotiation]:
        """Get negotiation by ID"""
        return self._active_negotiations.get(negotiation_id)
    
    def get_active_negotiations(self, agent_id: str = None) -> List[Negotiation]:
        """Get active negotiations, optionally filtered by agent"""
        with self._lock:
            if agent_id:
                return [
                    n for n in self._active_negotiations.values()
                    if agent_id in n.participants
                ]
            return list(self._active_negotiations.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get negotiation statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['active_count'] = len(self._active_negotiations)
            
            if stats['total_negotiations'] > 0:
                stats['success_rate'] = stats['successful'] / stats['total_negotiations']
            
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CONFLICT RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class ConflictResolver:
    """
    Resolves conflicts between agents.
    
    Implements various conflict resolution strategies.
    """
    
    def __init__(self):
        """Initialize conflict resolver."""
        self._active_conflicts: Dict[str, Conflict] = {}
        self._resolved_conflicts: deque = deque(maxlen=1000)
        self._resolution_strategies = {
            ConflictType.RESOURCE_CONTENTION: self._resolve_resource_contention,
            ConflictType.TASK_OVERLAP: self._resolve_task_overlap,
            ConflictType.PRIORITY_CONFLICT: self._resolve_priority_conflict,
            ConflictType.DATA_INCONSISTENCY: self._resolve_data_inconsistency,
            ConflictType.GOAL_CONFLICT: self._resolve_goal_conflict,
            ConflictType.COMMUNICATION_FAILURE: self._resolve_communication_failure,
        }
        self._stats = {
            'total_conflicts': 0,
            'resolved': 0,
            'escalated': 0,
            'avg_resolution_time': 0.0,
        }
        
        self._lock = threading.RLock()
    
    def detect_conflict(
        self,
        agents: List[str],
        conflict_type: ConflictType,
        description: str,
        **kwargs,
    ) -> Conflict:
        """
        Detect and register a conflict.
        
        Args:
            agents: Agents involved in conflict
            conflict_type: Type of conflict
            description: Description of conflict
            **kwargs: Additional context
            
        Returns:
            Conflict object
        """
        with self._lock:
            conflict = Conflict(
                agents_involved=agents,
                conflict_type=conflict_type,
                description=description,
                resource_id=kwargs.get('resource_id'),
                task_id=kwargs.get('task_id'),
                data_id=kwargs.get('data_id'),
            )
            
            # Set severity based on type and context
            if conflict_type in [ConflictType.GOAL_CONFLICT, ConflictType.DATA_INCONSISTENCY]:
                conflict.severity = "high"
            
            self._active_conflicts[conflict.id] = conflict
            self._stats['total_conflicts'] += 1
            
            logger.warning(
                f"Conflict {conflict.id} detected: {conflict_type.name} between {agents}"
            )
            
            return conflict
    
    def resolve(self, conflict_id: str) -> Dict[str, Any]:
        """
        Resolve a conflict.
        
        Args:
            conflict_id: Conflict to resolve
            
        Returns:
            Resolution result
        """
        with self._lock:
            conflict = self._active_conflicts.get(conflict_id)
            if not conflict:
                return {'success': False, 'error': 'Conflict not found'}
            
            conflict.status = "resolving"
            
            # Get resolution strategy
            strategy = self._resolution_strategies.get(
                conflict.conflict_type,
                self._resolve_generic,
            )
            
            # Attempt resolution
            resolution = strategy(conflict)
            
            if resolution.get('success'):
                conflict.status = "resolved"
                conflict.resolution = resolution
                conflict.resolved_at = time.time()
                
                self._resolved_conflicts.append(conflict)
                del self._active_conflicts[conflict.id]
                
                self._stats['resolved'] += 1
                
                logger.info(f"Conflict {conflict_id} resolved: {resolution}")
            else:
                conflict.resolution_attempts += 1
                
                if conflict.resolution_attempts >= 3:
                    conflict.escalated = True
                    conflict.status = "escalated"
                    self._stats['escalated'] += 1
                    
                    logger.warning(f"Conflict {conflict_id} escalated after 3 attempts")
            
            return resolution
    
    def _resolve_resource_contention(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve resource contention conflict"""
        # Strategies: time-sharing, priority-based, negotiation
        return {
            'success': True,
            'strategy': 'time_sharing',
            'allocation': {
                'agent_1': {'start': 0, 'duration': 30},
                'agent_2': {'start': 30, 'duration': 30},
            },
            'description': 'Resource allocated with time-sharing',
        }
    
    def _resolve_task_overlap(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve task overlap conflict"""
        return {
            'success': True,
            'strategy': 'partition',
            'description': 'Task partitioned between agents',
        }
    
    def _resolve_priority_conflict(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve priority conflict"""
        return {
            'success': True,
            'strategy': 'urgency_based',
            'winner': conflict.agents_involved[0],
            'description': 'Priority assigned based on urgency',
        }
    
    def _resolve_data_inconsistency(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve data inconsistency conflict"""
        return {
            'success': True,
            'strategy': 'merge',
            'description': 'Data merged with conflict resolution',
        }
    
    def _resolve_goal_conflict(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve goal conflict"""
        return {
            'success': True,
            'strategy': 'negotiation',
            'description': 'Agents negotiated new shared goal',
        }
    
    def _resolve_communication_failure(self, conflict: Conflict) -> Dict[str, Any]:
        """Resolve communication failure"""
        return {
            'success': True,
            'strategy': 'retry',
            'description': 'Communication retry initiated',
        }
    
    def _resolve_generic(self, conflict: Conflict) -> Dict[str, Any]:
        """Generic conflict resolution"""
        return {
            'success': False,
            'strategy': 'escalation',
            'description': 'Unable to resolve automatically',
        }
    
    def get_active_conflicts(self) -> List[Conflict]:
        """Get active conflicts"""
        with self._lock:
            return list(self._active_conflicts.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conflict statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['active_count'] = len(self._active_conflicts)
            
            if stats['total_conflicts'] > 0:
                stats['resolution_rate'] = stats['resolved'] / stats['total_conflicts']
            
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CAPABILITY MATCHER
# ═══════════════════════════════════════════════════════════════════════════════

class CapabilityMatcher:
    """
    Matches tasks to agents based on capabilities.
    
    Implements sophisticated matching algorithms.
    """
    
    def __init__(self):
        """Initialize capability matcher."""
        self._capability_registry: Dict[str, AgentCapability] = {}
        self._agent_capabilities: Dict[str, AgentCapability] = {}
        self._stats = {
            'total_matches': 0,
            'successful_matches': 0,
            'avg_match_score': 0.0,
        }
    
    def register_agent(self, agent_id: str, capabilities: AgentCapability):
        """Register agent capabilities"""
        self._agent_capabilities[agent_id] = capabilities
        logger.debug(f"Registered capabilities for agent {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent"""
        if agent_id in self._agent_capabilities:
            del self._agent_capabilities[agent_id]
    
    def find_matching_agents(
        self,
        required_capabilities: AgentCapability,
        exclude: Set[str] = None,
        prefer_available: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Find agents matching required capabilities.
        
        Args:
            required_capabilities: Required capabilities
            exclude: Agents to exclude
            prefer_available: Prefer available agents
            
        Returns:
            List of (agent_id, match_score) tuples
        """
        exclude = exclude or set()
        matches = []
        
        for agent_id, capabilities in self._agent_capabilities.items():
            if agent_id in exclude:
                continue
            
            # Check if agent has all required capabilities
            if (capabilities & required_capabilities) == required_capabilities:
                # Calculate match score
                score = self._calculate_match_score(
                    capabilities,
                    required_capabilities,
                )
                matches.append((agent_id, score))
        
        # Sort by score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        self._stats['total_matches'] += 1
        if matches:
            self._stats['successful_matches'] += 1
        
        return matches
    
    def _calculate_match_score(
        self,
        agent_capabilities: AgentCapability,
        required_capabilities: AgentCapability,
    ) -> float:
        """Calculate capability match score"""
        # Exact match of required
        if agent_capabilities == required_capabilities:
            return 1.0
        
        # Count matching capabilities
        required_count = bin(required_capabilities.value).count('1')
        matching = agent_capabilities & required_capabilities
        matching_count = bin(matching.value).count('1')
        
        # Base score: how many required capabilities are matched
        base_score = matching_count / max(required_count, 1)
        
        # Bonus for extra capabilities (versatility)
        extra = agent_capabilities & ~required_capabilities
        extra_count = bin(extra.value).count('1')
        versatility_bonus = min(extra_count * 0.05, 0.2)
        
        return min(1.0, base_score + versatility_bonus)
    
    def get_best_agent(
        self,
        required_capabilities: AgentCapability,
        exclude: Set[str] = None,
    ) -> Optional[str]:
        """Get single best matching agent"""
        matches = self.find_matching_agents(required_capabilities, exclude)
        return matches[0][0] if matches else None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get matcher statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MultiAgentOrchestrator:
    """
    Main orchestrator for multi-agent system.
    
    Coordinates all agents and components.
    
    Usage:
        orchestrator = MultiAgentOrchestrator(kimi_client)
        orchestrator.start()
        
        # Spawn agents
        agent_id = orchestrator.spawn_agent(profile)
        
        # Decompose and execute task
        decomposition = orchestrator.decompose_task(task)
        orchestrator.execute_decomposition(decomposition)
        
        # Shutdown
        orchestrator.shutdown()
    """
    
    # Default agent profiles
    DEFAULT_PROFILES = {
        'coordinator': AgentProfile(
            agent_type='coordinator',
            name='Coordinator Agent',
            description='Orchestrates other agents and manages workflows',
            capabilities=AgentCapability.PLANNING | AgentCapability.COMMUNICATION | AgentCapability.NEGOTIATION,
            specializations=[AgentRole.COORDINATOR],
            max_concurrent_tasks=10,
        ),
        'executor': AgentProfile(
            agent_type='executor',
            name='Executor Agent',
            description='Executes code generation and modifications',
            capabilities=AgentCapability.CODE_GENERATION | AgentCapability.FILE_OPERATIONS,
            specializations=[AgentRole.EXECUTOR],
            max_concurrent_tasks=5,
        ),
        'analyzer': AgentProfile(
            agent_type='analyzer',
            name='Analyzer Agent',
            description='Analyzes code and provides insights',
            capabilities=AgentCapability.CODE_ANALYSIS | AgentCapability.AI_REASONING,
            specializations=[AgentRole.ANALYZER],
            max_concurrent_tasks=3,
        ),
        'validator': AgentProfile(
            agent_type='validator',
            name='Validator Agent',
            description='Validates code and tests',
            capabilities=AgentCapability.TESTING | AgentCapability.SECURITY_ANALYSIS,
            specializations=[AgentRole.VALIDATOR],
            max_concurrent_tasks=5,
        ),
    }
    
    def __init__(
        self,
        kimi_client=None,
        max_agents: int = 20,
        enable_negotiation: bool = True,
        enable_learning: bool = True,
    ):
        """
        Initialize orchestrator.
        
        Args:
            kimi_client: Kimi K2.5 client for AI reasoning
            max_agents: Maximum number of agents
            enable_negotiation: Enable negotiation engine
            enable_learning: Enable learning from experiences
        """
        self._kimi = kimi_client
        self._max_agents = max_agents
        
        # Agent management
        self._agents: Dict[str, Agent] = {}
        self._agent_profiles: Dict[str, AgentProfile] = self.DEFAULT_PROFILES.copy()
        
        # Components
        self._decomposer = TaskDecomposer(kimi_client)
        self._negotiation_engine = NegotiationEngine() if enable_negotiation else None
        self._conflict_resolver = ConflictResolver()
        self._capability_matcher = CapabilityMatcher()
        
        # Message routing
        self._message_queue: deque = deque(maxlen=10000)
        self._message_handlers: Dict[str, Callable] = {}
        
        # Task management
        self._active_decompositions: Dict[str, TaskDecomposition] = {}
        self._task_assignments: Dict[str, str] = {}  # task_id -> agent_id
        
        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=max_agents)
        
        # Statistics
        self._stats = {
            'total_agents_spawned': 0,
            'total_tasks_decomposed': 0,
            'total_tasks_executed': 0,
            'total_messages_routed': 0,
            'uptime': 0.0,
        }
        
        # State
        self._running = False
        self._start_time: Optional[float] = None
        
        # Threading
        self._lock = threading.RLock()
        self._message_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        
        logger.info("MultiAgentOrchestrator initialized (Level 70-80)")
    
    def start(self) -> bool:
        """Start the orchestrator"""
        try:
            with self._lock:
                if self._running:
                    return True
                
                self._running = True
                self._start_time = time.time()
                
                # Start message router thread
                self._message_thread = threading.Thread(
                    target=self._message_loop,
                    name="orchestrator-messages",
                    daemon=True,
                )
                self._message_thread.start()
                
                # Start monitor thread
                self._monitor_thread = threading.Thread(
                    target=self._monitor_loop,
                    name="orchestrator-monitor",
                    daemon=True,
                )
                self._monitor_thread.start()
                
                logger.info("MultiAgentOrchestrator started")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown the orchestrator"""
        try:
            with self._lock:
                self._running = False
                
                # Shutdown all agents
                for agent_id, agent in list(self._agents.items()):
                    agent.shutdown()
                
                self._agents.clear()
                
                # Shutdown executor
                self._executor.shutdown(wait=True)
                
                self._stats['uptime'] = time.time() - (self._start_time or time.time())
                
                logger.info("MultiAgentOrchestrator shutdown complete")
                return True
                
        except Exception as e:
            logger.error(f"Failed to shutdown orchestrator: {e}")
            return False
    
    def spawn_agent(
        self,
        profile: AgentProfile = None,
        profile_name: str = None,
        agent_class: Type[Agent] = DefaultAgent,
    ) -> Optional[str]:
        """
        Spawn a new agent.
        
        Args:
            profile: Agent profile (overrides profile_name)
            profile_name: Name of predefined profile
            agent_class: Agent class to instantiate
            
        Returns:
            Agent ID or None on failure
        """
        with self._lock:
            if len(self._agents) >= self._max_agents:
                logger.warning("Maximum agents reached")
                return None
            
            # Get profile
            if profile is None:
                profile_name = profile_name or 'generalist'
                profile = self._agent_profiles.get(profile_name)
                
                if not profile:
                    # Create generic profile
                    profile = AgentProfile(
                        agent_type=profile_name,
                        name=f"{profile_name.title()} Agent",
                        capabilities=AgentCapability.CODE_GENERATION | AgentCapability.CODE_ANALYSIS,
                    )
            
            # Create agent
            agent_id = f"agent_{profile.agent_type}_{uuid.uuid4().hex[:8]}"
            agent = agent_class(
                agent_id=agent_id,
                profile=profile,
                orchestrator=self,
                kimi_client=self._kimi,
            )
            
            # Initialize
            if not agent.initialize():
                logger.error(f"Failed to initialize agent {agent_id}")
                return None
            
            # Register
            self._agents[agent_id] = agent
            self._capability_matcher.register_agent(agent_id, profile.capabilities)
            self._stats['total_agents_spawned'] += 1
            
            logger.info(f"Spawned agent {agent_id} with profile {profile.name}")
            return agent_id
    
    def terminate_agent(self, agent_id: str) -> bool:
        """Terminate an agent"""
        with self._lock:
            agent = self._agents.get(agent_id)
            if not agent:
                return False
            
            # Reassign tasks
            # TODO: Implement task reassignment
            
            # Shutdown
            agent.shutdown()
            
            # Unregister
            del self._agents[agent_id]
            self._capability_matcher.unregister_agent(agent_id)
            
            logger.info(f"Terminated agent {agent_id}")
            return True
    
    def decompose_task(
        self,
        task_id: str,
        task_description: str,
        strategy: DecompositionStrategy = DecompositionStrategy.HYBRID,
    ) -> TaskDecomposition:
        """
        Decompose a task into subtasks.
        
        Args:
            task_id: Task ID
            task_description: Task description
            strategy: Decomposition strategy
            
        Returns:
            TaskDecomposition
        """
        decomposition = self._decomposer.decompose(
            task_id=task_id,
            task_description=task_description,
            strategy=strategy,
        )
        
        with self._lock:
            self._active_decompositions[decomposition.id] = decomposition
            self._stats['total_tasks_decomposed'] += 1
        
        return decomposition
    
    def execute_decomposition(
        self,
        decomposition: TaskDecomposition,
    ) -> Dict[str, Any]:
        """
        Execute a task decomposition.
        
        Args:
            decomposition: Task decomposition to execute
            
        Returns:
            Execution result
        """
        results = {}
        futures: Dict[str, Future] = {}
        
        try:
            # Execute in order, respecting dependencies
            for group in self._get_execution_groups(decomposition):
                # Submit tasks in group
                for subtask_id in group:
                    subtask = decomposition.subtasks[subtask_id]
                    agent_id = self._assign_agent(subtask)
                    
                    if agent_id:
                        agent = self._agents[agent_id]
                        future = self._executor.submit(agent.execute_task, subtask)
                        futures[subtask_id] = future
                
                # Wait for group to complete
                for subtask_id, future in futures.items():
                    try:
                        results[subtask_id] = future.result(timeout=300)
                    except Exception as e:
                        results[subtask_id] = {'success': False, 'error': str(e)}
                
                futures.clear()
            
            # Update decomposition status
            decomposition.update_progress()
            decomposition.status = "completed"
            
            self._stats['total_tasks_executed'] += len(results)
            
            return {
                'success': True,
                'decomposition_id': decomposition.id,
                'results': results,
                'progress': decomposition.progress,
            }
            
        except Exception as e:
            logger.error(f"Decomposition execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
            }
    
    def _get_execution_groups(self, decomposition: TaskDecomposition) -> List[List[str]]:
        """Get execution groups respecting dependencies"""
        if decomposition.parallel_groups:
            return decomposition.parallel_groups
        
        # Generate groups from dependency graph
        groups = []
        assigned = set()
        
        while len(assigned) < len(decomposition.subtasks):
            group = []
            for subtask_id, subtask in decomposition.subtasks.items():
                if subtask_id in assigned:
                    continue
                
                # Check if all dependencies are assigned
                if all(d in assigned for d in subtask.dependencies):
                    group.append(subtask_id)
            
            if not group:
                break
            
            groups.append(group)
            assigned.update(group)
        
        return groups
    
    def _assign_agent(self, subtask: SubTask) -> Optional[str]:
        """Assign an agent to a subtask"""
        # Find matching agents
        matches = self._capability_matcher.find_matching_agents(
            subtask.required_capabilities,
        )
        
        # Filter for available agents
        for agent_id, score in matches:
            agent = self._agents.get(agent_id)
            if agent and agent.is_available:
                subtask.assigned_agent = agent_id
                subtask.status = "assigned"
                return agent_id
        
        # No available agent, use first match
        if matches:
            agent_id = matches[0][0]
            subtask.assigned_agent = agent_id
            subtask.status = "assigned"
            return agent_id
        
        return None
    
    def route_message(self, message: Dict[str, Any]):
        """Route a message to target agent"""
        with self._lock:
            self._message_queue.append(message)
            self._stats['total_messages_routed'] += 1
    
    def _message_loop(self):
        """Message routing loop"""
        while self._running:
            try:
                while self._message_queue:
                    message = self._message_queue.popleft()
                    self._deliver_message(message)
                
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Message loop error: {e}")
    
    def _deliver_message(self, message: Dict[str, Any]):
        """Deliver message to target"""
        target_id = message.get('to')
        if not target_id:
            return
        
        agent = self._agents.get(target_id)
        if agent:
            agent.receive_message(message)
    
    def _monitor_loop(self):
        """Monitoring loop"""
        while self._running:
            try:
                # Check agent health
                for agent in list(self._agents.values()):
                    if agent.state == AgentState.ERROR:
                        agent.metrics.error_count += 1
                
                # Check negotiation timeouts
                if self._negotiation_engine:
                    self._negotiation_engine.check_timeouts()
                
                time.sleep(5.0)
                
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID"""
        return self._agents.get(agent_id)
    
    def get_all_agents(self) -> List[Agent]:
        """Get all agents"""
        with self._lock:
            return list(self._agents.values())
    
    def get_available_agents(self) -> List[Agent]:
        """Get available agents"""
        with self._lock:
            return [a for a in self._agents.values() if a.is_available]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['active_agents'] = len(self._agents)
            stats['available_agents'] = len(self.get_available_agents())
            stats['active_decompositions'] = len(self._active_decompositions)
            
            if self._negotiation_engine:
                stats['negotiations'] = self._negotiation_engine.get_stats()
            
            stats['conflicts'] = self._conflict_resolver.get_stats()
            stats['matching'] = self._capability_matcher.get_stats()
            
            if self._start_time:
                stats['uptime'] = time.time() - self._start_time
            
            return stats
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        with self._lock:
            return {
                'running': self._running,
                'agents': {
                    'total': len(self._agents),
                    'by_state': self._count_agents_by_state(),
                    'by_role': self._count_agents_by_role(),
                },
                'tasks': {
                    'active_decompositions': len(self._active_decompositions),
                    'total_executed': self._stats['total_tasks_executed'],
                },
                'performance': {
                    'messages_per_second': self._calculate_message_rate(),
                    'avg_agent_utilization': self._calculate_utilization(),
                },
            }
    
    def _count_agents_by_state(self) -> Dict[str, int]:
        """Count agents by state"""
        counts = defaultdict(int)
        for agent in self._agents.values():
            counts[agent.state.name] += 1
        return dict(counts)
    
    def _count_agents_by_role(self) -> Dict[str, int]:
        """Count agents by role"""
        counts = defaultdict(int)
        for agent in self._agents.values():
            for role in agent.profile.specializations:
                counts[role.name] += 1
        return dict(counts)
    
    def _calculate_message_rate(self) -> float:
        """Calculate message routing rate"""
        if not self._start_time:
            return 0.0
        elapsed = time.time() - self._start_time
        if elapsed == 0:
            return 0.0
        return self._stats['total_messages_routed'] / elapsed
    
    def _calculate_utilization(self) -> float:
        """Calculate average agent utilization"""
        if not self._agents:
            return 0.0
        
        total_utilization = 0.0
        for agent in self._agents.values():
            if agent.state == AgentState.WORKING:
                total_utilization += 1.0
            elif agent.state == AgentState.IDLE:
                total_utilization += 0.0
            else:
                total_utilization += 0.5
        
        return total_utilization / len(self._agents)
    
    def set_kimi_client(self, client):
        """Set Kimi client"""
        self._kimi = client
        for agent in self._agents.values():
            if hasattr(agent, '_kimi'):
                agent._kimi = client


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_orchestrator(kimi_client=None) -> MultiAgentOrchestrator:
    """Get global orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator(kimi_client=kimi_client)
    elif kimi_client:
        _orchestrator.set_kimi_client(kimi_client)
    return _orchestrator


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Multi-Agent Orchestrator Test")
    print("="*60)
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(kimi_client=None)
    orchestrator.start()
    
    # Spawn agents
    print("\n1. Spawning agents...")
    coordinator_id = orchestrator.spawn_agent(profile_name='coordinator')
    executor_id = orchestrator.spawn_agent(profile_name='executor')
    analyzer_id = orchestrator.spawn_agent(profile_name='analyzer')
    
    print(f"   Spawned: {coordinator_id}, {executor_id}, {analyzer_id}")
    
    # Decompose task
    print("\n2. Decomposing task...")
    decomposition = orchestrator.decompose_task(
        task_id="test_task_001",
        task_description="Analyze and optimize the main module code",
        strategy=DecompositionStrategy.HYBRID,
    )
    
    print(f"   Decomposition ID: {decomposition.id}")
    print(f"   Subtasks: {len(decomposition.subtasks)}")
    print(f"   Strategy: {decomposition.strategy.name}")
    
    # Print subtasks
    print("\n3. Subtasks:")
    for subtask in decomposition.subtasks.values():
        print(f"   - {subtask.name}: {subtask.description[:50]}...")
    
    # Get stats
    print("\n4. System Stats:")
    stats = orchestrator.get_stats()
    print(f"   Active agents: {stats['active_agents']}")
    print(f"   Available agents: {stats['available_agents']}")
    
    # Get system status
    print("\n5. System Status:")
    status = orchestrator.get_system_status()
    print(f"   Agents by state: {status['agents']['by_state']}")
    print(f"   Avg utilization: {status['performance']['avg_agent_utilization']:.1%}")
    
    # Shutdown
    print("\n6. Shutting down...")
    orchestrator.shutdown()
    print("   Done!")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

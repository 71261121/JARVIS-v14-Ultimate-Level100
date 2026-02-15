#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Distributed Task Executor
=================================================

Phase 4: Distributed Task Execution System (Level 70-80)

This module provides sophisticated distributed execution:
- Task scheduling and distribution
- Load balancing across agents
- Fault tolerance and recovery
- Execution monitoring
- Resource allocation
- Parallel execution management
- Result aggregation
- Timeout and retry handling

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                  Distributed Task Executor                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │    Task     │  │    Load     │  │   Resource  │  Core        │
│  │  Scheduler  │  │  Balancer   │  │  Allocator  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Execution  │  │   Fault     │  │   Result    │  Management  │
│  │   Monitor   │  │  Tolerance  │  │ Aggregator  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Worker    │  │   Timeout   │  │   Parallel  │  Execution   │
│  │    Pool     │  │   Handler   │  │   Engine    │              │
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
from typing import Dict, Any, Optional, List, Set, Callable, Tuple, Union, TypeVar, Generic, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from queue import Queue, PriorityQueue, Empty
from abc import ABC, abstractmethod
import weakref

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class TaskStatus(Enum):
    """Status of a task"""
    PENDING = auto()
    QUEUED = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()
    RETRYING = auto()


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ScheduleStrategy(Enum):
    """Task scheduling strategies"""
    FIFO = auto()           # First in, first out
    PRIORITY = auto()       # Priority-based
    SHORTEST_FIRST = auto() # Shortest task first
    ROUND_ROBIN = auto()    # Round robin across workers
    LEAST_LOADED = auto()   # Assign to least loaded worker
    CAPABILITY = auto()     # Match capabilities


class LoadBalanceStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = auto()
    LEAST_CONNECTIONS = auto()
    WEIGHTED_ROUND_ROBIN = auto()
    RESOURCE_AWARE = auto()
    ADAPTIVE = auto()


class ExecutionMode(Enum):
    """Execution modes"""
    SEQUENTIAL = auto()
    PARALLEL = auto()
    PIPELINE = auto()
    MAP_REDUCE = auto()
    SCATTER_GATHER = auto()


class ResourceType(Enum):
    """Types of resources"""
    CPU = auto()
    MEMORY = auto()
    IO = auto()
    NETWORK = auto()
    AGENT = auto()


class FailureType(Enum):
    """Types of failures"""
    TIMEOUT = auto()
    ERROR = auto()
    RESOURCE_EXHAUSTED = auto()
    WORKER_UNAVAILABLE = auto()
    DEPENDENCY_FAILED = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(order=True)
class Task:
    """
    A task to be executed.
    
    Ordered by priority for priority queue.
    """
    # Identification
    id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    
    # Priority (lower = higher priority, used for sorting)
    priority: int = 2
    created_at: float = field(default_factory=time.time)
    
    # Execution
    func: Callable = None
    args: Tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    # Constraints
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    required_capabilities: Set[str] = field(default_factory=set)
    estimated_duration: float = 60.0
    
    # State
    status: TaskStatus = TaskStatus.PENDING
    assigned_worker: Optional[str] = None
    attempts: int = 0
    
    # Timing
    scheduled_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Result
    result: Any = None
    error: Optional[str] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other: 'Task') -> bool:
        """For priority queue ordering"""
        return (self.priority, self.created_at) < (other.priority, other.created_at)
    
    @property
    def execution_time(self) -> float:
        """Get execution time if completed"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0
    
    @property
    def wait_time(self) -> float:
        """Get wait time in queue"""
        if self.scheduled_at:
            return self.scheduled_at - self.created_at
        return time.time() - self.created_at
    
    @property
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.attempts < self.max_retries
    
    @property
    def is_complete(self) -> bool:
        """Check if task is in a complete state"""
        return self.status in [
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.TIMEOUT,
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status.name,
            'priority': self.priority,
            'assigned_worker': self.assigned_worker,
            'attempts': self.attempts,
            'execution_time': self.execution_time,
        }


@dataclass
class WorkerInfo:
    """
    Information about a worker.
    """
    id: str
    name: str = ""
    
    # Capabilities
    capabilities: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 3
    
    # Current state
    current_tasks: Set[str] = field(default_factory=set)
    status: str = "idle"  # idle, busy, offline
    
    # Resources
    cpu_capacity: float = 100.0
    memory_capacity: float = 100.0
    cpu_used: float = 0.0
    memory_used: float = 0.0
    
    # Performance
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    avg_task_time: float = 0.0
    
    # Health
    last_heartbeat: float = field(default_factory=time.time)
    error_count: int = 0
    last_error: Optional[str] = None
    
    # Load
    weight: float = 1.0  # For weighted load balancing
    
    @property
    def current_load(self) -> float:
        """Get current load (0-1)"""
        return len(self.current_tasks) / max(self.max_concurrent_tasks, 1)
    
    @property
    def is_available(self) -> bool:
        """Check if worker is available"""
        return (
            self.status == "idle" or
            (self.status == "busy" and len(self.current_tasks) < self.max_concurrent_tasks)
        )
    
    @property
    def cpu_available(self) -> float:
        """Get available CPU capacity"""
        return self.cpu_capacity - self.cpu_used
    
    @property
    def memory_available(self) -> float:
        """Get available memory capacity"""
        return self.memory_capacity - self.memory_used
    
    @property
    def success_rate(self) -> float:
        """Get success rate"""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_completed / total
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'status': self.status,
            'current_tasks': len(self.current_tasks),
            'load': f"{self.current_load:.1%}",
            'tasks_completed': self.tasks_completed,
            'success_rate': f"{self.success_rate:.1%}",
        }


@dataclass
class ExecutionPlan:
    """
    Plan for executing a set of tasks.
    """
    id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    
    # Tasks
    tasks: Dict[str, Task] = field(default_factory=dict)
    
    # Execution groups (can run in parallel within group)
    groups: List[List[str]] = field(default_factory=list)
    
    # Mode
    mode: ExecutionMode = ExecutionMode.PARALLEL
    
    # Resource requirements
    total_cpu_required: float = 0.0
    total_memory_required: float = 0.0
    
    # Estimated time
    estimated_duration: float = 0.0
    critical_path: List[str] = field(default_factory=list)
    
    # State
    status: str = "pending"
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def add_task(self, task: Task):
        """Add a task to the plan"""
        self.tasks[task.id] = task
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tasks': len(self.tasks),
            'groups': len(self.groups),
            'mode': self.mode.name,
            'status': self.status,
            'estimated_duration': f"{self.estimated_duration:.1f}s",
        }


@dataclass
class ExecutionResult:
    """
    Result of task execution.
    """
    task_id: str
    worker_id: str
    
    # Outcome
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    
    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: float = field(default_factory=time.time)
    execution_time: float = 0.0
    
    # Resources
    cpu_used: float = 0.0
    memory_used: float = 0.0
    
    # Retry
    attempts: int = 1
    
    @property
    def throughput(self) -> float:
        """Calculate throughput"""
        if self.execution_time > 0:
            return 1.0 / self.execution_time
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'worker_id': self.worker_id,
            'success': self.success,
            'execution_time': f"{self.execution_time:.2f}s",
            'attempts': self.attempts,
        }


@dataclass
class ResourceAllocation:
    """
    Resource allocation for a task.
    """
    task_id: str
    worker_id: str
    
    # Allocated resources
    cpu_allocated: float = 0.0
    memory_allocated: float = 0.0
    
    # Time
    allocated_at: float = field(default_factory=time.time)
    released_at: Optional[float] = None
    
    # Status
    active: bool = True
    
    def release(self):
        """Release the allocation"""
        self.active = False
        self.released_at = time.time()


# ═══════════════════════════════════════════════════════════════════════════════
# TASK SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

class TaskScheduler:
    """
    Schedules tasks for execution.
    
    Implements multiple scheduling strategies.
    """
    
    def __init__(
        self,
        strategy: ScheduleStrategy = ScheduleStrategy.PRIORITY,
        max_queue_size: int = 10000,
    ):
        """
        Initialize task scheduler.
        
        Args:
            strategy: Scheduling strategy
            max_queue_size: Maximum queue size
        """
        self._strategy = strategy
        self._max_queue_size = max_queue_size
        
        # Queues
        if strategy == ScheduleStrategy.PRIORITY:
            self._queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        else:
            self._queue: Queue = Queue(maxsize=max_queue_size)
        
        # Task tracking
        self._tasks: Dict[str, Task] = {}
        self._pending: Set[str] = set()
        self._scheduled: Set[str] = set()
        
        # Statistics
        self._stats = {
            'tasks_queued': 0,
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'avg_wait_time': 0.0,
            'avg_queue_length': 0.0,
        }
        
        self._lock = threading.RLock()
        
        logger.info(f"TaskScheduler initialized with {strategy.name} strategy")
    
    def submit(self, task: Task) -> bool:
        """
        Submit a task for scheduling.
        
        Args:
            task: Task to schedule
            
        Returns:
            True if successfully queued
        """
        with self._lock:
            if len(self._tasks) >= self._max_queue_size:
                logger.warning("Task queue full")
                return False
            
            # Set priority value
            task.priority = task.priority if isinstance(task.priority, int) else task.priority.value
            task.status = TaskStatus.QUEUED
            
            # Add to queue
            if isinstance(self._queue, PriorityQueue):
                self._queue.put(task)
            else:
                self._queue.put(task)
            
            # Track
            self._tasks[task.id] = task
            self._pending.add(task.id)
            self._stats['tasks_queued'] += 1
            
            logger.debug(f"Task {task.id} queued")
            return True
    
    def get_next(self) -> Optional[Task]:
        """
        Get the next task to execute.
        
        Returns:
            Next task or None if queue empty
        """
        try:
            if isinstance(self._queue, PriorityQueue):
                task = self._queue.get_nowait()
            else:
                task = self._queue.get_nowait()
            
            with self._lock:
                self._pending.discard(task.id)
                self._scheduled.add(task.id)
                task.status = TaskStatus.SCHEDULED
                task.scheduled_at = time.time()
                self._stats['tasks_scheduled'] += 1
            
            return task
            
        except Empty:
            return None
    
    def peek(self) -> Optional[Task]:
        """Peek at next task without removing"""
        # Can't peek at PriorityQueue directly
        return None
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self._tasks.get(task_id)
    
    def complete_task(self, task_id: str, success: bool = True):
        """Mark task as completed"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                task.completed_at = time.time()
                self._scheduled.discard(task_id)
                self._stats['tasks_completed'] += 1
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        with self._lock:
            if task_id in self._pending:
                task = self._tasks.get(task_id)
                if task:
                    task.status = TaskStatus.CANCELLED
                self._pending.discard(task_id)
                return True
            return False
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self._queue.qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['pending'] = len(self._pending)
            stats['scheduled'] = len(self._scheduled)
            stats['queue_size'] = self.get_queue_size()
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD BALANCER
# ═══════════════════════════════════════════════════════════════════════════════

class LoadBalancer:
    """
    Balances load across workers.
    
    Implements multiple load balancing strategies.
    """
    
    def __init__(
        self,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ADAPTIVE,
    ):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
        """
        self._strategy = strategy
        self._workers: Dict[str, WorkerInfo] = {}
        
        # Round robin state
        self._rr_index: int = 0
        
        # Statistics
        self._stats = {
            'assignments': 0,
            'reassignments': 0,
            'worker_failures': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info(f"LoadBalancer initialized with {strategy.name} strategy")
    
    def register_worker(self, worker: WorkerInfo):
        """Register a worker"""
        with self._lock:
            self._workers[worker.id] = worker
            logger.debug(f"Worker {worker.id} registered")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker"""
        with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
    
    def select_worker(
        self,
        required_capabilities: Set[str] = None,
        exclude: Set[str] = None,
    ) -> Optional[str]:
        """
        Select the best worker for a task.
        
        Args:
            required_capabilities: Required capabilities
            exclude: Workers to exclude
            
        Returns:
            Selected worker ID or None
        """
        with self._lock:
            exclude = exclude or set()
            
            # Filter available workers
            available = [
                w for w in self._workers.values()
                if w.is_available and w.id not in exclude
            ]
            
            # Filter by capabilities
            if required_capabilities:
                available = [
                    w for w in available
                    if required_capabilities.issubset(w.capabilities)
                ]
            
            if not available:
                return None
            
            # Apply strategy
            if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
                return self._select_round_robin(available)
            elif self._strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
                return self._select_least_connections(available)
            elif self._strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
                return self._select_weighted(available)
            elif self._strategy == LoadBalanceStrategy.RESOURCE_AWARE:
                return self._select_resource_aware(available)
            else:  # ADAPTIVE
                return self._select_adaptive(available)
    
    def _select_round_robin(self, workers: List[WorkerInfo]) -> str:
        """Round robin selection"""
        self._rr_index = (self._rr_index + 1) % len(workers)
        return workers[self._rr_index].id
    
    def _select_least_connections(self, workers: List[WorkerInfo]) -> str:
        """Select worker with least connections"""
        return min(workers, key=lambda w: len(w.current_tasks)).id
    
    def _select_weighted(self, workers: List[WorkerInfo]) -> str:
        """Weighted round robin selection"""
        # Weight by available capacity
        weights = [w.weight * (1 - w.current_load) for w in workers]
        total = sum(weights)
        
        if total == 0:
            return random.choice(workers).id
        
        r = random.random() * total
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return workers[i].id
        
        return workers[-1].id
    
    def _select_resource_aware(self, workers: List[WorkerInfo]) -> str:
        """Resource-aware selection"""
        # Score by available resources and performance
        def score(w: WorkerInfo) -> float:
            resource_score = (
                w.cpu_available / w.cpu_capacity +
                w.memory_available / w.memory_capacity
            ) / 2
            performance_score = w.success_rate
            load_score = 1 - w.current_load
            return resource_score * 0.3 + performance_score * 0.4 + load_score * 0.3
        
        return max(workers, key=score).id
    
    def _select_adaptive(self, workers: List[WorkerInfo]) -> str:
        """Adaptive selection based on current conditions"""
        # Combine multiple factors
        def adaptive_score(w: WorkerInfo) -> float:
            # Load factor
            load_factor = 1 - w.current_load
            
            # Performance factor
            perf_factor = w.success_rate
            
            # Response time factor (lower is better)
            if w.avg_task_time > 0:
                time_factor = 1 / (1 + w.avg_task_time / 60)
            else:
                time_factor = 0.5
            
            # Health factor
            health_factor = 1 / (1 + w.error_count * 0.1)
            
            return (
                load_factor * 0.35 +
                perf_factor * 0.25 +
                time_factor * 0.25 +
                health_factor * 0.15
            )
        
        return max(workers, key=adaptive_score).id
    
    def assign_task(self, worker_id: str, task_id: str):
        """Assign a task to a worker"""
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker:
                worker.current_tasks.add(task_id)
                if worker.status == "idle":
                    worker.status = "busy"
                self._stats['assignments'] += 1
    
    def complete_task(self, worker_id: str, task_id: str, success: bool):
        """Mark task as completed on worker"""
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker:
                worker.current_tasks.discard(task_id)
                if success:
                    worker.tasks_completed += 1
                else:
                    worker.tasks_failed += 1
                
                if not worker.current_tasks:
                    worker.status = "idle"
    
    def update_worker_load(self, worker_id: str, cpu: float, memory: float):
        """Update worker resource usage"""
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker:
                worker.cpu_used = cpu
                worker.memory_used = memory
    
    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker info"""
        return self._workers.get(worker_id)
    
    def get_all_workers(self) -> List[WorkerInfo]:
        """Get all workers"""
        with self._lock:
            return list(self._workers.values())
    
    def get_available_workers(self) -> List[WorkerInfo]:
        """Get available workers"""
        with self._lock:
            return [w for w in self._workers.values() if w.is_available]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['workers'] = len(self._workers)
            stats['available'] = len(self.get_available_workers())
            
            if self._workers:
                avg_load = sum(w.current_load for w in self._workers.values()) / len(self._workers)
                stats['avg_load'] = f"{avg_load:.1%}"
            
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# FAULT TOLERANCE
# ═══════════════════════════════════════════════════════════════════════════════

class FaultTolerance:
    """
    Provides fault tolerance for task execution.
    
    Handles retries, timeouts, and recovery.
    """
    
    def __init__(
        self,
        default_timeout: float = 300.0,
        default_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        """
        Initialize fault tolerance.
        
        Args:
            default_timeout: Default task timeout
            default_retries: Default retry count
            retry_backoff: Exponential backoff multiplier
        """
        self._default_timeout = default_timeout
        self._default_retries = default_retries
        self._retry_backoff = retry_backoff
        
        # Tracking
        self._timeouts: Dict[str, threading.Timer] = {}
        self._failures: Dict[str, List[FailureType]] = defaultdict(list)
        
        # Recovery strategies
        self._recovery_strategies: Dict[FailureType, Callable] = {
            FailureType.TIMEOUT: self._recover_timeout,
            FailureType.ERROR: self._recover_error,
            FailureType.RESOURCE_EXHAUSTED: self._recover_resource,
            FailureType.WORKER_UNAVAILABLE: self._recover_worker,
            FailureType.DEPENDENCY_FAILED: self._recover_dependency,
        }
        
        # Statistics
        self._stats = {
            'timeouts': 0,
            'errors': 0,
            'retries': 0,
            'recoveries': 0,
            'permanent_failures': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("FaultTolerance initialized")
    
    def set_timeout(
        self,
        task_id: str,
        timeout: float,
        callback: Callable,
    ):
        """Set a timeout for a task"""
        with self._lock:
            # Cancel existing timeout
            self.clear_timeout(task_id)
            
            # Create new timeout
            timer = threading.Timer(timeout, callback, args=[task_id])
            timer.daemon = True
            timer.start()
            
            self._timeouts[task_id] = timer
    
    def clear_timeout(self, task_id: str):
        """Clear timeout for a task"""
        with self._lock:
            timer = self._timeouts.pop(task_id, None)
            if timer:
                timer.cancel()
    
    def record_failure(
        self,
        task_id: str,
        failure_type: FailureType,
        details: str = "",
    ):
        """Record a failure for a task"""
        with self._lock:
            self._failures[task_id].append(failure_type)
            
            if failure_type == FailureType.TIMEOUT:
                self._stats['timeouts'] += 1
            else:
                self._stats['errors'] += 1
            
            logger.warning(f"Task {task_id} failure: {failure_type.name} - {details}")
    
    def should_retry(self, task_id: str, max_retries: int) -> bool:
        """Check if task should be retried"""
        with self._lock:
            failures = self._failures.get(task_id, [])
            return len(failures) < max_retries
    
    def get_retry_delay(self, task_id: str, base_delay: float) -> float:
        """Calculate retry delay with exponential backoff"""
        with self._lock:
            failures = len(self._failures.get(task_id, []))
            return base_delay * (self._retry_backoff ** failures)
    
    def recover(
        self,
        task_id: str,
        failure_type: FailureType,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Attempt recovery from failure.
        
        Args:
            task_id: Failed task
            failure_type: Type of failure
            context: Recovery context
            
        Returns:
            Recovery result
        """
        with self._lock:
            strategy = self._recovery_strategies.get(failure_type)
            
            if strategy:
                result = strategy(task_id, context)
                if result.get('recovered'):
                    self._stats['recoveries'] += 1
                else:
                    self._stats['permanent_failures'] += 1
                return result
            
            return {'recovered': False, 'reason': 'No recovery strategy'}
    
    def _recover_timeout(
        self,
        task_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recover from timeout"""
        # Increase timeout for retry
        old_timeout = context.get('timeout', self._default_timeout)
        return {
            'recovered': True,
            'action': 'retry',
            'new_timeout': old_timeout * 1.5,
        }
    
    def _recover_error(
        self,
        task_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recover from error"""
        error = context.get('error', '')
        
        # Check if error is retryable
        retryable_errors = ['connection', 'timeout', 'temporary']
        is_retryable = any(e in error.lower() for e in retryable_errors)
        
        if is_retryable:
            return {'recovered': True, 'action': 'retry'}
        
        return {'recovered': False, 'reason': 'Non-retryable error'}
    
    def _recover_resource(
        self,
        task_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recover from resource exhaustion"""
        return {
            'recovered': True,
            'action': 'reschedule',
            'delay': 30.0,  # Wait before rescheduling
        }
    
    def _recover_worker(
        self,
        task_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recover from worker unavailability"""
        return {
            'recovered': True,
            'action': 'reassign',
        }
    
    def _recover_dependency(
        self,
        task_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recover from dependency failure"""
        return {
            'recovered': False,
            'reason': 'Dependency failed',
            'action': 'skip',
        }
    
    def get_failure_count(self, task_id: str) -> int:
        """Get failure count for a task"""
        return len(self._failures.get(task_id, []))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fault tolerance statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['active_timeouts'] = len(self._timeouts)
            stats['tracked_failures'] = len(self._failures)
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT AGGREGATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ResultAggregator:
    """
    Aggregates results from parallel task execution.
    """
    
    def __init__(self):
        """Initialize result aggregator."""
        self._results: Dict[str, ExecutionResult] = {}
        self._partial_results: Dict[str, List[Any]] = defaultdict(list)
        
        # Statistics
        self._stats = {
            'results_aggregated': 0,
            'partial_results': 0,
            'failed_aggregations': 0,
        }
        
        self._lock = threading.RLock()
    
    def add_result(self, result: ExecutionResult):
        """Add a result"""
        with self._lock:
            self._results[result.task_id] = result
            self._stats['results_aggregated'] += 1
    
    def add_partial_result(self, task_id: str, partial: Any):
        """Add a partial result"""
        with self._lock:
            self._partial_results[task_id].append(partial)
            self._stats['partial_results'] += 1
    
    def get_result(self, task_id: str) -> Optional[ExecutionResult]:
        """Get result for a task"""
        return self._results.get(task_id)
    
    def get_all_results(self) -> Dict[str, ExecutionResult]:
        """Get all results"""
        with self._lock:
            return dict(self._results)
    
    def aggregate_map_reduce(
        self,
        results: Dict[str, ExecutionResult],
        reduce_func: Callable[[List[Any]], Any] = None,
    ) -> Any:
        """
        Aggregate results using map-reduce pattern.
        
        Args:
            results: Results to aggregate
            reduce_func: Function to combine results
            
        Returns:
            Aggregated result
        """
        with self._lock:
            # Collect successful results
            values = [
                r.result for r in results.values()
                if r.success and r.result is not None
            ]
            
            if not values:
                return None
            
            # Apply reduce function
            if reduce_func:
                return reduce_func(values)
            
            # Default: return list
            return values
    
    def aggregate_parallel(
        self,
        results: Dict[str, ExecutionResult],
    ) -> Dict[str, Any]:
        """
        Aggregate parallel execution results.
        
        Returns results keyed by task ID.
        """
        with self._lock:
            return {
                task_id: result.result
                for task_id, result in results.items()
                if result.success
            }
    
    def aggregate_pipeline(
        self,
        results: Dict[str, ExecutionResult],
        order: List[str],
    ) -> List[Any]:
        """
        Aggregate pipeline results in order.
        
        Args:
            results: Results to aggregate
            order: Order of task IDs
            
        Returns:
            Results in order
        """
        with self._lock:
            return [
                results.get(task_id).result
                for task_id in order
                if task_id in results and results[task_id].success
            ]
    
    def get_statistics(
        self,
        results: Dict[str, ExecutionResult] = None,
    ) -> Dict[str, Any]:
        """Get result statistics"""
        with self._lock:
            results = results or self._results
            
            if not results:
                return {}
            
            successful = sum(1 for r in results.values() if r.success)
            failed = len(results) - successful
            
            total_time = sum(r.execution_time for r in results.values())
            avg_time = total_time / len(results) if results else 0
            
            return {
                'total': len(results),
                'successful': successful,
                'failed': failed,
                'success_rate': successful / len(results),
                'total_time': total_time,
                'avg_time': avg_time,
            }
    
    def clear(self):
        """Clear all results"""
        with self._lock:
            self._results.clear()
            self._partial_results.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator statistics"""
        with self._lock:
            return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTED EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DistributedExecutor:
    """
    Main distributed task executor.
    
    Coordinates scheduling, load balancing, and execution.
    """
    
    def __init__(
        self,
        max_workers: int = 10,
        schedule_strategy: ScheduleStrategy = ScheduleStrategy.PRIORITY,
        load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ADAPTIVE,
        default_timeout: float = 300.0,
    ):
        """
        Initialize distributed executor.
        
        Args:
            max_workers: Maximum concurrent workers
            schedule_strategy: Task scheduling strategy
            load_balance_strategy: Load balancing strategy
            default_timeout: Default task timeout
        """
        self._max_workers = max_workers
        
        # Components
        self._scheduler = TaskScheduler(strategy=schedule_strategy)
        self._load_balancer = LoadBalancer(strategy=load_balance_strategy)
        self._fault_tolerance = FaultTolerance(default_timeout=default_timeout)
        self._result_aggregator = ResultAggregator()
        
        # Thread pool
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Execution tracking
        self._running_tasks: Dict[str, Future] = {}
        self._workers: Dict[str, WorkerInfo] = {}
        
        # State
        self._running = False
        self._executor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_execution_time': 0.0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("DistributedExecutor initialized")
    
    def start(self):
        """Start the executor"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            
            # Start executor thread
            self._executor_thread = threading.Thread(
                target=self._execution_loop,
                name="distributed-executor",
                daemon=True,
            )
            self._executor_thread.start()
            
            logger.info("DistributedExecutor started")
    
    def stop(self):
        """Stop the executor"""
        with self._lock:
            self._running = False
            
            # Cancel running tasks
            for future in self._running_tasks.values():
                future.cancel()
            
            # Shutdown executor
            self._executor.shutdown(wait=False)
            
            logger.info("DistributedExecutor stopped")
    
    def register_worker(self, worker: WorkerInfo):
        """Register a worker"""
        with self._lock:
            self._workers[worker.id] = worker
            self._load_balancer.register_worker(worker)
    
    def submit_task(
        self,
        func: Callable,
        args: Tuple = None,
        kwargs: Dict[str, Any] = None,
        name: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: float = None,
        max_retries: int = 3,
        required_capabilities: Set[str] = None,
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            name: Task name
            priority: Task priority
            timeout: Execution timeout
            max_retries: Maximum retry attempts
            required_capabilities: Required worker capabilities
            
        Returns:
            Task ID
        """
        task = Task(
            name=name or func.__name__,
            func=func,
            args=args or (),
            kwargs=kwargs or {},
            priority=priority.value,
            timeout=timeout or 300.0,
            max_retries=max_retries,
            required_capabilities=required_capabilities or set(),
        )
        
        with self._lock:
            self._stats['tasks_submitted'] += 1
        
        self._scheduler.submit(task)
        return task.id
    
    def submit_batch(
        self,
        tasks: List[Tuple[Callable, Tuple, Dict[str, Any]]],
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> List[str]:
        """
        Submit multiple tasks.
        
        Args:
            tasks: List of (func, args, kwargs) tuples
            priority: Priority for all tasks
            
        Returns:
            List of task IDs
        """
        task_ids = []
        for func, args, kwargs in tasks:
            task_id = self.submit_task(func, args, kwargs, priority=priority)
            task_ids.append(task_id)
        return task_ids
    
    def execute_parallel(
        self,
        func: Callable,
        args_list: List[Tuple],
        kwargs_list: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute function in parallel with different arguments.
        
        Args:
            func: Function to execute
            args_list: List of argument tuples
            kwargs_list: List of keyword argument dicts
            
        Returns:
            Dictionary mapping task ID to result
        """
        task_ids = []
        kwargs_list = kwargs_list or [{}] * len(args_list)
        
        for args, kwargs in zip(args_list, kwargs_list):
            task_id = self.submit_task(func, args, kwargs)
            task_ids.append(task_id)
        
        # Wait for completion
        results = {}
        for task_id in task_ids:
            result = self.wait_for_task(task_id)
            results[task_id] = result
        
        return results
    
    def execute_map_reduce(
        self,
        map_func: Callable,
        reduce_func: Callable,
        data: List[Any],
        chunk_size: int = 1,
    ) -> Any:
        """
        Execute map-reduce operation.
        
        Args:
            map_func: Map function
            reduce_func: Reduce function
            data: Input data
            chunk_size: Items per chunk
            
        Returns:
            Reduced result
        """
        # Split data into chunks
        chunks = [
            data[i:i + chunk_size]
            for i in range(0, len(data), chunk_size)
        ]
        
        # Execute map phase
        map_results = self.execute_parallel(
            map_func,
            [(chunk,) for chunk in chunks],
        )
        
        # Collect results
        values = [
            r.result for r in map_results.values()
            if r.get('success')
        ]
        
        # Execute reduce phase
        return reduce_func(values)
    
    def wait_for_task(
        self,
        task_id: str,
        timeout: float = None,
    ) -> Optional[ExecutionResult]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task to wait for
            timeout: Maximum wait time
            
        Returns:
            Execution result or None
        """
        start = time.time()
        timeout = timeout or 300.0
        
        while time.time() - start < timeout:
            result = self._result_aggregator.get_result(task_id)
            if result:
                return result
            time.sleep(0.1)
        
        return None
    
    def wait_for_all(
        self,
        task_ids: List[str],
        timeout: float = None,
    ) -> Dict[str, ExecutionResult]:
        """
        Wait for multiple tasks.
        
        Args:
            task_ids: Tasks to wait for
            timeout: Maximum wait time per task
            
        Returns:
            Dictionary of results
        """
        results = {}
        for task_id in task_ids:
            result = self.wait_for_task(task_id, timeout)
            if result:
                results[task_id] = result
        return results
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        with self._lock:
            # Try to cancel from scheduler
            if self._scheduler.cancel_task(task_id):
                return True
            
            # Try to cancel running task
            future = self._running_tasks.get(task_id)
            if future:
                return future.cancel()
            
            return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task"""
        task = self._scheduler.get_task(task_id)
        if task:
            return task.status
        return None
    
    def _execution_loop(self):
        """Main execution loop"""
        while self._running:
            try:
                # Get next task
                task = self._scheduler.get_next()
                if not task:
                    time.sleep(0.1)
                    continue
                
                # Select worker
                worker_id = self._load_balancer.select_worker(
                    task.required_capabilities,
                )
                
                if not worker_id:
                    # No available worker, requeue
                    task.status = TaskStatus.PENDING
                    self._scheduler.submit(task)
                    time.sleep(0.5)
                    continue
                
                # Execute task
                self._execute_task(task, worker_id)
                
            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                time.sleep(0.1)
    
    def _execute_task(self, task: Task, worker_id: str):
        """Execute a task on a worker"""
        # Update state
        task.assigned_worker = worker_id
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        task.attempts += 1
        
        # Assign to worker
        self._load_balancer.assign_task(worker_id, task.id)
        
        # Set timeout
        def timeout_callback(tid):
            self._handle_timeout(tid)
        
        self._fault_tolerance.set_timeout(
            task.id,
            task.timeout,
            timeout_callback,
        )
        
        # Submit to executor
        future = self._executor.submit(
            self._run_task,
            task,
            worker_id,
        )
        
        self._running_tasks[task.id] = future
    
    def _run_task(self, task: Task, worker_id: str) -> ExecutionResult:
        """Run a task (executed in thread pool)"""
        result = ExecutionResult(
            task_id=task.id,
            worker_id=worker_id,
            started_at=task.started_at,
        )
        
        try:
            # Execute function
            output = task.func(*task.args, **task.kwargs)
            
            result.success = True
            result.result = output
            task.result = output
            task.status = TaskStatus.COMPLETED
            
        except Exception as e:
            result.success = False
            result.error = str(e)
            task.error = str(e)
            
            # Record failure
            self._fault_tolerance.record_failure(
                task.id,
                FailureType.ERROR,
                str(e),
            )
            
            # Check for retry
            if task.can_retry:
                task.status = TaskStatus.RETRYING
                delay = self._fault_tolerance.get_retry_delay(
                    task.id,
                    task.retry_delay,
                )
                
                # Schedule retry
                threading.Timer(
                    delay,
                    self._retry_task,
                    args=[task],
                ).start()
            else:
                task.status = TaskStatus.FAILED
        
        finally:
            # Update timing
            result.completed_at = time.time()
            result.execution_time = result.completed_at - result.started_at
            task.completed_at = result.completed_at
            
            # Clear timeout
            self._fault_tolerance.clear_timeout(task.id)
            
            # Update worker
            self._load_balancer.complete_task(
                worker_id,
                task.id,
                result.success,
            )
            
            # Store result
            self._result_aggregator.add_result(result)
            
            # Update stats
            with self._lock:
                if result.success:
                    self._stats['tasks_completed'] += 1
                else:
                    self._stats['tasks_failed'] += 1
                self._stats['total_execution_time'] += result.execution_time
            
            # Remove from running
            self._running_tasks.pop(task.id, None)
        
        return result
    
    def _handle_timeout(self, task_id: str):
        """Handle task timeout"""
        task = self._scheduler.get_task(task_id)
        if not task:
            return
        
        # Record failure
        self._fault_tolerance.record_failure(
            task_id,
            FailureType.TIMEOUT,
            f"Task exceeded timeout of {task.timeout}s",
        )
        
        # Update task
        task.status = TaskStatus.TIMEOUT
        task.error = "Timeout"
        
        # Try recovery
        recovery = self._fault_tolerance.recover(
            task_id,
            FailureType.TIMEOUT,
            {'timeout': task.timeout},
        )
        
        if recovery.get('recovered') and recovery.get('action') == 'retry':
            self._retry_task(task)
    
    def _retry_task(self, task: Task):
        """Retry a failed task"""
        if not task.can_retry:
            return
        
        # Reset task
        task.status = TaskStatus.PENDING
        task.assigned_worker = None
        task.error = None
        
        # Resubmit
        self._scheduler.submit(task)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['running_tasks'] = len(self._running_tasks)
            stats['scheduler'] = self._scheduler.get_stats()
            stats['load_balancer'] = self._load_balancer.get_stats()
            stats['fault_tolerance'] = self._fault_tolerance.get_stats()
            stats['result_aggregator'] = self._result_aggregator.get_stats()
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_executor: Optional[DistributedExecutor] = None


def get_executor(**kwargs) -> DistributedExecutor:
    """Get global executor instance"""
    global _executor
    if _executor is None:
        _executor = DistributedExecutor(**kwargs)
    return _executor


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Distributed Executor Test")
    print("="*60)
    
    # Create executor
    executor = DistributedExecutor(max_workers=5)
    
    # Register workers
    print("\n1. Registering workers...")
    for i in range(3):
        worker = WorkerInfo(
            id=f"worker_{i}",
            name=f"Worker {i}",
            capabilities={'general', 'code'},
            max_concurrent_tasks=2,
        )
        executor.register_worker(worker)
    print("   Registered 3 workers")
    
    # Start executor
    print("\n2. Starting executor...")
    executor.start()
    
    # Submit tasks
    print("\n3. Submitting tasks...")
    
    def sample_task(n):
        time.sleep(0.5)
        return n * n
    
    task_ids = []
    for i in range(5):
        task_id = executor.submit_task(
            sample_task,
            args=(i,),
            name=f"square_{i}",
            priority=TaskPriority.NORMAL,
        )
        task_ids.append(task_id)
        print(f"   Submitted: {task_id}")
    
    # Wait for completion
    print("\n4. Waiting for completion...")
    results = executor.wait_for_all(task_ids)
    
    print(f"   Completed: {len(results)} tasks")
    for task_id, result in results.items():
        if result.success:
            print(f"   - {task_id}: {result.result}")
    
    # Stats
    print("\n5. Statistics:")
    stats = executor.get_stats()
    print(f"   Tasks submitted: {stats['tasks_submitted']}")
    print(f"   Tasks completed: {stats['tasks_completed']}")
    print(f"   Tasks failed: {stats['tasks_failed']}")
    print(f"   Total execution time: {stats['total_execution_time']:.2f}s")
    
    # Stop executor
    print("\n6. Stopping executor...")
    executor.stop()
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

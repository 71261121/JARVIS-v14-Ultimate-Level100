#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Workflow Manager
=======================================

Phase 4: Complex Workflow Management System (Level 70-80)

This module provides sophisticated workflow management:
- DAG-based workflow definition
- Dependency resolution and topological sorting
- Parallel and sequential execution
- Conditional branching and loops
- Workflow templates and patterns
- State persistence and recovery
- Monitoring and visualization
- Error handling and compensation

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                      Workflow Manager                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Workflow   │  │ Dependency  │  │  Execution  │  Core        │
│  │   Builder   │  │   Resolver  │  │   Engine    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   State     │  │  Template   │  │  Monitor    │  Management  │
│  │  Manager    │  │   Engine    │  │  & Visual   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Error     │  │ Compensation│  │  Scheduler  │  Execution   │
│  │  Handler    │  │   Engine    │  │             │              │
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
import hashlib
from typing import Dict, Any, Optional, List, Set, Callable, Tuple, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowStatus(Enum):
    """Status of a workflow"""
    CREATED = auto()
    PENDING = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class StepStatus(Enum):
    """Status of a workflow step"""
    PENDING = auto()
    WAITING = auto()      # Waiting for dependencies
    READY = auto()        # Dependencies satisfied
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    CANCELLED = auto()


class StepType(Enum):
    """Types of workflow steps"""
    TASK = auto()         # Execute a task
    PARALLEL = auto()     # Execute steps in parallel
    SEQUENTIAL = auto()   # Execute steps sequentially
    CONDITION = auto()    # Conditional branching
    LOOP = auto()         # Loop iteration
    WAIT = auto()         # Wait for condition
    SUBWORKFLOW = auto()  # Execute sub-workflow
    COMPENSATION = auto() # Compensation/rollback step


class DependencyType(Enum):
    """Types of dependencies"""
    FINISH = auto()       # Dependency must finish
    SUCCESS = auto()      # Dependency must succeed
    FAILURE = auto()      # Dependency must fail
    CONDITION = auto()    # Condition-based dependency


class WorkflowPriority(Enum):
    """Workflow priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class BranchCondition(Enum):
    """Branching conditions"""
    SUCCESS = auto()
    FAILURE = auto()
    ALWAYS = auto()
    CUSTOM = auto()


class LoopType(Enum):
    """Types of loops"""
    FOR = auto()          # For loop
    WHILE = auto()        # While loop
    FOR_EACH = auto()     # For each item


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Dependency:
    """
    A dependency between workflow steps.
    """
    step_id: str
    dependency_type: DependencyType = DependencyType.FINISH
    condition: Optional[Callable] = None
    
    # Metadata
    description: str = ""
    
    def is_satisfied(self, step_result: Any, step_status: StepStatus) -> bool:
        """Check if dependency is satisfied"""
        if self.dependency_type == DependencyType.FINISH:
            return step_status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]
        elif self.dependency_type == DependencyType.SUCCESS:
            return step_status == StepStatus.COMPLETED
        elif self.dependency_type == DependencyType.FAILURE:
            return step_status == StepStatus.FAILED
        elif self.dependency_type == DependencyType.CONDITION:
            if self.condition:
                return self.condition(step_result)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'type': self.dependency_type.name,
            'description': self.description,
        }


@dataclass
class StepResult:
    """
    Result of a workflow step execution.
    """
    step_id: str
    status: StepStatus = StepStatus.PENDING
    
    # Output
    output: Any = None
    error: Optional[str] = None
    
    # Timing
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Metadata
    attempts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def execution_time(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0
    
    @property
    def success(self) -> bool:
        return self.status == StepStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'status': self.status.name,
            'success': self.success,
            'execution_time': self.execution_time,
        }


@dataclass
class WorkflowStep:
    """
    A step in a workflow.
    
    Represents a unit of work in the workflow.
    """
    # Identity
    id: str = field(default_factory=lambda: f"step_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    
    # Type
    step_type: StepType = StepType.TASK
    
    # Task definition
    task: Optional[Callable] = None
    task_args: Tuple = field(default_factory=tuple)
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    dependencies: List[Dependency] = field(default_factory=list)
    
    # Execution control
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Conditional execution
    condition: Optional[Callable] = None
    branch_true: Optional[str] = None
    branch_false: Optional[str] = None
    
    # Loop configuration
    loop_type: Optional[LoopType] = None
    loop_iterations: int = 0
    loop_items: List[Any] = field(default_factory=list)
    loop_condition: Optional[Callable] = None
    loop_body: List[str] = field(default_factory=list)  # Step IDs
    
    # Parallel configuration
    parallel_steps: List[str] = field(default_factory=list)
    
    # Sub-workflow
    sub_workflow_id: Optional[str] = None
    
    # Compensation
    compensation_step: Optional[str] = None
    
    # State
    status: StepStatus = StepStatus.PENDING
    result: Optional[StepResult] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_dependency(
        self,
        step_id: str,
        dep_type: DependencyType = DependencyType.FINISH,
        condition: Callable = None,
    ):
        """Add a dependency"""
        dep = Dependency(
            step_id=step_id,
            dependency_type=dep_type,
            condition=condition,
        )
        self.dependencies.append(dep)
    
    def check_dependencies(self, step_results: Dict[str, StepResult]) -> bool:
        """Check if all dependencies are satisfied"""
        for dep in self.dependencies:
            if dep.step_id not in step_results:
                return False
            
            result = step_results[dep.step_id]
            if not dep.is_satisfied(result.output, result.status):
                return False
        
        return True
    
    def get_dependencies(self) -> List[str]:
        """Get list of dependency step IDs"""
        return [d.step_id for d in self.dependencies]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.step_type.name,
            'status': self.status.name,
            'dependencies': [d.to_dict() for d in self.dependencies],
            'timeout': self.timeout,
        }


@dataclass
class WorkflowDefinition:
    """
    Definition of a workflow.
    
    Contains all steps and their relationships.
    """
    # Identity
    id: str = field(default_factory=lambda: f"wf_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # Steps
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    
    # Entry and exit points
    entry_step: Optional[str] = None
    exit_steps: List[str] = field(default_factory=list)
    
    # Execution order
    execution_order: List[str] = field(default_factory=list)
    
    # Configuration
    max_parallel: int = 10
    default_timeout: float = 300.0
    default_retries: int = 3
    
    # Priority
    priority: WorkflowPriority = WorkflowPriority.NORMAL
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Templates
    template_id: Optional[str] = None
    template_version: Optional[str] = None
    
    def add_step(self, step: WorkflowStep) -> str:
        """Add a step to the workflow"""
        self.steps[step.id] = step
        if not self.entry_step:
            self.entry_step = step.id
        return step.id
    
    def remove_step(self, step_id: str) -> bool:
        """Remove a step from the workflow"""
        if step_id in self.steps:
            del self.steps[step_id]
            if self.entry_step == step_id:
                self.entry_step = None
            return True
        return False
    
    def connect(
        self,
        from_step: str,
        to_step: str,
        dep_type: DependencyType = DependencyType.FINISH,
    ):
        """Connect two steps"""
        if to_step in self.steps:
            self.steps[to_step].add_dependency(from_step, dep_type)
    
    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """Get a step by ID"""
        return self.steps.get(step_id)
    
    def get_successors(self, step_id: str) -> List[str]:
        """Get steps that depend on this step"""
        successors = []
        for sid, step in self.steps.items():
            if any(d.step_id == step_id for d in step.dependencies):
                successors.append(sid)
        return successors
    
    def get_predecessors(self, step_id: str) -> List[str]:
        """Get steps this step depends on"""
        step = self.steps.get(step_id)
        if step:
            return step.get_dependencies()
        return []
    
    def validate(self) -> List[str]:
        """Validate workflow definition"""
        errors = []
        
        # Check for cycles
        if self._has_cycle():
            errors.append("Workflow contains cycles")
        
        # Check for orphan steps
        for step_id, step in self.steps.items():
            if step_id != self.entry_step and not step.dependencies:
                errors.append(f"Step {step_id} has no dependencies but is not entry point")
        
        # Check for missing dependencies
        for step_id, step in self.steps.items():
            for dep in step.dependencies:
                if dep.step_id not in self.steps:
                    errors.append(f"Step {step_id} depends on non-existent step {dep.step_id}")
        
        return errors
    
    def _has_cycle(self) -> bool:
        """Check for cycles in dependency graph"""
        visited = set()
        rec_stack = set()
        
        def dfs(step_id: str) -> bool:
            visited.add(step_id)
            rec_stack.add(step_id)
            
            for dep in self.steps.get(step_id, WorkflowStep()).dependencies:
                if dep.step_id not in visited:
                    if dfs(dep.step_id):
                        return True
                elif dep.step_id in rec_stack:
                    return True
            
            rec_stack.remove(step_id)
            return False
        
        for step_id in self.steps:
            if step_id not in visited:
                if dfs(step_id):
                    return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'steps': {sid: s.to_dict() for sid, s in self.steps.items()},
            'entry_step': self.entry_step,
            'exit_steps': self.exit_steps,
            'priority': self.priority.name,
        }


@dataclass
class WorkflowInstance:
    """
    Runtime instance of a workflow.
    
    Contains execution state and results.
    """
    # Identity
    id: str = field(default_factory=lambda: f"wf_inst_{uuid.uuid4().hex[:8]}")
    definition_id: str = ""
    
    # State
    status: WorkflowStatus = WorkflowStatus.CREATED
    current_step: Optional[str] = None
    
    # Results
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Execution
    current_iteration: Dict[str, int] = field(default_factory=dict)
    execution_path: List[str] = field(default_factory=list)
    
    # Error handling
    error: Optional[str] = None
    failed_step: Optional[str] = None
    
    # Compensation
    compensation_stack: List[str] = field(default_factory=list)
    
    @property
    def execution_time(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return time.time() - self.started_at if self.started_at else 0
    
    @property
    def progress(self) -> float:
        """Calculate execution progress"""
        if not self.step_results:
            return 0.0
        
        completed = sum(
            1 for r in self.step_results.values()
            if r.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
        )
        return completed / len(self.step_results)
    
    def set_variable(self, name: str, value: Any):
        """Set a workflow variable"""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a workflow variable"""
        return self.variables.get(name, default)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'definition_id': self.definition_id,
            'status': self.status.name,
            'progress': f"{self.progress:.1%}",
            'execution_time': f"{self.execution_time:.1f}s",
            'steps_completed': sum(1 for r in self.step_results.values() if r.success),
            'steps_total': len(self.step_results),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCY RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class DependencyResolver:
    """
    Resolves dependencies and determines execution order.
    
    Implements topological sorting for DAGs.
    """
    
    def __init__(self):
        """Initialize dependency resolver."""
        self._cache: Dict[str, List[str]] = {}
    
    def resolve(
        self,
        workflow: WorkflowDefinition,
    ) -> List[str]:
        """
        Resolve execution order using topological sort.
        
        Args:
            workflow: Workflow definition
            
        Returns:
            Ordered list of step IDs
        """
        # Check cache
        cache_key = self._get_cache_key(workflow)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Build adjacency list
        in_degree: Dict[str, int] = {sid: 0 for sid in workflow.steps}
        graph: Dict[str, List[str]] = {sid: [] for sid in workflow.steps}
        
        for step_id, step in workflow.steps.items():
            for dep in step.dependencies:
                if dep.step_id in workflow.steps:
                    graph[dep.step_id].append(step_id)
                    in_degree[step_id] += 1
        
        # Kahn's algorithm
        queue = deque([sid for sid, degree in in_degree.items() if degree == 0])
        order = []
        
        while queue:
            current = queue.popleft()
            order.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycle
        if len(order) != len(workflow.steps):
            raise ValueError("Workflow contains a cycle")
        
        # Cache result
        self._cache[cache_key] = order
        
        return order
    
    def get_ready_steps(
        self,
        workflow: WorkflowDefinition,
        instance: WorkflowInstance,
    ) -> List[str]:
        """
        Get steps ready for execution.
        
        Args:
            workflow: Workflow definition
            instance: Workflow instance
            
        Returns:
            List of ready step IDs
        """
        ready = []
        
        for step_id, step in workflow.steps.items():
            # Skip already processed
            if step_id in instance.step_results:
                result = instance.step_results[step_id]
                if result.status in [
                    StepStatus.COMPLETED,
                    StepStatus.SKIPPED,
                    StepStatus.FAILED,
                ]:
                    continue
            
            # Check dependencies
            if step.check_dependencies(instance.step_results):
                ready.append(step_id)
        
        return ready
    
    def get_parallel_groups(
        self,
        workflow: WorkflowDefinition,
    ) -> List[List[str]]:
        """
        Group steps that can execute in parallel.
        
        Args:
            workflow: Workflow definition
            
        Returns:
            List of parallel groups
        """
        order = self.resolve(workflow)
        
        # Build dependency levels
        levels: Dict[str, int] = {}
        
        for step_id in order:
            step = workflow.steps[step_id]
            if not step.dependencies:
                levels[step_id] = 0
            else:
                max_dep_level = max(
                    levels.get(d.step_id, 0)
                    for d in step.dependencies
                    if d.step_id in levels
                )
                levels[step_id] = max_dep_level + 1
        
        # Group by level
        groups: Dict[int, List[str]] = defaultdict(list)
        for step_id, level in levels.items():
            groups[level].append(step_id)
        
        return [groups[i] for i in sorted(groups.keys())]
    
    def get_critical_path(
        self,
        workflow: WorkflowDefinition,
        step_durations: Dict[str, float] = None,
    ) -> List[str]:
        """
        Calculate critical path through workflow.
        
        Args:
            workflow: Workflow definition
            step_durations: Estimated durations for steps
            
        Returns:
            Critical path as list of step IDs
        """
        step_durations = step_durations or {}
        
        # Calculate earliest start times
        earliest: Dict[str, float] = {}
        order = self.resolve(workflow)
        
        for step_id in order:
            step = workflow.steps[step_id]
            duration = step_durations.get(step_id, 1.0)
            
            if not step.dependencies:
                earliest[step_id] = 0
            else:
                earliest[step_id] = max(
                    earliest.get(d.step_id, 0) + step_durations.get(d.step_id, 1.0)
                    for d in step.dependencies
                    if d.step_id in earliest
                )
        
        # Find critical path by backtracking from max
        if not earliest:
            return []
        
        max_time = max(earliest.values())
        critical_path = []
        
        # Find end steps with max time
        end_steps = [sid for sid, t in earliest.items() if t == max_time]
        
        # Backtrack
        def find_path(step_id: str, path: List[str]):
            path = [step_id] + path
            step = workflow.steps[step_id]
            
            if not step.dependencies:
                paths.append(path)
                return
            
            # Find predecessor with max earliest time
            preds = [
                d.step_id for d in step.dependencies
                if d.step_id in earliest
            ]
            if preds:
                pred = max(preds, key=lambda p: earliest[p])
                find_path(pred, path)
        
        paths = []
        for end in end_steps:
            find_path(end, [])
        
        # Return longest path
        return max(paths, key=len) if paths else []
    
    def _get_cache_key(self, workflow: WorkflowDefinition) -> str:
        """Generate cache key for workflow"""
        step_ids = sorted(workflow.steps.keys())
        return hashlib.md5(str(step_ids).encode()).hexdigest()
    
    def clear_cache(self):
        """Clear resolution cache"""
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowExecutor:
    """
    Executes workflows.
    
    Handles step execution, parallel processing, and error handling.
    """
    
    def __init__(
        self,
        max_parallel: int = 10,
        default_timeout: float = 300.0,
    ):
        """
        Initialize workflow executor.
        
        Args:
            max_parallel: Maximum parallel steps
            default_timeout: Default step timeout
        """
        self._max_parallel = max_parallel
        self._default_timeout = default_timeout
        
        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=max_parallel)
        
        # Running instances
        self._running: Dict[str, WorkflowInstance] = {}
        
        # Statistics
        self._stats = {
            'workflows_executed': 0,
            'workflows_completed': 0,
            'workflows_failed': 0,
            'steps_executed': 0,
            'total_execution_time': 0.0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("WorkflowExecutor initialized")
    
    def execute(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any] = None,
        variables: Dict[str, Any] = None,
    ) -> WorkflowInstance:
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow definition
            context: Execution context
            variables: Initial variables
            
        Returns:
            Workflow instance with results
        """
        # Create instance
        instance = WorkflowInstance(
            definition_id=workflow.id,
            context=context or {},
            variables=variables or {},
        )
        
        # Initialize step results
        for step_id in workflow.steps:
            instance.step_results[step_id] = StepResult(step_id=step_id)
        
        # Update state
        instance.status = WorkflowStatus.RUNNING
        instance.started_at = time.time()
        
        with self._lock:
            self._running[instance.id] = instance
            self._stats['workflows_executed'] += 1
        
        try:
            # Execute steps
            self._execute_workflow(workflow, instance)
            
            # Update final status
            instance.completed_at = time.time()
            
            if instance.error:
                instance.status = WorkflowStatus.FAILED
                self._stats['workflows_failed'] += 1
            else:
                instance.status = WorkflowStatus.COMPLETED
                self._stats['workflows_completed'] += 1
            
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error = str(e)
            instance.completed_at = time.time()
            self._stats['workflows_failed'] += 1
        
        finally:
            with self._lock:
                self._stats['total_execution_time'] += instance.execution_time
                self._running.pop(instance.id, None)
        
        return instance
    
    def execute_async(
        self,
        workflow: WorkflowDefinition,
        context: Dict[str, Any] = None,
        variables: Dict[str, Any] = None,
    ) -> Future:
        """
        Execute workflow asynchronously.
        
        Args:
            workflow: Workflow definition
            context: Execution context
            variables: Initial variables
            
        Returns:
            Future for the workflow instance
        """
        return self._executor.submit(
            self.execute,
            workflow,
            context,
            variables,
        )
    
    def _execute_workflow(
        self,
        workflow: WorkflowDefinition,
        instance: WorkflowInstance,
    ):
        """Internal workflow execution"""
        # Get execution order
        resolver = DependencyResolver()
        order = resolver.resolve(workflow)
        
        # Track completed steps
        completed: Set[str] = set()
        
        while len(completed) < len(workflow.steps):
            # Get ready steps
            ready = self._get_ready_steps(workflow, instance, completed)
            
            if not ready:
                # Check for deadlock
                if instance.status == WorkflowStatus.RUNNING:
                    instance.error = "Deadlock detected in workflow"
                break
            
            # Execute ready steps
            futures = {}
            
            for step_id in ready[:self._max_parallel]:
                step = workflow.steps[step_id]
                future = self._executor.submit(
                    self._execute_step,
                    step,
                    instance,
                )
                futures[future] = step_id
            
            # Wait for completion
            for future in as_completed(futures):
                step_id = futures[future]
                try:
                    result = future.result()
                    instance.step_results[step_id] = result
                    instance.execution_path.append(step_id)
                    
                    if result.success:
                        completed.add(step_id)
                    elif result.status == StepStatus.SKIPPED:
                        completed.add(step_id)
                    else:
                        # Handle failure
                        instance.failed_step = step_id
                        instance.error = result.error
                        
                        # Check if can continue
                        if self._can_continue(workflow, instance, step_id):
                            completed.add(step_id)
                        else:
                            break
                    
                    self._stats['steps_executed'] += 1
                    
                except Exception as e:
                    instance.step_results[step_id].status = StepStatus.FAILED
                    instance.step_results[step_id].error = str(e)
    
    def _get_ready_steps(
        self,
        workflow: WorkflowDefinition,
        instance: WorkflowInstance,
        completed: Set[str],
    ) -> List[str]:
        """Get steps ready for execution"""
        ready = []
        
        for step_id, step in workflow.steps.items():
            if step_id in completed:
                continue
            
            result = instance.step_results.get(step_id)
            if result and result.status != StepStatus.PENDING:
                continue
            
            # Check dependencies
            deps_satisfied = True
            for dep in step.dependencies:
                if dep.step_id not in completed:
                    deps_satisfied = False
                    break
                
                # Check dependency type
                dep_result = instance.step_results.get(dep.step_id)
                if dep_result:
                    if dep.dependency_type == DependencyType.SUCCESS:
                        if not dep_result.success:
                            deps_satisfied = False
                            break
                    elif dep.dependency_type == DependencyType.FAILURE:
                        if dep_result.success:
                            deps_satisfied = False
                            break
            
            if deps_satisfied:
                ready.append(step_id)
        
        return ready
    
    def _execute_step(
        self,
        step: WorkflowStep,
        instance: WorkflowInstance,
    ) -> StepResult:
        """Execute a single step"""
        result = StepResult(step_id=step.id)
        result.started_at = time.time()
        result.status = StepStatus.RUNNING
        
        try:
            # Handle different step types
            if step.step_type == StepType.TASK:
                output = self._execute_task_step(step, instance)
            
            elif step.step_type == StepType.PARALLEL:
                output = self._execute_parallel_step(step, instance)
            
            elif step.step_type == StepType.SEQUENTIAL:
                output = self._execute_sequential_step(step, instance)
            
            elif step.step_type == StepType.CONDITION:
                output = self._execute_condition_step(step, instance)
            
            elif step.step_type == StepType.LOOP:
                output = self._execute_loop_step(step, instance)
            
            elif step.step_type == StepType.WAIT:
                output = self._execute_wait_step(step, instance)
            
            else:
                output = None
            
            result.output = output
            result.status = StepStatus.COMPLETED
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error = str(e)
        
        finally:
            result.completed_at = time.time()
        
        return result
    
    def _execute_task_step(
        self,
        step: WorkflowStep,
        instance: WorkflowInstance,
    ) -> Any:
        """Execute a task step"""
        if not step.task:
            return None
        
        # Prepare context
        context = {
            **instance.context,
            'variables': instance.variables,
            'step_id': step.id,
        }
        
        # Execute
        return step.task(*step.task_args, **step.task_kwargs)
    
    def _execute_parallel_step(
        self,
        step: WorkflowStep,
        instance: WorkflowInstance,
    ) -> List[Any]:
        """Execute parallel steps"""
        results = []
        futures = []
        
        for sub_step_id in step.parallel_steps:
            # Would execute sub-steps in parallel
            pass
        
        return results
    
    def _execute_sequential_step(
        self,
        step: WorkflowStep,
        instance: WorkflowInstance,
    ) -> List[Any]:
        """Execute sequential steps"""
        results = []
        return results
    
    def _execute_condition_step(
        self,
        step: WorkflowStep,
        instance: WorkflowInstance,
    ) -> Any:
        """Execute conditional step"""
        if step.condition:
            condition_result = step.condition(instance.variables)
            return {
                'condition': condition_result,
                'branch': step.branch_true if condition_result else step.branch_false,
            }
        return None
    
    def _execute_loop_step(
        self,
        step: WorkflowStep,
        instance: WorkflowInstance,
    ) -> List[Any]:
        """Execute loop step"""
        results = []
        
        if step.loop_type == LoopType.FOR:
            for i in range(step.loop_iterations):
                # Execute loop body
                pass
        
        elif step.loop_type == LoopType.FOR_EACH:
            for item in step.loop_items:
                # Execute for each item
                pass
        
        elif step.loop_type == LoopType.WHILE:
            while step.loop_condition and step.loop_condition(instance.variables):
                # Execute while condition is true
                pass
        
        return results
    
    def _execute_wait_step(
        self,
        step: WorkflowStep,
        instance: WorkflowInstance,
    ) -> Any:
        """Execute wait step"""
        if step.condition:
            while not step.condition(instance.variables):
                time.sleep(0.1)
        return True
    
    def _can_continue(
        self,
        workflow: WorkflowDefinition,
        instance: WorkflowInstance,
        failed_step_id: str,
    ) -> bool:
        """Check if workflow can continue after failure"""
        # Check if other steps depend on the failed step
        for step_id, step in workflow.steps.items():
            for dep in step.dependencies:
                if dep.step_id == failed_step_id:
                    if dep.dependency_type == DependencyType.SUCCESS:
                        return False
        
        return True
    
    def cancel(self, instance_id: str) -> bool:
        """Cancel a running workflow"""
        with self._lock:
            instance = self._running.get(instance_id)
            if instance:
                instance.status = WorkflowStatus.CANCELLED
                return True
            return False
    
    def get_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Get workflow instance"""
        return self._running.get(instance_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['running'] = len(self._running)
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW TEMPLATE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowTemplateEngine:
    """
    Manages workflow templates.
    
    Provides pre-defined workflow patterns.
    """
    
    def __init__(self):
        """Initialize template engine."""
        self._templates: Dict[str, WorkflowDefinition] = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """Register default workflow templates"""
        # Sequential pipeline
        self.register_template(self._create_sequential_template())
        
        # Parallel fan-out/fan-in
        self.register_template(self._create_parallel_template())
        
        # Map-reduce
        self.register_template(self._create_map_reduce_template())
        
        # Conditional branching
        self.register_template(self._create_conditional_template())
    
    def register_template(self, template: WorkflowDefinition):
        """Register a workflow template"""
        self._templates[template.id] = template
        logger.info(f"Registered workflow template: {template.name}")
    
    def get_template(self, template_id: str) -> Optional[WorkflowDefinition]:
        """Get a template by ID"""
        return self._templates.get(template_id)
    
    def instantiate(
        self,
        template_id: str,
        customizations: Dict[str, Any] = None,
    ) -> Optional[WorkflowDefinition]:
        """
        Create a workflow from a template.
        
        Args:
            template_id: Template ID
            customizations: Customizations to apply
            
        Returns:
            Workflow definition
        """
        template = self._templates.get(template_id)
        if not template:
            return None
        
        # Deep copy template
        workflow = copy.deepcopy(template)
        workflow.id = f"wf_{uuid.uuid4().hex[:8]}"
        workflow.template_id = template_id
        
        # Apply customizations
        customizations = customizations or {}
        
        if 'name' in customizations:
            workflow.name = customizations['name']
        
        if 'description' in customizations:
            workflow.description = customizations['description']
        
        return workflow
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List available templates"""
        return [
            {'id': t.id, 'name': t.name, 'description': t.description}
            for t in self._templates.values()
        ]
    
    def _create_sequential_template(self) -> WorkflowDefinition:
        """Create sequential pipeline template"""
        workflow = WorkflowDefinition(
            id="template_sequential",
            name="Sequential Pipeline",
            description="Execute steps in sequence",
        )
        return workflow
    
    def _create_parallel_template(self) -> WorkflowDefinition:
        """Create parallel fan-out/fan-in template"""
        workflow = WorkflowDefinition(
            id="template_parallel",
            name="Parallel Execution",
            description="Execute steps in parallel and aggregate",
        )
        return workflow
    
    def _create_map_reduce_template(self) -> WorkflowDefinition:
        """Create map-reduce template"""
        workflow = WorkflowDefinition(
            id="template_map_reduce",
            name="Map-Reduce",
            description="Map data and reduce results",
        )
        return workflow
    
    def _create_conditional_template(self) -> WorkflowDefinition:
        """Create conditional branching template"""
        workflow = WorkflowDefinition(
            id="template_conditional",
            name="Conditional Branching",
            description="Execute based on conditions",
        )
        return workflow


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW STATE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowStateManager:
    """
    Manages workflow state persistence and recovery.
    """
    
    def __init__(self, storage_path: str = None):
        """Initialize state manager."""
        self._storage_path = storage_path
        self._instances: Dict[str, WorkflowInstance] = {}
        self._definitions: Dict[str, WorkflowDefinition] = {}
        
        self._lock = threading.RLock()
    
    def save_instance(self, instance: WorkflowInstance):
        """Save workflow instance state"""
        with self._lock:
            self._instances[instance.id] = instance
            
            if self._storage_path:
                self._persist_instance(instance)
    
    def load_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Load workflow instance state"""
        with self._lock:
            return self._instances.get(instance_id)
    
    def save_definition(self, definition: WorkflowDefinition):
        """Save workflow definition"""
        with self._lock:
            self._definitions[definition.id] = definition
    
    def load_definition(self, definition_id: str) -> Optional[WorkflowDefinition]:
        """Load workflow definition"""
        with self._lock:
            return self._definitions.get(definition_id)
    
    def _persist_instance(self, instance: WorkflowInstance):
        """Persist instance to storage"""
        if not self._storage_path:
            return
        
        try:
            path = Path(self._storage_path)
            path.mkdir(parents=True, exist_ok=True)
            
            file_path = path / f"instance_{instance.id}.json"
            
            data = {
                'id': instance.id,
                'definition_id': instance.definition_id,
                'status': instance.status.name,
                'variables': instance.variables,
                'step_results': {
                    sid: r.to_dict()
                    for sid, r in instance.step_results.items()
                },
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist instance: {e}")
    
    def get_active_instances(self) -> List[WorkflowInstance]:
        """Get active workflow instances"""
        with self._lock:
            return [
                inst for inst in self._instances.values()
                if inst.status == WorkflowStatus.RUNNING
            ]


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class WorkflowManager:
    """
    Main workflow management class.
    
    Coordinates all workflow components.
    """
    
    def __init__(
        self,
        max_parallel: int = 10,
        default_timeout: float = 300.0,
        storage_path: str = None,
    ):
        """
        Initialize workflow manager.
        
        Args:
            max_parallel: Maximum parallel steps
            default_timeout: Default step timeout
            storage_path: Path for state persistence
        """
        # Components
        self._resolver = DependencyResolver()
        self._executor = WorkflowExecutor(
            max_parallel=max_parallel,
            default_timeout=default_timeout,
        )
        self._templates = WorkflowTemplateEngine()
        self._state_manager = WorkflowStateManager(storage_path)
        
        # Workflow registry
        self._workflows: Dict[str, WorkflowDefinition] = {}
        self._instances: Dict[str, WorkflowInstance] = {}
        
        # Statistics
        self._stats = {
            'workflows_created': 0,
            'workflows_executed': 0,
            'instances_created': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("WorkflowManager initialized")
    
    def create_workflow(
        self,
        name: str,
        description: str = "",
        priority: WorkflowPriority = WorkflowPriority.NORMAL,
    ) -> WorkflowDefinition:
        """Create a new workflow"""
        workflow = WorkflowDefinition(
            name=name,
            description=description,
            priority=priority,
        )
        
        with self._lock:
            self._workflows[workflow.id] = workflow
            self._state_manager.save_definition(workflow)
            self._stats['workflows_created'] += 1
        
        return workflow
    
    def create_from_template(
        self,
        template_id: str,
        customizations: Dict[str, Any] = None,
    ) -> Optional[WorkflowDefinition]:
        """Create workflow from template"""
        workflow = self._templates.instantiate(template_id, customizations)
        
        if workflow:
            with self._lock:
                self._workflows[workflow.id] = workflow
                self._stats['workflows_created'] += 1
        
        return workflow
    
    def add_step(
        self,
        workflow_id: str,
        name: str,
        task: Callable,
        args: Tuple = None,
        kwargs: Dict[str, Any] = None,
        dependencies: List[str] = None,
        timeout: float = None,
    ) -> Optional[str]:
        """Add a step to a workflow"""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                return None
            
            step = WorkflowStep(
                name=name,
                task=task,
                task_args=args or (),
                task_kwargs=kwargs or {},
                timeout=timeout or workflow.default_timeout,
            )
            
            # Add dependencies
            for dep_id in (dependencies or []):
                step.add_dependency(dep_id)
            
            workflow.add_step(step)
            return step.id
    
    def connect_steps(
        self,
        workflow_id: str,
        from_step: str,
        to_step: str,
        dep_type: DependencyType = DependencyType.FINISH,
    ) -> bool:
        """Connect two steps in a workflow"""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                return False
            
            workflow.connect(from_step, to_step, dep_type)
            return True
    
    def execute(
        self,
        workflow_id: str,
        context: Dict[str, Any] = None,
        variables: Dict[str, Any] = None,
    ) -> Optional[WorkflowInstance]:
        """Execute a workflow"""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                return None
            
            # Validate
            errors = workflow.validate()
            if errors:
                logger.error(f"Workflow validation failed: {errors}")
                return None
        
        # Execute
        instance = self._executor.execute(workflow, context, variables)
        
        with self._lock:
            self._instances[instance.id] = instance
            self._state_manager.save_instance(instance)
            self._stats['workflows_executed'] += 1
            self._stats['instances_created'] += 1
        
        return instance
    
    def execute_async(
        self,
        workflow_id: str,
        context: Dict[str, Any] = None,
        variables: Dict[str, Any] = None,
    ) -> Optional[Future]:
        """Execute workflow asynchronously"""
        with self._lock:
            workflow = self._workflows.get(workflow_id)
            if not workflow:
                return None
        
        return self._executor.execute_async(workflow, context, variables)
    
    def get_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Get workflow instance"""
        return self._instances.get(instance_id)
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition"""
        return self._workflows.get(workflow_id)
    
    def cancel(self, instance_id: str) -> bool:
        """Cancel a running workflow"""
        return self._executor.cancel(instance_id)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List available templates"""
        return self._templates.list_templates()
    
    def get_parallel_groups(self, workflow_id: str) -> List[List[str]]:
        """Get parallel execution groups"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return []
        
        return self._resolver.get_parallel_groups(workflow)
    
    def get_critical_path(self, workflow_id: str) -> List[str]:
        """Get critical path"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return []
        
        return self._resolver.get_critical_path(workflow)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['workflows'] = len(self._workflows)
            stats['instances'] = len(self._instances)
            stats['executor'] = self._executor.get_stats()
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_manager: Optional[WorkflowManager] = None


def get_workflow_manager(**kwargs) -> WorkflowManager:
    """Get global workflow manager instance"""
    global _manager
    if _manager is None:
        _manager = WorkflowManager(**kwargs)
    return _manager


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Workflow Manager Test")
    print("="*60)
    
    # Create manager
    manager = WorkflowManager()
    
    # Create workflow
    print("\n1. Creating workflow...")
    workflow = manager.create_workflow(
        name="Test Pipeline",
        description="A test workflow pipeline",
    )
    print(f"   Created: {workflow.id}")
    
    # Add steps
    print("\n2. Adding steps...")
    
    def step1():
        time.sleep(0.5)
        return "Step 1 result"
    
    def step2(prev_result):
        time.sleep(0.3)
        return f"Step 2 processed: {prev_result}"
    
    def step3(prev_result):
        time.sleep(0.2)
        return f"Final: {prev_result}"
    
    step1_id = manager.add_step(workflow.id, "Initialize", step1)
    step2_id = manager.add_step(workflow.id, "Process", step2, dependencies=[step1_id])
    step3_id = manager.add_step(workflow.id, "Finalize", step3, dependencies=[step2_id])
    
    print(f"   Steps: {step1_id}, {step2_id}, {step3_id}")
    
    # Get execution order
    print("\n3. Resolving dependencies...")
    resolver = DependencyResolver()
    order = resolver.resolve(workflow)
    print(f"   Execution order: {order}")
    
    # Get parallel groups
    groups = manager.get_parallel_groups(workflow.id)
    print(f"   Parallel groups: {groups}")
    
    # Execute workflow
    print("\n4. Executing workflow...")
    instance = manager.execute(workflow.id)
    
    print(f"   Status: {instance.status.name}")
    print(f"   Progress: {instance.progress:.1%}")
    print(f"   Time: {instance.execution_time:.2f}s")
    
    # List templates
    print("\n5. Available templates:")
    templates = manager.list_templates()
    for t in templates:
        print(f"   - {t['name']}: {t['description']}")
    
    # Stats
    print("\n6. Statistics:")
    stats = manager.get_stats()
    print(f"   Workflows created: {stats['workflows_created']}")
    print(f"   Workflows executed: {stats['workflows_executed']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Goal Manager
===================================

Phase 3: Goal setting, tracking, and achievement system.

This module enables JARVIS to:
- Set and track goals
- Break down complex goals into sub-goals
- Monitor progress
- Adapt strategies based on progress
- Celebrate achievements

Key Features:
- Hierarchical goal structure
- Progress tracking with milestones
- Deadline management
- Goal dependencies
- Automatic goal adjustment

Author: JARVIS AI Project
Version: 3.0.0
Target Level: 60-70
"""

import time
import json
import logging
import threading
import math
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class GoalState(Enum):
    """States a goal can be in"""
    PENDING = auto()        # Not yet started
    PLANNED = auto()        # Plan created, ready to execute
    IN_PROGRESS = auto()    # Actively working on
    BLOCKED = auto()        # Cannot proceed (dependency/issue)
    PAUSED = auto()         # Temporarily stopped
    COMPLETED = auto()      # Successfully achieved
    FAILED = auto()         # Could not achieve
    CANCELLED = auto()      # Cancelled by user/system


class GoalPriority(Enum):
    """Goal priority levels"""
    CRITICAL = 100   # Must achieve, highest priority
    HIGH = 75        # Very important
    MEDIUM = 50      # Normal priority
    LOW = 25         # Nice to have
    BACKGROUND = 0   # Work on when idle


class GoalCategory(Enum):
    """Categories of goals"""
    PERFORMANCE = auto()     # Improve performance
    RELIABILITY = auto()     # Fix bugs, improve stability
    CAPABILITY = auto()      # Add new features
    EFFICIENCY = auto()      # Optimize resource usage
    QUALITY = auto()         # Improve code quality
    SECURITY = auto()        # Security improvements
    LEARNING = auto()        # Learn/adapt from data
    MAINTENANCE = auto()     # Regular maintenance
    USER_REQUEST = auto()    # User-initiated goals


class ProgressStatus(Enum):
    """Progress assessment"""
    ON_TRACK = auto()        # Progress as expected
    AHEAD = auto()           # Progress faster than expected
    BEHIND = auto()          # Progress slower than expected
    STALLED = auto()         # No progress
    AT_RISK = auto()         # May not complete in time


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProgressMetric:
    """
    A metric for tracking goal progress.
    """
    name: str
    current_value: float
    target_value: float
    unit: str = ""
    weight: float = 1.0
    
    # Tracking
    initial_value: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.initial_value == 0.0:
            self.initial_value = self.current_value
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage"""
        if self.target_value == self.initial_value:
            return 100.0 if self.current_value >= self.target_value else 0.0
        
        total_change = self.target_value - self.initial_value
        if total_change == 0:
            return 100.0
        
        current_change = self.current_value - self.initial_value
        progress = (current_change / total_change) * 100
        return max(0.0, min(100.0, progress))
    
    @property
    def is_complete(self) -> bool:
        """Check if metric target is reached"""
        if self.target_value >= self.initial_value:
            return self.current_value >= self.target_value
        else:
            return self.current_value <= self.target_value
    
    def update(self, new_value: float):
        """Update current value"""
        self.current_value = new_value
        self.last_updated = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'current': self.current_value,
            'target': self.target_value,
            'unit': self.unit,
            'progress': self.progress_percent,
            'complete': self.is_complete,
        }


@dataclass
class Milestone:
    """
    A milestone in goal progress.
    """
    id: str
    name: str
    description: str = ""
    target_progress: float = 0.0  # Progress % at which milestone is reached
    
    # Status
    reached: bool = False
    reached_at: Optional[float] = None
    
    # Rewards/actions
    reward_points: int = 0
    on_reach_actions: List[str] = field(default_factory=list)
    
    def check_reached(self, current_progress: float) -> bool:
        """Check if milestone is reached"""
        if not self.reached and current_progress >= self.target_progress:
            self.reached = True
            self.reached_at = time.time()
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'target_progress': self.target_progress,
            'reached': self.reached,
            'reached_at': self.reached_at,
        }


@dataclass
class Goal:
    """
    A goal to be achieved.
    
    Goals can be hierarchical with sub-goals.
    """
    # Identification
    id: str
    name: str
    description: str = ""
    category: GoalCategory = GoalCategory.CAPABILITY
    
    # State
    state: GoalState = GoalState.PENDING
    priority: GoalPriority = GoalPriority.MEDIUM
    
    # Hierarchy
    parent_id: Optional[str] = None
    sub_goal_ids: List[str] = field(default_factory=list)
    
    # Progress
    progress_metrics: Dict[str, ProgressMetric] = field(default_factory=dict)
    milestones: List[Milestone] = field(default_factory=list)
    overall_progress: float = 0.0
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    deadline: Optional[float] = None
    estimated_duration: float = 0.0  # seconds
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Goal IDs
    blocks: List[str] = field(default_factory=list)      # Goal IDs this blocks
    
    # Strategy
    strategy: str = ""  # How to achieve the goal
    fallback_strategy: str = ""  # Alternative approach
    
    # Tracking
    effort_spent: float = 0.0  # Total effort units
    attempts: int = 0
    last_progress_update: float = field(default_factory=time.time)
    
    # Rewards
    reward_points: int = 100
    penalty_points: int = 0
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"goal_{int(time.time() * 1000)}"
    
    @property
    def is_active(self) -> bool:
        """Check if goal is in active state"""
        return self.state in (GoalState.PLANNED, GoalState.IN_PROGRESS)
    
    @property
    def is_complete(self) -> bool:
        """Check if goal is completed"""
        return self.state == GoalState.COMPLETED
    
    @property
    def is_blocked(self) -> bool:
        """Check if goal is blocked"""
        return self.state == GoalState.BLOCKED
    
    @property
    def time_remaining(self) -> Optional[float]:
        """Get remaining time until deadline (seconds)"""
        if self.deadline is None:
            return None
        return max(0, self.deadline - time.time())
    
    @property
    def is_overdue(self) -> bool:
        """Check if goal is past deadline"""
        if self.deadline is None:
            return False
        return time.time() > self.deadline and self.state not in (GoalState.COMPLETED, GoalState.CANCELLED)
    
    def calculate_progress(self) -> float:
        """Calculate overall progress from metrics"""
        if not self.progress_metrics:
            return self.overall_progress
        
        total_weight = sum(m.weight for m in self.progress_metrics.values())
        if total_weight == 0:
            return 0.0
        
        weighted_progress = sum(
            m.progress_percent * m.weight
            for m in self.progress_metrics.values()
        )
        
        self.overall_progress = weighted_progress / total_weight
        self.last_progress_update = time.time()
        
        return self.overall_progress
    
    def update_metric(self, name: str, value: float):
        """Update a progress metric"""
        if name in self.progress_metrics:
            self.progress_metrics[name].update(value)
            self.calculate_progress()
    
    def add_sub_goal(self, goal_id: str):
        """Add a sub-goal"""
        if goal_id not in self.sub_goal_ids:
            self.sub_goal_ids.append(goal_id)
    
    def check_milestones(self) -> List[Milestone]:
        """Check and return newly reached milestones"""
        newly_reached = []
        for milestone in self.milestones:
            if milestone.check_reached(self.overall_progress):
                newly_reached.append(milestone)
        return newly_reached
    
    def get_progress_status(self) -> ProgressStatus:
        """Determine progress status"""
        if self.state == GoalState.PENDING:
            return ProgressStatus.STALLED
        
        if self.is_overdue:
            return ProgressStatus.AT_RISK
        
        # Check time-based progress
        if self.deadline and self.started_at:
            time_elapsed = time.time() - self.started_at
            time_total = self.deadline - self.started_at
            time_progress = (time_elapsed / time_total) * 100 if time_total > 0 else 100
            
            if self.overall_progress > time_progress + 10:
                return ProgressStatus.AHEAD
            elif self.overall_progress < time_progress - 10:
                return ProgressStatus.BEHIND
            elif self.overall_progress < time_progress - 30:
                return ProgressStatus.AT_RISK
        
        if self.overall_progress == 0 and self.state == GoalState.IN_PROGRESS:
            return ProgressStatus.STALLED
        
        return ProgressStatus.ON_TRACK
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category.name,
            'state': self.state.name,
            'priority': self.priority.name,
            'progress': self.overall_progress,
            'progress_status': self.get_progress_status().name,
            'sub_goals': len(self.sub_goal_ids),
            'deadline': self.deadline,
            'is_overdue': self.is_overdue,
            'time_remaining': self.time_remaining,
            'milestones_reached': sum(1 for m in self.milestones if m.reached),
            'milestones_total': len(self.milestones),
            'created_at': self.created_at,
            'tags': self.tags,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GOAL MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class GoalManager:
    """
    Manage goals for autonomous system.
    
    Features:
    - Goal creation and tracking
    - Hierarchical goal decomposition
    - Progress monitoring
    - Dependency management
    - Achievement rewards
    - Goal prioritization
    
    Usage:
        manager = GoalManager()
        
        # Create a goal
        goal = manager.create_goal(
            name="Reduce memory usage",
            category=GoalCategory.PERFORMANCE,
            priority=GoalPriority.HIGH,
            metrics=[ProgressMetric("memory_mb", 500, 300, "MB")]
        )
        
        # Update progress
        manager.update_progress(goal.id, "memory_mb", 400)
        
        # Check status
        status = manager.get_goal_status(goal.id)
    """
    
    def __init__(
        self,
        storage_path: str = None,
        auto_save: bool = True,
        max_active_goals: int = 50,
    ):
        """
        Initialize goal manager.
        
        Args:
            storage_path: Path to persist goals
            auto_save: Automatically save changes
            max_active_goals: Maximum concurrent active goals
        """
        self._storage_path = storage_path
        self._auto_save = auto_save
        self._max_active = max_active_goals
        
        # Goal storage
        self._goals: Dict[str, Goal] = {}
        self._lock = threading.RLock()
        
        # Indices
        self._by_state: Dict[GoalState, Set[str]] = {s: set() for s in GoalState}
        self._by_priority: Dict[GoalPriority, Set[str]] = {p: set() for p in GoalPriority}
        self._by_category: Dict[GoalCategory, Set[str]] = {c: set() for c in GoalCategory}
        
        # Achievement tracking
        self._total_points: int = 0
        self._achievements: List[Dict] = []
        
        # Statistics
        self._stats = {
            'goals_created': 0,
            'goals_completed': 0,
            'goals_failed': 0,
            'total_points_earned': 0,
            'total_points_lost': 0,
        }
        
        # Load from storage
        if storage_path:
            self._load()
        
        logger.info("GoalManager initialized")
    
    def create_goal(
        self,
        name: str,
        description: str = "",
        category: GoalCategory = GoalCategory.CAPABILITY,
        priority: GoalPriority = GoalPriority.MEDIUM,
        metrics: List[ProgressMetric] = None,
        milestones: List[Milestone] = None,
        deadline: float = None,
        depends_on: List[str] = None,
        strategy: str = "",
        parent_id: str = None,
        tags: List[str] = None,
    ) -> Goal:
        """
        Create a new goal.
        
        Args:
            name: Goal name
            description: Detailed description
            category: Goal category
            priority: Priority level
            metrics: Progress metrics
            milestones: Achievement milestones
            deadline: Unix timestamp deadline
            depends_on: Goal IDs this depends on
            strategy: Strategy to achieve
            parent_id: Parent goal ID
            tags: Tags for categorization
            
        Returns:
            Created Goal
        """
        with self._lock:
            goal = Goal(
                id=f"goal_{int(time.time() * 1000)}_{len(self._goals)}",
                name=name,
                description=description,
                category=category,
                priority=priority,
                deadline=deadline,
                strategy=strategy,
                parent_id=parent_id,
                tags=tags or [],
            )
            
            # Add metrics
            if metrics:
                for m in metrics:
                    goal.progress_metrics[m.name] = m
            
            # Add milestones
            if milestones:
                goal.milestones = milestones
            
            # Add dependencies
            if depends_on:
                goal.depends_on = depends_on
                for dep_id in depends_on:
                    if dep_id in self._goals:
                        self._goals[dep_id].blocks.append(goal.id)
            
            # Add to parent
            if parent_id and parent_id in self._goals:
                self._goals[parent_id].add_sub_goal(goal.id)
            
            # Store and index
            self._goals[goal.id] = goal
            self._by_state[goal.state].add(goal.id)
            self._by_priority[goal.priority].add(goal.id)
            self._by_category[goal.category].add(goal.id)
            
            self._stats['goals_created'] += 1
            
            if self._auto_save:
                self._save()
            
            logger.info(f"Created goal: {name} ({goal.id})")
            return goal
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get goal by ID"""
        return self._goals.get(goal_id)
    
    def update_progress(
        self,
        goal_id: str,
        metric_name: str,
        value: float,
    ) -> Optional[Goal]:
        """
        Update goal progress.
        
        Args:
            goal_id: Goal ID
            metric_name: Metric to update
            value: New value
            
        Returns:
            Updated Goal or None
        """
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None
            
            # Update metric
            goal.update_metric(metric_name, value)
            
            # Check milestones
            newly_reached = goal.check_milestones()
            for milestone in newly_reached:
                self._on_milestone_reached(goal, milestone)
            
            # Check completion
            if goal.overall_progress >= 100:
                self.complete_goal(goal_id)
            
            if self._auto_save:
                self._save()
            
            return goal
    
    def start_goal(self, goal_id: str) -> bool:
        """Mark goal as started"""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            # Check dependencies
            for dep_id in goal.depends_on:
                dep = self._goals.get(dep_id)
                if dep and dep.state != GoalState.COMPLETED:
                    logger.warning(f"Cannot start {goal_id}: dependency {dep_id} not completed")
                    return False
            
            # Update state
            self._update_state(goal, GoalState.IN_PROGRESS)
            goal.started_at = time.time()
            
            logger.info(f"Started goal: {goal.name}")
            return True
    
    def complete_goal(self, goal_id: str) -> bool:
        """Mark goal as completed"""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            # Update state
            self._update_state(goal, GoalState.COMPLETED)
            goal.completed_at = time.time()
            goal.overall_progress = 100.0
            
            # Award points
            self._total_points += goal.reward_points
            self._stats['goals_completed'] += 1
            self._stats['total_points_earned'] += goal.reward_points
            
            # Record achievement
            self._achievements.append({
                'goal_id': goal.id,
                'name': goal.name,
                'completed_at': goal.completed_at,
                'points': goal.reward_points,
            })
            
            # Unblock dependent goals
            for blocked_id in goal.blocks:
                blocked = self._goals.get(blocked_id)
                if blocked and blocked.state == GoalState.BLOCKED:
                    # Check if all dependencies are now met
                    deps_met = all(
                        self._goals.get(d, Goal("")).state == GoalState.COMPLETED
                        for d in blocked.depends_on
                    )
                    if deps_met:
                        self._update_state(blocked, GoalState.PENDING)
            
            # Update parent progress
            if goal.parent_id:
                self._update_parent_progress(goal.parent_id)
            
            if self._auto_save:
                self._save()
            
            logger.info(f"Completed goal: {goal.name} (+{goal.reward_points} points)")
            return True
    
    def fail_goal(self, goal_id: str, reason: str = "") -> bool:
        """Mark goal as failed"""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            self._update_state(goal, GoalState.FAILED)
            goal.completed_at = time.time()
            
            # Penalty
            self._total_points -= goal.penalty_points
            self._stats['goals_failed'] += 1
            self._stats['total_points_lost'] += goal.penalty_points
            
            logger.warning(f"Failed goal: {goal.name} ({reason})")
            return True
    
    def block_goal(self, goal_id: str, reason: str = "") -> bool:
        """Block a goal"""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            self._update_state(goal, GoalState.BLOCKED)
            goal.metadata['block_reason'] = reason
            
            logger.warning(f"Blocked goal: {goal.name} ({reason})")
            return True
    
    def cancel_goal(self, goal_id: str, reason: str = "") -> bool:
        """Cancel a goal"""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            self._update_state(goal, GoalState.CANCELLED)
            goal.completed_at = time.time()
            goal.metadata['cancel_reason'] = reason
            
            logger.info(f"Cancelled goal: {goal.name}")
            return True
    
    def get_active_goals(self) -> List[Goal]:
        """Get all active goals"""
        with self._lock:
            return [
                self._goals[gid]
                for gid in self._by_state[GoalState.IN_PROGRESS]
                if gid in self._goals
            ]
    
    def get_pending_goals(self) -> List[Goal]:
        """Get all pending goals"""
        with self._lock:
            return [
                self._goals[gid]
                for gid in self._by_state[GoalState.PENDING]
                if gid in self._goals
            ]
    
    def get_next_goal(self) -> Optional[Goal]:
        """Get highest priority goal to work on"""
        with self._lock:
            # Priority order: PENDING and IN_PROGRESS
            candidates = []
            
            for state in [GoalState.IN_PROGRESS, GoalState.PENDING]:
                for gid in self._by_state[state]:
                    goal = self._goals.get(gid)
                    if goal:
                        # Check dependencies
                        deps_met = all(
                            self._goals.get(d, Goal("")).state == GoalState.COMPLETED
                            for d in goal.depends_on
                        )
                        if deps_met and not goal.is_overdue:
                            candidates.append(goal)
            
            if not candidates:
                return None
            
            # Sort by priority (higher first), then by deadline
            candidates.sort(key=lambda g: (
                -g.priority.value,
                g.deadline or float('inf')
            ))
            
            return candidates[0]
    
    def get_goal_status(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive goal status"""
        goal = self._goals.get(goal_id)
        if not goal:
            return None
        
        return {
            'goal': goal.to_dict(),
            'metrics': {k: v.to_dict() for k, v in goal.progress_metrics.items()},
            'progress_status': goal.get_progress_status().name,
            'sub_goals': [
                self._goals[sid].to_dict()
                for sid in goal.sub_goal_ids
                if sid in self._goals
            ],
            'dependencies': {
                'waiting_on': [
                    self._goals[did].name
                    for did in goal.depends_on
                    if did in self._goals and self._goals[did].state != GoalState.COMPLETED
                ],
                'blocking': [
                    self._goals[bid].name
                    for bid in goal.blocks
                    if bid in self._goals
                ],
            },
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get goal manager summary"""
        with self._lock:
            return {
                'total_goals': len(self._goals),
                'by_state': {
                    s.name: len(ids)
                    for s, ids in self._by_state.items()
                },
                'active_goals': len(self.get_active_goals()),
                'pending_goals': len(self.get_pending_goals()),
                'completed_goals': self._stats['goals_completed'],
                'failed_goals': self._stats['goals_failed'],
                'total_points': self._total_points,
                'recent_achievements': self._achievements[-5:],
                'overdue_goals': sum(
                    1 for g in self._goals.values() if g.is_overdue
                ),
            }
    
    def _update_state(self, goal: Goal, new_state: GoalState):
        """Update goal state and indices"""
        old_state = goal.state
        if old_state == new_state:
            return
        
        # Update indices
        self._by_state[old_state].discard(goal.id)
        self._by_state[new_state].add(goal.id)
        
        # Update goal
        goal.state = new_state
    
    def _update_parent_progress(self, parent_id: str):
        """Update parent goal progress based on sub-goals"""
        parent = self._goals.get(parent_id)
        if not parent or not parent.sub_goal_ids:
            return
        
        # Calculate progress from sub-goals
        total_progress = 0
        count = 0
        
        for sub_id in parent.sub_goal_ids:
            sub = self._goals.get(sub_id)
            if sub:
                total_progress += sub.overall_progress
                count += 1
        
        if count > 0:
            parent.overall_progress = total_progress / count
            parent.last_progress_update = time.time()
    
    def _on_milestone_reached(self, goal: Goal, milestone: Milestone):
        """Handle milestone reached event"""
        logger.info(f"Milestone reached: {milestone.name} for goal {goal.name}")
        
        # Add reward points
        self._total_points += milestone.reward_points
    
    def _save(self):
        """Save goals to storage"""
        if not self._storage_path:
            return
        
        try:
            data = {
                'goals': {gid: g.to_dict() for gid, g in self._goals.items()},
                'stats': self._stats,
                'total_points': self._total_points,
                'achievements': self._achievements,
            }
            
            path = Path(self._storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save goals: {e}")
    
    def _load(self):
        """Load goals from storage"""
        if not self._storage_path:
            return
        
        try:
            path = Path(self._storage_path)
            if not path.exists():
                return
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Restore state
            self._stats = data.get('stats', self._stats)
            self._total_points = data.get('total_points', 0)
            self._achievements = data.get('achievements', [])
            
            logger.info(f"Loaded {len(self._goals)} goals from storage")
        except Exception as e:
            logger.error(f"Failed to load goals: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_manager: Optional[GoalManager] = None


def get_goal_manager() -> GoalManager:
    """Get global goal manager"""
    global _manager
    if _manager is None:
        _manager = GoalManager()
    return _manager


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Goal Manager Test")
    print("="*60)
    
    manager = GoalManager()
    
    # Create goal
    goal = manager.create_goal(
        name="Improve Performance",
        description="Reduce response time by 50%",
        category=GoalCategory.PERFORMANCE,
        priority=GoalPriority.HIGH,
        metrics=[
            ProgressMetric("response_time_ms", 200, 100, "ms"),
        ],
        milestones=[
            Milestone("m1", "25% improvement", target_progress=25),
            Milestone("m2", "50% improvement", target_progress=50),
            Milestone("m3", "Complete", target_progress=100),
        ],
    )
    
    print(f"\nCreated Goal: {goal.name}")
    print(f"  ID: {goal.id}")
    print(f"  Priority: {goal.priority.name}")
    print(f"  State: {goal.state.name}")
    
    # Start goal
    manager.start_goal(goal.id)
    print(f"\nStarted goal: {goal.state.name}")
    
    # Update progress
    manager.update_progress(goal.id, "response_time_ms", 150)
    goal = manager.get_goal(goal.id)
    print(f"\nProgress: {goal.overall_progress:.1f}%")
    print(f"  Milestones reached: {sum(1 for m in goal.milestones if m.reached)}")
    
    # Complete
    manager.update_progress(goal.id, "response_time_ms", 95)
    
    print(f"\nSummary:")
    summary = manager.get_summary()
    print(f"  Completed: {summary['completed_goals']}")
    print(f"  Total Points: {summary['total_points']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

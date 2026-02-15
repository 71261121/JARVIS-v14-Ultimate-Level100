#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Autonomous System Package
=================================================

Phase 3: Autonomous Decision Engine (Level 60-70)

This package enables JARVIS to make autonomous decisions:
- Self-monitoring and diagnostics
- Goal setting and tracking
- Autonomous decision making
- Self-improvement cycles

Modules:
    - decision_engine: Core autonomous decision making
    - self_monitor: System health monitoring
    - goal_manager: Goal setting and tracking
    - action_executor: Execute decisions safely

Author: JARVIS AI Project
Version: 3.0.0
Target Level: 60-70
"""

from .decision_engine import (
    AutonomousDecisionEngine,
    DecisionContext,
    Decision,
    DecisionType,
    DecisionPriority,
    get_decision_engine,
)

from .self_monitor import (
    SelfMonitor,
    SystemHealth,
    HealthMetric,
    AlertLevel,
    get_self_monitor,
)

from .goal_manager import (
    GoalManager,
    Goal,
    GoalState,
    GoalPriority,
    ProgressMetric,
    get_goal_manager,
)

__all__ = [
    # Decision Engine
    'AutonomousDecisionEngine',
    'DecisionContext',
    'Decision',
    'DecisionType',
    'DecisionPriority',
    'get_decision_engine',
    
    # Self Monitor
    'SelfMonitor',
    'SystemHealth',
    'HealthMetric',
    'AlertLevel',
    'get_self_monitor',
    
    # Goal Manager
    'GoalManager',
    'Goal',
    'GoalState',
    'GoalPriority',
    'ProgressMetric',
    'get_goal_manager',
]

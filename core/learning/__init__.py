#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Learning Package
=======================================

Phase 3: Reinforcement learning from outcomes.

This package enables JARVIS to learn from its decisions:
- Track modification outcomes
- Build experience database
- Improve decision making over time
- Adapt strategies based on results

Author: JARVIS AI Project
Version: 3.0.0
Target Level: 60-70
"""

from .reinforcement_engine import (
    ReinforcementEngine,
    Experience,
    State,
    Action,
    Reward,
    get_reinforcement_engine,
)

from .outcome_evaluator import (
    OutcomeEvaluator,
    ModificationOutcome,
    OutcomeType,
    get_outcome_evaluator,
)

__all__ = [
    # Reinforcement Engine
    'ReinforcementEngine',
    'Experience',
    'State',
    'Action',
    'Reward',
    'get_reinforcement_engine',
    
    # Outcome Evaluator
    'OutcomeEvaluator',
    'ModificationOutcome',
    'OutcomeType',
    'get_outcome_evaluator',
]

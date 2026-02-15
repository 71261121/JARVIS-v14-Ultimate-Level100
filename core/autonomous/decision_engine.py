#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Autonomous Decision Engine
==================================================

Phase 3: Core autonomous decision making system.

This is the brain of JARVIS Level 60-70, enabling:
- Autonomous decision making
- Risk assessment
- Action planning
- Outcome prediction
- Learning from decisions

Key Features:
- Multi-factor decision analysis
- Risk/benefit evaluation
- Constraint satisfaction
- Context-aware decisions
- Decision history and learning

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
from typing import Dict, Any, Optional, List, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class DecisionType(Enum):
    """Types of decisions the system can make"""
    # Code modifications
    CODE_GENERATION = auto()
    CODE_IMPROVEMENT = auto()
    BUG_FIX = auto()
    REFACTORING = auto()
    
    # System operations
    RESOURCE_ALLOCATION = auto()
    TASK_SCHEDULING = auto()
    PRIORITY_CHANGE = auto()
    
    # Learning
    STRATEGY_UPDATE = auto()
    MODEL_UPDATE = auto()
    PARAMETER_TUNING = auto()
    
    # Recovery
    ERROR_RECOVERY = auto()
    FALLBACK_SWITCH = auto()
    ROLLBACK = auto()
    
    # User interaction
    USER_QUERY = auto()
    USER_NOTIFICATION = auto()
    PERMISSION_REQUEST = auto()


class DecisionPriority(Enum):
    """Priority levels for decisions"""
    IMMEDIATE = 0     # Must execute now
    HIGH = 1          # Execute soon
    NORMAL = 2        # Normal priority
    LOW = 3           # Can wait
    BACKGROUND = 4    # Execute when idle


class DecisionStatus(Enum):
    """Status of a decision"""
    PENDING = auto()
    EVALUATING = auto()
    APPROVED = auto()
    REJECTED = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    ROLLED_BACK = auto()


class RiskLevel(Enum):
    """Risk levels for decisions"""
    MINIMAL = 1       # Almost no risk
    LOW = 2           # Minor risk
    MODERATE = 3      # Some risk
    HIGH = 4          # Significant risk
    CRITICAL = 5      # Major risk


class ConfidenceLevel(Enum):
    """Confidence in decision"""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RiskAssessment:
    """
    Risk assessment for a decision.
    """
    level: RiskLevel = RiskLevel.LOW
    score: float = 0.0  # 0-100
    
    # Risk factors
    code_impact: float = 0.0      # How much code affected
    system_impact: float = 0.0    # System-wide impact
    reversibility: float = 100.0  # How easy to reverse (100 = easy)
    test_coverage: float = 0.0    # Test coverage for affected code
    
    # Specific risks
    risks: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    
    def calculate_overall(self) -> float:
        """Calculate overall risk score"""
        # Higher impact = higher risk
        impact_score = (self.code_impact + self.system_impact) / 2
        
        # Lower reversibility = higher risk
        reversibility_risk = 100 - self.reversibility
        
        # Lower test coverage = higher risk
        test_risk = 100 - self.test_coverage
        
        # Weighted average
        self.score = (
            impact_score * 0.4 +
            reversibility_risk * 0.35 +
            test_risk * 0.25
        )
        
        # Determine level
        if self.score < 20:
            self.level = RiskLevel.MINIMAL
        elif self.score < 40:
            self.level = RiskLevel.LOW
        elif self.score < 60:
            self.level = RiskLevel.MODERATE
        elif self.score < 80:
            self.level = RiskLevel.HIGH
        else:
            self.level = RiskLevel.CRITICAL
        
        return self.score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level.name,
            'score': self.score,
            'code_impact': self.code_impact,
            'system_impact': self.system_impact,
            'reversibility': self.reversibility,
            'test_coverage': self.test_coverage,
            'risks': self.risks,
            'mitigations': self.mitigations,
        }


@dataclass
class BenefitAssessment:
    """
    Benefit assessment for a decision.
    """
    score: float = 0.0  # 0-100
    
    # Benefit categories
    performance_gain: float = 0.0
    reliability_gain: float = 0.0
    maintainability_gain: float = 0.0
    user_experience_gain: float = 0.0
    
    # Expected outcomes
    expected_outcomes: List[str] = field(default_factory=list)
    
    def calculate_overall(self) -> float:
        """Calculate overall benefit score"""
        self.score = (
            self.performance_gain * 0.25 +
            self.reliability_gain * 0.25 +
            self.maintainability_gain * 0.25 +
            self.user_experience_gain * 0.25
        )
        return self.score
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'score': self.score,
            'performance_gain': self.performance_gain,
            'reliability_gain': self.reliability_gain,
            'maintainability_gain': self.maintainability_gain,
            'user_experience_gain': self.user_experience_gain,
            'expected_outcomes': self.expected_outcomes,
        }


@dataclass
class Action:
    """
    An action to be executed as part of a decision.
    """
    id: str
    action_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Execution
    precondition: Optional[str] = None
    postcondition: Optional[str] = None
    
    # Status
    executed: bool = False
    success: Optional[bool] = None
    execution_time: Optional[float] = None
    error: Optional[str] = None
    
    # Rollback
    rollback_action: Optional[str] = None
    rollback_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.action_type,
            'description': self.description,
            'executed': self.executed,
            'success': self.success,
        }


@dataclass 
class Decision:
    """
    A complete decision with all metadata.
    """
    # Identification
    id: str
    decision_type: DecisionType
    description: str
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    trigger: str = ""  # What triggered this decision
    
    # Assessment
    risk: RiskAssessment = field(default_factory=RiskAssessment)
    benefit: BenefitAssessment = field(default_factory=BenefitAssessment)
    confidence: ConfidenceLevel = ConfidenceLevel.MODERATE
    confidence_score: float = 0.5  # 0-1
    
    # Priority
    priority: DecisionPriority = DecisionPriority.NORMAL
    
    # Status
    status: DecisionStatus = DecisionStatus.PENDING
    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None
    executed_at: Optional[float] = None
    
    # Actions
    actions: List[Action] = field(default_factory=list)
    current_action_index: int = 0
    
    # Reasoning
    reasoning: str = ""
    alternatives_considered: List[str] = field(default_factory=list)
    
    # Outcome
    outcome: str = ""
    outcome_score: float = 0.0  # Actual outcome vs expected
    
    # Learning
    learned: bool = False
    lessons: List[str] = field(default_factory=list)
    
    # Approval
    requires_approval: bool = False
    approved_by: Optional[str] = None
    
    @property
    def should_execute(self) -> bool:
        """Determine if decision should be executed"""
        if self.status != DecisionStatus.APPROVED:
            return False
        
        if self.requires_approval and not self.approved_by:
            return False
        
        return True
    
    @property
    def risk_benefit_ratio(self) -> float:
        """Calculate risk/benefit ratio"""
        if self.benefit.score == 0:
            return float('inf')
        return self.risk.score / self.benefit.score
    
    @property
    def expected_value(self) -> float:
        """Calculate expected value of decision"""
        # EV = (Benefit * Confidence) - (Risk * (1 - Confidence))
        return (
            self.benefit.score * self.confidence_score -
            self.risk.score * (1 - self.confidence_score)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.decision_type.name,
            'description': self.description,
            'trigger': self.trigger,
            'risk': self.risk.to_dict(),
            'benefit': self.benefit.to_dict(),
            'confidence': self.confidence.name,
            'priority': self.priority.name,
            'status': self.status.name,
            'expected_value': self.expected_value,
            'risk_benefit_ratio': self.risk_benefit_ratio,
            'reasoning': self.reasoning,
            'actions_count': len(self.actions),
            'created_at': self.created_at,
        }


@dataclass
class DecisionContext:
    """
    Context for making a decision.
    
    Contains all relevant information for the decision engine.
    """
    # System state
    health_score: float = 100.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    active_goals: List[str] = field(default_factory=list)
    
    # Recent history
    recent_decisions: List[str] = field(default_factory=list)
    recent_errors: List[str] = field(default_factory=list)
    
    # Constraints
    constraints: Dict[str, Any] = field(default_factory=dict)
    max_risk_level: RiskLevel = RiskLevel.MODERATE
    
    # User context
    user_present: bool = False
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Time context
    time_of_day: str = "day"  # day, evening, night
    system_load: float = 0.5
    
    # Code context
    code_being_analyzed: Optional[str] = None
    detected_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'health_score': self.health_score,
            'resource_usage': self.resource_usage,
            'active_goals': len(self.active_goals),
            'recent_decisions': len(self.recent_decisions),
            'recent_errors': len(self.recent_errors),
            'max_risk_level': self.max_risk_level.name,
            'user_present': self.user_present,
            'system_load': self.system_load,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION POLICIES
# ═══════════════════════════════════════════════════════════════════════════════

class DecisionPolicy:
    """
    Policies for decision making.
    
    Defines rules and constraints for autonomous decisions.
    """
    
    def __init__(self):
        """Initialize decision policies"""
        # Risk thresholds
        self.max_auto_risk = RiskLevel.MODERATE
        self.max_auto_risk_score = 60.0
        
        # Confidence thresholds
        self.min_confidence_for_auto = ConfidenceLevel.MODERATE
        self.min_confidence_score = 0.5
        
        # Approval requirements
        self.approval_required_types = {
            DecisionType.ROLLBACK,
            DecisionType.MODEL_UPDATE,
        }
        
        self.approval_required_risk = RiskLevel.HIGH
        
        # Rate limits
        self.max_decisions_per_minute = 10
        self.max_decisions_per_hour = 100
        
        # Cooldown periods (seconds)
        self.cooldown_after_failure = 60.0
        self.cooldown_after_critical = 300.0
    
    def requires_approval(self, decision: Decision) -> bool:
        """Check if decision requires approval"""
        # By risk level
        if decision.risk.level.value >= self.approval_required_risk.value:
            return True
        
        # By type
        if decision.decision_type in self.approval_required_types:
            return True
        
        # By confidence
        if decision.confidence.value < self.min_confidence_for_auto.value:
            return True
        
        return False
    
    def can_auto_execute(self, decision: Decision) -> bool:
        """Check if decision can be auto-executed"""
        # Risk check
        if decision.risk.score > self.max_auto_risk_score:
            return False
        
        # Confidence check
        if decision.confidence_score < self.min_confidence_score:
            return False
        
        # Risk level check
        if decision.risk.level.value > self.max_auto_risk.value:
            return False
        
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class AutonomousDecisionEngine:
    """
    Core autonomous decision making engine.
    
    This is the brain of JARVIS Level 60-70, making intelligent
    decisions about:
    - Code modifications
    - System operations
    - Learning updates
    - Recovery actions
    
    Usage:
        engine = AutonomousDecisionEngine(kimi_client)
        
        # Evaluate a potential action
        decision = engine.evaluate(
            decision_type=DecisionType.CODE_IMPROVEMENT,
            context=current_context,
            description="Optimize database queries"
        )
        
        # Execute if approved
        if decision.status == DecisionStatus.APPROVED:
            engine.execute(decision)
    """
    
    def __init__(
        self,
        kimi_client=None,
        policy: DecisionPolicy = None,
        enable_learning: bool = True,
        history_size: int = 1000,
    ):
        """
        Initialize decision engine.
        
        Args:
            kimi_client: Kimi K2.5 client for AI reasoning
            policy: Decision policy
            enable_learning: Learn from decision outcomes
            history_size: Max decisions in history
        """
        self._kimi = kimi_client
        self._policy = policy or DecisionPolicy()
        self._enable_learning = enable_learning
        
        # Decision storage
        self._decisions: Dict[str, Decision] = {}
        self._history: deque = deque(maxlen=history_size)
        self._pending: deque = deque(maxlen=100)
        
        # Learning data
        self._outcomes: Dict[str, float] = {}  # decision_type -> avg outcome
        self._success_rate: Dict[DecisionType, float] = {}
        
        # Rate limiting
        self._recent_times: deque = deque(maxlen=1000)
        
        # Statistics
        self._stats = {
            'total_decisions': 0,
            'auto_approved': 0,
            'manual_approved': 0,
            'rejected': 0,
            'executed': 0,
            'successful': 0,
            'failed': 0,
            'rolled_back': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("AutonomousDecisionEngine initialized (Level 60-70)")
    
    def set_kimi_client(self, client):
        """Set Kimi client for AI reasoning"""
        self._kimi = client
    
    def evaluate(
        self,
        decision_type: DecisionType,
        context: DecisionContext,
        description: str,
        trigger: str = "",
        actions: List[Action] = None,
    ) -> Decision:
        """
        Evaluate a potential decision.
        
        Args:
            decision_type: Type of decision
            context: Decision context
            description: What the decision is about
            trigger: What triggered this decision
            actions: Actions to execute
            
        Returns:
            Decision with assessment
        """
        with self._lock:
            # Create decision
            decision = Decision(
                id=f"dec_{int(time.time() * 1000)}_{len(self._decisions)}",
                decision_type=decision_type,
                description=description,
                trigger=trigger,
                context=context.to_dict(),
                actions=actions or [],
            )
            
            # Assess risk
            decision.risk = self._assess_risk(decision, context)
            
            # Assess benefit
            decision.benefit = self._assess_benefit(decision, context)
            
            # Calculate confidence
            decision.confidence, decision.confidence_score = self._calculate_confidence(
                decision, context
            )
            
            # Determine priority
            decision.priority = self._determine_priority(decision, context)
            
            # Generate reasoning
            decision.reasoning = self._generate_reasoning(decision, context)
            
            # Check if requires approval
            decision.requires_approval = self._policy.requires_approval(decision)
            
            # Make decision
            if self._policy.can_auto_execute(decision):
                decision.status = DecisionStatus.APPROVED
                self._stats['auto_approved'] += 1
            elif decision.requires_approval:
                decision.status = DecisionStatus.PENDING
                self._pending.append(decision.id)
            else:
                decision.status = DecisionStatus.REJECTED
                self._stats['rejected'] += 1
            
            # Store
            self._decisions[decision.id] = decision
            self._history.append(decision.id)
            self._stats['total_decisions'] += 1
            
            logger.info(
                f"Decision {decision.id}: {decision.status.name} "
                f"(risk={decision.risk.level.name}, benefit={decision.benefit.score:.1f})"
            )
            
            return decision
    
    def approve(self, decision_id: str, approver: str = "user") -> bool:
        """Approve a pending decision"""
        with self._lock:
            decision = self._decisions.get(decision_id)
            if not decision or decision.status != DecisionStatus.PENDING:
                return False
            
            decision.status = DecisionStatus.APPROVED
            decision.approved_by = approver
            decision.decided_at = time.time()
            
            self._stats['manual_approved'] += 1
            
            logger.info(f"Decision {decision_id} approved by {approver}")
            return True
    
    def reject(self, decision_id: str, reason: str = "") -> bool:
        """Reject a pending decision"""
        with self._lock:
            decision = self._decisions.get(decision_id)
            if not decision:
                return False
            
            decision.status = DecisionStatus.REJECTED
            decision.decided_at = time.time()
            decision.outcome = f"Rejected: {reason}"
            
            self._stats['rejected'] += 1
            
            logger.info(f"Decision {decision_id} rejected: {reason}")
            return True
    
    def execute(self, decision_id: str, executor: Callable = None) -> bool:
        """
        Execute an approved decision.
        
        Args:
            decision_id: Decision to execute
            executor: Optional execution function
            
        Returns:
            Success status
        """
        with self._lock:
            decision = self._decisions.get(decision_id)
            if not decision or decision.status != DecisionStatus.APPROVED:
                return False
            
            # Rate limit check
            if not self._check_rate_limit():
                logger.warning("Rate limit exceeded, delaying execution")
                return False
            
            decision.status = DecisionStatus.EXECUTING
            decision.executed_at = time.time()
            
            # Execute actions
            success = True
            for action in decision.actions:
                action.executed = True
                action.execution_time = time.time()
                
                try:
                    if executor:
                        action.success = executor(action)
                    else:
                        # Default: assume success for now
                        action.success = True
                    
                    if not action.success:
                        success = False
                        break
                        
                except Exception as e:
                    action.success = False
                    action.error = str(e)
                    success = False
                    break
            
            # Update status
            if success:
                decision.status = DecisionStatus.COMPLETED
                self._stats['successful'] += 1
            else:
                decision.status = DecisionStatus.FAILED
                self._stats['failed'] += 1
                decision.outcome = "Execution failed"
            
            self._stats['executed'] += 1
            
            # Record outcome for learning
            if self._enable_learning:
                self._record_outcome(decision)
            
            logger.info(f"Decision {decision_id} executed: {decision.status.name}")
            return success
    
    def rollback(self, decision_id: str) -> bool:
        """Rollback a completed decision"""
        with self._lock:
            decision = self._decisions.get(decision_id)
            if not decision or decision.status != DecisionStatus.COMPLETED:
                return False
            
            # Execute rollback actions
            for action in reversed(decision.actions):
                if action.executed and action.rollback_action:
                    # Would execute rollback here
                    pass
            
            decision.status = DecisionStatus.ROLLED_BACK
            self._stats['rolled_back'] += 1
            
            logger.info(f"Decision {decision_id} rolled back")
            return True
    
    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Get decision by ID"""
        return self._decisions.get(decision_id)
    
    def get_pending(self) -> List[Decision]:
        """Get pending decisions"""
        with self._lock:
            return [
                self._decisions[did]
                for did in self._pending
                if did in self._decisions
            ]
    
    def get_recent(self, count: int = 10) -> List[Decision]:
        """Get recent decisions"""
        with self._lock:
            recent_ids = list(self._history)[-count:]
            return [
                self._decisions[did]
                for did in recent_ids
                if did in self._decisions
            ]
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # INTERNAL METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def _assess_risk(self, decision: Decision, context: DecisionContext) -> RiskAssessment:
        """Assess risk of a decision"""
        assessment = RiskAssessment()
        
        # Base risk by decision type
        type_risk = {
            DecisionType.CODE_GENERATION: 30,
            DecisionType.CODE_IMPROVEMENT: 25,
            DecisionType.BUG_FIX: 35,
            DecisionType.REFACTORING: 40,
            DecisionType.RESOURCE_ALLOCATION: 20,
            DecisionType.ERROR_RECOVERY: 45,
            DecisionType.ROLLBACK: 50,
        }
        
        base_risk = type_risk.get(decision.decision_type, 30)
        
        # Factors
        # Health impact
        if context.health_score < 70:
            assessment.system_impact = 30
        
        # Recent errors increase risk
        if len(context.recent_errors) > 3:
            assessment.system_impact += 20
        
        # Actions count
        if len(decision.actions) > 5:
            assessment.code_impact = 30
        
        # Calculate
        assessment.calculate_overall()
        
        # Adjust based on type base risk
        assessment.score = (assessment.score + base_risk) / 2
        assessment.calculate_overall()
        
        # Add specific risks
        if decision.decision_type == DecisionType.CODE_GENERATION:
            assessment.risks.append("Generated code may have bugs")
            assessment.mitigations.append("Test before integration")
        
        if decision.decision_type == DecisionType.REFACTORING:
            assessment.risks.append("May break existing functionality")
            assessment.mitigations.append("Run full test suite")
        
        return assessment
    
    def _assess_benefit(self, decision: Decision, context: DecisionContext) -> BenefitAssessment:
        """Assess benefit of a decision"""
        assessment = BenefitAssessment()
        
        # Base benefit by decision type
        type_benefit = {
            DecisionType.CODE_GENERATION: 70,
            DecisionType.CODE_IMPROVEMENT: 75,
            DecisionType.BUG_FIX: 85,
            DecisionType.REFACTORING: 65,
            DecisionType.ERROR_RECOVERY: 80,
        }
        
        base_benefit = type_benefit.get(decision.decision_type, 50)
        
        # Factors
        if context.health_score < 80:
            assessment.reliability_gain = 20
        
        if len(context.detected_issues) > 0:
            assessment.maintainability_gain = 15
        
        # Calculate
        assessment.calculate_overall()
        
        # Adjust with base
        assessment.score = (assessment.score + base_benefit) / 2
        
        # Expected outcomes
        if decision.decision_type == DecisionType.BUG_FIX:
            assessment.expected_outcomes.append("Issue resolved")
            assessment.expected_outcomes.append("Improved stability")
        
        if decision.decision_type == DecisionType.CODE_IMPROVEMENT:
            assessment.expected_outcomes.append("Better performance")
            assessment.expected_outcomes.append("Cleaner code")
        
        return assessment
    
    def _calculate_confidence(
        self,
        decision: Decision,
        context: DecisionContext
    ) -> Tuple[ConfidenceLevel, float]:
        """Calculate confidence in decision"""
        score = 0.5  # Base confidence
        
        # Historical success rate
        if decision.decision_type in self._success_rate:
            historical_rate = self._success_rate[decision.decision_type]
            score = score * 0.6 + historical_rate * 0.4
        
        # Context factors
        if context.health_score > 80:
            score += 0.1
        else:
            score -= 0.1
        
        if len(context.recent_errors) > 0:
            score -= 0.05 * min(len(context.recent_errors), 5)
        
        # Ensure bounds
        score = max(0.1, min(1.0, score))
        
        # Determine level
        if score >= 0.9:
            level = ConfidenceLevel.VERY_HIGH
        elif score >= 0.7:
            level = ConfidenceLevel.HIGH
        elif score >= 0.5:
            level = ConfidenceLevel.MODERATE
        elif score >= 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW
        
        return level, score
    
    def _determine_priority(
        self,
        decision: Decision,
        context: DecisionContext
    ) -> DecisionPriority:
        """Determine decision priority"""
        # Critical situations
        if context.health_score < 50:
            return DecisionPriority.IMMEDIATE
        
        if decision.decision_type == DecisionType.ERROR_RECOVERY:
            return DecisionPriority.HIGH
        
        if decision.decision_type == DecisionType.BUG_FIX:
            if "critical" in decision.description.lower():
                return DecisionPriority.HIGH
            return DecisionPriority.NORMAL
        
        # Default
        return DecisionPriority.NORMAL
    
    def _generate_reasoning(self, decision: Decision, context: DecisionContext) -> str:
        """Generate human-readable reasoning"""
        parts = []
        
        # Risk assessment
        parts.append(f"Risk level: {decision.risk.level.name} ({decision.risk.score:.1f}/100)")
        
        # Benefit assessment
        parts.append(f"Expected benefit: {decision.benefit.score:.1f}/100")
        
        # Confidence
        parts.append(f"Confidence: {decision.confidence.name} ({decision.confidence_score:.0%})")
        
        # Expected value
        parts.append(f"Expected value: {decision.expected_value:.1f}")
        
        # Key factors
        if decision.risk.risks:
            parts.append(f"Key risks: {', '.join(decision.risk.risks[:2])}")
        
        if decision.benefit.expected_outcomes:
            parts.append(f"Expected: {', '.join(decision.benefit.expected_outcomes[:2])}")
        
        return " | ".join(parts)
    
    def _check_rate_limit(self) -> bool:
        """Check if within rate limits"""
        now = time.time()
        
        # Clean old entries
        while self._recent_times and now - self._recent_times[0] > 3600:
            self._recent_times.popleft()
        
        # Check hourly limit
        if len(self._recent_times) >= self._policy.max_decisions_per_hour:
            return False
        
        # Check minute limit
        recent_minute = sum(1 for t in self._recent_times if now - t < 60)
        if recent_minute >= self._policy.max_decisions_per_minute:
            return False
        
        # Record
        self._recent_times.append(now)
        return True
    
    def _record_outcome(self, decision: Decision):
        """Record decision outcome for learning"""
        # Calculate outcome score
        if decision.status == DecisionStatus.COMPLETED:
            outcome = 1.0
        elif decision.status == DecisionStatus.FAILED:
            outcome = 0.0
        else:
            outcome = 0.5
        
        # Update type-specific rate
        if decision.decision_type not in self._outcomes:
            self._outcomes[decision.decision_type] = outcome
        else:
            # Moving average
            current = self._outcomes[decision.decision_type]
            self._outcomes[decision.decision_type] = current * 0.8 + outcome * 0.2
        
        self._success_rate[decision.decision_type] = self._outcomes[decision.decision_type]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['pending_count'] = len(self._pending)
            stats['history_count'] = len(self._history)
            stats['success_rates'] = {
                dt.name: rate for dt, rate in self._success_rate.items()
            }
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[AutonomousDecisionEngine] = None


def get_decision_engine(kimi_client=None) -> AutonomousDecisionEngine:
    """Get global decision engine"""
    global _engine
    if _engine is None:
        _engine = AutonomousDecisionEngine(kimi_client=kimi_client)
    elif kimi_client:
        _engine.set_kimi_client(kimi_client)
    return _engine


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Autonomous Decision Engine Test")
    print("="*60)
    
    engine = AutonomousDecisionEngine(kimi_client=None)
    
    # Create context
    context = DecisionContext(
        health_score=85.0,
        resource_usage={'memory': 45.0, 'cpu': 30.0},
        active_goals=['improve_performance'],
    )
    
    # Evaluate decision
    decision = engine.evaluate(
        decision_type=DecisionType.CODE_IMPROVEMENT,
        context=context,
        description="Optimize database query performance",
        trigger="Performance metric below threshold",
    )
    
    print(f"\nDecision Created:")
    print(f"  ID: {decision.id}")
    print(f"  Type: {decision.decision_type.name}")
    print(f"  Status: {decision.status.name}")
    print(f"  Priority: {decision.priority.name}")
    
    print(f"\nAssessment:")
    print(f"  Risk: {decision.risk.level.name} ({decision.risk.score:.1f})")
    print(f"  Benefit: {decision.benefit.score:.1f}")
    print(f"  Confidence: {decision.confidence.name} ({decision.confidence_score:.0%})")
    print(f"  Expected Value: {decision.expected_value:.1f}")
    
    print(f"\nReasoning:")
    print(f"  {decision.reasoning}")
    
    print(f"\nStatistics:")
    stats = engine.get_stats()
    print(f"  Total decisions: {stats['total_decisions']}")
    print(f"  Auto approved: {stats['auto_approved']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

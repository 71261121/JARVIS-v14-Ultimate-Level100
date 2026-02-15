#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Outcome Evaluator
=======================================

Phase 3: Evaluate modification outcomes for learning.

This module evaluates the results of code modifications:
- Success/failure assessment
- Impact measurement
- Quality change detection
- Learning signal generation

Key Features:
- Multi-dimensional evaluation
- Before/after comparison
- Automatic test execution
- Performance benchmarking

Author: JARVIS AI Project
Version: 3.0.0
Target Level: 60-70
"""

import time
import logging
import threading
import ast
import subprocess
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════.SyntaxError═══════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class OutcomeType(Enum):
    """Types of modification outcomes"""
    SUCCESS = auto()           # Modification successful
    PARTIAL_SUCCESS = auto()   # Partially successful
    FAILURE = auto()           # Modification failed
    ROLLBACK = auto()          # Had to rollback
    NO_CHANGE = auto()         # No significant change
    ERROR = auto()             # Error during modification


class ImpactType(Enum):
    """Types of impacts"""
    POSITIVE = auto()    # Improvement
    NEGATIVE = auto()    # Regression
    NEUTRAL = auto()     # No significant change


class EvaluationDimension(Enum):
    """Dimensions for evaluation"""
    FUNCTIONALITY = auto()     # Does it work?
    PERFORMANCE = auto()       # Is it faster?
    READABILITY = auto()       # Is it cleaner?
    MAINTAINABILITY = auto()   # Is it easier to maintain?
    TESTABILITY = auto()       # Is it easier to test?
    SECURITY = auto()          # Is it more secure?


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricChange:
    """
    Change in a metric before/after modification.
    """
    metric_name: str
    before_value: float
    after_value: float
    unit: str = ""
    
    @property
    def absolute_change(self) -> float:
        return self.after_value - self.before_value
    
    @property
    def relative_change(self) -> float:
        if self.before_value == 0:
            return 0.0
        return (self.after_value - self.before_value) / abs(self.before_value) * 100
    
    @property
    def impact_type(self) -> ImpactType:
        """Determine impact type (metric-specific)"""
        # For most metrics, higher is better
        # But for some (like complexity), lower is better
        lower_is_better = {'complexity', 'error_count', 'execution_time', 'memory_usage'}
        
        if self.metric_name.lower() in lower_is_better:
            if self.after_value < self.before_value:
                return ImpactType.POSITIVE
            elif self.after_value > self.before_value:
                return ImpactType.NEGATIVE
        else:
            if self.after_value > self.before_value:
                return ImpactType.POSITIVE
            elif self.after_value < self.before_value:
                return ImpactType.NEGATIVE
        
        return ImpactType.NEUTRAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'metric': self.metric_name,
            'before': self.before_value,
            'after': self.after_value,
            'change': self.absolute_change,
            'change_percent': self.relative_change,
            'impact': self.impact_type.name,
            'unit': self.unit,
        }


@dataclass
class DimensionScore:
    """
    Score for an evaluation dimension.
    """
    dimension: EvaluationDimension
    score: float = 0.0  # -100 to +100
    
    # Details
    before_score: float = 0.0
    after_score: float = 0.0
    confidence: float = 0.5
    
    # Reasoning
    factors: List[str] = field(default_factory=list)
    
    @property
    def impact(self) -> ImpactType:
        if self.score > 10:
            return ImpactType.POSITIVE
        elif self.score < -10:
            return ImpactType.NEGATIVE
        return ImpactType.NEUTRAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dimension': self.dimension.name,
            'score': self.score,
            'impact': self.impact.name,
            'confidence': self.confidence,
            'factors': self.factors,
        }


@dataclass
class ModificationOutcome:
    """
    Complete outcome of a modification.
    """
    # Identification
    modification_id: str
    file_path: str
    modification_type: str
    
    # Overall
    outcome_type: OutcomeType = OutcomeType.SUCCESS
    overall_score: float = 0.0  # -100 to +100
    
    # Dimension scores
    dimensions: Dict[EvaluationDimension, DimensionScore] = field(default_factory=dict)
    
    # Metric changes
    metric_changes: Dict[str, MetricChange] = field(default_factory=dict)
    
    # Testing
    tests_passed: bool = False
    tests_run: int = 0
    tests_failed: int = 0
    test_output: str = ""
    
    # Errors
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Timing
    evaluation_time_ms: float = 0.0
    evaluated_at: float = field(default_factory=time.time)
    
    # Learning signal
    reward_value: float = 0.0
    reward_reasons: List[str] = field(default_factory=list)
    
    @property
    def is_success(self) -> bool:
        return self.outcome_type in (OutcomeType.SUCCESS, OutcomeType.PARTIAL_SUCCESS)
    
    @property
    def is_improvement(self) -> bool:
        return self.overall_score > 10
    
    @property
    def is_regression(self) -> bool:
        return self.overall_score < -10
    
    def get_reward_signal(self) -> float:
        """Calculate reward signal for reinforcement learning"""
        if not self.tests_passed:
            return -1.0  # Failed tests = negative reward
        
        if self.outcome_type == OutcomeType.ERROR:
            return -0.5
        
        if self.outcome_type == OutcomeType.ROLLBACK:
            return -0.3
        
        # Base reward from overall score
        base_reward = self.overall_score / 100.0
        
        # Adjust for test results
        if self.tests_run > 0:
            test_pass_rate = (self.tests_run - self.tests_failed) / self.tests_run
            base_reward = base_reward * 0.7 + test_pass_rate * 0.3
        
        return base_reward
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'modification_id': self.modification_id,
            'file_path': self.file_path,
            'outcome_type': self.outcome_type.name,
            'overall_score': self.overall_score,
            'dimensions': {k.name: v.to_dict() for k, v in self.dimensions.items()},
            'metric_changes': {k: v.to_dict() for k, v in self.metric_changes.items()},
            'tests_passed': self.tests_passed,
            'tests_run': self.tests_run,
            'tests_failed': self.tests_failed,
            'is_success': self.is_success,
            'is_improvement': self.is_improvement,
            'reward_value': self.get_reward_signal(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# OUTCOME EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class OutcomeEvaluator:
    """
    Evaluate modification outcomes for learning.
    
    Features:
    - Multi-dimensional evaluation
    - Test execution
    - Performance comparison
    - Quality metrics
    - Learning signal generation
    
    Usage:
        evaluator = OutcomeEvaluator()
        
        # Evaluate a modification
        outcome = evaluator.evaluate(
            modification_id="mod_123",
            file_path="core/ai/client.py",
            before_code=original_code,
            after_code=modified_code,
        )
        
        print(f"Outcome: {outcome.outcome_type.name}")
        print(f"Score: {outcome.overall_score}")
        print(f"Reward: {outcome.get_reward_signal()}")
    """
    
    def __init__(
        self,
        run_tests: bool = True,
        test_timeout: int = 60,
        benchmark_performance: bool = False,
    ):
        """
        Initialize outcome evaluator.
        
        Args:
            run_tests: Execute tests during evaluation
            test_timeout: Timeout for test execution
            benchmark_performance: Run performance benchmarks
        """
        self._run_tests = run_tests
        self._test_timeout = test_timeout
        self._benchmark = benchmark_performance
        
        # Statistics
        self._stats = {
            'evaluations': 0,
            'successful': 0,
            'failed': 0,
            'regressions': 0,
            'improvements': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("OutcomeEvaluator initialized")
    
    def evaluate(
        self,
        modification_id: str,
        file_path: str,
        before_code: str,
        after_code: str,
        modification_type: str = "unknown",
        run_tests: bool = None,
    ) -> ModificationOutcome:
        """
        Evaluate a code modification.
        
        Args:
            modification_id: Unique identifier
            file_path: Path to modified file
            before_code: Code before modification
            after_code: Code after modification
            modification_type: Type of modification
            run_tests: Override run_tests setting
            
        Returns:
            ModificationOutcome with evaluation results
        """
        start_time = time.time()
        
        outcome = ModificationOutcome(
            modification_id=modification_id,
            file_path=file_path,
            modification_type=modification_type,
        )
        
        # Evaluate each dimension
        outcome.dimensions = self._evaluate_dimensions(
            before_code, after_code, file_path
        )
        
        # Calculate metric changes
        outcome.metric_changes = self._calculate_metrics(
            before_code, after_code
        )
        
        # Run tests
        should_run_tests = run_tests if run_tests is not None else self._run_tests
        if should_run_tests:
            test_result = self._run_tests_for_file(file_path)
            outcome.tests_passed = test_result['passed']
            outcome.tests_run = test_result['total']
            outcome.tests_failed = test_result['failed']
            outcome.test_output = test_result['output']
        
        # Calculate overall score
        outcome.overall_score = self._calculate_overall_score(outcome)
        
        # Determine outcome type
        outcome.outcome_type = self._determine_outcome_type(outcome)
        
        # Calculate reward
        outcome.reward_value = outcome.get_reward_signal()
        outcome.reward_reasons = self._generate_reward_reasons(outcome)
        
        # Timing
        outcome.evaluation_time_ms = (time.time() - start_time) * 1000
        
        # Update stats
        with self._lock:
            self._stats['evaluations'] += 1
            if outcome.is_success:
                self._stats['successful'] += 1
            else:
                self._stats['failed'] += 1
            if outcome.is_improvement:
                self._stats['improvements'] += 1
            if outcome.is_regression:
                self._stats['regressions'] += 1
        
        logger.info(
            f"Evaluation complete: {outcome.outcome_type.name} "
            f"(score={outcome.overall_score:.1f})"
        )
        
        return outcome
    
    def _evaluate_dimensions(
        self,
        before_code: str,
        after_code: str,
        file_path: str,
    ) -> Dict[EvaluationDimension, DimensionScore]:
        """Evaluate all dimensions"""
        dimensions = {}
        
        # Functionality
        dimensions[EvaluationDimension.FUNCTIONALITY] = self._evaluate_functionality(
            before_code, after_code
        )
        
        # Performance
        dimensions[EvaluationDimension.PERFORMANCE] = self._evaluate_performance(
            before_code, after_code
        )
        
        # Readability
        dimensions[EvaluationDimension.READABILITY] = self._evaluate_readability(
            before_code, after_code
        )
        
        # Maintainability
        dimensions[EvaluationDimension.MAINTAINABILITY] = self._evaluate_maintainability(
            before_code, after_code
        )
        
        # Testability
        dimensions[EvaluationDimension.TESTABILITY] = self._evaluate_testability(
            before_code, after_code
        )
        
        # Security
        dimensions[EvaluationDimension.SECURITY] = self._evaluate_security(
            before_code, after_code
        )
        
        return dimensions
    
    def _evaluate_functionality(
        self,
        before_code: str,
        after_code: str,
    ) -> DimensionScore:
        """Evaluate functionality preservation"""
        score = DimensionScore(dimension=EvaluationDimension.FUNCTIONALITY)
        
        try:
            # Parse both
            before_ast = ast.parse(before_code)
            after_ast = ast.parse(after_code)
            
            # Count functions/classes
            before_funcs = sum(1 for n in ast.walk(before_ast) if isinstance(n, ast.FunctionDef))
            after_funcs = sum(1 for n in ast.walk(after_ast) if isinstance(n, ast.FunctionDef))
            
            before_classes = sum(1 for n in ast.walk(before_ast) if isinstance(n, ast.ClassDef))
            after_classes = sum(1 for n in ast.walk(after_ast) if isinstance(n, ast.ClassDef))
            
            # Functionality preserved if counts similar
            func_ratio = after_funcs / max(before_funcs, 1)
            class_ratio = after_classes / max(before_classes, 1)
            
            if func_ratio >= 1.0 and class_ratio >= 1.0:
                score.score = 50  # Added functionality
                score.factors.append("Added functions or classes")
            elif func_ratio >= 0.9 and class_ratio >= 0.9:
                score.score = 30  # Mostly preserved
                score.factors.append("Functionality preserved")
            else:
                score.score = -30  # Removed functionality
                score.factors.append("Some functionality removed")
            
            score.confidence = 0.7
            
        except SyntaxError as e:
            score.score = -50
            score.factors.append(f"Syntax error: {e}")
            score.confidence = 0.9
        
        return score
    
    def _evaluate_performance(
        self,
        before_code: str,
        after_code: str,
    ) -> DimensionScore:
        """Evaluate performance impact"""
        score = DimensionScore(dimension=EvaluationDimension.PERFORMANCE)
        
        # Count loops and comprehensions
        before_loops = before_code.count('for ') + before_code.count('while ')
        after_loops = after_code.count('for ') + after_code.count('while ')
        
        before_comp = before_code.count('[') - before_code.count('{')  # Rough estimate
        after_comp = after_code.count('[') - after_code.count('{')
        
        # Lower loops/comprehensions might indicate optimization
        if after_loops < before_loops:
            score.score += 20
            score.factors.append("Reduced loop count")
        elif after_loops > before_loops:
            score.score -= 10
            score.factors.append("Increased loop count")
        
        # Async additions
        if 'async ' in after_code and 'async ' not in before_code:
            score.score += 15
            score.factors.append("Added async support")
        
        score.confidence = 0.5
        return score
    
    def _evaluate_readability(
        self,
        before_code: str,
        after_code: str,
    ) -> DimensionScore:
        """Evaluate readability impact"""
        score = DimensionScore(dimension=EvaluationDimension.READABILITY)
        
        # Docstrings
        before_docs = before_code.count('"""') + before_code.count("'''")
        after_docs = after_code.count('"""') + after_code.count("'''")
        
        if after_docs > before_docs:
            score.score += 20
            score.factors.append("Added documentation")
        
        # Comments
        before_comments = sum(1 for line in before_code.split('\n') if line.strip().startswith('#'))
        after_comments = sum(1 for line in after_code.split('\n') if line.strip().startswith('#'))
        
        if after_comments > before_comments:
            score.score += 10
            score.factors.append("Added comments")
        
        # Line length (rough measure)
        before_avg_line = len(before_code) / max(len(before_code.split('\n')), 1)
        after_avg_line = len(after_code) / max(len(after_code.split('\n')), 1)
        
        if after_avg_line < before_avg_line * 0.9:
            score.score += 15
            score.factors.append("Shorter lines (better readability)")
        
        score.confidence = 0.6
        return score
    
    def _evaluate_maintainability(
        self,
        before_code: str,
        after_code: str,
    ) -> DimensionScore:
        """Evaluate maintainability impact"""
        score = DimensionScore(dimension=EvaluationDimension.MAINTAINABILITY)
        
        # Function length
        def get_avg_func_length(code):
            try:
                tree = ast.parse(code)
                funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                if not funcs:
                    return 0
                lengths = [n.end_lineno - n.lineno for n in funcs if hasattr(n, 'end_lineno')]
                return sum(lengths) / len(lengths) if lengths else 0
            except:
                return 0
        
        before_avg = get_avg_func_length(before_code)
        after_avg = get_avg_func_length(after_code)
        
        if after_avg < before_avg * 0.9:
            score.score += 25
            score.factors.append("Shorter functions")
        elif after_avg > before_avg * 1.1:
            score.score -= 15
            score.factors.append("Longer functions")
        
        # Type hints
        before_hints = before_code.count(': ') + before_code.count('-> ')
        after_hints = after_code.count(': ') + after_code.count('-> ')
        
        if after_hints > before_hints:
            score.score += 20
            score.factors.append("Added type hints")
        
        score.confidence = 0.6
        return score
    
    def _evaluate_testability(
        self,
        before_code: str,
        after_code: str,
    ) -> DimensionScore:
        """Evaluate testability impact"""
        score = DimensionScore(dimension=EvaluationDimension.TESTABILITY)
        
        # Dependency injection patterns
        if '__init__' in after_code and 'self.' in after_code:
            score.score += 10
            score.factors.append("Class structure for testing")
        
        # Pure functions (no side effects)
        if 'global ' not in after_code and 'global ' in before_code:
            score.score += 15
            score.factors.append("Removed global state")
        
        score.confidence = 0.5
        return score
    
    def _evaluate_security(
        self,
        before_code: str,
        after_code: str,
    ) -> DimensionScore:
        """Evaluate security impact"""
        score = DimensionScore(dimension=EvaluationDimension.SECURITY)
        
        # Dangerous functions
        dangerous = ['eval(', 'exec(', 'compile(', '__import__(', 'input(']
        
        for func in dangerous:
            if func in after_code and func not in before_code:
                score.score -= 30
                score.factors.append(f"Added dangerous function: {func}")
            elif func not in after_code and func in before_code:
                score.score += 20
                score.factors.append(f"Removed dangerous function: {func}")
        
        # SQL injection patterns
        if 'execute(' in after_code and 'f"' in after_code:
            if '%' not in after_code.split('execute(')[1].split(')')[0]:
                score.score -= 20
                score.factors.append("Potential SQL injection")
        
        score.confidence = 0.7
        return score
    
    def _calculate_metrics(
        self,
        before_code: str,
        after_code: str,
    ) -> Dict[str, MetricChange]:
        """Calculate metric changes"""
        metrics = {}
        
        # Lines of code
        before_lines = len(before_code.split('\n'))
        after_lines = len(after_code.split('\n'))
        metrics['lines_of_code'] = MetricChange(
            metric_name='lines_of_code',
            before_value=before_lines,
            after_value=after_lines,
            unit='lines',
        )
        
        # Character count
        metrics['char_count'] = MetricChange(
            metric_name='char_count',
            before_value=len(before_code),
            after_value=len(after_code),
            unit='chars',
        )
        
        # Complexity (rough estimate)
        before_complexity = (
            before_code.count('if ') +
            before_code.count('for ') +
            before_code.count('while ') +
            before_code.count('try:') +
            before_code.count('except')
        )
        after_complexity = (
            after_code.count('if ') +
            after_code.count('for ') +
            after_code.count('while ') +
            after_code.count('try:') +
            after_code.count('except')
        )
        metrics['complexity'] = MetricChange(
            metric_name='complexity',
            before_value=before_complexity,
            after_value=after_complexity,
            unit='points',
        )
        
        return metrics
    
    def _run_tests_for_file(self, file_path: str) -> Dict[str, Any]:
        """Run tests for a file"""
        result = {
            'passed': False,
            'total': 0,
            'failed': 0,
            'output': '',
        }
        
        try:
            # Find test file
            test_file = self._find_test_file(file_path)
            if not test_file:
                result['output'] = "No test file found"
                return result
            
            # Run pytest
            proc = subprocess.run(
                ['python', '-m', 'pytest', test_file, '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=self._test_timeout,
            )
            
            result['output'] = proc.stdout + proc.stderr
            result['passed'] = proc.returncode == 0
            
            # Parse results
            output = proc.stdout
            if 'passed' in output:
                import re
                match = re.search(r'(\d+) passed', output)
                if match:
                    result['total'] = int(match.group(1))
            
            if 'failed' in output:
                match = re.search(r'(\d+) failed', output)
                if match:
                    result['failed'] = int(match.group(1))
                    result['total'] += result['failed']
            
        except subprocess.TimeoutExpired:
            result['output'] = "Test execution timed out"
        except Exception as e:
            result['output'] = f"Test execution error: {e}"
        
        return result
    
    def _find_test_file(self, file_path: str) -> Optional[str]:
        """Find test file for a source file"""
        path = Path(file_path)
        
        # Common test file patterns
        test_patterns = [
            f"test_{path.stem}.py",
            f"{path.stem}_test.py",
            f"tests/test_{path.stem}.py",
            f"tests/{path.stem}_test.py",
        ]
        
        for pattern in test_patterns:
            test_path = path.parent / pattern
            if test_path.exists():
                return str(test_path)
        
        return None
    
    def _calculate_overall_score(self, outcome: ModificationOutcome) -> float:
        """Calculate overall score from dimensions"""
        if not outcome.dimensions:
            return 0.0
        
        # Weighted average
        weights = {
            EvaluationDimension.FUNCTIONALITY: 0.30,
            EvaluationDimension.PERFORMANCE: 0.15,
            EvaluationDimension.READABILITY: 0.15,
            EvaluationDimension.MAINTAINABILITY: 0.20,
            EvaluationDimension.TESTABILITY: 0.10,
            EvaluationDimension.SECURITY: 0.10,
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for dim, score in outcome.dimensions.items():
            weight = weights.get(dim, 0.1)
            total_score += score.score * weight * score.confidence
            total_weight += weight * score.confidence
        
        if total_weight == 0:
            return 0.0
        
        base_score = total_score / total_weight
        
        # Adjust for test results
        if outcome.tests_run > 0:
            pass_rate = (outcome.tests_run - outcome.tests_failed) / outcome.tests_run
            if pass_rate < 1.0:
                base_score -= (1 - pass_rate) * 30  # Penalty for failed tests
        
        return base_score
    
    def _determine_outcome_type(self, outcome: ModificationOutcome) -> OutcomeType:
        """Determine outcome type from evaluation"""
        # Failed tests
        if outcome.tests_failed > 0:
            if outcome.tests_failed == outcome.tests_run:
                return OutcomeType.FAILURE
            return OutcomeType.PARTIAL_SUCCESS
        
        # Syntax errors
        if any("Syntax error" in f for f in outcome.dimensions.get(EvaluationDimension.FUNCTIONALITY, DimensionScore(EvaluationDimension.FUNCTIONALITY)).factors):
            return OutcomeType.ERROR
        
        # Score-based
        if outcome.overall_score >= 20:
            return OutcomeType.SUCCESS
        elif outcome.overall_score >= 0:
            return OutcomeType.PARTIAL_SUCCESS
        elif outcome.overall_score >= -20:
            return OutcomeType.NO_CHANGE
        else:
            return OutcomeType.FAILURE
    
    def _generate_reward_reasons(self, outcome: ModificationOutcome) -> List[str]:
        """Generate reasons for the reward signal"""
        reasons = []
        
        if outcome.tests_passed:
            reasons.append("All tests passed")
        else:
            reasons.append("Some tests failed")
        
        for dim, score in outcome.dimensions.items():
            if score.impact == ImpactType.POSITIVE:
                reasons.append(f"{dim.name}: improved ({score.score:+.0f})")
            elif score.impact == ImpactType.NEGATIVE:
                reasons.append(f"{dim.name}: regressed ({score.score:+.0f})")
        
        return reasons
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics"""
        with self._lock:
            return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_evaluator: Optional[OutcomeEvaluator] = None


def get_outcome_evaluator() -> OutcomeEvaluator:
    """Get global outcome evaluator"""
    global _evaluator
    if _evaluator is None:
        _evaluator = OutcomeEvaluator()
    return _evaluator


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Outcome Evaluator Test")
    print("="*60)
    
    evaluator = OutcomeEvaluator(run_tests=False)
    
    before = '''
def calculate(x, y):
    return x + y
'''
    
    after = '''
def calculate(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
'''
    
    outcome = evaluator.evaluate(
        modification_id="test_1",
        file_path="test.py",
        before_code=before,
        after_code=after,
        modification_type="improvement",
    )
    
    print(f"\nOutcome: {outcome.outcome_type.name}")
    print(f"Overall Score: {outcome.overall_score:.1f}")
    print(f"Reward Signal: {outcome.reward_value:.3f}")
    
    print(f"\nDimensions:")
    for dim, score in outcome.dimensions.items():
        print(f"  {dim.name}: {score.score:+.1f} ({score.impact.name})")
        for factor in score.factors:
            print(f"    - {factor}")
    
    print(f"\nMetrics:")
    for name, change in outcome.metric_changes.items():
        print(f"  {name}: {change.before_value} → {change.after_value} ({change.relative_change:+.1f}%)")
    
    print(f"\nReasons:")
    for reason in outcome.reward_reasons:
        print(f"  - {reason}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Self-Evolution Engine
============================================

Phase 5: Ultimate Self-Evolution System (Level 85-100+)

This module implements autonomous self-improvement:
- Genetic Algorithm-based Code Evolution
- Self-Modifying Code with Safety Constraints
- Architecture Evolution and Optimization
- Mutation Operators for Code Transformation
- Fitness Functions for Code Quality
- Population Management and Selection
- Crossover and Recombination Operators
- Safe Code Modification Pipeline

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                     Self-Evolution Engine                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Genetic   │  │   Mutation  │  │   Fitness   │  Core        │
│  │   Engine    │  │   Operators │  │  Evaluator  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Population │  │  Selection  │  │  Crossover  │  Evolution   │
│  │   Manager   │  │   Engine    │  │   Engine    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │    Safe     │  │  rollback   │  │   Code      │  Safety      │
│  │  Modifier   │  │   Manager   │  │  Validator  │              │
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
import uuid
import math
import random
import hashlib
import ast
import re
import copy
from typing import Dict, Any, Optional, List, Set, Callable, Tuple, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
import difflib

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class MutationType(Enum):
    """Types of code mutations"""
    # Structural mutations
    INSERT_LINE = auto()
    DELETE_LINE = auto()
    MODIFY_LINE = auto()
    SWAP_LINES = auto()
    DUPLICATE_LINE = auto()
    
    # Code transformations
    RENAME_VARIABLE = auto()
    RENAME_FUNCTION = auto()
    EXTRACT_FUNCTION = auto()
    INLINE_FUNCTION = auto()
    ADD_PARAMETER = auto()
    REMOVE_PARAMETER = auto()
    
    # Logic mutations
    INVERT_CONDITION = auto()
    CHANGE_OPERATOR = auto()
    MODIFY_CONSTANT = auto()
    ADD_STATEMENT = auto()
    REMOVE_STATEMENT = auto()
    
    # Optimization mutations
    LOOP_UNROLL = auto()
    LOOP_FUSION = auto()
    DEAD_CODE_REMOVAL = auto()
    CONSTANT_FOLDING = auto()
    
    # Architecture mutations
    ADD_CLASS = auto()
    REMOVE_CLASS = auto()
    ADD_METHOD = auto()
    REMOVE_METHOD = auto()
    CHANGE_INHERITANCE = auto()


class CrossoverType(Enum):
    """Types of crossover operations"""
    SINGLE_POINT = auto()
    TWO_POINT = auto()
    UNIFORM = auto()
    ARITHMETIC = auto()
    BLEND = auto()
    SIMULATED_BINARY = auto()


class SelectionType(Enum):
    """Types of selection mechanisms"""
    TOURNAMENT = auto()
    ROULETTE = auto()
    RANK = auto()
    ELITIST = auto()
    STOCHASTIC_UNIVERSAL = auto()
    TRUNCATION = auto()


class FitnessMetric(Enum):
    """Metrics for fitness evaluation"""
    CORRECTNESS = auto()
    PERFORMANCE = auto()
    READABILITY = auto()
    MAINTAINABILITY = auto()
    SECURITY = auto()
    COMPLEXITY = auto()
    COVERAGE = auto()
    EFFICIENCY = auto()


class EvolutionPhase(Enum):
    """Phases of evolution"""
    INITIALIZATION = auto()
    EVALUATION = auto()
    SELECTION = auto()
    CROSSOVER = auto()
    MUTATION = auto()
    REPLACEMENT = auto()
    TERMINATION = auto()


class IndividualStatus(Enum):
    """Status of an individual in population"""
    CREATED = auto()
    EVALUATING = auto()
    EVALUATED = auto()
    SELECTED = auto()
    REPRODUCED = auto()
    MUTATED = auto()
    DISCARDED = auto()


class SafetyLevel(Enum):
    """Safety levels for modifications"""
    SAFE = auto()          # Guaranteed safe
    LOW_RISK = auto()      # Low risk modification
    MEDIUM_RISK = auto()   # Medium risk, needs review
    HIGH_RISK = auto()     # High risk, requires approval
    DANGEROUS = auto()     # Potentially dangerous, manual review required


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CodeGene:
    """
    A gene representing a unit of code.
    
    The basic building block for genetic operations.
    """
    id: str = field(default_factory=lambda: f"gene_{uuid.uuid4().hex[:8]}")
    
    # Code content
    code: str = ""
    gene_type: str = "statement"  # statement, function, class, module
    
    # Location
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    
    # AST representation
    ast_node: Optional[str] = None  # Serialized AST
    
    # Properties
    complexity: float = 1.0
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    # Metadata
    language: str = "python"
    is_mutable: bool = True
    
    def mutate(self, mutation_type: MutationType) -> 'CodeGene':
        """Create a mutated copy"""
        mutated = copy.deepcopy(self)
        mutated.id = f"gene_{uuid.uuid4().hex[:8]}"
        
        # Apply mutation based on type
        if mutation_type == MutationType.MODIFY_CONSTANT:
            mutated.code = self._mutate_constant(mutated.code)
        elif mutation_type == MutationType.INVERT_CONDITION:
            mutated.code = self._invert_condition(mutated.code)
        elif mutation_type == MutationType.CHANGE_OPERATOR:
            mutated.code = self._change_operator(mutated.code)
        
        return mutated
    
    def _mutate_constant(self, code: str) -> str:
        """Mutate numeric constants"""
        def replace_number(match):
            num = float(match.group())
            mutation = random.uniform(0.5, 2.0)
            return str(num * mutation)
        
        return re.sub(r'\b\d+\.?\d*\b', replace_number, code)
    
    def _invert_condition(self, code: str) -> str:
        """Invert conditional expressions"""
        if 'if ' in code:
            if ' not ' in code:
                code = code.replace(' not ', ' ')
            else:
                code = code.replace('if ', 'if not ')
        return code
    
    def _change_operator(self, code: str) -> str:
        """Change operators randomly"""
        operators = ['+', '-', '*', '/', '//', '%', '**',
                     '==', '!=', '<', '>', '<=', '>=',
                     'and', 'or']
        
        for op in operators:
            if op in code:
                new_op = random.choice(operators)
                code = code.replace(op, new_op, 1)
                break
        
        return code
    
    def crossover(self, other: 'CodeGene', point: int = None) -> 'CodeGene':
        """Perform crossover with another gene"""
        child = CodeGene(
            code=self.code[:point] + other.code[point:] if point else self.code,
            gene_type=self.gene_type,
            file_path=self.file_path,
            language=self.language,
        )
        return child
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'gene_type': self.gene_type,
            'code_length': len(self.code),
            'complexity': self.complexity,
            'dependencies': list(self.dependencies),
        }


@dataclass
class Genome:
    """
    Complete genome representing an individual's code.
    
    Collection of genes forming a complete solution.
    """
    id: str = field(default_factory=lambda: f"genome_{uuid.uuid4().hex[:8]}")
    
    # Genes
    genes: Dict[str, CodeGene] = field(default_factory=dict)
    
    # Structure
    gene_order: List[str] = field(default_factory=list)
    
    # Properties
    total_complexity: float = 0.0
    total_lines: int = 0
    
    # Metadata
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    
    def add_gene(self, gene: CodeGene):
        """Add a gene to the genome"""
        self.genes[gene.id] = gene
        if gene.id not in self.gene_order:
            self.gene_order.append(gene.id)
        self._update_properties()
    
    def remove_gene(self, gene_id: str):
        """Remove a gene from the genome"""
        if gene_id in self.genes:
            del self.genes[gene_id]
            if gene_id in self.gene_order:
                self.gene_order.remove(gene_id)
            self._update_properties()
    
    def _update_properties(self):
        """Update genome properties"""
        self.total_complexity = sum(g.complexity for g in self.genes.values())
        self.total_lines = sum(g.line_end - g.line_start + 1 for g in self.genes.values())
    
    def get_gene_at(self, index: int) -> Optional[CodeGene]:
        """Get gene at index"""
        if 0 <= index < len(self.gene_order):
            gene_id = self.gene_order[index]
            return self.genes.get(gene_id)
        return None
    
    def to_code(self) -> str:
        """Convert genome to code string"""
        lines = []
        for gene_id in self.gene_order:
            gene = self.genes.get(gene_id)
            if gene:
                lines.append(gene.code)
        return '\n'.join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'genes': len(self.genes),
            'total_complexity': self.total_complexity,
            'total_lines': self.total_lines,
            'generation': self.generation,
        }


@dataclass
class FitnessScore:
    """
    Fitness score for an individual.
    
    Multi-dimensional fitness evaluation.
    """
    individual_id: str = ""
    
    # Individual metrics
    metrics: Dict[FitnessMetric, float] = field(default_factory=dict)
    
    # Aggregate
    total_fitness: float = 0.0
    weighted_fitness: float = 0.0
    
    # Weights for each metric
    weights: Dict[FitnessMetric, float] = field(default_factory=lambda: {
        FitnessMetric.CORRECTNESS: 0.30,
        FitnessMetric.PERFORMANCE: 0.20,
        FitnessMetric.READABILITY: 0.15,
        FitnessMetric.MAINTAINABILITY: 0.15,
        FitnessMetric.SECURITY: 0.10,
        FitnessMetric.COMPLEXITY: 0.10,
    })
    
    # Ranking
    rank: int = 0
    percentile: float = 0.0
    
    # Timestamps
    evaluated_at: float = field(default_factory=time.time)
    
    def calculate_total(self) -> float:
        """Calculate total fitness score"""
        self.total_fitness = sum(self.metrics.values())
        return self.total_fitness
    
    def calculate_weighted(self) -> float:
        """Calculate weighted fitness score"""
        self.weighted_fitness = sum(
            self.metrics.get(metric, 0.0) * weight
            for metric, weight in self.weights.items()
        )
        return self.weighted_fitness
    
    def set_metric(self, metric: FitnessMetric, value: float):
        """Set a metric value"""
        self.metrics[metric] = value
        self.calculate_total()
        self.calculate_weighted()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'individual_id': self.individual_id,
            'total_fitness': self.total_fitness,
            'weighted_fitness': self.weighted_fitness,
            'rank': self.rank,
            'metrics': {m.name: v for m, v in self.metrics.items()},
        }


@dataclass
class Individual:
    """
    An individual in the population.
    
    Represents a candidate solution.
    """
    id: str = field(default_factory=lambda: f"ind_{uuid.uuid4().hex[:8]}")
    
    # Genome
    genome: Optional[Genome] = None
    
    # Original code
    original_code: str = ""
    evolved_code: str = ""
    
    # Fitness
    fitness: Optional[FitnessScore] = None
    
    # Status
    status: IndividualStatus = IndividualStatus.CREATED
    generation: int = 0
    
    # Lineage
    parent_ids: List[str] = field(default_factory=list)
    children_ids: List[str] = field(default_factory=list)
    
    # Mutations applied
    mutations: List[Tuple[MutationType, str]] = field(default_factory=list)
    
    # Safety
    safety_level: SafetyLevel = SafetyLevel.SAFE
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    modified_at: float = field(default_factory=time.time)
    
    @property
    def is_evaluated(self) -> bool:
        return self.fitness is not None
    
    @property
    def fitness_score(self) -> float:
        return self.fitness.weighted_fitness if self.fitness else 0.0
    
    def apply_mutation(self, mutation_type: MutationType, description: str = ""):
        """Record a mutation"""
        self.mutations.append((mutation_type, description))
        self.modified_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'generation': self.generation,
            'status': self.status.name,
            'fitness_score': self.fitness_score,
            'mutations_count': len(self.mutations),
            'safety_level': self.safety_level.name,
        }


@dataclass
class Population:
    """
    Population of individuals.
    
    Manages the collection of candidate solutions.
    """
    id: str = field(default_factory=lambda: f"pop_{uuid.uuid4().hex[:8]}")
    
    # Individuals
    individuals: Dict[str, Individual] = field(default_factory=dict)
    
    # Configuration
    max_size: int = 100
    elitism_count: int = 5
    
    # Statistics
    generation: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity: float = 1.0
    
    # History
    fitness_history: deque = field(default_factory=lambda: deque(maxlen=100))
    best_individual_id: Optional[str] = None
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    
    def add_individual(self, individual: Individual):
        """Add an individual to the population"""
        self.individuals[individual.id] = individual
        self._update_statistics()
    
    def remove_individual(self, individual_id: str):
        """Remove an individual"""
        if individual_id in self.individuals:
            del self.individuals[individual_id]
            self._update_statistics()
    
    def get_individual(self, individual_id: str) -> Optional[Individual]:
        """Get individual by ID"""
        return self.individuals.get(individual_id)
    
    def get_best(self) -> Optional[Individual]:
        """Get best individual"""
        if not self.individuals:
            return None
        
        return max(self.individuals.values(), key=lambda x: x.fitness_score)
    
    def get_sorted(self, descending: bool = True) -> List[Individual]:
        """Get individuals sorted by fitness"""
        return sorted(
            self.individuals.values(),
            key=lambda x: x.fitness_score,
            reverse=descending,
        )
    
    def _update_statistics(self):
        """Update population statistics"""
        if not self.individuals:
            return
        
        fitnesses = [ind.fitness_score for ind in self.individuals.values()]
        
        self.best_fitness = max(fitnesses)
        self.avg_fitness = sum(fitnesses) / len(fitnesses)
        
        # Update best individual
        best = self.get_best()
        if best:
            self.best_individual_id = best.id
        
        # Record in history
        self.fitness_history.append({
            'generation': self.generation,
            'best': self.best_fitness,
            'avg': self.avg_fitness,
            'size': len(self.individuals),
        })
    
    def calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.individuals) < 2:
            return 1.0
        
        # Calculate average pairwise distance
        codes = [ind.evolved_code for ind in self.individuals.values()]
        
        total_distance = 0.0
        comparisons = 0
        
        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                # Use sequence matcher for distance
                similarity = difflib.SequenceMatcher(None, code1, code2).ratio()
                total_distance += 1 - similarity
                comparisons += 1
        
        self.diversity = total_distance / comparisons if comparisons > 0 else 1.0
        return self.diversity
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'size': len(self.individuals),
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'diversity': self.diversity,
        }


@dataclass
class EvolutionConfig:
    """
    Configuration for evolution process.
    """
    # Population
    population_size: int = 50
    max_generations: int = 100
    elitism_rate: float = 0.1
    
    # Selection
    selection_type: SelectionType = SelectionType.TOURNAMENT
    tournament_size: int = 3
    selection_pressure: float = 1.5
    
    # Crossover
    crossover_type: CrossoverType = CrossoverType.UNIFORM
    crossover_rate: float = 0.8
    
    # Mutation
    mutation_rate: float = 0.2
    mutation_types: List[MutationType] = field(default_factory=lambda: [
        MutationType.MODIFY_CONSTANT,
        MutationType.INVERT_CONDITION,
        MutationType.CHANGE_OPERATOR,
        MutationType.ADD_STATEMENT,
    ])
    
    # Termination
    fitness_threshold: float = 0.95
    stagnation_limit: int = 20
    
    # Safety
    max_code_size: int = 10000
    allowed_modules: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        'exec(', 'eval(', 'compile(',
        '__import__', 'subprocess', 'os.system',
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'population_size': self.population_size,
            'max_generations': self.max_generations,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'selection_type': self.selection_type.name,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MUTATION OPERATORS
# ═══════════════════════════════════════════════════════════════════════════════

class MutationOperators:
    """
    Collection of mutation operators for code transformation.
    """
    
    def __init__(self):
        """Initialize mutation operators."""
        self._operators: Dict[MutationType, Callable] = {
            MutationType.MODIFY_CONSTANT: self._modify_constant,
            MutationType.INVERT_CONDITION: self._invert_condition,
            MutationType.CHANGE_OPERATOR: self._change_operator,
            MutationType.ADD_STATEMENT: self._add_statement,
            MutationType.REMOVE_STATEMENT: self._remove_statement,
            MutationType.INSERT_LINE: self._insert_line,
            MutationType.DELETE_LINE: self._delete_line,
            MutationType.MODIFY_LINE: self._modify_line,
            MutationType.SWAP_LINES: self._swap_lines,
            MutationType.RENAME_VARIABLE: self._rename_variable,
        }
        
        self._stats = {
            'mutations_applied': 0,
            'successful_mutations': 0,
            'failed_mutations': 0,
        }
    
    def mutate(
        self,
        code: str,
        mutation_type: MutationType = None,
    ) -> Tuple[str, MutationType]:
        """
        Apply mutation to code.
        
        Args:
            code: Source code to mutate
            mutation_type: Specific mutation type (or random)
            
        Returns:
            Tuple of (mutated_code, applied_mutation_type)
        """
        # Select random mutation if not specified
        if mutation_type is None:
            mutation_type = random.choice(list(self._operators.keys()))
        
        operator = self._operators.get(mutation_type)
        
        if operator:
            try:
                mutated = operator(code)
                self._stats['successful_mutations'] += 1
                return mutated, mutation_type
            except Exception as e:
                logger.warning(f"Mutation failed: {e}")
                self._stats['failed_mutations'] += 1
                return code, mutation_type
        
        self._stats['mutations_applied'] += 1
        return code, mutation_type
    
    def _modify_constant(self, code: str) -> str:
        """Modify numeric constants"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            # Find numeric constants
            numbers = re.findall(r'\b\d+\.?\d*\b', line)
            if numbers:
                # Modify a random number
                old_num = random.choice(numbers)
                new_num = str(float(old_num) * random.uniform(0.5, 2.0))
                lines[i] = line.replace(old_num, new_num, 1)
                break
        
        return '\n'.join(lines)
    
    def _invert_condition(self, code: str) -> str:
        """Invert conditional statements"""
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            if 'if ' in line and 'elif ' not in line:
                if ' not ' in line:
                    lines[i] = line.replace(' not ', ' ', 1)
                else:
                    # Find condition
                    match = re.search(r'if\s+(.+):', line)
                    if match:
                        condition = match.group(1)
                        new_condition = f'not ({condition})'
                        lines[i] = line.replace(condition, new_condition, 1)
                break
        
        return '\n'.join(lines)
    
    def _change_operator(self, code: str) -> str:
        """Change comparison or arithmetic operators"""
        operators = {
            '+': ['-', '*', '/'],
            '-': ['+', '*', '/'],
            '*': ['+', '-', '/'],
            '/': ['+', '-', '*'],
            '==': ['!=', '<=', '>='],
            '!=': ['==', '<=', '>='],
            '<': ['>', '<=', '>='],
            '>': ['<', '<=', '>='],
            '<=': ['<', '>', '>='],
            '>=': ['<', '>', '<='],
        }
        
        for old_op, alternatives in operators.items():
            if old_op in code:
                new_op = random.choice(alternatives)
                return code.replace(old_op, new_op, 1)
        
        return code
    
    def _add_statement(self, code: str) -> str:
        """Add a new statement"""
        # Possible statements to add
        statements = [
            'pass',
            '# TODO: optimize',
            '# Added by evolution',
            'assert True',
            'result = None',
            'value = 0',
        ]
        
        lines = code.split('\n')
        
        # Find a good insertion point
        insert_pos = random.randint(0, len(lines))
        new_statement = random.choice(statements)
        
        # Indent appropriately
        if insert_pos > 0 and lines[insert_pos - 1].startswith('    '):
            new_statement = '    ' + new_statement
        
        lines.insert(insert_pos, new_statement)
        return '\n'.join(lines)
    
    def _remove_statement(self, code: str) -> str:
        """Remove a statement"""
        lines = code.split('\n')
        
        # Don't remove if too few lines
        if len(lines) <= 3:
            return code
        
        # Find removable lines (not function/class definitions)
        removable = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith(('def ', 'class ', 'import ', 'from ')):
                removable.append(i)
        
        if removable:
            remove_idx = random.choice(removable)
            del lines[remove_idx]
        
        return '\n'.join(lines)
    
    def _insert_line(self, code: str) -> str:
        """Insert a random line"""
        return self._add_statement(code)
    
    def _delete_line(self, code: str) -> str:
        """Delete a random line"""
        return self._remove_statement(code)
    
    def _modify_line(self, code: str) -> str:
        """Modify a random line"""
        # Combine multiple modifications
        code, _ = self.mutate(code, MutationType.CHANGE_OPERATOR)
        return code
    
    def _swap_lines(self, code: str) -> str:
        """Swap two lines"""
        lines = code.split('\n')
        
        if len(lines) < 2:
            return code
        
        i, j = random.sample(range(len(lines)), 2)
        lines[i], lines[j] = lines[j], lines[i]
        
        return '\n'.join(lines)
    
    def _rename_variable(self, code: str) -> str:
        """Rename a variable"""
        # Find variable names
        variables = re.findall(r'\b([a-z_][a-z0-9_]*)\s*=', code)
        
        # Filter out keywords
        keywords = {'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 'import', 'from'}
        variables = [v for v in variables if v not in keywords]
        
        if variables:
            old_var = random.choice(variables)
            new_var = f"var_{random.randint(1, 100)}"
            return code.replace(old_var, new_var)
        
        return code
    
    def get_stats(self) -> Dict[str, Any]:
        """Get operator statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class FitnessEvaluator:
    """
    Evaluates fitness of individuals.
    """
    
    def __init__(self, weights: Dict[FitnessMetric, float] = None):
        """
        Initialize fitness evaluator.
        
        Args:
            weights: Weights for each metric
        """
        self._weights = weights or {
            FitnessMetric.CORRECTNESS: 0.30,
            FitnessMetric.PERFORMANCE: 0.20,
            FitnessMetric.READABILITY: 0.15,
            FitnessMetric.MAINTAINABILITY: 0.15,
            FitnessMetric.SECURITY: 0.10,
            FitnessMetric.COMPLEXITY: 0.10,
        }
        
        self._evaluators: Dict[FitnessMetric, Callable] = {
            FitnessMetric.CORRECTNESS: self._evaluate_correctness,
            FitnessMetric.PERFORMANCE: self._evaluate_performance,
            FitnessMetric.READABILITY: self._evaluate_readability,
            FitnessMetric.MAINTAINABILITY: self._evaluate_maintainability,
            FitnessMetric.SECURITY: self._evaluate_security,
            FitnessMetric.COMPLEXITY: self._evaluate_complexity,
        }
        
        self._stats = {
            'evaluations': 0,
            'avg_fitness': 0.0,
        }
    
    def evaluate(
        self,
        individual: Individual,
        metrics: List[FitnessMetric] = None,
    ) -> FitnessScore:
        """
        Evaluate fitness of an individual.
        
        Args:
            individual: Individual to evaluate
            metrics: Specific metrics to evaluate
            
        Returns:
            FitnessScore
        """
        metrics = metrics or list(self._evaluators.keys())
        
        score = FitnessScore(
            individual_id=individual.id,
            weights=self._weights,
        )
        
        for metric in metrics:
            evaluator = self._evaluators.get(metric)
            if evaluator:
                value = evaluator(individual.evolved_code)
                score.set_metric(metric, value)
        
        score.calculate_weighted()
        
        self._stats['evaluations'] += 1
        self._stats['avg_fitness'] = (
            self._stats['avg_fitness'] * 0.9 +
            score.weighted_fitness * 0.1
        )
        
        return score
    
    def _evaluate_correctness(self, code: str) -> float:
        """Evaluate code correctness"""
        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return 0.0
    
    def _evaluate_performance(self, code: str) -> float:
        """Evaluate code performance (heuristic)"""
        score = 1.0
        
        # Penalize nested loops
        nested_loop_penalty = code.count('for ') * 0.05 + code.count('while ') * 0.05
        score -= nested_loop_penalty
        
        # Reward efficient patterns
        if 'set(' in code:
            score += 0.05
        if 'dict' in code:
            score += 0.03
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_readability(self, code: str) -> float:
        """Evaluate code readability"""
        score = 1.0
        
        lines = code.split('\n')
        
        # Line length penalty
        long_lines = sum(1 for line in lines if len(line) > 80)
        score -= long_lines * 0.02
        
        # Reward comments
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        score += comment_lines * 0.02
        
        # Reward docstrings
        if '"""' in code or "'''" in code:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_maintainability(self, code: str) -> float:
        """Evaluate code maintainability"""
        score = 1.0
        
        # Penalize magic numbers
        magic_numbers = len(re.findall(r'(?<!["\'])\b\d{2,}\b(?!["\'])', code))
        score -= magic_numbers * 0.02
        
        # Reward function decomposition
        function_count = code.count('def ')
        if function_count > 1:
            score += 0.1
        
        # Penalize deeply nested code
        max_indent = 0
        for line in code.split('\n'):
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent // 4)
        
        if max_indent > 3:
            score -= (max_indent - 3) * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_security(self, code: str) -> float:
        """Evaluate code security"""
        score = 1.0
        
        # Check for dangerous patterns
        dangerous = ['exec(', 'eval(', '__import__', 'subprocess', 'os.system']
        for pattern in dangerous:
            if pattern in code:
                score -= 0.2
        
        # Check for input validation
        if 'assert' in code or 'raise' in code:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_complexity(self, code: str) -> float:
        """Evaluate code complexity (inverse - lower is better)"""
        lines = code.split('\n')
        
        # Cyclomatic complexity estimate
        complexity = 1
        complexity += code.count('if ')
        complexity += code.count('elif ')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('and ')
        complexity += code.count('or ')
        
        # Convert to score (lower complexity = higher score)
        score = 1.0 / (1 + complexity * 0.1)
        
        return max(0.0, min(1.0, score))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# SELECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SelectionEngine:
    """
    Selection mechanisms for genetic algorithm.
    """
    
    def __init__(
        self,
        selection_type: SelectionType = SelectionType.TOURNAMENT,
        tournament_size: int = 3,
    ):
        """
        Initialize selection engine.
        
        Args:
            selection_type: Type of selection
            tournament_size: Size for tournament selection
        """
        self._selection_type = selection_type
        self._tournament_size = tournament_size
        
        self._stats = {
            'selections': 0,
        }
    
    def select(
        self,
        population: Population,
        count: int = 2,
    ) -> List[Individual]:
        """
        Select individuals for reproduction.
        
        Args:
            population: Population to select from
            count: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        individuals = list(population.individuals.values())
        
        if not individuals:
            return []
        
        selected = []
        
        for _ in range(count):
            if self._selection_type == SelectionType.TOURNAMENT:
                ind = self._tournament_selection(individuals)
            elif self._selection_type == SelectionType.ROULETTE:
                ind = self._roulette_selection(individuals)
            elif self._selection_type == SelectionType.RANK:
                ind = self._rank_selection(individuals)
            else:
                ind = self._elitist_selection(individuals)
            
            if ind:
                selected.append(ind)
        
        self._stats['selections'] += len(selected)
        return selected
    
    def _tournament_selection(self, individuals: List[Individual]) -> Individual:
        """Tournament selection"""
        tournament = random.sample(
            individuals,
            min(self._tournament_size, len(individuals))
        )
        return max(tournament, key=lambda x: x.fitness_score)
    
    def _roulette_selection(self, individuals: List[Individual]) -> Individual:
        """Roulette wheel selection"""
        total_fitness = sum(ind.fitness_score for ind in individuals)
        
        if total_fitness == 0:
            return random.choice(individuals)
        
        r = random.random() * total_fitness
        cumulative = 0
        
        for ind in individuals:
            cumulative += ind.fitness_score
            if cumulative >= r:
                return ind
        
        return individuals[-1]
    
    def _rank_selection(self, individuals: List[Individual]) -> Individual:
        """Rank-based selection"""
        sorted_inds = sorted(individuals, key=lambda x: x.fitness_score)
        ranks = list(range(1, len(sorted_inds) + 1))
        total_rank = sum(ranks)
        
        r = random.random() * total_rank
        cumulative = 0
        
        for i, ind in enumerate(sorted_inds):
            cumulative += ranks[i]
            if cumulative >= r:
                return ind
        
        return sorted_inds[-1]
    
    def _elitist_selection(self, individuals: List[Individual]) -> Individual:
        """Elitist selection (select best)"""
        return max(individuals, key=lambda x: x.fitness_score)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get selection statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# CROSSOVER ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class CrossoverEngine:
    """
    Crossover operations for genetic algorithm.
    """
    
    def __init__(
        self,
        crossover_type: CrossoverType = CrossoverType.UNIFORM,
        crossover_rate: float = 0.8,
    ):
        """
        Initialize crossover engine.
        
        Args:
            crossover_type: Type of crossover
            crossover_rate: Probability of crossover
        """
        self._crossover_type = crossover_type
        self._crossover_rate = crossover_rate
        
        self._stats = {
            'crossovers': 0,
            'successful': 0,
        }
    
    def crossover(
        self,
        parent1: Individual,
        parent2: Individual,
    ) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of (child1, child2)
        """
        self._stats['crossovers'] += 1
        
        if random.random() > self._crossover_rate:
            # No crossover, return copies of parents
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)
            child1.id = f"ind_{uuid.uuid4().hex[:8]}"
            child2.id = f"ind_{uuid.uuid4().hex[:8]}"
            return child1, child2
        
        code1 = parent1.evolved_code
        code2 = parent2.evolved_code
        
        if self._crossover_type == CrossoverType.SINGLE_POINT:
            child1_code, child2_code = self._single_point_crossover(code1, code2)
        elif self._crossover_type == CrossoverType.TWO_POINT:
            child1_code, child2_code = self._two_point_crossover(code1, code2)
        else:
            child1_code, child2_code = self._uniform_crossover(code1, code2)
        
        # Create children
        child1 = Individual(
            evolved_code=child1_code,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )
        
        child2 = Individual(
            evolved_code=child2_code,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id],
        )
        
        self._stats['successful'] += 1
        return child1, child2
    
    def _single_point_crossover(
        self,
        code1: str,
        code2: str,
    ) -> Tuple[str, str]:
        """Single point crossover"""
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        point = random.randint(1, min(len(lines1), len(lines2)) - 1)
        
        child1 = lines1[:point] + lines2[point:]
        child2 = lines2[:point] + lines1[point:]
        
        return '\n'.join(child1), '\n'.join(child2)
    
    def _two_point_crossover(
        self,
        code1: str,
        code2: str,
    ) -> Tuple[str, str]:
        """Two point crossover"""
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        points = sorted(random.sample(range(1, min(len(lines1), len(lines2))), 2))
        
        child1 = lines1[:points[0]] + lines2[points[0]:points[1]] + lines1[points[1]:]
        child2 = lines2[:points[0]] + lines1[points[0]:points[1]] + lines2[points[1]:]
        
        return '\n'.join(child1), '\n'.join(child2)
    
    def _uniform_crossover(
        self,
        code1: str,
        code2: str,
    ) -> Tuple[str, str]:
        """Uniform crossover"""
        lines1 = code1.split('\n')
        lines2 = code2.split('\n')
        
        max_len = max(len(lines1), len(lines2))
        
        child1 = []
        child2 = []
        
        for i in range(max_len):
            line1 = lines1[i] if i < len(lines1) else ''
            line2 = lines2[i] if i < len(lines2) else ''
            
            if random.random() < 0.5:
                child1.append(line1)
                child2.append(line2)
            else:
                child1.append(line2)
                child2.append(line1)
        
        return '\n'.join(child1), '\n'.join(child2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get crossover statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# SAFE MODIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class SafeModifier:
    """
    Safe code modification with validation.
    """
    
    def __init__(self, forbidden_patterns: List[str] = None):
        """
        Initialize safe modifier.
        
        Args:
            forbidden_patterns: Patterns to forbid
        """
        self._forbidden_patterns = forbidden_patterns or [
            'exec(', 'eval(', 'compile(',
            '__import__', 'subprocess', 'os.system',
            'open(', 'file(', 'input(',
        ]
        
        self._stats = {
            'validations': 0,
            'safe_modifications': 0,
            'blocked_modifications': 0,
        }
    
    def validate(self, code: str) -> Tuple[bool, List[str], SafetyLevel]:
        """
        Validate code for safety.
        
        Args:
            code: Code to validate
            
        Returns:
            Tuple of (is_valid, errors, safety_level)
        """
        self._stats['validations'] += 1
        errors = []
        safety_level = SafetyLevel.SAFE
        
        # Check syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return False, errors, SafetyLevel.DANGEROUS
        
        # Check forbidden patterns
        for pattern in self._forbidden_patterns:
            if pattern in code:
                errors.append(f"Forbidden pattern found: {pattern}")
                safety_level = SafetyLevel.HIGH_RISK
        
        # Check for dangerous attributes
        if '__' in code:
            if safety_level.value < SafetyLevel.MEDIUM_RISK.value:
                safety_level = SafetyLevel.MEDIUM_RISK
        
        is_valid = len(errors) == 0
        return is_valid, errors, safety_level
    
    def apply_modification(
        self,
        original_code: str,
        modified_code: str,
        require_safe: bool = True,
    ) -> Tuple[str, bool, str]:
        """
        Apply a modification with safety checks.
        
        Args:
            original_code: Original code
            modified_code: Modified code
            require_safe: Require safe modification
            
        Returns:
            Tuple of (final_code, success, message)
        """
        is_valid, errors, safety_level = self.validate(modified_code)
        
        if not is_valid:
            self._stats['blocked_modifications'] += 1
            return original_code, False, f"Validation failed: {errors}"
        
        if require_safe and safety_level != SafetyLevel.SAFE:
            self._stats['blocked_modifications'] += 1
            return original_code, False, f"Safety level too low: {safety_level.name}"
        
        self._stats['safe_modifications'] += 1
        return modified_code, True, "Modification applied successfully"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get modifier statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-EVOLUTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SelfEvolutionEngine:
    """
    Main self-evolution engine.
    
    Orchestrates the entire evolution process.
    """
    
    def __init__(
        self,
        config: EvolutionConfig = None,
    ):
        """
        Initialize self-evolution engine.
        
        Args:
            config: Evolution configuration
        """
        self._config = config or EvolutionConfig()
        
        # Components
        self._mutation_operators = MutationOperators()
        self._fitness_evaluator = FitnessEvaluator()
        self._selection_engine = SelectionEngine(
            selection_type=self._config.selection_type,
            tournament_size=self._config.tournament_size,
        )
        self._crossover_engine = CrossoverEngine(
            crossover_type=self._config.crossover_type,
            crossover_rate=self._config.crossover_rate,
        )
        self._safe_modifier = SafeModifier(
            forbidden_patterns=self._config.forbidden_patterns,
        )
        
        # Population
        self._population: Optional[Population] = None
        
        # State
        self._running = False
        self._current_generation = 0
        self._best_individual: Optional[Individual] = None
        
        # History
        self._evolution_history: deque = deque(maxlen=1000)
        
        # Statistics
        self._stats = {
            'generations': 0,
            'total_mutations': 0,
            'total_crossovers': 0,
            'improvements': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("SelfEvolutionEngine initialized")
    
    def initialize_population(
        self,
        seed_code: str,
        population_size: int = None,
    ) -> Population:
        """
        Initialize population from seed code.
        
        Args:
            seed_code: Initial code to evolve
            population_size: Size of population
            
        Returns:
            Initialized population
        """
        population_size = population_size or self._config.population_size
        
        population = Population(
            max_size=population_size,
            elitism_count=self._config.elitism_rate * population_size,
        )
        
        # Create initial individual
        initial = Individual(
            original_code=seed_code,
            evolved_code=seed_code,
            generation=0,
        )
        
        # Evaluate initial
        initial.fitness = self._fitness_evaluator.evaluate(initial)
        population.add_individual(initial)
        
        # Create variations
        for i in range(population_size - 1):
            # Apply random mutation
            mutated_code, mutation_type = self._mutation_operators.mutate(seed_code)
            
            # Validate
            is_valid, _, _ = self._safe_modifier.validate(mutated_code)
            
            if is_valid:
                ind = Individual(
                    original_code=seed_code,
                    evolved_code=mutated_code,
                    generation=0,
                )
                ind.apply_mutation(mutation_type, f"Initial variation {i}")
                ind.fitness = self._fitness_evaluator.evaluate(ind)
                population.add_individual(ind)
        
        self._population = population
        self._best_individual = population.get_best()
        
        logger.info(f"Initialized population with {len(population.individuals)} individuals")
        
        return population
    
    def evolve(
        self,
        generations: int = None,
        target_fitness: float = None,
    ) -> Dict[str, Any]:
        """
        Run evolution process.
        
        Args:
            generations: Number of generations (or use config)
            target_fitness: Target fitness to achieve
            
        Returns:
            Evolution result
        """
        generations = generations or self._config.max_generations
        target_fitness = target_fitness or self._config.fitness_threshold
        
        if not self._population:
            return {'success': False, 'error': 'Population not initialized'}
        
        self._running = True
        self._current_generation = 0
        
        stagnation_count = 0
        last_best_fitness = 0.0
        
        results = {
            'generations': 0,
            'final_fitness': 0.0,
            'improvements': 0,
            'best_individual': None,
        }
        
        for gen in range(generations):
            if not self._running:
                break
            
            self._current_generation = gen
            self._population.generation = gen
            
            # Selection
            parents = self._selection_engine.select(
                self._population,
                count=max(2, len(self._population.individuals) // 2),
            )
            
            # Crossover
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self._crossover_engine.crossover(
                    parents[i],
                    parents[i + 1],
                )
                offspring.extend([child1, child2])
            
            # Mutation
            for ind in offspring:
                if random.random() < self._config.mutation_rate:
                    mutated_code, mutation_type = self._mutation_operators.mutate(
                        ind.evolved_code
                    )
                    
                    is_valid, _, safety = self._safe_modifier.validate(mutated_code)
                    
                    if is_valid and safety != SafetyLevel.DANGEROUS:
                        ind.evolved_code = mutated_code
                        ind.apply_mutation(mutation_type)
                        self._stats['total_mutations'] += 1
            
            # Evaluate offspring
            for ind in offspring:
                ind.fitness = self._fitness_evaluator.evaluate(ind)
            
            # Replacement
            self._replace_population(offspring)
            
            # Update best
            current_best = self._population.get_best()
            if current_best:
                if current_best.fitness_score > last_best_fitness:
                    self._best_individual = current_best
                    last_best_fitness = current_best.fitness_score
                    stagnation_count = 0
                    self._stats['improvements'] += 1
                else:
                    stagnation_count += 1
            
            # Record history
            self._evolution_history.append({
                'generation': gen,
                'best_fitness': self._population.best_fitness,
                'avg_fitness': self._population.avg_fitness,
                'diversity': self._population.calculate_diversity(),
            })
            
            # Check termination conditions
            if last_best_fitness >= target_fitness:
                logger.info(f"Target fitness achieved at generation {gen}")
                break
            
            if stagnation_count >= self._config.stagnation_limit:
                logger.info(f"Evolution stagnated at generation {gen}")
                break
        
        self._stats['generations'] = self._current_generation
        
        results['generations'] = self._current_generation
        results['final_fitness'] = last_best_fitness
        results['improvements'] = self._stats['improvements']
        results['best_individual'] = self._best_individual.to_dict() if self._best_individual else None
        
        return results
    
    def _replace_population(self, offspring: List[Individual]):
        """Replace population with offspring (elitism)"""
        # Get elites
        sorted_individuals = self._population.get_sorted()
        elites = sorted_individuals[:int(self._config.elitism_rate * self._population.max_size)]
        
        # Clear population
        self._population.individuals.clear()
        
        # Add elites
        for elite in elites:
            self._population.add_individual(elite)
        
        # Add best offspring
        sorted_offspring = sorted(offspring, key=lambda x: x.fitness_score, reverse=True)
        
        for ind in sorted_offspring:
            if len(self._population.individuals) >= self._population.max_size:
                break
            
            # Validate before adding
            is_valid, _, _ = self._safe_modifier.validate(ind.evolved_code)
            if is_valid:
                self._population.add_individual(ind)
        
        self._population._update_statistics()
    
    def stop(self):
        """Stop evolution"""
        self._running = False
    
    def get_best(self) -> Optional[Individual]:
        """Get best individual"""
        return self._best_individual
    
    def get_population(self) -> Optional[Population]:
        """Get current population"""
        return self._population
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['current_generation'] = self._current_generation
            stats['population_size'] = len(self._population.individuals) if self._population else 0
            stats['mutation_operators'] = self._mutation_operators.get_stats()
            stats['fitness_evaluator'] = self._fitness_evaluator.get_stats()
            stats['selection_engine'] = self._selection_engine.get_stats()
            stats['crossover_engine'] = self._crossover_engine.get_stats()
            stats['safe_modifier'] = self._safe_modifier.get_stats()
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[SelfEvolutionEngine] = None


def get_evolution_engine(**kwargs) -> SelfEvolutionEngine:
    """Get global evolution engine"""
    global _engine
    if _engine is None:
        _engine = SelfEvolutionEngine(**kwargs)
    return _engine


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Self-Evolution Engine Test")
    print("="*60)
    
    # Create engine
    config = EvolutionConfig(
        population_size=20,
        max_generations=10,
        mutation_rate=0.3,
    )
    engine = SelfEvolutionEngine(config=config)
    
    # Seed code
    seed_code = '''
def calculate(x, y):
    """Calculate sum of two numbers."""
    if x > 0 and y > 0:
        result = x + y
        return result
    else:
        return 0
'''
    
    # Initialize population
    print("\n1. Initializing population...")
    population = engine.initialize_population(seed_code, population_size=20)
    print(f"   Population size: {len(population.individuals)}")
    print(f"   Initial best fitness: {population.best_fitness:.3f}")
    
    # Run evolution
    print("\n2. Running evolution...")
    result = engine.evolve(generations=10)
    print(f"   Generations: {result['generations']}")
    print(f"   Final fitness: {result['final_fitness']:.3f}")
    print(f"   Improvements: {result['improvements']}")
    
    # Get best
    print("\n3. Best individual:")
    best = engine.get_best()
    if best:
        print(f"   Fitness: {best.fitness_score:.3f}")
        print(f"   Mutations: {len(best.mutations)}")
    
    # Statistics
    print("\n4. Statistics:")
    stats = engine.get_stats()
    print(f"   Total mutations: {stats['total_mutations']}")
    print(f"   Safe modifications: {stats['safe_modifier']['safe_modifications']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

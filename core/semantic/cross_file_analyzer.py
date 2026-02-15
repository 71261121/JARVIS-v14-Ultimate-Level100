#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Cross-File Analyzer
==========================================

Phase 2: Project-level code understanding.

This module provides:
- Dependency graph construction
- Cross-file relationship analysis
- Project-wide pattern detection
- Impact analysis for modifications

Key Features:
- Import/export tracking
- Function call graph
- Class inheritance analysis
- Circular dependency detection
- Modification impact prediction

Author: JARVIS AI Project
Version: 2.0.0
Target Level: 40-50
"""

import ast
import os
import logging
import time
import json
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter
from enum import Enum, auto

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class RelationType(Enum):
    """Types of file relationships"""
    IMPORTS = auto()           # File A imports from File B
    IMPORTED_BY = auto()       # File A is imported by File B
    CALLS = auto()             # File A calls function from File B
    CALLED_BY = auto()         # File A function is called by File B
    INHERITS = auto()          # Class in A inherits from B
    INHERITED_BY = auto()      # Class in B is inherited by A
    USES = auto()              # General usage relationship
    USED_BY = auto()           # General usage relationship
    TESTS = auto()             # A tests B
    TESTED_BY = auto()         # A is tested by B
    CONFIGURES = auto()        # A configures B
    CONFIGURED_BY = auto()     # A is configured by B


class DependencyStrength(Enum):
    """Strength of dependency between files"""
    WEAK = "weak"        # Indirect, optional dependency
    MODERATE = "moderate"  # Direct but replaceable
    STRONG = "strong"    # Direct, essential dependency
    CRITICAL = "critical"  # Breaking change would cascade


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FileRelation:
    """
    Relationship between two files.
    """
    source_file: str
    target_file: str
    relation_type: RelationType
    strength: DependencyStrength = DependencyStrength.MODERATE
    
    # Details
    imported_symbols: List[str] = field(default_factory=list)
    called_functions: List[str] = field(default_factory=list)
    inherited_classes: List[str] = field(default_factory=list)
    
    # Line numbers for the relationship
    line_numbers: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source': self.source_file,
            'target': self.target_file,
            'type': self.relation_type.name,
            'strength': self.strength.value,
            'imported_symbols': self.imported_symbols,
            'called_functions': self.called_functions,
            'inherited_classes': self.inherited_classes,
        }


@dataclass
class FileInfo:
    """
    Information about a single file.
    """
    file_path: str
    file_name: str
    
    # Content info
    line_count: int = 0
    char_count: int = 0
    
    # Structure
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)  # Public symbols
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)  # Files this depends on
    dependents: List[str] = field(default_factory=list)    # Files that depend on this
    
    # Metrics
    complexity_score: float = 0.0
    coupling_score: float = 0.0  # How many files depend on this
    cohesion_score: float = 0.0  # How related are its functions
    
    # Categories
    file_type: str = "module"  # module, test, config, init, main
    
    # Issues
    circular_deps: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'line_count': self.line_count,
            'function_count': len(self.functions),
            'class_count': len(self.classes),
            'import_count': len(self.imports),
            'complexity': self.complexity_score,
            'coupling': self.coupling_score,
            'file_type': self.file_type,
            'dependencies': len(self.dependencies),
            'dependents': len(self.dependents),
            'has_circular_deps': bool(self.circular_deps),
        }


@dataclass
class DependencyGraph:
    """
    Complete dependency graph for a project.
    """
    # Files
    files: Dict[str, FileInfo] = field(default_factory=dict)
    
    # Relations
    relations: List[FileRelation] = field(default_factory=list)
    
    # Quick lookup
    file_relations: Dict[str, List[FileRelation]] = field(default_factory=dict)
    
    # Project metrics
    total_files: int = 0
    total_relations: int = 0
    average_coupling: float = 0.0
    circular_dependencies: List[Tuple[str, str]] = field(default_factory=list)
    
    # Entry points
    entry_points: List[str] = field(default_factory=list)
    
    # Critical files (high coupling)
    critical_files: List[str] = field(default_factory=list)
    
    def get_file(self, path: str) -> Optional[FileInfo]:
        """Get file info by path"""
        return self.files.get(path)
    
    def get_dependencies(self, file_path: str) -> List[str]:
        """Get all files that the given file depends on"""
        info = self.files.get(file_path)
        return info.dependencies if info else []
    
    def get_dependents(self, file_path: str) -> List[str]:
        """Get all files that depend on the given file"""
        info = self.files.get(file_path)
        return info.dependents if info else []
    
    def get_impact(self, file_path: str) -> List[str]:
        """
        Get all files that would be impacted by changes to the given file.
        This includes transitive dependents.
        """
        impacted = set()
        to_visit = [file_path]
        
        while to_visit:
            current = to_visit.pop()
            for dep in self.get_dependents(current):
                if dep not in impacted:
                    impacted.add(dep)
                    to_visit.append(dep)
        
        return list(impacted)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_files': self.total_files,
            'total_relations': self.total_relations,
            'average_coupling': self.average_coupling,
            'circular_dependency_count': len(self.circular_dependencies),
            'entry_points': self.entry_points,
            'critical_files': self.critical_files,
            'files': {k: v.to_dict() for k, v in self.files.items()},
        }


@dataclass
class ProjectContext:
    """
    Complete context for a project.
    """
    project_root: str
    project_name: str = ""
    
    # Graph
    dependency_graph: DependencyGraph = field(default_factory=DependencyGraph)
    
    # Patterns
    common_patterns: List[str] = field(default_factory=list)
    architecture_style: str = "unknown"  # monolith, microservice, library, etc.
    
    # Metrics
    total_lines: int = 0
    total_functions: int = 0
    total_classes: int = 0
    
    # Tech stack (detected)
    frameworks: List[str] = field(default_factory=list)
    libraries: List[str] = field(default_factory=list)
    
    # Structure
    main_modules: List[str] = field(default_factory=list)
    test_modules: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'project_root': self.project_root,
            'project_name': self.project_name,
            'total_lines': self.total_lines,
            'total_functions': self.total_functions,
            'total_classes': self.total_classes,
            'architecture_style': self.architecture_style,
            'frameworks': self.frameworks,
            'main_modules': self.main_modules,
            'test_modules': self.test_modules,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-FILE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class CrossFileAnalyzer:
    """
    Analyze cross-file relationships and dependencies.
    
    Provides:
    - Complete dependency graph
    - Impact analysis for modifications
    - Circular dependency detection
    - Architecture analysis
    
    Usage:
        analyzer = CrossFileAnalyzer()
        context = analyzer.analyze_project("/path/to/project")
        
        # Get impact of changing a file
        impact = context.dependency_graph.get_impact("core/ai/client.py")
        print(f"Changing client.py would impact: {impact}")
    """
    
    # Framework detection patterns
    FRAMEWORK_PATTERNS = {
        'flask': ['flask', 'Flask'],
        'django': ['django', 'Django'],
        'fastapi': ['fastapi', 'FastAPI'],
        'pytest': ['pytest', 'unittest'],
        'sqlalchemy': ['sqlalchemy', 'SQLAlchemy'],
        'pandas': ['pandas', 'pd'],
        'numpy': ['numpy', 'np'],
        'requests': ['requests'],
        'asyncio': ['asyncio', 'aiohttp'],
        'tensorflow': ['tensorflow', 'tf'],
        'pytorch': ['torch', 'pytorch'],
    }
    
    def __init__(self, kimi_client=None):
        """Initialize cross-file analyzer"""
        self._kimi = kimi_client
        
        # Statistics
        self._stats = {
            'projects_analyzed': 0,
            'files_processed': 0,
            'relations_found': 0,
            'circular_deps_found': 0,
        }
        
        logger.info("CrossFileAnalyzer initialized")
    
    def analyze_project(
        self,
        project_root: str,
        exclude_dirs: List[str] = None,
        file_patterns: List[str] = None,
    ) -> ProjectContext:
        """
        Analyze entire project.
        
        Args:
            project_root: Root directory of project
            exclude_dirs: Directories to exclude
            file_patterns: File patterns to include (default: *.py)
            
        Returns:
            ProjectContext with complete analysis
        """
        start_time = time.time()
        
        project_root = os.path.abspath(project_root)
        exclude_dirs = exclude_dirs or ['__pycache__', '.git', 'node_modules', 'venv', '.venv']
        file_patterns = file_patterns or ['*.py']
        
        # Create context
        context = ProjectContext(
            project_root=project_root,
            project_name=os.path.basename(project_root),
        )
        
        # Find all files
        python_files = self._find_files(project_root, exclude_dirs, file_patterns)
        
        logger.info(f"Found {len(python_files)} files to analyze")
        
        # Analyze each file
        for file_path in python_files:
            file_info = self._analyze_file(file_path, project_root)
            if file_info:
                context.dependency_graph.files[file_path] = file_info
                context.total_lines += file_info.line_count
                context.total_functions += len(file_info.functions)
                context.total_classes += len(file_info.classes)
                self._stats['files_processed'] += 1
        
        # Build relations
        self._build_relations(context)
        
        # Detect circular dependencies
        self._detect_circular_deps(context)
        
        # Detect frameworks
        self._detect_frameworks(context)
        
        # Identify entry points
        self._identify_entry_points(context)
        
        # Calculate metrics
        self._calculate_metrics(context)
        
        # Finalize graph
        context.dependency_graph.total_files = len(context.dependency_graph.files)
        context.dependency_graph.total_relations = len(context.dependency_graph.relations)
        
        self._stats['projects_analyzed'] += 1
        
        logger.info(f"Project analysis completed in {(time.time() - start_time):.2f}s")
        
        return context
    
    def _find_files(
        self,
        root: str,
        exclude: List[str],
        patterns: List[str]
    ) -> List[str]:
        """Find all matching files"""
        files = []
        
        for dirpath, dirnames, filenames in os.walk(root):
            # Filter excluded directories
            dirnames[:] = [d for d in dirnames if d not in exclude and not d.startswith('.')]
            
            # Find matching files
            for pattern in patterns:
                for filename in filenames:
                    if self._matches_pattern(filename, pattern):
                        files.append(os.path.join(dirpath, filename))
        
        return files
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches pattern"""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def _analyze_file(self, file_path: str, project_root: str) -> Optional[FileInfo]:
        """Analyze a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return None
        
        # Create file info
        rel_path = os.path.relpath(file_path, project_root)
        info = FileInfo(
            file_path=rel_path,
            file_name=os.path.basename(file_path),
            line_count=len(content.split('\n')),
            char_count=len(content),
        )
        
        # Determine file type
        info.file_type = self._determine_file_type(file_path)
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            info.issues.append(f"Syntax error: {e}")
            return info
        
        # Extract structure
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                info.functions.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                info.functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                info.classes.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                info.imports.append(module)
        
        # Calculate complexity
        info.complexity_score = self._calculate_file_complexity(tree, content)
        
        return info
    
    def _determine_file_type(self, file_path: str) -> str:
        """Determine the type of file"""
        name = os.path.basename(file_path).lower()
        
        if name.startswith('test_') or name.endswith('_test.py'):
            return 'test'
        elif name == '__init__.py':
            return 'init'
        elif name in ('setup.py', 'conftest.py', 'config.py'):
            return 'config'
        elif name == 'main.py':
            return 'main'
        elif 'cli' in name:
            return 'cli'
        else:
            return 'module'
    
    def _calculate_file_complexity(self, tree: ast.AST, content: str) -> float:
        """Calculate complexity score for file"""
        score = 0.0
        
        # Count nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While)):
                score += 1
            elif isinstance(node, ast.FunctionDef):
                score += 2
            elif isinstance(node, ast.ClassDef):
                score += 3
            elif isinstance(node, ast.Try):
                score += 1.5
        
        # Normalize by file size
        lines = len(content.split('\n'))
        if lines > 0:
            score = score / (lines ** 0.5)
        
        return round(score, 2)
    
    def _build_relations(self, context: ProjectContext) -> None:
        """Build file relationships from imports"""
        graph = context.dependency_graph
        files = graph.files
        
        # Create file lookup
        file_modules = {}
        for path, info in files.items():
            # Convert path to module name
            module = path.replace('/', '.').replace('\\', '.').replace('.py', '')
            file_modules[module] = path
            
            # Also add parent modules
            parts = module.split('.')
            for i in range(len(parts)):
                parent = '.'.join(parts[:i+1])
                if parent not in file_modules:
                    file_modules[parent] = path
        
        # Build relations from imports
        for file_path, info in files.items():
            for imp in info.imports:
                # Check if import is in project
                if imp in file_modules:
                    target_path = file_modules[imp]
                    if target_path != file_path:  # Don't self-reference
                        # Create relation
                        relation = FileRelation(
                            source_file=file_path,
                            target_file=target_path,
                            relation_type=RelationType.IMPORTS,
                            strength=DependencyStrength.MODERATE,
                        )
                        
                        graph.relations.append(relation)
                        
                        # Update file dependencies
                        if target_path not in info.dependencies:
                            info.dependencies.append(target_path)
                        
                        target_info = files.get(target_path)
                        if target_info and file_path not in target_info.dependents:
                            target_info.dependents.append(file_path)
                        
                        self._stats['relations_found'] += 1
    
    def _detect_circular_deps(self, context: ProjectContext) -> None:
        """Detect circular dependencies"""
        graph = context.dependency_graph
        
        def find_cycle(start: str, current: str, visited: Set, path: List) -> Optional[List]:
            if current in visited:
                if current == start and len(path) > 1:
                    return path
                return None
            
            visited.add(current)
            path.append(current)
            
            info = graph.files.get(current)
            if info:
                for dep in info.dependencies:
                    cycle = find_cycle(start, dep, visited.copy(), path.copy())
                    if cycle:
                        return cycle
            
            return None
        
        # Check each file
        for file_path in graph.files:
            cycle = find_cycle(file_path, file_path, set(), [])
            if cycle:
                # Record cycle
                cycle_tuple = tuple(cycle)
                if cycle_tuple not in graph.circular_dependencies:
                    graph.circular_dependencies.append(cycle_tuple)
                    self._stats['circular_deps_found'] += 1
                
                # Mark files as having circular deps
                for f in cycle:
                    info = graph.files.get(f)
                    if info and f not in info.circular_deps:
                        info.circular_deps.append(f)
    
    def _detect_frameworks(self, context: ProjectContext) -> None:
        """Detect frameworks and libraries used"""
        all_imports = set()
        
        for info in context.dependency_graph.files.values():
            all_imports.update(info.imports)
        
        detected = []
        for framework, patterns in self.FRAMEWORK_PATTERNS.items():
            for pattern in patterns:
                if any(pattern.lower() in imp.lower() for imp in all_imports):
                    detected.append(framework)
                    break
        
        context.frameworks = list(set(detected))
    
    def _identify_entry_points(self, context: ProjectContext) -> None:
        """Identify project entry points"""
        graph = context.dependency_graph
        
        for file_path, info in graph.files.items():
            # Main files
            if info.file_type == 'main':
                graph.entry_points.append(file_path)
            
            # Files with no dependencies but many dependents
            if len(info.dependencies) == 0 and len(info.dependents) > 3:
                if file_path not in graph.entry_points:
                    graph.entry_points.append(file_path)
    
    def _calculate_metrics(self, context: ProjectContext) -> None:
        """Calculate project metrics"""
        graph = context.dependency_graph
        
        # Calculate coupling
        total_coupling = 0
        for info in graph.files.values():
            info.coupling_score = len(info.dependents)
            total_coupling += info.coupling_score
            
            # Identify critical files
            if info.coupling_score > 5:
                graph.critical_files.append(info.file_path)
        
        if graph.files:
            graph.average_coupling = total_coupling / len(graph.files)
        
        # Determine architecture style
        if len(graph.entry_points) == 1:
            context.architecture_style = "monolith"
        elif len(graph.entry_points) > 3:
            context.architecture_style = "microservice"
        else:
            context.architecture_style = "library"
    
    def get_modification_impact(
        self,
        context: ProjectContext,
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Analyze impact of modifying a file.
        
        Args:
            context: Project context
            file_path: File being modified
            
        Returns:
            Impact analysis result
        """
        graph = context.dependency_graph
        
        # Get direct dependents
        direct = graph.get_dependents(file_path)
        
        # Get transitive impact
        all_impact = graph.get_impact(file_path)
        
        # Get file info
        info = graph.get_file(file_path)
        
        return {
            'file': file_path,
            'direct_impact_count': len(direct),
            'total_impact_count': len(all_impact),
            'direct_impact': direct,
            'all_impact': all_impact,
            'coupling_score': info.coupling_score if info else 0,
            'is_critical': file_path in graph.critical_files,
            'has_circular_deps': bool(info.circular_deps) if info else False,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_analyzer: Optional[CrossFileAnalyzer] = None


def get_cross_file_analyzer(kimi_client=None) -> CrossFileAnalyzer:
    """Get or create global cross-file analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = CrossFileAnalyzer(kimi_client=kimi_client)
    elif kimi_client:
        _analyzer._kimi = kimi_client
    return _analyzer


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Cross-File Analyzer Test")
    print("="*60)
    
    # Use this project as test
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    print(f"\nAnalyzing project: {project_root}")
    
    analyzer = CrossFileAnalyzer()
    context = analyzer.analyze_project(
        project_root,
        exclude_dirs=['__pycache__', '.git', 'node_modules', 'venv', '.venv', 'skills']
    )
    
    print(f"\nProject Analysis:")
    print(f"  Total files: {context.dependency_graph.total_files}")
    print(f"  Total relations: {context.dependency_graph.total_relations}")
    print(f"  Total lines: {context.total_lines}")
    print(f"  Total functions: {context.total_functions}")
    print(f"  Total classes: {context.total_classes}")
    
    print(f"\nFrameworks detected: {context.frameworks}")
    print(f"Architecture style: {context.architecture_style}")
    
    print(f"\nEntry points: {context.dependency_graph.entry_points}")
    print(f"Critical files: {context.dependency_graph.critical_files[:5]}")
    
    if context.dependency_graph.circular_dependencies:
        print(f"\n⚠️  Circular dependencies found: {len(context.dependency_graph.circular_dependencies)}")
        for cycle in context.dependency_graph.circular_dependencies[:3]:
            print(f"   {' -> '.join(cycle)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

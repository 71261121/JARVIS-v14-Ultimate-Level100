#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Universal Code Analyzer
==============================================

Phase 5: Universal Code Understanding (Level 85-100+)

Multi-language code analysis with semantic understanding.

Author: JARVIS AI Project
Version: 5.0.0
Target Level: 85-100+
"""

import time
import logging
import re
import ast
import hashlib
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported languages"""
    PYTHON = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    JAVA = auto()
    CPP = auto()
    C = auto()
    GO = auto()
    RUST = auto()
    RUBY = auto()
    PHP = auto()
    KOTLIN = auto()
    SWIFT = auto()
    UNKNOWN = auto()


class CodeElement(Enum):
    """Types of code elements"""
    FUNCTION = auto()
    CLASS = auto()
    METHOD = auto()
    VARIABLE = auto()
    CONSTANT = auto()
    IMPORT = auto()
    COMMENT = auto()
    DECORATOR = auto()
    INTERFACE = auto()
    TRAIT = auto()
    MODULE = auto()
    NAMESPACE = auto()


@dataclass
class CodeSymbol:
    """Represents a code symbol"""
    name: str
    element_type: CodeElement
    line_start: int
    line_end: int
    signature: str = ""
    docstring: str = ""
    parameters: List[str] = field(default_factory=list)
    return_type: str = ""
    visibility: str = "public"
    complexity: int = 1
    dependencies: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.element_type.name,
            'lines': f"{self.line_start}-{self.line_end}",
            'complexity': self.complexity,
        }


@dataclass
class FileAnalysis:
    """Analysis result for a file"""
    file_path: str
    language: Language
    lines_of_code: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    symbols: List[CodeSymbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    
    # Metrics
    cyclomatic_complexity: int = 1
    maintainability_index: float = 100.0
    cognitive_complexity: int = 0
    
    # Issues
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def function_count(self) -> int:
        return sum(1 for s in self.symbols if s.element_type == CodeElement.FUNCTION)
    
    @property
    def class_count(self) -> int:
        return sum(1 for s in self.symbols if s.element_type == CodeElement.CLASS)


class UniversalCodeAnalyzer:
    """
    Universal multi-language code analyzer.
    
    Provides deep understanding of code across languages.
    """
    
    LANGUAGE_PATTERNS = {
        Language.PYTHON: {
            'extensions': ['.py', '.pyw', '.pyi'],
            'function_pattern': r'def\s+(\w+)\s*\(([^)]*)\)',
            'class_pattern': r'class\s+(\w+)(?:\(([^)]*)\))?:',
            'import_pattern': r'^(?:from\s+(\S+)\s+)?import\s+(.+)$',
            'comment_pattern': r'#.*$',
        },
        Language.JAVASCRIPT: {
            'extensions': ['.js', '.mjs', '.cjs'],
            'function_pattern': r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
            'class_pattern': r'class\s+(\w+)(?:\s+extends\s+(\w+))?',
            'import_pattern': r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]',
            'comment_pattern': r'(?://.*$|/\*[\s\S]*?\*/)',
        },
        Language.TYPESCRIPT: {
            'extensions': ['.ts', '.tsx'],
            'function_pattern': r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*(?::\s*\w+)?\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
            'class_pattern': r'class\s+(\w+)(?:\s+(?:extends|implements)\s+(\w+))?',
            'import_pattern': r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]',
            'comment_pattern': r'(?://.*$|/\*[\s\S]*?\*/)',
        },
        Language.JAVA: {
            'extensions': ['.java'],
            'function_pattern': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(([^)]*)\)',
            'class_pattern': r'class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+(.+))?',
            'import_pattern': r'import\s+([\w.]+);',
            'comment_pattern': r'(?://.*$|/\*[\s\S]*?\*/)',
        },
        Language.CPP: {
            'extensions': ['.cpp', '.cc', '.cxx', '.hpp', '.h'],
            'function_pattern': r'(?:\w+\s+)+(\w+)\s*\(([^)]*)\)\s*(?:const)?\s*\{',
            'class_pattern': r'class\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+(\w+))?',
            'import_pattern': r'#include\s*[<"]([^>"]+)[>"]',
            'comment_pattern': r'(?://.*$|/\*[\s\S]*?\*/)',
        },
        Language.GO: {
            'extensions': ['.go'],
            'function_pattern': r'func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(([^)]*)\)',
            'class_pattern': r'type\s+(\w+)\s+struct',
            'import_pattern': r'import\s+(?:\([\s\S]*?\)|"([^"]+)")',
            'comment_pattern': r'//.*$',
        },
        Language.RUST: {
            'extensions': ['.rs'],
            'function_pattern': r'fn\s+(\w+)\s*(?:<[^>]+>)?\s*\(([^)]*)\)',
            'class_pattern': r'(?:pub\s+)?struct\s+(\w+)',
            'import_pattern': r'use\s+([\w:]+)',
            'comment_pattern': r'(?://.*$|/\*[\s\S]*?\*/)',
        },
    }
    
    def __init__(self):
        """Initialize analyzer."""
        self._cache: Dict[str, FileAnalysis] = {}
        self._stats = {
            'files_analyzed': 0,
            'total_symbols': 0,
            'cache_hits': 0,
        }
    
    def detect_language(self, file_path: str, content: str = None) -> Language:
        """Detect language from file extension or content"""
        ext = Path(file_path).suffix.lower()
        
        for lang, patterns in self.LANGUAGE_PATTERNS.items():
            if ext in patterns['extensions']:
                return lang
        
        # Try content-based detection
        if content:
            if 'def ' in content and 'import ' in content:
                return Language.PYTHON
            if 'function ' in content and 'const ' in content:
                return Language.JAVASCRIPT
            if 'package ' in content and 'public class' in content:
                return Language.JAVA
            if '#include' in content:
                return Language.CPP
            if 'func ' in content and 'package ' in content:
                return Language.GO
            if 'fn ' in content and 'let ' in content:
                return Language.RUST
        
        return Language.UNKNOWN
    
    def analyze_file(
        self,
        file_path: str,
        content: str = None,
        use_cache: bool = True,
    ) -> FileAnalysis:
        """Analyze a code file"""
        # Check cache
        cache_key = hashlib.md5(file_path.encode()).hexdigest()
        if use_cache and cache_key in self._cache:
            self._stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        # Read content if needed
        if content is None:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                return FileAnalysis(file_path=file_path, language=Language.UNKNOWN)
        
        # Detect language
        language = self.detect_language(file_path, content)
        
        # Create analysis
        analysis = FileAnalysis(
            file_path=file_path,
            language=language,
        )
        
        # Count lines
        lines = content.split('\n')
        analysis.lines_of_code = len(lines)
        
        # Get patterns
        patterns = self.LANGUAGE_PATTERNS.get(language, {})
        
        # Extract symbols
        if language == Language.PYTHON:
            self._analyze_python(content, analysis)
        else:
            self._analyze_generic(content, analysis, patterns)
        
        # Calculate metrics
        self._calculate_metrics(analysis)
        
        # Cache
        self._cache[cache_key] = analysis
        self._stats['files_analyzed'] += 1
        self._stats['total_symbols'] += len(analysis.symbols)
        
        return analysis
    
    def _analyze_python(self, content: str, analysis: FileAnalysis):
        """Deep Python analysis using AST"""
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    symbol = CodeSymbol(
                        name=node.name,
                        element_type=CodeElement.METHOD if isinstance(node, ast.AsyncFunctionDef) else CodeElement.FUNCTION,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        parameters=[arg.arg for arg in node.args.args],
                        docstring=ast.get_docstring(node) or "",
                    )
                    analysis.symbols.append(symbol)
                    
                elif isinstance(node, ast.ClassDef):
                    symbol = CodeSymbol(
                        name=node.name,
                        element_type=CodeElement.CLASS,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                    )
                    analysis.symbols.append(symbol)
                    
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        analysis.imports.append(alias.name)
                        analysis.dependencies.add(alias.name.split('.')[0])
        
        except SyntaxError as e:
            analysis.issues.append({
                'type': 'syntax_error',
                'message': str(e),
                'line': e.lineno,
            })
    
    def _analyze_generic(
        self,
        content: str,
        analysis: FileAnalysis,
        patterns: Dict[str, str],
    ):
        """Generic pattern-based analysis"""
        lines = content.split('\n')
        
        # Extract functions
        func_pattern = patterns.get('function_pattern', '')
        if func_pattern:
            for match in re.finditer(func_pattern, content, re.MULTILINE):
                name = match.group(1) or match.group(2)
                if name:
                    line_num = content[:match.start()].count('\n') + 1
                    symbol = CodeSymbol(
                        name=name,
                        element_type=CodeElement.FUNCTION,
                        line_start=line_num,
                        line_end=line_num,
                    )
                    analysis.symbols.append(symbol)
        
        # Extract classes
        class_pattern = patterns.get('class_pattern', '')
        if class_pattern:
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                name = match.group(1)
                if name:
                    line_num = content[:match.start()].count('\n') + 1
                    symbol = CodeSymbol(
                        name=name,
                        element_type=CodeElement.CLASS,
                        line_start=line_num,
                        line_end=line_num,
                    )
                    analysis.symbols.append(symbol)
        
        # Extract imports
        import_pattern = patterns.get('import_pattern', '')
        if import_pattern:
            for match in re.finditer(import_pattern, content, re.MULTILINE):
                imp = match.group(1)
                if imp:
                    analysis.imports.append(imp)
                    analysis.dependencies.add(imp.split('.')[0].split('/')[0])
    
    def _calculate_metrics(self, analysis: FileAnalysis):
        """Calculate code metrics"""
        # Cyclomatic complexity estimate
        analysis.cyclomatic_complexity = sum(
            s.complexity for s in analysis.symbols
        ) + 1
        
        # Maintainability index (simplified)
        if analysis.lines_of_code > 0:
            vol = analysis.lines_of_code * len(analysis.symbols)
            analysis.maintainability_index = max(0, 100 - vol / 100)
    
    def analyze_project(
        self,
        project_path: str,
        recursive: bool = True,
    ) -> Dict[str, FileAnalysis]:
        """Analyze entire project"""
        results = {}
        path = Path(project_path)
        
        extensions = []
        for patterns in self.LANGUAGE_PATTERNS.values():
            extensions.extend(patterns['extensions'])
        
        if recursive:
            files = path.rglob('*')
        else:
            files = path.glob('*')
        
        for file_path in files:
            if file_path.is_file() and file_path.suffix in extensions:
                analysis = self.analyze_file(str(file_path))
                results[str(file_path)] = analysis
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            **self._stats,
            'cache_size': len(self._cache),
            'languages_supported': len(self.LANGUAGE_PATTERNS),
        }


# Global instance
_analyzer: Optional[UniversalCodeAnalyzer] = None

def get_universal_analyzer() -> UniversalCodeAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = UniversalCodeAnalyzer()
    return _analyzer


if __name__ == "__main__":
    print("Universal Code Analyzer initialized")

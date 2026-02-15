#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 - Termux-Compatible Code Embedder
=============================================

CONSTRAINT: Pure Python implementation
- NO torch, tensorflow, sklearn, numpy, scipy
- Uses AST analysis + feature hashing
- Bounded memory usage

Memory Budget: 30 MB maximum
"""

import hashlib
import ast
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


def _verify_imports():
    """Verify no forbidden heavy dependencies are loaded."""
    forbidden = ['torch', 'tensorflow', 'sklearn', 'numpy', 'scipy', 'chromadb']
    for pkg in forbidden:
        try:
            __import__(pkg)
            # Package exists - warn but don't halt
            import warnings
            warnings.warn(f'Heavy package {pkg} is installed - may impact memory')
        except ImportError:
            pass  # Correct - package not available


_verify_imports()


@dataclass
class CodeFingerprint:
    """Semantic fingerprint for code - pure Python."""
    
    function_count: int
    class_count: int
    import_count: int
    max_nesting: int
    total_lines: int
    structure_hash: str
    name_hash: str
    pattern_hash: str
    embedding: Tuple[float, ...]
    
    def to_dict(self) -> dict:
        return {
            'function_count': self.function_count,
            'class_count': self.class_count,
            'import_count': self.import_count,
            'max_nesting': self.max_nesting,
            'total_lines': self.total_lines,
            'structure_hash': self.structure_hash,
        }


class TermuxCodeEmbedder:
    """
    Pure Python code embedder for Termux.
    
    CONSTRAINTS:
    - NO external ML dependencies
    - Uses AST analysis + feature hashing
    - Bounded cache for memory safety
    """
    
    __slots__ = ['_cache', '_cache_max_size', '_pattern_weights']
    
    def __init__(self, cache_max_size: int = 100):
        """
        Initialize with bounded cache.
        
        Args:
            cache_max_size: Maximum cached fingerprints (memory budget)
        
        Raises:
            AssertionError: If cache size exceeds memory budget
        """
        # RUNTIME ASSERTION - memory budget
        assert cache_max_size <= 1000, f'HALT: Cache size {cache_max_size} exceeds memory budget'
        
        self._cache: Dict[str, CodeFingerprint] = {}
        self._cache_max_size = cache_max_size
        self._pattern_weights = {
            'loop': 0.15,
            'condition': 0.12,
            'function_def': 0.18,
            'class_def': 0.20,
            'import': 0.08,
            'try_except': 0.15,
            'async': 0.12,
        }
    
    def embed(self, code: str) -> CodeFingerprint:
        """
        Generate semantic fingerprint for code.
        
        Args:
            code: Source code string
            
        Returns:
            CodeFingerprint with structural and semantic features
            
        Raises:
            ValueError: If code is invalid or unparseable
        """
        # Input validation
        if not code or not isinstance(code, str):
            raise ValueError('HALT: Invalid code input - must be non-empty string')
        
        if len(code) > 100000:  # 100KB limit
            raise ValueError('HALT: Code exceeds size limit (100KB)')
        
        # Check cache
        cache_key = hashlib.sha256(code.encode()).hexdigest()[:16]
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f'HALT: Code syntax error: {e}')
        
        # Extract features
        features = self._extract_features(tree, code)
        
        # Generate hashes
        structure_hash = self._hash_structure(tree)
        name_hash = self._hash_names(tree)
        pattern_hash = self._hash_patterns(tree)
        
        # Generate embedding
        embedding = self._generate_embedding(features)
        
        # Create fingerprint
        fingerprint = CodeFingerprint(
            function_count=features['function_count'],
            class_count=features['class_count'],
            import_count=features['import_count'],
            max_nesting=features['max_nesting'],
            total_lines=len(code.split('\n')),
            structure_hash=structure_hash,
            name_hash=name_hash,
            pattern_hash=pattern_hash,
            embedding=embedding,
        )
        
        # Cache with eviction
        if len(self._cache) >= self._cache_max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[cache_key] = fingerprint
        return fingerprint
    
    def _extract_features(self, tree: ast.AST, code: str) -> Dict[str, int]:
        """Extract structural features from AST."""
        features = {
            'function_count': 0,
            'class_count': 0,
            'import_count': 0,
            'max_nesting': 0,
            'loop_count': 0,
            'condition_count': 0,
            'try_count': 0,
            'async_count': 0,
        }
        
        def visit(node, depth=0):
            features['max_nesting'] = max(features['max_nesting'], depth)
            
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                features['function_count'] += 1
            elif isinstance(node, ast.ClassDef):
                features['class_count'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                features['import_count'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                features['loop_count'] += 1
            elif isinstance(node, ast.If):
                features['condition_count'] += 1
            elif isinstance(node, ast.Try):
                features['try_count'] += 1
            elif isinstance(node, ast.AsyncFunctionDef):
                features['async_count'] += 1
            
            for child in ast.iter_child_nodes(node):
                visit(child, depth + 1)
        
        visit(tree)
        return features
    
    def _hash_structure(self, tree: ast.AST) -> str:
        """Generate hash of AST structure."""
        structure = []
        
        def visit(node):
            structure.append(type(node).__name__)
            for child in ast.iter_child_nodes(node):
                visit(child)
        
        visit(tree)
        return hashlib.md5('|'.join(structure).encode()).hexdigest()[:16]
    
    def _hash_names(self, tree: ast.AST) -> str:
        """Generate hash of names used in code."""
        names = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.FunctionDef):
                names.append(node.name)
            elif isinstance(node, ast.ClassDef):
                names.append(node.name)
        
        unique_names = sorted(set(names))
        return hashlib.md5('|'.join(unique_names).encode()).hexdigest()[:16]
    
    def _hash_patterns(self, tree: ast.AST) -> str:
        """Generate hash of code patterns."""
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                patterns.append('for_loop')
            elif isinstance(node, ast.While):
                patterns.append('while_loop')
            elif isinstance(node, ast.If):
                patterns.append('condition')
            elif isinstance(node, ast.Try):
                patterns.append('exception_handling')
            elif isinstance(node, ast.With):
                patterns.append('context_manager')
            elif isinstance(node, ast.AsyncFunctionDef):
                patterns.append('async_function')
        
        return hashlib.md5('|'.join(sorted(patterns)).encode()).hexdigest()[:16]
    
    def _generate_embedding(self, features: Dict[str, int]) -> Tuple[float, ...]:
        """Generate fixed-size embedding (16 floats)."""
        normalized = [
            min(features['function_count'] / 20, 1.0),
            min(features['class_count'] / 10, 1.0),
            min(features['import_count'] / 30, 1.0),
            min(features['max_nesting'] / 10, 1.0),
            min(features['loop_count'] / 10, 1.0),
            min(features['condition_count'] / 20, 1.0),
            min(features['try_count'] / 5, 1.0),
            min(features['async_count'] / 5, 1.0),
            self._pattern_weights['loop'] * min(features['loop_count'], 1),
            self._pattern_weights['condition'] * min(features['condition_count'], 1),
            self._pattern_weights['function_def'] * min(features['function_count'], 1),
            self._pattern_weights['class_def'] * min(features['class_count'], 1),
            self._pattern_weights['import'] * min(features['import_count'], 1),
            self._pattern_weights['try_except'] * min(features['try_count'], 1),
            self._pattern_weights['async'] * min(features['async_count'], 1),
            min((features['function_count'] + features['class_count'] * 2 + features['max_nesting']) / 30, 1.0),
        ]
        
        return tuple(normalized)
    
    def similarity(self, fp1: CodeFingerprint, fp2: CodeFingerprint) -> float:
        """
        Calculate similarity between fingerprints.
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Structural similarity
        structural_sim = 1.0 if fp1.structure_hash == fp2.structure_hash else 0.0
        
        # Name similarity
        name_sim = 1.0 if fp1.name_hash == fp2.name_hash else 0.0
        
        # Pattern similarity
        pattern_sim = 1.0 if fp1.pattern_hash == fp2.pattern_hash else 0.0
        
        # Embedding similarity (cosine)
        dot = sum(a * b for a, b in zip(fp1.embedding, fp2.embedding))
        norm1 = sum(a * a for a in fp1.embedding) ** 0.5
        norm2 = sum(b * b for b in fp2.embedding) ** 0.5
        
        embedding_sim = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
        # Weighted combination
        return (
            structural_sim * 0.3 +
            name_sim * 0.2 +
            pattern_sim * 0.2 +
            embedding_sim * 0.3
        )
    
    def clear_cache(self):
        """Clear cache to free memory."""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Return current cache size."""
        return len(self._cache)


# SELF-TEST
if __name__ == '__main__':
    import sys
    
    print('=== TERMUX CODE EMBEDDER SELF-TEST ===')
    print()
    
    code1 = '''
def calculate_sum(items):
    total = 0
    for item in items:
        total += item.price
    return total
'''
    
    code2 = '''
def compute_total(products):
    result = 0
    for product in products:
        result += product.cost
    return result
'''
    
    code3 = '''
class UserManager:
    def __init__(self):
        self.users = []
    
    def add_user(self, user):
        self.users.append(user)
'''
    
    try:
        embedder = TermuxCodeEmbedder(cache_max_size=10)
        
        fp1 = embedder.embed(code1)
        fp2 = embedder.embed(code2)
        fp3 = embedder.embed(code3)
        
        sim_12 = embedder.similarity(fp1, fp2)
        sim_13 = embedder.similarity(fp1, fp3)
        
        print(f'Code 1 - functions: {fp1.function_count}, lines: {fp1.total_lines}')
        print(f'Code 3 - classes: {fp3.class_count}, lines: {fp3.total_lines}')
        print()
        print(f'Similarity (code1 vs code2): {sim_12:.2f} [should be HIGH]')
        print(f'Similarity (code1 vs code3): {sim_13:.2f} [should be LOW]')
        print()
        
        # Assertions
        assert fp1.function_count == 1, f'Expected 1 function, got {fp1.function_count}'
        assert fp3.class_count == 1, f'Expected 1 class, got {fp3.class_count}'
        assert sim_12 > sim_13, f'Similar code should have higher similarity'
        
        print('=== ALL TESTS PASSED ===')
        
    except AssertionError as e:
        print(f'ASSERTION FAILED: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

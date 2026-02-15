#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Semantic Code Understanding Package
==========================================================

Phase 2: Semantic Code Understanding (Level 40-50)

This package provides deep semantic understanding of code:
- Code embeddings for similarity detection
- Intent analysis for understanding code purpose
- Cross-file dependency analysis
- Pattern recognition and code clustering

Modules:
    - code_embedder: Generate semantic embeddings for code
    - similarity_engine: Find similar code patterns
    - intent_analyzer: Understand code intent and purpose
    - cross_file_analyzer: Project-level code understanding

Author: JARVIS AI Project
Version: 2.0.0
Target Level: 40-50
"""

from .code_embedder import (
    SemanticCodeEmbedder,
    CodeEmbedding,
    EmbeddingCache,
    get_embedder,
)

from .similarity_engine import (
    SimilarityEngine,
    SimilarCodeMatch,
    CodeCluster,
    get_similarity_engine,
)

from .intent_analyzer import (
    IntentAnalyzer,
    CodeIntent,
    IntentType,
    get_intent_analyzer,
)

from .cross_file_analyzer import (
    CrossFileAnalyzer,
    DependencyGraph,
    FileRelation,
    ProjectContext,
    get_cross_file_analyzer,
)

__all__ = [
    # Code Embedder
    'SemanticCodeEmbedder',
    'CodeEmbedding',
    'EmbeddingCache',
    'get_embedder',
    
    # Similarity Engine
    'SimilarityEngine',
    'SimilarCodeMatch',
    'CodeCluster',
    'get_similarity_engine',
    
    # Intent Analyzer
    'IntentAnalyzer',
    'CodeIntent',
    'IntentType',
    'get_intent_analyzer',
    
    # Cross-File Analyzer
    'CrossFileAnalyzer',
    'DependencyGraph',
    'FileRelation',
    'ProjectContext',
    'get_cross_file_analyzer',
]

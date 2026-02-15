#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Code Similarity Engine
=============================================

Phase 2: Find similar code patterns and clusters.

This module enables:
- Finding similar code across files/projects
- Code deduplication detection
- Pattern clustering
- Code search by example

Key Features:
- Vector-based similarity search
- Structural similarity comparison
- Semantic similarity via AI
- Code clustering algorithms

Author: JARVIS AI Project
Version: 2.0.0
Target Level: 40-50
"""

import logging
import threading
import time
import math
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimilarCodeMatch:
    """
    A match found by similarity search.
    """
    # The matching embedding
    embedding: Any  # CodeEmbedding
    
    # Similarity scores
    overall_similarity: float = 0.0
    structural_similarity: float = 0.0
    semantic_similarity: float = 0.0
    
    # Match details
    match_type: str = "similar"  # exact, near_duplicate, similar, related
    
    # Explanation
    matching_features: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code_hash': self.embedding.code_hash if self.embedding else None,
            'file_path': self.embedding.file_path if self.embedding else None,
            'function_name': self.embedding.function_name if self.embedding else None,
            'overall_similarity': self.overall_similarity,
            'structural_similarity': self.structural_similarity,
            'semantic_similarity': self.semantic_similarity,
            'match_type': self.match_type,
            'matching_features': self.matching_features,
            'differences': self.differences,
        }


@dataclass
class CodeCluster:
    """
    A cluster of similar code snippets.
    """
    cluster_id: str
    cluster_name: str = ""
    
    # Members
    members: List[Any] = field(default_factory=list)  # List of CodeEmbedding
    
    # Cluster features
    common_patterns: List[str] = field(default_factory=list)
    representative_embedding: Any = None  # Most central embedding
    
    # Statistics
    avg_similarity: float = 0.0
    size: int = 0
    
    # Categories
    cluster_type: str = "unknown"  # utility, pattern, duplicate, api_call, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cluster_id': self.cluster_id,
            'cluster_name': self.cluster_name,
            'size': self.size,
            'avg_similarity': self.avg_similarity,
            'cluster_type': self.cluster_type,
            'common_patterns': self.common_patterns,
            'members': [
                {'hash': e.code_hash, 'name': e.function_name}
                for e in self.members
            ] if self.members else [],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SIMILARITY CALCULATORS
# ═══════════════════════════════════════════════════════════════════════════════

class VectorSimilarity:
    """
    Calculate similarity between embedding vectors.
    
    Supports multiple similarity metrics:
    - Cosine similarity
    - Euclidean distance
    - Dot product
    """
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Range: -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)
        """
        if not vec1 or not vec2:
            return 0.0
        
        # Pad shorter vector
        max_len = max(len(vec1), len(vec2))
        v1 = vec1 + [0.0] * (max_len - len(vec1))
        v2 = vec2 + [0.0] * (max_len - len(vec2))
        
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = math.sqrt(sum(a ** 2 for a in v1))
        magnitude2 = math.sqrt(sum(b ** 2 for b in v2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    @staticmethod
    def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """Calculate Euclidean distance between vectors"""
        if not vec1 or not vec2:
            return float('inf')
        
        max_len = max(len(vec1), len(vec2))
        v1 = vec1 + [0.0] * (max_len - len(vec1))
        v2 = vec2 + [0.0] * (max_len - len(vec2))
        
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))
    
    @staticmethod
    def euclidean_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Convert Euclidean distance to similarity (0-1)"""
        distance = VectorSimilarity.euclidean_distance(vec1, vec2)
        # Convert to similarity: similarity = 1 / (1 + distance)
        return 1.0 / (1.0 + distance)


class StructuralSimilarity:
    """
    Calculate structural similarity between code snippets.
    
    Compares:
    - Function/class counts
    - Complexity metrics
    - Pattern usage
    - Import structure
    """
    
    @staticmethod
    def calculate(feat1: Any, feat2: Any) -> Tuple[float, List[str], List[str]]:
        """
        Calculate structural similarity.
        
        Returns:
            Tuple of (similarity, matching_features, differences)
        """
        if feat1 is None or feat2 is None:
            return 0.0, [], []
        
        score = 0.0
        matches = []
        diffs = []
        
        # Compare function/class counts
        if feat1.function_count == feat2.function_count:
            score += 0.1
            if feat1.function_count > 0:
                matches.append(f"Same function count: {feat1.function_count}")
        else:
            diffs.append(f"Different function count: {feat1.function_count} vs {feat2.function_count}")
        
        if feat1.class_count == feat2.class_count:
            score += 0.1
            if feat1.class_count > 0:
                matches.append(f"Same class count: {feat1.class_count}")
        else:
            diffs.append(f"Different class count: {feat1.class_count} vs {feat2.class_count}")
        
        # Compare patterns
        patterns1 = {
            'try_except': feat1.has_try_except,
            'context_manager': feat1.has_context_manager,
            'decorator': feat1.has_decorator,
            'comprehension': feat1.has_comprehension,
            'generator': feat1.has_generator,
            'lambda': feat1.has_lambda,
        }
        
        patterns2 = {
            'try_except': feat2.has_try_except,
            'context_manager': feat2.has_context_manager,
            'decorator': feat2.has_decorator,
            'comprehension': feat2.has_comprehension,
            'generator': feat2.has_generator,
            'lambda': feat2.has_lambda,
        }
        
        pattern_matches = 0
        pattern_total = len(patterns1)
        
        for pattern, has1 in patterns1.items():
            has2 = patterns2.get(pattern, False)
            if has1 and has2:
                pattern_matches += 1
                matches.append(f"Both use {pattern}")
            elif has1 != has2:
                diffs.append(f"Only one uses {pattern}")
        
        score += 0.3 * (pattern_matches / pattern_total)
        
        # Compare imports
        imports1 = set(feat1.imports or [])
        imports2 = set(feat2.imports or [])
        
        if imports1 and imports2:
            common_imports = imports1 & imports2
            if common_imports:
                import_sim = len(common_imports) / max(len(imports1), len(imports2))
                score += 0.2 * import_sim
                matches.append(f"Common imports: {list(common_imports)[:3]}")
        
        # Compare complexity
        depth_diff = abs(feat1.max_nesting_depth - feat2.max_nesting_depth)
        if depth_diff == 0:
            score += 0.1
            matches.append("Same nesting depth")
        elif depth_diff <= 1:
            score += 0.05
        else:
            diffs.append(f"Different complexity: depth {feat1.max_nesting_depth} vs {feat2.max_nesting_depth}")
        
        # Compare code size
        size_ratio = min(feat1.code_lines, feat2.code_lines) / max(feat1.code_lines, feat2.code_lines, 1)
        score += 0.1 * size_ratio
        
        return min(score, 1.0), matches, diffs


# ═══════════════════════════════════════════════════════════════════════════════
# SIMILARITY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SimilarityEngine:
    """
    Main engine for finding similar code.
    
    Features:
    - Vector similarity search
    - Structural comparison
    - Semantic comparison via AI
    - Code clustering
    - Duplicate detection
    
    Usage:
        engine = SimilarityEngine()
        
        # Find similar code
        matches = engine.find_similar(query_embedding, corpus)
        
        # Cluster code
        clusters = engine.cluster(corpus)
    """
    
    # Similarity thresholds
    EXACT_THRESHOLD = 0.99
    NEAR_DUPLICATE_THRESHOLD = 0.85
    SIMILAR_THRESHOLD = 0.70
    RELATED_THRESHOLD = 0.50
    
    def __init__(
        self,
        kimi_client=None,
        use_vector_similarity: bool = True,
        use_structural_similarity: bool = True,
    ):
        """
        Initialize similarity engine.
        
        Args:
            kimi_client: Kimi K2.5 client for semantic comparison
            use_vector_similarity: Use vector-based similarity
            use_structural_similarity: Use structural comparison
        """
        self._kimi = kimi_client
        self._use_vector = use_vector_similarity
        self._use_structural = use_structural_similarity
        
        # Index for fast search
        self._index: Dict[str, Any] = {}  # hash -> embedding
        self._vector_index: List[Tuple[List[float], str]] = []  # (vector, hash)
        
        # Statistics
        self._stats = {
            'total_searches': 0,
            'total_comparisons': 0,
            'exact_matches': 0,
            'similar_matches': 0,
            'clusters_created': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("SimilarityEngine initialized")
    
    def set_kimi_client(self, client):
        """Set Kimi client for semantic comparison"""
        self._kimi = client
    
    def index(self, embeddings: List[Any]) -> None:
        """
        Index embeddings for fast search.
        
        Args:
            embeddings: List of CodeEmbedding objects
        """
        with self._lock:
            for emb in embeddings:
                self._index[emb.code_hash] = emb
                if emb.embedding_vector:
                    self._vector_index.append((emb.embedding_vector, emb.code_hash))
        
        logger.info(f"Indexed {len(embeddings)} embeddings")
    
    def clear_index(self):
        """Clear the search index"""
        with self._lock:
            self._index.clear()
            self._vector_index.clear()
    
    def find_similar(
        self,
        query: Any,  # CodeEmbedding
        corpus: List[Any] = None,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> List[SimilarCodeMatch]:
        """
        Find similar code to query.
        
        Args:
            query: CodeEmbedding to search for
            corpus: List of CodeEmbedding to search in (uses index if None)
            top_k: Maximum results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of SimilarCodeMatch sorted by similarity
        """
        start_time = time.time()
        
        # Use indexed corpus if not provided
        search_corpus = corpus or list(self._index.values())
        
        if query.code_hash in search_corpus:
            search_corpus = [e for e in search_corpus if e.code_hash != query.code_hash]
        
        results = []
        
        for candidate in search_corpus:
            self._stats['total_comparisons'] += 1
            
            # Calculate similarities
            vector_sim = 0.0
            struct_sim = 0.0
            semantic_sim = 0.0
            matches = []
            diffs = []
            
            # Vector similarity
            if self._use_vector and query.embedding_vector and candidate.embedding_vector:
                vector_sim = VectorSimilarity.cosine_similarity(
                    query.embedding_vector,
                    candidate.embedding_vector
                )
            
            # Structural similarity
            if self._use_structural:
                struct_sim, matches, diffs = StructuralSimilarity.calculate(
                    query.structural_features,
                    candidate.structural_features
                )
            
            # Combine scores
            overall = (vector_sim * 0.5 + struct_sim * 0.5) if vector_sim > 0 else struct_sim
            
            if overall < min_similarity:
                continue
            
            # Determine match type
            match_type = self._determine_match_type(overall)
            
            # Create match
            match = SimilarCodeMatch(
                embedding=candidate,
                overall_similarity=overall,
                structural_similarity=struct_sim,
                semantic_similarity=semantic_sim,
                match_type=match_type,
                matching_features=matches,
                differences=diffs,
            )
            
            results.append(match)
        
        # Sort by similarity
        results.sort(key=lambda x: x.overall_similarity, reverse=True)
        
        # Update stats
        self._stats['total_searches'] += 1
        if results:
            if results[0].match_type == 'exact':
                self._stats['exact_matches'] += 1
            elif results[0].match_type in ('near_duplicate', 'similar'):
                self._stats['similar_matches'] += 1
        
        logger.debug(f"Found {len(results)} similar code in {(time.time() - start_time)*1000:.1f}ms")
        
        return results[:top_k]
    
    def find_duplicates(
        self,
        embeddings: List[Any],
        threshold: float = None,
    ) -> List[Tuple[Any, Any, float]]:
        """
        Find duplicate code pairs.
        
        Args:
            embeddings: List of CodeEmbedding
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of (embedding1, embedding2, similarity) tuples
        """
        threshold = threshold or self.NEAR_DUPLICATE_THRESHOLD
        duplicates = []
        
        for i, emb1 in enumerate(embeddings):
            for emb2 in embeddings[i+1:]:
                if not emb1.embedding_vector or not emb2.embedding_vector:
                    continue
                
                sim = VectorSimilarity.cosine_similarity(
                    emb1.embedding_vector,
                    emb2.embedding_vector
                )
                
                if sim >= threshold:
                    duplicates.append((emb1, emb2, sim))
        
        return duplicates
    
    def cluster(
        self,
        embeddings: List[Any],
        min_cluster_size: int = 2,
        similarity_threshold: float = 0.7,
    ) -> List[CodeCluster]:
        """
        Cluster code by similarity.
        
        Uses a simple greedy clustering algorithm:
        1. Start with first embedding as cluster center
        2. Add similar embeddings to cluster
        3. Start new cluster for dissimilar embeddings
        4. Repeat until all embeddings processed
        
        Args:
            embeddings: List of CodeEmbedding
            min_cluster_size: Minimum cluster size
            similarity_threshold: Threshold for cluster membership
            
        Returns:
            List of CodeCluster
        """
        if not embeddings:
            return []
        
        clusters: List[CodeCluster] = []
        assigned = set()
        
        for i, emb in enumerate(embeddings):
            if emb.code_hash in assigned:
                continue
            
            # Start new cluster
            cluster = CodeCluster(
                cluster_id=f"cluster_{len(clusters)}",
                members=[emb],
            )
            assigned.add(emb.code_hash)
            
            # Find similar embeddings
            for other in embeddings[i+1:]:
                if other.code_hash in assigned:
                    continue
                
                if emb.embedding_vector and other.embedding_vector:
                    sim = VectorSimilarity.cosine_similarity(
                        emb.embedding_vector,
                        other.embedding_vector
                    )
                    
                    if sim >= similarity_threshold:
                        cluster.members.append(other)
                        assigned.add(other.code_hash)
            
            # Only keep clusters with minimum size
            if len(cluster.members) >= min_cluster_size:
                cluster.size = len(cluster.members)
                cluster.representative_embedding = emb
                cluster.cluster_type = self._determine_cluster_type(cluster.members)
                cluster.common_patterns = self._find_common_patterns(cluster.members)
                clusters.append(cluster)
                self._stats['clusters_created'] += 1
        
        return clusters
    
    def _determine_match_type(self, similarity: float) -> str:
        """Determine match type from similarity score"""
        if similarity >= self.EXACT_THRESHOLD:
            return 'exact'
        elif similarity >= self.NEAR_DUPLICATE_THRESHOLD:
            return 'near_duplicate'
        elif similarity >= self.SIMILAR_THRESHOLD:
            return 'similar'
        elif similarity >= self.RELATED_THRESHOLD:
            return 'related'
        else:
            return 'weak'
    
    def _determine_cluster_type(self, members: List[Any]) -> str:
        """Determine the type of code cluster"""
        # Analyze member features
        has_api_calls = False
        has_utilities = False
        has_data_processing = False
        
        for member in members:
            if member.structural_features:
                api_calls = member.structural_features.api_calls or []
                if any('.' in call for call in api_calls):
                    has_api_calls = True
                
                func_names = member.structural_features.function_names or []
                if any(name in ['get', 'set', 'init', 'load', 'save', 'parse'] 
                       for name in ' '.join(func_names).lower().split('_')):
                    has_utilities = True
                
                if member.structural_features.has_comprehension:
                    has_data_processing = True
        
        if has_api_calls:
            return 'api_call'
        elif has_data_processing:
            return 'data_processing'
        elif has_utilities:
            return 'utility'
        else:
            return 'general'
    
    def _find_common_patterns(self, members: List[Any]) -> List[str]:
        """Find patterns common to all cluster members"""
        if not members:
            return []
        
        # Collect all patterns
        pattern_counts = defaultdict(int)
        
        for member in members:
            if member.structural_features:
                features = member.structural_features
                if features.has_try_except:
                    pattern_counts['error_handling'] += 1
                if features.has_context_manager:
                    pattern_counts['context_manager'] += 1
                if features.has_decorator:
                    pattern_counts['decorators'] += 1
                if features.has_comprehension:
                    pattern_counts['comprehensions'] += 1
                if features.has_generator:
                    pattern_counts['generators'] += 1
                if features.has_lambda:
                    pattern_counts['lambdas'] += 1
                if features.has_docstrings:
                    pattern_counts['documentation'] += 1
        
        # Return patterns present in at least half of members
        threshold = len(members) / 2
        return [p for p, c in pattern_counts.items() if c >= threshold]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats = self._stats.copy()
        stats['index_size'] = len(self._index)
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_engine: Optional[SimilarityEngine] = None


def get_similarity_engine(kimi_client=None) -> SimilarityEngine:
    """Get or create global similarity engine"""
    global _engine
    if _engine is None:
        _engine = SimilarityEngine(kimi_client=kimi_client)
    elif kimi_client:
        _engine.set_kimi_client(kimi_client)
    return _engine


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Similarity Engine Test")
    print("="*60)
    
    # Create mock embeddings
    from dataclasses import dataclass
    
    @dataclass
    class MockFeatures:
        function_count: int = 1
        class_count: int = 0
        max_nesting_depth: int = 2
        code_lines: int = 10
        has_try_except: bool = True
        has_context_manager: bool = False
        has_decorator: bool = False
        has_comprehension: bool = True
        has_generator: bool = False
        has_lambda: bool = False
        imports: List = None
        function_names: List = None
        
        def __post_init__(self):
            if self.imports is None:
                self.imports = []
            if self.function_names is None:
                self.function_names = []
    
    @dataclass
    class MockEmbedding:
        code_hash: str
        embedding_vector: List[float]
        structural_features: MockFeatures
        file_path: str = None
        function_name: str = None
    
    # Test vectors
    vec1 = [1.0, 0.0, 0.0, 0.5, 0.5]
    vec2 = [0.9, 0.1, 0.0, 0.5, 0.5]  # Similar to vec1
    vec3 = [0.0, 1.0, 0.0, 0.0, 1.0]  # Different
    
    # Create mock embeddings
    emb1 = MockEmbedding(
        code_hash="hash1",
        embedding_vector=vec1,
        structural_features=MockFeatures(),
    )
    
    emb2 = MockEmbedding(
        code_hash="hash2",
        embedding_vector=vec2,
        structural_features=MockFeatures(),
    )
    
    emb3 = MockEmbedding(
        code_hash="hash3",
        embedding_vector=vec3,
        structural_features=MockFeatures(function_count=2, has_comprehension=False),
    )
    
    # Test similarity
    engine = SimilarityEngine()
    engine.index([emb2, emb3])
    
    print("\nTest 1: Vector similarity...")
    sim = VectorSimilarity.cosine_similarity(vec1, vec2)
    print(f"  vec1 vs vec2: {sim:.3f}")
    
    sim = VectorSimilarity.cosine_similarity(vec1, vec3)
    print(f"  vec1 vs vec3: {sim:.3f}")
    
    print("\nTest 2: Find similar...")
    matches = engine.find_similar(emb1, top_k=5)
    for m in matches:
        print(f"  {m.embedding.code_hash}: {m.overall_similarity:.3f} ({m.match_type})")
    
    print("\nTest 3: Clustering...")
    clusters = engine.cluster([emb1, emb2, emb3])
    for c in clusters:
        print(f"  {c.cluster_id}: {c.size} members, type={c.cluster_type}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

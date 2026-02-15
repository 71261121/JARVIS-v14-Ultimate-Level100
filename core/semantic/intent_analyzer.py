#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Intent Analyzer
======================================

Phase 2: Understand code intent and purpose.

This module analyzes code to understand:
- What the code is trying to accomplish
- The intent behind code patterns
- Business logic understanding
- Code behavior prediction

Key Features:
- Intent classification
- Behavior analysis
- Semantic understanding via Kimi K2.5
- Pattern-to-intent mapping

Author: JARVIS AI Project
Version: 2.0.0
Target Level: 40-50
"""

import ast
import re
import logging
import time
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import Counter

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class IntentType(Enum):
    """Types of code intent"""
    
    # Data operations
    DATA_INPUT = auto()        # Reading/fetching data
    DATA_OUTPUT = auto()       # Writing/sending data
    DATA_PROCESSING = auto()   # Transforming data
    DATA_STORAGE = auto()      # Storing data
    DATA_RETRIEVAL = auto()    # Querying data
    
    # Logic operations
    VALIDATION = auto()        # Checking conditions
    CALCULATION = auto()       # Computing values
    TRANSFORMATION = auto()    # Converting formats
    FILTERING = auto()         # Selecting subset
    SORTING = auto()           # Ordering data
    
    # Control flow
    COORDINATION = auto()      # Orchestrating processes
    SYNCHRONIZATION = auto()   # Managing concurrency
    ERROR_HANDLING = auto()    # Managing errors
    LOGGING = auto()           # Recording events
    MONITORING = auto()        # Observing state
    
    # Communication
    API_CALL = auto()          # External API interaction
    USER_INTERACTION = auto()  # User interface
    MESSAGING = auto()         # Inter-process communication
    NETWORKING = auto()        # Network operations
    
    # Structure
    INITIALIZATION = auto()    # Setup/config
    CLEANUP = auto()           # Resource release
    CONFIGURATION = auto()     # Settings management
    TESTING = auto()           # Test code
    
    # Security
    AUTHENTICATION = auto()    # Identity verification
    AUTHORIZATION = auto()     # Permission checking
    ENCRYPTION = auto()        # Security operations
    
    # Unknown
    UNKNOWN = auto()           # Cannot determine


class ComplexityLevel(Enum):
    """Code complexity levels"""
    TRIVIAL = "trivial"      # Single operation
    SIMPLE = "simple"        # Few operations
    MODERATE = "moderate"    # Some logic
    COMPLEX = "complex"      # Significant logic
    VERY_COMPLEX = "very_complex"  # Multiple complex operations


class ConfidenceLevel(Enum):
    """Confidence in intent analysis"""
    HIGH = "high"          # Very confident
    MEDIUM = "medium"      # Reasonably confident
    LOW = "low"            # Uncertain
    GUESS = "guess"        # Best guess


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CodeIntent:
    """
    Complete intent analysis of code.
    
    Captures what the code does, why it does it,
    and how confident we are in the analysis.
    """
    # Primary intent
    primary_intent: IntentType = IntentType.UNKNOWN
    secondary_intents: List[IntentType] = field(default_factory=list)
    
    # Description
    summary: str = ""
    detailed_purpose: str = ""
    
    # Behavior
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    
    # Complexity
    complexity: ComplexityLevel = ComplexityLevel.MODERATE
    complexity_reasons: List[str] = field(default_factory=list)
    
    # Confidence
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    confidence_score: float = 0.5
    
    # Keywords and patterns
    keywords: List[str] = field(default_factory=list)
    patterns_detected: List[str] = field(default_factory=list)
    
    # Dependencies
    required_imports: List[str] = field(default_factory=list)
    external_dependencies: List[str] = field(default_factory=list)
    
    # Risks
    potential_issues: List[str] = field(default_factory=list)
    security_concerns: List[str] = field(default_factory=list)
    
    # AI insights
    ai_analysis: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'primary_intent': self.primary_intent.name,
            'secondary_intents': [i.name for i in self.secondary_intents],
            'summary': self.summary,
            'detailed_purpose': self.detailed_purpose,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'side_effects': self.side_effects,
            'complexity': self.complexity.value,
            'confidence': self.confidence.value,
            'confidence_score': self.confidence_score,
            'keywords': self.keywords,
            'patterns_detected': self.patterns_detected,
            'required_imports': self.required_imports,
            'external_dependencies': self.external_dependencies,
            'potential_issues': self.potential_issues,
            'security_concerns': self.security_concerns,
        }
    
    def is_data_operation(self) -> bool:
        """Check if this is a data operation"""
        return self.primary_intent in {
            IntentType.DATA_INPUT,
            IntentType.DATA_OUTPUT,
            IntentType.DATA_PROCESSING,
            IntentType.DATA_STORAGE,
            IntentType.DATA_RETRIEVAL,
        }
    
    def is_security_related(self) -> bool:
        """Check if this is security-related"""
        return self.primary_intent in {
            IntentType.AUTHENTICATION,
            IntentType.AUTHORIZATION,
            IntentType.ENCRYPTION,
        } or bool(self.security_concerns)


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT PATTERN MATCHER
# ═══════════════════════════════════════════════════════════════════════════════

class IntentPatternMatcher:
    """
    Match code patterns to intents.
    
    Uses heuristic rules to quickly identify common patterns.
    """
    
    # Function name patterns -> Intent
    FUNCTION_PATTERNS = {
        # Data operations
        r'^(get|fetch|retrieve|load|read)_': IntentType.DATA_RETRIEVAL,
        r'^(save|store|write|persist)_': IntentType.DATA_STORAGE,
        r'^(process|transform|convert|parse)_': IntentType.DATA_PROCESSING,
        r'^(send|post|submit|push)_': IntentType.DATA_OUTPUT,
        r'^(receive|accept|pull)_': IntentType.DATA_INPUT,
        
        # Validation
        r'^(validate|check|verify|ensure)_': IntentType.VALIDATION,
        r'^(is_|has_|can_)': IntentType.VALIDATION,
        
        # Calculation
        r'^(calculate|compute|evaluate|solve)_': IntentType.CALCULATION,
        r'^(count|sum|average|mean)_': IntentType.CALCULATION,
        
        # Filtering
        r'^(filter|select|find|search)_': IntentType.FILTERING,
        r'^(sort|order|rank)_': IntentType.SORTING,
        
        # Control
        r'^(init|setup|configure)_': IntentType.INITIALIZATION,
        r'^(cleanup|teardown|dispose)_': IntentType.CLEANUP,
        r'^(start|run|execute)_': IntentType.COORDINATION,
        
        # Logging
        r'^(log|debug|info|warn|error)_': IntentType.LOGGING,
        r'^(monitor|watch|observe)_': IntentType.MONITORING,
        
        # Auth
        r'^(auth|login|logout|signin)': IntentType.AUTHENTICATION,
        r'^(permit|allow|deny|grant)_': IntentType.AUTHORIZATION,
        
        # Test
        r'^test_': IntentType.TESTING,
        r'_test$': IntentType.TESTING,
    }
    
    # Import patterns -> Intent
    IMPORT_PATTERNS = {
        r'requests|httpx|aiohttp|urllib': IntentType.API_CALL,
        r'flask|django|fastapi|bottle': IntentType.USER_INTERACTION,
        r'sqlalchemy|sqlite|postgres|mysql': IntentType.DATA_STORAGE,
        r'pandas|numpy|polars': IntentType.DATA_PROCESSING,
        r'jwt|crypto|bcrypt|hashlib': IntentType.ENCRYPTION,
        r'logging|structlog': IntentType.LOGGING,
        r'multiprocessing|threading|asyncio': IntentType.SYNCHRONIZATION,
        r'pytest|unittest|nose': IntentType.TESTING,
    }
    
    # Method call patterns -> Intent
    METHOD_PATTERNS = {
        r'\.read\(|\.readlines\(|\.load\(': IntentType.DATA_INPUT,
        r'\.write\(|\.save\(|\.dump\(': IntentType.DATA_OUTPUT,
        r'\.get\(|\.fetch\(|\.request\(': IntentType.DATA_RETRIEVAL,
        r'\.post\(|\.put\(|\.send\(': IntentType.API_CALL,
        r'\.validate\(|\.check\(|\.verify\(': IntentType.VALIDATION,
        r'\.log\(|\.debug\(|\.info\(': IntentType.LOGGING,
        r'\.encrypt\(|\.decrypt\(|\.hash\(': IntentType.ENCRYPTION,
        r'\.sort\(|\.sorted\(': IntentType.SORTING,
        r'\.filter\(|\.select\(': IntentType.FILTERING,
    }
    
    @classmethod
    def match_function_name(cls, name: str) -> Optional[IntentType]:
        """Match function name to intent"""
        name_lower = name.lower()
        for pattern, intent in cls.FUNCTION_PATTERNS.items():
            if re.match(pattern, name_lower):
                return intent
        return None
    
    @classmethod
    def match_imports(cls, imports: List[str]) -> List[IntentType]:
        """Match imports to intents"""
        intents = []
        for imp in imports:
            for pattern, intent in cls.IMPORT_PATTERNS.items():
                if re.search(pattern, imp.lower()):
                    intents.append(intent)
        return intents
    
    @classmethod
    def match_method_calls(cls, code: str) -> List[IntentType]:
        """Match method calls to intents"""
        intents = []
        for pattern, intent in cls.METHOD_PATTERNS.items():
            if re.search(pattern, code):
                intents.append(intent)
        return intents


# ═══════════════════════════════════════════════════════════════════════════════
# INTENT ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class IntentAnalyzer:
    """
    Analyze code to understand intent.
    
    Combines:
    - Pattern-based heuristics (fast)
    - AST analysis (accurate)
    - AI analysis via Kimi K2.5 (deep understanding)
    
    Usage:
        analyzer = IntentAnalyzer(kimi_client)
        intent = analyzer.analyze(code)
        
        print(f"Intent: {intent.primary_intent.name}")
        print(f"Purpose: {intent.summary}")
    """
    
    def __init__(
        self,
        kimi_client=None,
        use_ai: bool = True,
        ai_timeout: int = 60,
    ):
        """
        Initialize intent analyzer.
        
        Args:
            kimi_client: Kimi K2.5 client for AI analysis
            use_ai: Whether to use AI analysis
            ai_timeout: Timeout for AI calls
        """
        self._kimi = kimi_client
        self._use_ai = use_ai
        self._ai_timeout = ai_timeout
        
        # Statistics
        self._stats = {
            'total_analyzed': 0,
            'pattern_matches': 0,
            'ai_analyses': 0,
            'unknown_intents': 0,
        }
        
        logger.info("IntentAnalyzer initialized")
    
    def set_kimi_client(self, client):
        """Set Kimi client"""
        self._kimi = client
    
    def analyze(
        self,
        code: str,
        function_name: str = None,
        file_path: str = None,
        use_ai: bool = None,
    ) -> CodeIntent:
        """
        Analyze code intent.
        
        Args:
            code: Source code to analyze
            function_name: Optional function name
            file_path: Optional file path
            use_ai: Override use_ai setting
            
        Returns:
            CodeIntent with analysis results
        """
        start_time = time.time()
        
        intent = CodeIntent()
        self._stats['total_analyzed'] += 1
        
        # Step 1: Pattern-based analysis
        self._analyze_patterns(code, function_name, intent)
        
        # Step 2: AST-based analysis
        self._analyze_ast(code, intent)
        
        # Step 3: AI-based analysis (if enabled)
        should_use_ai = use_ai if use_ai is not None else self._use_ai
        if should_use_ai and self._kimi:
            self._analyze_with_ai(code, intent)
        
        # Finalize
        if intent.primary_intent == IntentType.UNKNOWN:
            self._stats['unknown_intents'] += 1
        
        # Set confidence based on analysis depth
        if intent.ai_analysis:
            intent.confidence = ConfidenceLevel.HIGH
            intent.confidence_score = 0.9
        elif intent.patterns_detected:
            intent.confidence = ConfidenceLevel.MEDIUM
            intent.confidence_score = 0.7
        else:
            intent.confidence = ConfidenceLevel.LOW
            intent.confidence_score = 0.4
        
        logger.debug(f"Intent analysis completed in {(time.time() - start_time)*1000:.1f}ms")
        
        return intent
    
    def _analyze_patterns(
        self,
        code: str,
        function_name: str,
        intent: CodeIntent
    ) -> None:
        """Analyze code using pattern matching"""
        
        # Match function name
        if function_name:
            matched = IntentPatternMatcher.match_function_name(function_name)
            if matched:
                intent.primary_intent = matched
                intent.patterns_detected.append(f"function_name:{function_name}")
                self._stats['pattern_matches'] += 1
        
        # Match imports
        import_matches = re.findall(r'^(?:from|import)\s+(\S+)', code, re.MULTILINE)
        if import_matches:
            import_intents = IntentPatternMatcher.match_imports(import_matches)
            if import_intents:
                intent.secondary_intents.extend(import_intents)
                intent.patterns_detected.append("imports")
                intent.required_imports = import_matches
        
        # Match method calls
        method_intents = IntentPatternMatcher.match_method_calls(code)
        if method_intents:
            if intent.primary_intent == IntentType.UNKNOWN:
                intent.primary_intent = method_intents[0]
            intent.secondary_intents.extend(method_intents[1:])
            intent.patterns_detected.append("method_calls")
        
        # Extract keywords
        keywords = self._extract_keywords(code)
        intent.keywords = keywords
        
        # Generate summary
        if not intent.summary:
            intent.summary = self._generate_summary(intent, function_name)
    
    def _analyze_ast(self, code: str, intent: CodeIntent) -> None:
        """Analyze code using AST"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return
        
        # Analyze nodes
        for node in ast.walk(tree):
            # Function definitions
            if isinstance(node, ast.FunctionDef):
                # Check arguments
                for arg in node.args.args:
                    intent.inputs.append(arg.arg)
                
                # Check for return
                if any(isinstance(n, ast.Return) for n in ast.walk(node)):
                    intent.outputs.append("return_value")
                
                # Check docstring
                if (node.body and isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant)):
                    docstring = node.body[0].value.value
                    if isinstance(docstring, str):
                        intent.detailed_purpose = docstring[:200]
            
            # API calls
            elif isinstance(node, ast.Call):
                call_name = self._get_call_name(node)
                if call_name:
                    # Detect external API calls
                    if '.' in call_name and not call_name.startswith('_'):
                        parts = call_name.split('.')
                        if parts[0] not in {'self', 'cls', 'list', 'dict', 'str', 'int', 'float'}:
                            intent.external_dependencies.append(parts[0])
            
            # Side effects detection
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        intent.side_effects.append(f"modifies {self._get_attr_name(target)}")
            
            # Try/except - error handling
            elif isinstance(node, ast.Try):
                if IntentType.ERROR_HANDLING not in intent.secondary_intents:
                    intent.secondary_intents.append(IntentType.ERROR_HANDLING)
        
        # Calculate complexity
        intent.complexity, intent.complexity_reasons = self._calculate_complexity(tree, code)
    
    def _analyze_with_ai(self, code: str, intent: CodeIntent) -> None:
        """Analyze code using Kimi K2.5"""
        if not self._kimi:
            return
        
        prompt = f"""Analyze this Python code and determine its intent.

Code:
```python
{code[:3000]}
```

Current analysis:
- Primary intent detected: {intent.primary_intent.name}
- Patterns found: {intent.patterns_detected}

Provide a detailed analysis in this JSON format:
{{
    "primary_intent": "INTENT_TYPE",
    "summary": "One-line description",
    "detailed_purpose": "What this code accomplishes",
    "inputs": ["list of", "inputs"],
    "outputs": ["list of", "outputs"],
    "side_effects": ["list of", "side effects"],
    "potential_issues": ["list of potential", "problems"],
    "security_concerns": ["list of security", "concerns if any"]
}}

Valid INTENT_TYPE values: DATA_INPUT, DATA_OUTPUT, DATA_PROCESSING, DATA_STORAGE, DATA_RETRIEVAL, VALIDATION, CALCULATION, TRANSFORMATION, FILTERING, SORTING, COORDINATION, SYNCHRONIZATION, ERROR_HANDLING, LOGGING, MONITORING, API_CALL, USER_INTERACTION, MESSAGING, NETWORKING, INITIALIZATION, CLEANUP, CONFIGURATION, TESTING, AUTHENTICATION, AUTHORIZATION, ENCRYPTION, UNKNOWN

Output ONLY the JSON."""
        
        try:
            response = self._kimi.chat(
                prompt,
                temperature=0.2,
                max_tokens=1024,
            )
            
            if response.success:
                # Parse JSON from response
                import json
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    
                    # Update intent with AI analysis
                    if 'primary_intent' in data:
                        try:
                            intent.primary_intent = IntentType[data['primary_intent']]
                        except KeyError:
                            pass
                    
                    if 'summary' in data:
                        intent.summary = data['summary']
                    
                    if 'detailed_purpose' in data:
                        intent.detailed_purpose = data['detailed_purpose']
                    
                    if 'inputs' in data:
                        intent.inputs = data['inputs']
                    
                    if 'outputs' in data:
                        intent.outputs = data['outputs']
                    
                    if 'side_effects' in data:
                        intent.side_effects = data['side_effects']
                    
                    if 'potential_issues' in data:
                        intent.potential_issues = data['potential_issues']
                    
                    if 'security_concerns' in data:
                        intent.security_concerns = data['security_concerns']
                    
                    intent.ai_analysis = response.content
                    self._stats['ai_analyses'] += 1
                    
        except Exception as e:
            logger.warning(f"AI analysis failed: {e}")
    
    def _get_call_name(self, node: ast.Call) -> str:
        """Get function call name"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        return ""
    
    def _get_attr_name(self, node: ast.Attribute) -> str:
        """Get attribute name"""
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))
    
    def _extract_keywords(self, code: str) -> List[str]:
        """Extract meaningful keywords from code"""
        # Remove comments and strings
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'["\'].*?["\']', '', code)
        
        # Extract words
        words = re.findall(r'\b[a-z_][a-z0-9_]{2,}\b', code.lower())
        
        # Filter common words
        stop_words = {
            'self', 'the', 'and', 'for', 'not', 'none', 'true', 'false',
            'return', 'def', 'class', 'import', 'from', 'with', 'this',
        }
        
        words = [w for w in words if w not in stop_words]
        
        # Return most common
        counter = Counter(words)
        return [w for w, _ in counter.most_common(10)]
    
    def _generate_summary(self, intent: CodeIntent, function_name: str) -> str:
        """Generate summary from intent"""
        parts = []
        
        if intent.primary_intent != IntentType.UNKNOWN:
            parts.append(intent.primary_intent.name.replace('_', ' ').lower())
        
        if function_name:
            parts.append(f"in {function_name}")
        
        if intent.patterns_detected:
            parts.append(f"via {', '.join(intent.patterns_detected[:2])}")
        
        return ' '.join(parts) if parts else "Unknown purpose"
    
    def _calculate_complexity(
        self,
        tree: ast.AST,
        code: str
    ) -> tuple:
        """Calculate code complexity"""
        reasons = []
        score = 0
        
        # Count lines
        lines = len(code.split('\n'))
        if lines > 50:
            score += 2
            reasons.append(f"Large file ({lines} lines)")
        elif lines > 20:
            score += 1
            reasons.append(f"Moderate size ({lines} lines)")
        
        # Count branches
        branches = sum(1 for _ in ast.walk(tree) if isinstance(_, (ast.If, ast.For, ast.While)))
        if branches > 5:
            score += 2
            reasons.append(f"Many branches ({branches})")
        elif branches > 2:
            score += 1
            reasons.append(f"Some branching ({branches})")
        
        # Count functions
        functions = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
        if functions > 5:
            score += 1
            reasons.append(f"Multiple functions ({functions})")
        
        # Check nesting
        max_depth = self._get_max_depth(tree)
        if max_depth > 4:
            score += 2
            reasons.append(f"Deep nesting ({max_depth})")
        elif max_depth > 2:
            score += 1
            reasons.append(f"Some nesting ({max_depth})")
        
        # Determine level
        if score >= 6:
            level = ComplexityLevel.VERY_COMPLEX
        elif score >= 4:
            level = ComplexityLevel.COMPLEX
        elif score >= 2:
            level = ComplexityLevel.MODERATE
        elif score >= 1:
            level = ComplexityLevel.SIMPLE
        else:
            level = ComplexityLevel.TRIVIAL
        
        return level, reasons
    
    def _get_max_depth(self, tree: ast.AST) -> int:
        """Get maximum nesting depth"""
        def visit(node, depth):
            max_d = depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    max_d = max(max_d, visit(child, depth + 1))
                else:
                    max_d = max(max_d, visit(child, depth))
            return max_d
        
        return visit(tree, 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return self._stats.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_analyzer: Optional[IntentAnalyzer] = None


def get_intent_analyzer(kimi_client=None) -> IntentAnalyzer:
    """Get or create global intent analyzer"""
    global _analyzer
    if _analyzer is None:
        _analyzer = IntentAnalyzer(kimi_client=kimi_client)
    elif kimi_client:
        _analyzer.set_kimi_client(kimi_client)
    return _analyzer


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Intent Analyzer Test")
    print("="*60)
    
    analyzer = IntentAnalyzer(kimi_client=None, use_ai=False)
    
    # Test cases
    test_cases = [
        ('''
def get_user_data(user_id: int) -> dict:
    """Fetch user data from database."""
    return db.query(User).filter(User.id == user_id).first()
''', "get_user_data"),
        
        ('''
def calculate_total(items: list) -> float:
    """Calculate total price of items."""
    total = 0
    for item in items:
        total += item.price * item.quantity
    return total
''', "calculate_total"),
        
        ('''
def validate_email(email: str) -> bool:
    """Check if email is valid."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
''', "validate_email"),
    ]
    
    for code, name in test_cases:
        print(f"\nAnalyzing: {name}")
        intent = analyzer.analyze(code, function_name=name)
        print(f"  Primary Intent: {intent.primary_intent.name}")
        print(f"  Summary: {intent.summary}")
        print(f"  Complexity: {intent.complexity.value}")
        print(f"  Patterns: {intent.patterns_detected}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()

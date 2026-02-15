#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - AI Code Generator
=======================================

Generate code from specifications using Kimi K2.5.

Features:
- Generate code from natural language specifications
- Support multiple programming languages
- Context-aware generation
- Code style customization
- Automatic validation

Author: JARVIS AI Project
Version: 1.0.0
Device: Realme Pad 2 Lite | Termux
"""

import time
import logging
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class Language(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    BASH = "bash"
    SQL = "sql"


class CodeStyle(Enum):
    """Code style preferences"""
    PYTHONIC = "pythonic"
    VERBOSE = "verbose"
    MINIMAL = "minimal"
    ENTERPRISE = "enterprise"
    FUNCTIONAL = "functional"


@dataclass
class GenerationConfig:
    """Configuration for code generation"""
    language: Language = Language.PYTHON
    style: CodeStyle = CodeStyle.PYTHONIC
    max_tokens: int = 16384
    temperature: float = 0.3
    include_tests: bool = False
    include_docs: bool = True
    include_type_hints: bool = True
    include_error_handling: bool = True
    max_retries: int = 2


@dataclass
class GenerationResult:
    """Result of code generation"""
    code: str
    language: str
    success: bool = True
    error: Optional[str] = None
    explanation: str = ""
    confidence: float = 0.0
    tokens_used: int = 0
    latency_ms: float = 0.0
    tests: str = ""
    imports_needed: List[str] = field(default_factory=list)
    
    @property
    def has_code(self) -> bool:
        return bool(self.code.strip())
    
    def is_valid_python(self) -> bool:
        """Check if generated code is valid Python"""
        if self.language != "python":
            return True  # Can't validate non-Python
        
        try:
            compile(self.code, '<generated>', 'exec')
            return True
        except SyntaxError:
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# CODE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class CodeGenerator:
    """
    AI-powered code generator using Kimi K2.5.
    
    Usage:
        generator = CodeGenerator(kimi_client)
        result = generator.generate("Function to calculate fibonacci")
        
        if result.success:
            print(result.code)
    """
    
    # System prompts for different languages
    LANGUAGE_PROMPTS = {
        Language.PYTHON: """You are an expert Python programmer.
Generate clean, efficient, well-documented Python code.
Follow PEP 8 style guidelines.
Include type hints for function signatures.
Include docstrings for classes and functions.
Include proper error handling with try/except blocks.""",
        
        Language.JAVASCRIPT: """You are an expert JavaScript programmer.
Generate clean, efficient, well-documented JavaScript code.
Follow modern ES6+ conventions.
Include JSDoc comments for functions.
Handle errors appropriately.""",
        
        Language.TYPESCRIPT: """You are an expert TypeScript programmer.
Generate clean, efficient, well-documented TypeScript code.
Include proper type annotations.
Follow TSLint recommended rules.
Handle errors appropriately.""",
        
        Language.BASH: """You are an expert Bash scripter.
Generate clean, efficient shell scripts.
Follow Google Shell Style Guide.
Include error handling and proper quoting.""",
    }
    
    # Style modifiers
    STYLE_MODIFIERS = {
        CodeStyle.PYTHONIC: "Write idiomatic, Pythonic code. Use list comprehensions, context managers, and generators where appropriate.",
        CodeStyle.VERBOSE: "Write explicit, easy-to-understand code. Avoid compact syntax. Comment liberally.",
        CodeStyle.MINIMAL: "Write concise code. Avoid unnecessary comments. Prefer one-liners when clear.",
        CodeStyle.ENTERPRISE: "Write enterprise-grade code. Include logging, metrics, config management, and comprehensive error handling.",
        CodeStyle.FUNCTIONAL: "Write functional-style code. Prefer immutability, pure functions, and higher-order functions.",
    }
    
    def __init__(self, kimi_client=None, config: GenerationConfig = None):
        """
        Initialize code generator.
        
        Args:
            kimi_client: Kimi K2.5 client instance
            config: Generation configuration
        """
        self._kimi = kimi_client
        self._config = config or GenerationConfig()
        
        # Statistics
        self._stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_tokens': 0,
            'total_latency_ms': 0.0,
        }
        
        logger.info("CodeGenerator initialized")
    
    def set_kimi_client(self, client):
        """Set Kimi client (for lazy initialization)"""
        self._kimi = client
    
    def generate(
        self,
        specification: str,
        context: str = None,
        config: GenerationConfig = None,
    ) -> GenerationResult:
        """
        Generate code from specification.
        
        Args:
            specification: What the code should do
            context: Additional context (existing code, requirements)
            config: Override default configuration
            
        Returns:
            GenerationResult with generated code
        """
        if not self._kimi:
            return GenerationResult(
                code="",
                language="",
                success=False,
                error="Kimi client not initialized",
            )
        
        config = config or self._config
        start_time = time.time()
        
        self._stats['total_generations'] += 1
        
        # Build prompt
        system_prompt = self._build_system_prompt(config)
        user_prompt = self._build_user_prompt(specification, context, config)
        
        # Generate code
        try:
            response = self._kimi.chat(
                message=user_prompt,
                system=system_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            if not response.success:
                self._stats['failed_generations'] += 1
                return GenerationResult(
                    code="",
                    language=config.language.value,
                    success=False,
                    error=response.error,
                )
            
            # Extract code from response
            code = self._extract_code(response.content)
            explanation = self._extract_explanation(response.content)
            imports = self._extract_imports(code, config.language)
            
            # Validate if Python
            is_valid = True
            if config.language == Language.PYTHON:
                try:
                    compile(code, '<generated>', 'exec')
                except SyntaxError as e:
                    is_valid = False
                    logger.warning(f"Generated code has syntax error: {e}")
            
            latency = (time.time() - start_time) * 1000
            
            # Update stats
            self._stats['successful_generations'] += 1
            self._stats['total_tokens'] += response.tokens_used
            self._stats['total_latency_ms'] += latency
            
            return GenerationResult(
                code=code,
                language=config.language.value,
                success=True,
                explanation=explanation,
                confidence=self._estimate_confidence(code, specification),
                tokens_used=response.tokens_used,
                latency_ms=latency,
                imports_needed=imports,
            )
            
        except Exception as e:
            self._stats['failed_generations'] += 1
            logger.error(f"Generation error: {e}")
            return GenerationResult(
                code="",
                language=config.language.value,
                success=False,
                error=str(e),
            )
    
    def generate_function(
        self,
        name: str,
        description: str,
        parameters: List[Dict] = None,
        return_type: str = None,
        config: GenerationConfig = None,
    ) -> GenerationResult:
        """
        Generate a single function.
        
        Args:
            name: Function name
            description: What the function should do
            parameters: List of parameter dicts with 'name' and 'type'
            return_type: Return type annotation
            config: Generation configuration
            
        Returns:
            GenerationResult with function code
        """
        params_str = ""
        if parameters:
            params_str = "Parameters:\n" + "\n".join(
                f"  - {p['name']}: {p.get('type', 'Any')}" for p in parameters
            )
        
        return_str = f"\nReturns: {return_type}" if return_type else ""
        
        spec = f"""Generate a function named '{name}' that:
{description}

{params_str}{return_str}"""
        
        return self.generate(spec, config=config or self._config)
    
    def generate_class(
        self,
        name: str,
        description: str,
        methods: List[str] = None,
        attributes: List[str] = None,
        config: GenerationConfig = None,
    ) -> GenerationResult:
        """
        Generate a class.
        
        Args:
            name: Class name
            description: What the class should do
            methods: List of method names to include
            attributes: List of attribute names
            config: Generation configuration
            
        Returns:
            GenerationResult with class code
        """
        methods_str = ""
        if methods:
            methods_str = "\nMethods to implement:\n" + "\n".join(f"  - {m}" for m in methods)
        
        attrs_str = ""
        if attributes:
            attrs_str = "\nAttributes:\n" + "\n".join(f"  - {a}" for a in attributes)
        
        spec = f"""Generate a class named '{name}' that:
{description}
{methods_str}{attrs_str}"""
        
        return self.generate(spec, config=config or self._config)
    
    def _build_system_prompt(self, config: GenerationConfig) -> str:
        """Build system prompt for generation"""
        base_prompt = self.LANGUAGE_PROMPTS.get(
            config.language,
            self.LANGUAGE_PROMPTS[Language.PYTHON]
        )
        
        style_modifier = self.STYLE_MODIFIERS.get(config.style, "")
        
        additional = []
        if config.include_docs:
            additional.append("Include comprehensive documentation.")
        if config.include_type_hints and config.language == Language.PYTHON:
            additional.append("Include type hints for all function parameters and return values.")
        if config.include_error_handling:
            additional.append("Include proper error handling.")
        if config.include_tests:
            additional.append("Include unit tests at the end.")
        
        additional_str = " " + " ".join(additional) if additional else ""
        
        return f"{base_prompt}\n\n{style_modifier}{additional_str}\n\nOutput ONLY the code in a code block. No explanations outside the code."
    
    def _build_user_prompt(
        self,
        specification: str,
        context: str,
        config: GenerationConfig
    ) -> str:
        """Build user prompt for generation"""
        prompt = f"Generate {config.language.value} code that:\n\n{specification}"
        
        if context:
            prompt = f"Context:\n{context}\n\n{prompt}"
        
        return prompt
    
    def _extract_code(self, text: str) -> str:
        """Extract code from markdown code block"""
        # Match code blocks with optional language specifier
        pattern = r'```(?:\w+)?\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            # Return largest code block (most likely the main code)
            return max(matches, key=len).strip()
        
        # No code block found, return original stripped
        return text.strip()
    
    def _extract_explanation(self, text: str) -> str:
        """Extract explanation from response"""
        # Remove code blocks
        text = re.sub(r'```\w*\n.*?```', '', text, flags=re.DOTALL)
        return text.strip()
    
    def _extract_imports(self, code: str, language: Language) -> List[str]:
        """Extract import statements from code"""
        imports = []
        
        if language == Language.PYTHON:
            # Match import statements
            import_pattern = r'^(?:from\s+\S+\s+import\s+.*|import\s+.*)$'
            for line in code.split('\n'):
                if re.match(import_pattern, line.strip()):
                    imports.append(line.strip())
        
        return imports
    
    def _estimate_confidence(self, code: str, spec: str) -> float:
        """Estimate confidence of generated code"""
        if not code:
            return 0.0
        
        confidence = 0.5
        
        # Penalize issues
        if "TODO" in code:
            confidence -= 0.2
        if "NotImplementedError" in code:
            confidence -= 0.3
        if "pass" in code and len(code.split('\n')) < 10:
            confidence -= 0.2
        if "..." in code and len(code.split('\n')) < 5:
            confidence -= 0.1
        
        # Reward good practices
        if "def " in code:
            confidence += 0.1
        if "try:" in code:
            confidence += 0.1
        if '"""' in code or "'''" in code:
            confidence += 0.05
        if "return" in code:
            confidence += 0.05
        if "class " in code:
            confidence += 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generator statistics"""
        stats = self._stats.copy()
        if stats['total_generations'] > 0:
            stats['success_rate'] = stats['successful_generations'] / stats['total_generations']
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['total_generations']
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_generator: Optional[CodeGenerator] = None


def get_code_generator(kimi_client=None) -> CodeGenerator:
    """Get or create global code generator"""
    global _generator
    if _generator is None:
        _generator = CodeGenerator(kimi_client=kimi_client)
    elif kimi_client:
        _generator.set_kimi_client(kimi_client)
    return _generator


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test(kimi_client=None):
    """Run self-test for code generator"""
    print("\n" + "="*60)
    print("Code Generator Self Test")
    print("="*60)
    
    if kimi_client is None:
        print("\n❌ No Kimi client provided")
        return
    
    generator = CodeGenerator(kimi_client=kimi_client)
    
    # Test 1: Simple function
    print("\nTest 1: Generate simple function...")
    result = generator.generate("Function to add two numbers")
    
    if result.success:
        print("✓ Success!")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Code preview:\n{result.code[:200]}...")
    else:
        print(f"✗ Failed: {result.error}")
    
    # Test 2: Class generation
    print("\nTest 2: Generate class...")
    result = generator.generate_class(
        name="Calculator",
        description="A simple calculator with basic operations",
        methods=["add", "subtract", "multiply", "divide"],
    )
    
    if result.success:
        print("✓ Success!")
        print(f"  Confidence: {result.confidence:.2f}")
    else:
        print(f"✗ Failed: {result.error}")
    
    # Stats
    print("\nStatistics:")
    stats = generator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    print("Code Generator module - use via JARVIS main.py")

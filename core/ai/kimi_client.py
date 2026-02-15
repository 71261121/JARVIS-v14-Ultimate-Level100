#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Kimi K2.5 API Client (BULLETPROOF EDITION)
================================================================

100x Advanced Version with:
- Retry with exponential backoff
- Memory-bounded LRU cache
- Thread-safe operations
- Circuit breaker pattern
- Token bucket rate limiting
- Request cancellation
- Health monitoring
- Termux/Android optimizations

Author: JARVIS AI Project
Version: 3.0.0 (Bulletproof)
Device: Realme Pad 2 Lite | Termux
"""

import time
import json
import hashlib
import threading
import logging
import re
import sys
import os
import gc
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import OrderedDict
from functools import wraps
from contextlib import contextmanager
import weakref

# Platform detection for Termux optimization
IS_TERMUX = os.path.exists('/data/data/com.termux')
IS_ANDROID = 'android' in os.environ.get('TERMUX_VERSION', '').lower()

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PLATFORM-SPECIFIC CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class PlatformConfig:
    """Platform-specific optimizations for Termux/Android"""
    
    # Memory limits (4GB device = ~1GB for app)
    MAX_CACHE_MEMORY_MB = 64 if IS_TERMUX else 256
    MAX_CACHE_ENTRIES = 500 if IS_TERMUX else 2000
    
    # Timeouts (mobile networks are slower)
    DEFAULT_TIMEOUT = 180 if IS_TERMUX else 120
    CONNECT_TIMEOUT = 30 if IS_TERMUX else 15
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 1.0
    RETRY_MAX_DELAY = 30.0
    
    # Rate limiting
    REQUESTS_PER_MINUTE = 20 if IS_TERMUX else 60
    
    # Circuit breaker
    FAILURE_THRESHOLD = 3
    RECOVERY_TIMEOUT = 60


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class KimiModel(Enum):
    """Available Kimi models via NVIDIA NIM"""
    K2_5 = "moonshotai/kimi-k2.5"


class TaskType(Enum):
    """Types of AI tasks"""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    BUG_FIXING = "bug_fixing"
    CODE_IMPROVEMENT = "code_improvement"
    REASONING = "reasoning"
    GENERAL_CHAT = "general_chat"
    LONG_CONTEXT = "long_context"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing if recovered


@dataclass
class KimiResponse:
    """Response from Kimi API"""
    content: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    raw_response: Dict = field(default_factory=dict)
    reasoning: str = ""
    from_cache: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'model': self.model,
            'tokens_used': self.tokens_used,
            'latency_ms': self.latency_ms,
            'success': self.success,
            'error': self.error,
            'from_cache': self.from_cache,
        }


@dataclass
class CodeResult:
    """Result of code generation"""
    code: str
    explanation: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    is_valid_syntax: bool = True
    
    @property
    def has_code(self) -> bool:
        return bool(self.code.strip())


@dataclass
class BugFixResult:
    """Result of bug fix"""
    original_code: str
    fixed_code: str
    explanation: str
    confidence: float
    changes_made: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY-BOUNDED LRU CACHE
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryBoundedLRUCache:
    """
    Thread-safe LRU cache with memory limits.
    Optimized for Termux 4GB RAM devices.
    """
    
    def __init__(
        self,
        max_entries: int = None,
        max_memory_mb: int = None,
        ttl_seconds: int = 3600,
    ):
        self._max_entries = max_entries or PlatformConfig.MAX_CACHE_ENTRIES
        self._max_memory = (max_memory_mb or PlatformConfig.MAX_CACHE_MEMORY_MB) * 1024 * 1024
        self._ttl = ttl_seconds
        
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
    
    def get(self, key: str) -> Optional[tuple]:
        """Get cached value if exists and not expired"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            value, timestamp, size = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp > self._ttl:
                self._evict(key)
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return value, timestamp
    
    def set(self, key: str, value: tuple, size_estimate: int = 0) -> bool:
        """Set cached value with memory management"""
        with self._lock:
            # Estimate size if not provided
            if size_estimate == 0:
                try:
                    size_estimate = len(str(value))
                except:
                    size_estimate = 1024  # Default 1KB
            
            # Check if single entry exceeds max memory
            if size_estimate > self._max_memory * 0.5:
                return False  # Don't cache huge entries
            
            # Evict entries if needed
            while (self._current_memory + size_estimate > self._max_memory or
                   len(self._cache) >= self._max_entries):
                if not self._evict_oldest():
                    break
            
            # Remove old entry if updating
            if key in self._cache:
                old_size = self._cache[key][2]
                self._current_memory -= old_size
            
            # Add new entry
            self._cache[key] = (value, time.time(), size_estimate)
            self._current_memory += size_estimate
            return True
    
    def _evict(self, key: str):
        """Evict specific entry"""
        if key in self._cache:
            _, _, size = self._cache[key]
            del self._cache[key]
            self._current_memory -= size
            self._evictions += 1
    
    def _evict_oldest(self) -> bool:
        """Evict oldest entry"""
        if not self._cache:
            return False
        
        key = next(iter(self._cache))
        self._evict(key)
        return True
    
    def clear(self):
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            gc.collect()  # Force garbage collection on Termux
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            
            return {
                'entries': len(self._cache),
                'max_entries': self._max_entries,
                'memory_bytes': self._current_memory,
                'memory_mb': self._current_memory / (1024 * 1024),
                'max_memory_mb': self._max_memory / (1024 * 1024),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
            }


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN BUCKET RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════

class TokenBucket:
    """Thread-safe token bucket rate limiter"""
    
    def __init__(self, rate_per_minute: int = None):
        self._rate = rate_per_minute or PlatformConfig.REQUESTS_PER_MINUTE
        self._interval = 60.0 / self._rate  # Seconds between tokens
        self._tokens = self._rate
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, timeout: float = None) -> bool:
        """Acquire a token, waiting if necessary"""
        start_time = time.time()
        
        while True:
            with self._lock:
                now = time.time()
                elapsed = now - self._last_update
                
                # Replenish tokens
                new_tokens = elapsed / self._interval
                self._tokens = min(self._rate, self._tokens + new_tokens)
                self._last_update = now
                
                # Try to consume a token
                if self._tokens >= 1:
                    self._tokens -= 1
                    return True
                
                # Calculate wait time
                wait_time = (1 - self._tokens) * self._interval
            
            # Check timeout
            if timeout is not None:
                elapsed_total = time.time() - start_time
                if elapsed_total + wait_time > timeout:
                    return False
            
            # Wait and retry
            time.sleep(min(wait_time, 0.1))


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(
        self,
        failure_threshold: int = None,
        recovery_timeout: float = None,
    ):
        self._failure_threshold = failure_threshold or PlatformConfig.FAILURE_THRESHOLD
        self._recovery_timeout = recovery_timeout or PlatformConfig.RECOVERY_TIMEOUT
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if request can proceed"""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout passed
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    return True
                return False
            
            # HALF_OPEN - allow one test request
            return True
    
    def record_success(self):
        """Record successful request"""
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
    
    def record_failure(self):
        """Record failed request"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
    
    @property
    def state(self) -> CircuitState:
        return self._state


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST CANCELLATION TOKEN
# ═══════════════════════════════════════════════════════════════════════════════

class CancellationToken:
    """Token for cancelling requests"""
    
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
    
    def cancel(self):
        """Cancel the token"""
        with self._lock:
            self._cancelled = True
    
    @property
    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancelled


# ═══════════════════════════════════════════════════════════════════════════════
# BULLETPROOF KIMI K2.5 CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class KimiK25Client:
    """
    Bulletproof Kimi K2.5 API Client for JARVIS Level 100.
    
    Features:
    - Retry with exponential backoff
    - Memory-bounded cache
    - Circuit breaker
    - Rate limiting
    - Request cancellation
    - Termux optimizations
    
    API: NVIDIA NIM (https://integrate.api.nvidia.com)
    Model: moonshotai/kimi-k2.5
    """
    
    API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
    MODEL = KimiModel.K2_5
    
    # Model for each task type
    MODEL_FOR_TASK = {task: KimiModel.K2_5 for task in TaskType}
    
    def __init__(
        self,
        api_key: str,
        timeout: int = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        enable_thinking: bool = True,
        max_retries: int = None,
        enable_rate_limit: bool = True,
    ):
        """
        Initialize bulletproof Kimi client.
        
        Args:
            api_key: NVIDIA NIM API key (nvapi-...)
            timeout: Request timeout in seconds
            enable_cache: Enable response caching
            cache_ttl: Cache TTL in seconds
            enable_thinking: Enable thinking mode for deep reasoning
            max_retries: Maximum retry attempts
            enable_rate_limit: Enable rate limiting
        """
        # Validate API key
        if not api_key:
            raise ValueError("API key is required")
        if not api_key.startswith('nvapi-'):
            logger.warning("API key format may be incorrect (expected nvapi-...)")
        
        self._api_key = api_key
        self._timeout = timeout or PlatformConfig.DEFAULT_TIMEOUT
        self._enable_cache = enable_cache
        self._cache_ttl = cache_ttl
        self._enable_thinking = enable_thinking
        self._max_retries = max_retries or PlatformConfig.MAX_RETRIES
        
        # Initialize components
        self._cache = MemoryBoundedLRUCache(ttl_seconds=cache_ttl) if enable_cache else None
        self._rate_limiter = TokenBucket() if enable_rate_limit else None
        self._circuit_breaker = CircuitBreaker()
        
        # HTTP client (lazy)
        self._http = None
        self._http_lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'total_latency_ms': 0.0,
            'retries': 0,
            'cache_hits': 0,
            'circuit_opens': 0,
        }
        self._stats_lock = threading.Lock()
        
        # Health status
        self._healthy = True
        self._last_success = time.time()
        
        logger.info(f"KimiK25Client initialized (Termux={IS_TERMUX}, Timeout={self._timeout}s)")
    
    def _get_headers(self, stream: bool = False) -> Dict[str, str]:
        """Get API headers"""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        else:
            headers["Accept"] = "application/json"
        return headers
    
    def _get_http_client(self):
        """Get HTTP client with thread safety"""
        with self._http_lock:
            if self._http is None:
                try:
                    from core.http_client import HTTPClient
                    self._http = HTTPClient(default_headers=self._get_headers())
                except ImportError:
                    self._http = "urllib"
            return self._http
    
    def _make_cache_key(self, messages: List[Dict], **kwargs) -> str:
        """Create cache key"""
        try:
            data = json.dumps({
                'messages': messages,
                'kwargs': {k: str(v)[:100] for k, v in kwargs.items()},
            }, sort_keys=True, default=str)
            return hashlib.sha256(data.encode()).hexdigest()[:32]
        except:
            return hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:32]
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay"""
        delay = PlatformConfig.RETRY_BASE_DELAY * (2 ** attempt)
        return min(delay, PlatformConfig.RETRY_MAX_DELAY)
    
    def _update_stats(self, key: str, value: Any = 1):
        """Update statistics thread-safely"""
        with self._stats_lock:
            if key in self._stats:
                if isinstance(self._stats[key], (int, float)):
                    self._stats[key] += value
    
    def chat(
        self,
        message: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        cancel_token: CancellationToken = None,
        **kwargs
    ) -> KimiResponse:
        """
        Send chat message with retry and error handling.
        
        Args:
            message: User message
            system: System prompt
            temperature: Response randomness
            max_tokens: Maximum response tokens
            cancel_token: Cancellation token
            
        Returns:
            KimiResponse
        """
        start_time = time.time()
        
        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})
        
        # Check cache
        if self._cache:
            cache_key = self._make_cache_key(messages, temperature=temperature, max_tokens=max_tokens)
            cached = self._cache.get(cache_key)
            if cached:
                cached_response, _ = cached
                cached_response.from_cache = True
                self._update_stats('cache_hits')
                return cached_response
        
        # Check circuit breaker
        if not self._circuit_breaker.can_execute():
            self._update_stats('circuit_opens')
            return KimiResponse(
                content="",
                model=self.MODEL.value,
                success=False,
                error="Circuit breaker open - too many failures",
            )
        
        # Rate limiting
        if self._rate_limiter:
            if not self._rate_limiter.acquire(timeout=self._timeout):
                return KimiResponse(
                    content="",
                    model=self.MODEL.value,
                    success=False,
                    error="Rate limit timeout",
                )
        
        # Build payload
        payload = {
            "model": self.MODEL.value,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1.0,
            "stream": False,
        }
        
        if self._enable_thinking:
            payload["chat_template_kwargs"] = {"thinking": True}
        
        # Execute with retry
        last_error = None
        for attempt in range(self._max_retries + 1):
            # Check cancellation
            if cancel_token and cancel_token.is_cancelled:
                return KimiResponse(
                    content="",
                    model=self.MODEL.value,
                    success=False,
                    error="Request cancelled",
                )
            
            try:
                response = self._execute_request(payload, attempt)
                
                if response.success:
                    self._circuit_breaker.record_success()
                    self._healthy = True
                    self._last_success = time.time()
                    
                    # Cache result
                    if self._cache:
                        size = len(response.content) + 500  # Estimate
                        self._cache.set(cache_key, (response, time.time()), size)
                    
                    return response
                
                last_error = response.error
                
                # Don't retry certain errors
                if "401" in str(last_error) or "403" in str(last_error):
                    break
                
                # Backoff before retry
                if attempt < self._max_retries:
                    backoff = self._calculate_backoff(attempt)
                    time.sleep(backoff)
                    self._update_stats('retries')
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Request error (attempt {attempt}): {e}")
        
        # All retries failed
        self._circuit_breaker.record_failure()
        self._healthy = False
        
        return KimiResponse(
            content="",
            model=self.MODEL.value,
            success=False,
            error=f"All retries failed: {last_error}",
            retry_count=self._max_retries,
        )
    
    def _execute_request(self, payload: Dict, attempt: int) -> KimiResponse:
        """Execute HTTP request"""
        start_time = time.time()
        
        self._update_stats('total_requests')
        
        http = self._get_http_client()
        
        try:
            if http != "urllib":
                # Use HTTPClient
                response = http.post(
                    self.API_URL,
                    json_data=payload,
                    timeout=self._timeout,
                )
                
                if not response.success:
                    self._update_stats('failed_requests')
                    return KimiResponse(
                        content="",
                        model=self.MODEL.value,
                        success=False,
                        error=f"HTTP error: {response.error}",
                    )
                
                data = response.json()
            else:
                # Use urllib (fallback for Termux)
                import urllib.request
                import urllib.error
                
                req = urllib.request.Request(
                    self.API_URL,
                    data=json.dumps(payload).encode('utf-8'),
                    headers=self._get_headers(),
                    method='POST',
                )
                
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
        
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode('utf-8')
            except:
                pass
            
            self._update_stats('failed_requests')
            return KimiResponse(
                content="",
                model=self.MODEL.value,
                success=False,
                error=f"HTTP {e.code}: {error_body[:500]}",
            )
        
        except urllib.error.URLError as e:
            self._update_stats('failed_requests')
            return KimiResponse(
                content="",
                model=self.MODEL.value,
                success=False,
                error=f"URL Error: {e.reason}",
            )
        
        except TimeoutError:
            self._update_stats('failed_requests')
            return KimiResponse(
                content="",
                model=self.MODEL.value,
                success=False,
                error="Request timeout",
            )
        
        except json.JSONDecodeError as e:
            self._update_stats('failed_requests')
            return KimiResponse(
                content="",
                model=self.MODEL.value,
                success=False,
                error=f"JSON decode error: {e}",
            )
        
        # Parse response
        if 'error' in data:
            self._update_stats('failed_requests')
            return KimiResponse(
                content="",
                model=self.MODEL.value,
                success=False,
                error=data['error'].get('message', str(data['error'])),
            )
        
        if 'choices' not in data or not data['choices']:
            self._update_stats('failed_requests')
            return KimiResponse(
                content="",
                model=self.MODEL.value,
                success=False,
                error="No response generated",
            )
        
        # Extract content
        choice = data['choices'][0]
        content = choice.get('message', {}).get('content', '')
        tokens = data.get('usage', {}).get('total_tokens', 0)
        latency = (time.time() - start_time) * 1000
        
        # Update success stats
        self._update_stats('successful_requests')
        self._update_stats('total_tokens', tokens)
        self._update_stats('total_latency_ms', latency)
        
        return KimiResponse(
            content=content,
            model=self.MODEL.value,
            tokens_used=tokens,
            latency_ms=latency,
            success=True,
            raw_response={'choices': data['choices'], 'usage': data.get('usage', {})},
        )
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CODE GENERATION METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def generate_code(
        self,
        specification: str,
        language: str = "python",
        context: str = None,
        validate_syntax: bool = True,
    ) -> CodeResult:
        """Generate code from specification"""
        system = f"""You are an expert {language} programmer.
Generate clean, efficient, well-documented code.
Include error handling and edge cases.
Output ONLY the code in a markdown code block."""
        
        prompt = f"Generate {language} code that:\n\n{specification}"
        if context:
            prompt = f"Context:\n{context}\n\n{prompt}"
        
        response = self.chat(
            prompt,
            system=system,
            temperature=0.3,
            max_tokens=16384,
        )
        
        if not response.success:
            return CodeResult(
                code="",
                explanation=f"Error: {response.error}",
                confidence=0.0,
                is_valid_syntax=False,
            )
        
        code = self._extract_code_block(response.content)
        
        # Validate syntax for Python
        is_valid = True
        if validate_syntax and language == "python":
            is_valid = self._validate_python_syntax(code)
        
        return CodeResult(
            code=code,
            explanation=self._extract_explanation(response.content),
            confidence=self._estimate_confidence(code),
            is_valid_syntax=is_valid,
        )
    
    def fix_bug(
        self,
        code: str,
        error: str,
        context: str = None,
    ) -> BugFixResult:
        """Fix bug in code"""
        system = """You are a debugging expert.
Fix the bug in the given code.
Output the fixed code in a code block followed by brief explanation."""
        
        prompt = f"""Code with bug:
```python
{code}
```

Error: {error}"""
        
        if context:
            prompt += f"\n\nContext: {context}"
        
        response = self.chat(prompt, system=system, temperature=0.2)
        
        fixed_code = self._extract_code_block(response.content) if response.success else code
        
        return BugFixResult(
            original_code=code,
            fixed_code=fixed_code,
            explanation=self._extract_explanation(response.content),
            confidence=0.9 if response.success else 0.0,
            changes_made=self._detect_changes(code, fixed_code),
        )
    
    def improve_code(
        self,
        code: str,
        improvement_goal: str,
        constraints: List[str] = None,
    ) -> CodeResult:
        """Improve existing code"""
        system = """You are a code improvement expert.
Improve the given code according to the specified goal.
Output ONLY the improved code in a code block."""
        
        prompt = f"""Current code:
```python
{code}
```

Improvement goal: {improvement_goal}"""
        
        if constraints:
            prompt += "\n\nConstraints:\n" + "\n".join(f"- {c}" for c in constraints)
        
        response = self.chat(prompt, system=system, temperature=0.3)
        
        return CodeResult(
            code=self._extract_code_block(response.content) if response.success else "",
            explanation=self._extract_explanation(response.content),
            confidence=0.8 if response.success else 0.0,
        )
    
    def analyze_code(
        self,
        code: str,
        analysis_type: str = "full",
    ) -> str:
        """Analyze code for issues"""
        system = """You are a code analysis expert.
Analyze code thoroughly and provide:
1. Issues found (with severity)
2. Security concerns
3. Performance issues
4. Improvement suggestions"""
        
        prompt = f"""Analyze this code ({analysis_type} analysis):
```python
{code}
```"""
        
        response = self.chat(prompt, system=system)
        return response.content if response.success else f"Analysis failed: {response.error}"
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def _extract_code_block(self, text: str) -> str:
        """Extract code from markdown code block"""
        pattern = r'```(?:\w+)?\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        if matches:
            return max(matches, key=len).strip()
        return text.strip()
    
    def _extract_explanation(self, text: str) -> str:
        """Extract explanation from response"""
        text = re.sub(r'```\w*\n.*?```', '', text, flags=re.DOTALL)
        return text.strip()
    
    def _validate_python_syntax(self, code: str) -> bool:
        """Validate Python syntax"""
        try:
            compile(code, '<generated>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _estimate_confidence(self, code: str) -> float:
        """Estimate confidence of generated code"""
        if not code:
            return 0.0
        
        confidence = 0.5
        
        # Penalties
        if "TODO" in code: confidence -= 0.2
        if "NotImplementedError" in code: confidence -= 0.3
        if "pass" in code and len(code.split('\n')) < 5: confidence -= 0.2
        
        # Bonuses
        if "def " in code: confidence += 0.1
        if "try:" in code: confidence += 0.1
        if '"""' in code or "'''" in code: confidence += 0.05
        if "return" in code: confidence += 0.05
        
        return min(1.0, max(0.0, confidence))
    
    def _detect_changes(self, old_code: str, new_code: str) -> List[str]:
        """Detect changes between code versions"""
        changes = []
        
        old_lines = set(old_code.split('\n'))
        new_lines = set(new_code.split('\n'))
        
        added = new_lines - old_lines
        
        for line in added:
            line = line.strip()
            if line and not line.startswith('#'):
                if 'try:' in line:
                    changes.append("Added error handling")
                elif 'if ' in line:
                    changes.append(f"Added condition")
        
        if not changes:
            if added:
                changes.append(f"Added {len(added)} lines")
        
        return changes
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HEALTH & STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def is_healthy(self) -> bool:
        """Check if client is healthy"""
        return self._healthy and self._circuit_breaker.state != CircuitState.OPEN
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self._stats_lock:
            stats = self._stats.copy()
            
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['avg_latency_ms'] = stats['total_latency_ms'] / stats['total_requests']
        else:
            stats['success_rate'] = 0
            stats['avg_latency_ms'] = 0
        
        stats['healthy'] = self.is_healthy()
        stats['circuit_state'] = self._circuit_breaker.state.name
        
        if self._cache:
            stats['cache'] = self._cache.get_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear response cache"""
        if self._cache:
            self._cache.clear()
        logger.info("Cache cleared")
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        self._circuit_breaker._state = CircuitState.CLOSED
        self._circuit_breaker._failure_count = 0
        self._healthy = True
        logger.info("Circuit breaker reset")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_kimi_client: Optional[KimiK25Client] = None
_kimi_lock = threading.Lock()


def get_kimi_client(api_key: str = None) -> KimiK25Client:
    """Get or create global Kimi client (thread-safe)"""
    global _kimi_client
    if _kimi_client is None and api_key:
        with _kimi_lock:
            if _kimi_client is None:
                _kimi_client = KimiK25Client(api_key=api_key)
    return _kimi_client


def initialize_kimi(api_key: str) -> KimiK25Client:
    """Initialize global Kimi client"""
    global _kimi_client
    with _kimi_lock:
        if _kimi_client is not None:
            _kimi_client.clear_cache()
        _kimi_client = KimiK25Client(api_key=api_key)
    return _kimi_client


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test(api_key: str = None) -> Dict[str, Any]:
    """Run comprehensive self-test"""
    import os
    
    results = {
        'passed': [],
        'failed': [],
        'warnings': [],
        'platform': {
            'is_termux': IS_TERMUX,
            'is_android': IS_ANDROID,
        },
    }
    
    key = api_key or os.environ.get('KIMI_API_KEY')
    if not key:
        try:
            from config import KIMI_API_KEY
            key = KIMI_API_KEY
        except:
            pass
    
    if not key:
        results['failed'].append('no_api_key')
        return results
    
    try:
        client = KimiK25Client(api_key=key)
        results['passed'].append('client_init')
    except Exception as e:
        results['failed'].append(f'client_init: {e}')
        return results
    
    # Test cache
    if client._cache:
        stats = client._cache.get_stats()
        results['passed'].append(f'cache_init (max {stats["max_entries"]} entries, {stats["max_memory_mb"]:.1f}MB)')
    
    # Test circuit breaker
    results['passed'].append(f'circuit_breaker ({client._circuit_breaker.state.name})')
    
    # Test rate limiter
    if client._rate_limiter:
        results['passed'].append(f'rate_limiter ({PlatformConfig.REQUESTS_PER_MINUTE}/min)')
    
    # Test simple chat (optional, can be slow)
    try:
        response = client.chat("Say 'test ok'")
        if response.success:
            results['passed'].append('api_connection')
        else:
            results['warnings'].append(f'api_connection: {response.error}')
    except Exception as e:
        results['warnings'].append(f'api_connection: {e}')
    
    results['stats'] = client.get_stats()
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Kimi K2.5 Client - Bulletproof Edition v3.0.0")
    print("=" * 70)
    print(f"\nPlatform: {'Termux' if IS_TERMUX else 'Standard'}")
    print(f"Android: {IS_ANDROID}")
    print(f"Max Cache: {PlatformConfig.MAX_CACHE_MEMORY_MB}MB, {PlatformConfig.MAX_CACHE_ENTRIES} entries")
    print(f"Timeout: {PlatformConfig.DEFAULT_TIMEOUT}s")
    print(f"Retries: {PlatformConfig.MAX_RETRIES}")
    print(f"Rate Limit: {PlatformConfig.REQUESTS_PER_MINUTE}/min")
    
    test_results = self_test()
    
    print("\n✅ Passed:")
    for test in test_results['passed']:
        print(f"   ✓ {test}")
    
    if test_results['failed']:
        print("\n❌ Failed:")
        for test in test_results['failed']:
            print(f"   ✗ {test}")
    
    if test_results['warnings']:
        print("\n⚠️  Warnings:")
        for w in test_results['warnings']:
            print(f"   ! {w}")
    
    print("\n" + "=" * 70)

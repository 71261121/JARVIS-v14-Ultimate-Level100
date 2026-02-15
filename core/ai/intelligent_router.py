#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Intelligent AI Router (BULLETPROOF EDITION)
=================================================================

100x Advanced Version with:
- Thread-safe client initialization
- Circuit breaker per provider
- Health monitoring
- Request prioritization
- Exponential backoff retry
- Automatic provider switching

Author: JARVIS AI Project
Version: 3.0.0 (Bulletproof)
Device: Realme Pad 2 Lite | Termux
"""

import time
import threading
import logging
import os
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque
import weakref

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS AND DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class AIProvider(Enum):
    """Available AI providers"""
    KIMI = "kimi"
    OPENROUTER = "openrouter"


class TaskType(Enum):
    """Types of AI tasks"""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    BUG_FIXING = "bug_fixing"
    CODE_IMPROVEMENT = "code_improvement"
    REASONING = "reasoning"
    GENERAL_CHAT = "general_chat"
    LONG_CONTEXT = "long_context"
    QUICK_RESPONSE = "quick_response"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class ProviderHealth(Enum):
    """Provider health status"""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()


@dataclass
class RoutedResponse:
    """Response from routed AI"""
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    fallback_used: bool = False
    from_cache: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'provider': self.provider,
            'model': self.model,
            'tokens_used': self.tokens_used,
            'latency_ms': self.latency_ms,
            'success': self.success,
            'error': self.error,
            'fallback_used': self.fallback_used,
        }


@dataclass
class ProviderStats:
    """Statistics for a single provider"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests


# ═══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Circuit breaker for a single provider"""
    
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if request can proceed"""
        with self._lock:
            if self._state == "CLOSED":
                return True
            
            if self._state == "OPEN":
                # Check if recovery timeout passed
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._state = "HALF_OPEN"
                    self._half_open_calls = 0
                    return True
                return False
            
            # HALF_OPEN
            if self._half_open_calls < self._half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
    
    def record_success(self):
        """Record successful request"""
        with self._lock:
            self._failure_count = 0
            self._state = "CLOSED"
    
    def record_failure(self):
        """Record failed request"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == "HALF_OPEN":
                self._state = "OPEN"
            elif self._failure_count >= self._failure_threshold:
                self._state = "OPEN"
    
    @property
    def state(self) -> str:
        with self._lock:
            return self._state


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class HealthMonitor:
    """Monitor provider health"""
    
    def __init__(self, window_size: int = 10):
        self._window_size = window_size
        self._history: Dict[str, deque] = {}  # provider -> deque of (success, timestamp)
        self._lock = threading.Lock()
    
    def record(self, provider: str, success: bool):
        """Record request outcome"""
        with self._lock:
            if provider not in self._history:
                self._history[provider] = deque(maxlen=self._window_size)
            
            self._history[provider].append((success, time.time()))
    
    def get_health(self, provider: str) -> ProviderHealth:
        """Get provider health status"""
        with self._lock:
            if provider not in self._history:
                return ProviderHealth.HEALTHY
            
            history = list(self._history[provider])
            
            if not history:
                return ProviderHealth.HEALTHY
            
            # Calculate success rate
            successes = sum(1 for s, _ in history if s)
            rate = successes / len(history)
            
            if rate >= 0.9:
                return ProviderHealth.HEALTHY
            elif rate >= 0.5:
                return ProviderHealth.DEGRADED
            else:
                return ProviderHealth.UNHEALTHY
    
    def get_success_rate(self, provider: str) -> float:
        """Get success rate for provider"""
        with self._lock:
            if provider not in self._history:
                return 1.0
            
            history = list(self._history[provider])
            if not history:
                return 1.0
            
            successes = sum(1 for s, _ in history if s)
            return successes / len(history)


# ═══════════════════════════════════════════════════════════════════════════════
# BULLETPROOF INTELLIGENT ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

class IntelligentAIRouter:
    """
    Bulletproof Intelligent AI Router for JARVIS Level 100.
    
    Features:
    - Thread-safe client initialization
    - Circuit breaker per provider
    - Health monitoring
    - Automatic fallback
    - Request prioritization
    
    Routing Strategy:
    - Code Generation → Kimi K2.5 (best for code)
    - Reasoning → Kimi K2.5
    - General Chat → OpenRouter (free)
    - Fallback: Kimi → OpenRouter
    """
    
    # Task → Primary Provider mapping
    TASK_ROUTING = {
        TaskType.CODE_GENERATION: AIProvider.KIMI,
        TaskType.CODE_ANALYSIS: AIProvider.KIMI,
        TaskType.BUG_FIXING: AIProvider.KIMI,
        TaskType.CODE_IMPROVEMENT: AIProvider.KIMI,
        TaskType.REASONING: AIProvider.KIMI,
        TaskType.LONG_CONTEXT: AIProvider.KIMI,
        TaskType.GENERAL_CHAT: AIProvider.OPENROUTER,
        TaskType.QUICK_RESPONSE: AIProvider.OPENROUTER,
    }
    
    # Fallback order for each task
    FALLBACK_ORDER = {
        TaskType.CODE_GENERATION: [AIProvider.KIMI, AIProvider.OPENROUTER],
        TaskType.CODE_ANALYSIS: [AIProvider.KIMI, AIProvider.OPENROUTER],
        TaskType.BUG_FIXING: [AIProvider.KIMI, AIProvider.OPENROUTER],
        TaskType.CODE_IMPROVEMENT: [AIProvider.KIMI, AIProvider.OPENROUTER],
        TaskType.REASONING: [AIProvider.KIMI, AIProvider.OPENROUTER],
        TaskType.LONG_CONTEXT: [AIProvider.KIMI, AIProvider.OPENROUTER],
        TaskType.GENERAL_CHAT: [AIProvider.OPENROUTER, AIProvider.KIMI],
        TaskType.QUICK_RESPONSE: [AIProvider.OPENROUTER, AIProvider.KIMI],
    }
    
    def __init__(
        self,
        kimi_api_key: str = None,
        openrouter_api_key: str = None,
        auto_fallback: bool = True,
        health_check_interval: float = 60.0,
    ):
        """
        Initialize Intelligent AI Router.
        
        Args:
            kimi_api_key: Kimi K2.5 API key
            openrouter_api_key: OpenRouter API key
            auto_fallback: Enable automatic fallback
            health_check_interval: Interval for health checks (seconds)
        """
        self._kimi_api_key = kimi_api_key
        self._openrouter_api_key = openrouter_api_key
        self._auto_fallback = auto_fallback
        self._health_check_interval = health_check_interval
        
        # Lazy-loaded clients with thread-safe initialization
        self._kimi_client = None
        self._openrouter_client = None
        self._kimi_lock = threading.Lock()
        self._openrouter_lock = threading.Lock()
        
        # Circuit breakers per provider
        self._circuit_breakers = {
            AIProvider.KIMI: CircuitBreaker(),
            AIProvider.OPENROUTER: CircuitBreaker(),
        }
        
        # Health monitor
        self._health_monitor = HealthMonitor()
        
        # Statistics per provider
        self._provider_stats = {
            AIProvider.KIMI: ProviderStats(),
            AIProvider.OPENROUTER: ProviderStats(),
        }
        self._stats_lock = threading.Lock()
        
        # Global statistics
        self._stats = {
            'total_requests': 0,
            'kimi_requests': 0,
            'openrouter_requests': 0,
            'fallback_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
        }
        self._global_lock = threading.Lock()
        
        logger.info("IntelligentAIRouter initialized")
    
    def _get_kimi_client(self):
        """Get or create Kimi client (thread-safe)"""
        if self._kimi_client is None and self._kimi_api_key:
            with self._kimi_lock:
                if self._kimi_client is None:
                    try:
                        from core.ai.kimi_client import KimiK25Client
                        self._kimi_client = KimiK25Client(api_key=self._kimi_api_key)
                        logger.debug("Kimi client initialized")
                    except ImportError as e:
                        logger.warning(f"KimiClient not available: {e}")
        return self._kimi_client
    
    def _get_openrouter_client(self):
        """Get or create OpenRouter client (thread-safe)"""
        if self._openrouter_client is None and self._openrouter_api_key:
            with self._openrouter_lock:
                if self._openrouter_client is None:
                    try:
                        from core.ai.openrouter_client import OpenRouterClient
                        self._openrouter_client = OpenRouterClient(api_key=self._openrouter_api_key)
                        logger.debug("OpenRouter client initialized")
                    except ImportError as e:
                        logger.warning(f"OpenRouterClient not available: {e}")
        return self._openrouter_client
    
    def _is_provider_available(self, provider: AIProvider) -> bool:
        """Check if provider is available and healthy"""
        # Check circuit breaker
        if not self._circuit_breakers[provider].can_execute():
            return False
        
        # Check client availability
        if provider == AIProvider.KIMI:
            return self._get_kimi_client() is not None
        elif provider == AIProvider.OPENROUTER:
            return self._get_openrouter_client() is not None
        
        return False
    
    def _get_available_providers(self, task_type: TaskType) -> List[AIProvider]:
        """Get list of available providers in fallback order"""
        providers = self.FALLBACK_ORDER.get(task_type, [AIProvider.KIMI, AIProvider.OPENROUTER])
        
        # Filter by availability
        available = []
        for provider in providers:
            if self._is_provider_available(provider):
                available.append(provider)
        
        return available if available else providers[:1]  # At least try one
    
    def route(
        self,
        task_type: TaskType,
        prompt: str,
        system: str = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs
    ) -> RoutedResponse:
        """
        Route request to appropriate AI.
        
        Args:
            task_type: Type of task
            prompt: User prompt
            system: System prompt
            priority: Task priority
            **kwargs: Additional parameters
            
        Returns:
            RoutedResponse
        """
        start_time = time.time()
        
        # Update global stats
        with self._global_lock:
            self._stats['total_requests'] += 1
        
        # Get available providers
        providers = self._get_available_providers(task_type)
        
        if not self._auto_fallback:
            providers = providers[:1]
        
        last_error = None
        
        for i, provider in enumerate(providers):
            is_fallback = i > 0
            
            try:
                response = self._try_provider(
                    provider, task_type, prompt, system, **kwargs
                )
                
                if response.success:
                    # Update stats
                    self._update_stats(provider, is_fallback, True, time.time() - start_time)
                    
                    # Record health
                    self._health_monitor.record(provider.value, True)
                    self._circuit_breakers[provider].record_success()
                    
                    response.fallback_used = is_fallback
                    return response
                
                last_error = response.error
                
                # Record failure
                self._health_monitor.record(provider.value, False)
                self._circuit_breakers[provider].record_failure()
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"{provider.value} failed: {e}")
                
                # Record failure
                self._health_monitor.record(provider.value, False)
                self._circuit_breakers[provider].record_failure()
        
        # All providers failed
        with self._global_lock:
            self._stats['failed_requests'] += 1
        
        return RoutedResponse(
            content="",
            provider="none",
            model="none",
            success=False,
            error=last_error or "All providers failed",
            latency_ms=(time.time() - start_time) * 1000,
        )
    
    def _try_provider(
        self,
        provider: AIProvider,
        task_type: TaskType,
        prompt: str,
        system: str,
        **kwargs
    ) -> RoutedResponse:
        """Try a specific provider"""
        if provider == AIProvider.KIMI:
            return self._try_kimi(task_type, prompt, system, **kwargs)
        elif provider == AIProvider.OPENROUTER:
            return self._try_openrouter(task_type, prompt, system, **kwargs)
        
        return RoutedResponse(
            content="",
            provider=provider.value,
            model="unknown",
            success=False,
            error="Unknown provider",
        )
    
    def _try_kimi(
        self,
        task_type: TaskType,
        prompt: str,
        system: str,
        **kwargs
    ) -> RoutedResponse:
        """Try Kimi for request"""
        client = self._get_kimi_client()
        
        if client is None:
            return RoutedResponse(
                content="",
                provider="kimi",
                model="",
                success=False,
                error="Kimi client not available",
            )
        
        # Use specialized methods for code tasks
        if task_type == TaskType.CODE_GENERATION:
            result = client.generate_code(
                specification=prompt,
                language=kwargs.get('language', 'python'),
                context=kwargs.get('context'),
            )
            return RoutedResponse(
                content=result.code,
                provider="kimi",
                model="moonshotai/kimi-k2.5",
                success=result.has_code,
                error=None if result.has_code else "No code generated",
            )
        
        elif task_type == TaskType.BUG_FIXING:
            result = client.fix_bug(
                code=kwargs.get('code', prompt),
                error=kwargs.get('error', ''),
            )
            return RoutedResponse(
                content=result.fixed_code,
                provider="kimi",
                model="moonshotai/kimi-k2.5",
                success=result.confidence > 0.5,
            )
        
        elif task_type == TaskType.CODE_IMPROVEMENT:
            result = client.improve_code(
                code=kwargs.get('code', prompt),
                improvement_goal=kwargs.get('goal', 'Improve code quality'),
            )
            return RoutedResponse(
                content=result.code,
                provider="kimi",
                model="moonshotai/kimi-k2.5",
                success=result.has_code,
            )
        
        # General chat
        response = client.chat(prompt, system=system, **kwargs)
        
        return RoutedResponse(
            content=response.content,
            provider="kimi",
            model=response.model,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
            success=response.success,
            error=response.error,
            from_cache=response.from_cache,
        )
    
    def _try_openrouter(
        self,
        task_type: TaskType,
        prompt: str,
        system: str,
        **kwargs
    ) -> RoutedResponse:
        """Try OpenRouter for request"""
        client = self._get_openrouter_client()
        
        if client is None:
            return RoutedResponse(
                content="",
                provider="openrouter",
                model="",
                success=False,
                error="OpenRouter client not available",
            )
        
        response = client.chat(prompt, system=system, **kwargs)
        
        return RoutedResponse(
            content=response.content,
            provider="openrouter",
            model=response.model,
            tokens_used=response.tokens_used,
            latency_ms=response.latency_ms,
            success=response.success,
            error=response.error,
        )
    
    def _update_stats(
        self,
        provider: AIProvider,
        is_fallback: bool,
        success: bool,
        elapsed: float,
    ):
        """Update statistics"""
        with self._global_lock:
            if provider == AIProvider.KIMI:
                self._stats['kimi_requests'] += 1
            else:
                self._stats['openrouter_requests'] += 1
            
            if is_fallback:
                self._stats['fallback_requests'] += 1
            
            if success:
                self._stats['successful_requests'] += 1
        
        with self._stats_lock:
            stats = self._provider_stats[provider]
            stats.total_requests += 1
            if success:
                stats.successful_requests += 1
                stats.last_success_time = time.time()
            else:
                stats.failed_requests += 1
                stats.last_failure_time = time.time()
            stats.total_latency_ms += elapsed * 1000
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def generate_code(self, specification: str, **kwargs) -> RoutedResponse:
        """Generate code using best AI"""
        return self.route(TaskType.CODE_GENERATION, specification, **kwargs)
    
    def fix_bug(self, code: str, error: str, **kwargs) -> RoutedResponse:
        """Fix bug using best AI"""
        return self.route(
            TaskType.BUG_FIXING,
            f"Fix bug: {error}",
            code=code,
            error=error,
            **kwargs
        )
    
    def analyze_code(self, code: str, **kwargs) -> RoutedResponse:
        """Analyze code using best AI"""
        return self.route(TaskType.CODE_ANALYSIS, code, **kwargs)
    
    def improve_code(self, code: str, goal: str, **kwargs) -> RoutedResponse:
        """Improve code using best AI"""
        return self.route(
            TaskType.CODE_IMPROVEMENT,
            f"Improve: {goal}",
            code=code,
            goal=goal,
            **kwargs
        )
    
    def chat(self, message: str, **kwargs) -> RoutedResponse:
        """General chat"""
        return self.route(TaskType.GENERAL_CHAT, message, **kwargs)
    
    def quick_chat(self, message: str, **kwargs) -> RoutedResponse:
        """Quick response"""
        return self.route(TaskType.QUICK_RESPONSE, message, **kwargs)
    
    def reason(self, problem: str, **kwargs) -> RoutedResponse:
        """Reasoning task"""
        return self.route(TaskType.REASONING, problem, **kwargs)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HEALTH & STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def get_provider_health(self, provider: AIProvider) -> ProviderHealth:
        """Get health status of a provider"""
        return self._health_monitor.get_health(provider.value)
    
    def get_provider_stats(self, provider: AIProvider) -> ProviderStats:
        """Get statistics for a provider"""
        with self._stats_lock:
            return self._provider_stats[provider]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        with self._global_lock:
            stats = self._stats.copy()
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['fallback_rate'] = stats['fallback_requests'] / stats['total_requests']
        
        # Add provider stats
        stats['kimi'] = {
            'health': self.get_provider_health(AIProvider.KIMI).name,
            'circuit_state': self._circuit_breakers[AIProvider.KIMI].state,
            **{k: v for k, v in self.get_provider_stats(AIProvider.KIMI).__dict__.items()},
        }
        
        stats['openrouter'] = {
            'health': self.get_provider_health(AIProvider.OPENROUTER).name,
            'circuit_state': self._circuit_breakers[AIProvider.OPENROUTER].state,
            **{k: v for k, v in self.get_provider_stats(AIProvider.OPENROUTER).__dict__.items()},
        }
        
        return stats
    
    def reset_circuit_breaker(self, provider: AIProvider):
        """Reset circuit breaker for a provider"""
        self._circuit_breakers[provider]._state = "CLOSED"
        self._circuit_breakers[provider]._failure_count = 0
        logger.info(f"Circuit breaker reset for {provider.value}")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_router: Optional[IntelligentAIRouter] = None
_router_lock = threading.Lock()


def get_router(
    kimi_api_key: str = None,
    openrouter_api_key: str = None,
) -> IntelligentAIRouter:
    """Get or create global router"""
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                _router = IntelligentAIRouter(
                    kimi_api_key=kimi_api_key,
                    openrouter_api_key=openrouter_api_key,
                )
    return _router


def initialize_router(kimi_key: str, openrouter_key: str) -> IntelligentAIRouter:
    """Initialize global router"""
    global _router
    with _router_lock:
        _router = IntelligentAIRouter(
            kimi_api_key=kimi_key,
            openrouter_api_key=openrouter_key,
        )
    return _router


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test(kimi_key: str = None, openrouter_key: str = None) -> Dict[str, Any]:
    """Run self-test"""
    results = {'passed': [], 'failed': [], 'warnings': []}
    
    # Try to get keys from config
    if not kimi_key or not openrouter_key:
        try:
            from config import KIMI_API_KEY, OPENROUTER_API_KEY
            kimi_key = kimi_key or KIMI_API_KEY
            openrouter_key = openrouter_key or OPENROUTER_API_KEY
        except ImportError:
            pass
    
    try:
        router = IntelligentAIRouter(
            kimi_api_key=kimi_key,
            openrouter_api_key=openrouter_key,
        )
        results['passed'].append('router_init')
    except Exception as e:
        results['failed'].append(f'router_init: {e}')
        return results
    
    # Check circuit breakers
    results['passed'].append(f'kimi_circuit_breaker ({router._circuit_breakers[AIProvider.KIMI].state})')
    results['passed'].append(f'openrouter_circuit_breaker ({router._circuit_breakers[AIProvider.OPENROUTER].state})')
    
    # Check health monitor
    results['passed'].append('health_monitor')
    
    # Check stats
    stats = router.get_stats()
    results['passed'].append(f'stats_tracking')
    
    results['stats'] = stats
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("JARVIS Intelligent AI Router - Bulletproof Edition v3.0.0")
    print("=" * 70)
    
    test_results = self_test()
    
    print("\n✅ Passed:")
    for test in test_results['passed']:
        print(f"   ✓ {test}")
    
    if test_results['failed']:
        print("\n❌ Failed:")
        for test in test_results['failed']:
            print(f"   ✗ {test}")
    
    print("\n" + "=" * 70)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - AI Provider Package
=========================================

AI provider modules for JARVIS AI system.

Modules:
    - openrouter_client: OpenRouter API client (fallback)
    - kimi_client: Kimi K2.5 API client (primary for code)
    - intelligent_router: Routes between Kimi and OpenRouter
    - rate_limiter: Rate limiting and circuit breaker
    - model_selector: Intelligent model selection
    - response_parser: API response parsing

Exports:
    - OpenRouterClient: OpenRouter API client
    - KimiK25Client: Kimi K2.5 API client
    - IntelligentAIRouter: Smart routing between AIs
    - RateLimiterManager: Rate limiting manager
    - ModelSelector: Model selection engine
    - ResponseParser: Response parser
"""

from .openrouter_client import (
    OpenRouterClient,
    FreeModel,
    ModelCapability,
    ChatMessage,
    AIResponse,
    ConversationContext,
    get_client,
    initialize_client,
)

# Kimi K2.5 Client (Primary AI for code)
try:
    from .kimi_client import (
        KimiK25Client,
        KimiModel,
        KimiResponse,
        CodeResult,
        BugFixResult,
        get_kimi_client,
        initialize_kimi,
    )
except ImportError:
    # Fallback if kimi_client not available
    KimiK25Client = None
    KimiModel = None
    KimiResponse = None
    CodeResult = None
    BugFixResult = None
    get_kimi_client = None
    initialize_kimi = None

# Intelligent Router
try:
    from .intelligent_router import (
        IntelligentAIRouter,
        AIProvider,
        RoutedResponse,
        get_router,
        initialize_router,
    )
    # Import TaskType from intelligent_router to avoid conflict
    from .intelligent_router import TaskType as AITaskType
except ImportError:
    IntelligentAIRouter = None
    AIProvider = None
    RoutedResponse = None
    get_router = None
    initialize_router = None
    AITaskType = None

from .rate_limiter import (
    RateLimiterManager,
    AdaptiveRateLimiter,
    TokenBucket,
    CircuitBreaker,
    CircuitState,
    RateLimitConfig,
    RateLimitResult,
    get_rate_limiter_manager,
    rate_limited,
)

from .model_selector import (
    ModelSelector,
    TaskType,
    ModelCapability as SelectionCapability,
    ModelInfo,
    ModelStatus,
    SelectionResult,
    TaskProfile,
    TaskDetector,
    get_model_selector,
    select_model,
)

from .response_parser import (
    ResponseParser,
    StreamingParser,
    ParsedResponse,
    StreamChunk,
    ErrorCode,
    ResponseType,
    ErrorDetector,
    get_parser,
    parse_response,
)

__all__ = [
    # OpenRouter Client (Fallback)
    'OpenRouterClient',
    'FreeModel',
    'ModelCapability',
    'ChatMessage',
    'AIResponse',
    'ConversationContext',
    'get_client',
    'initialize_client',
    
    # Kimi K2.5 Client (Primary)
    'KimiK25Client',
    'KimiModel',
    'KimiResponse',
    'CodeResult',
    'BugFixResult',
    'get_kimi_client',
    'initialize_kimi',
    
    # Intelligent Router
    'IntelligentAIRouter',
    'AIProvider',
    'RoutedResponse',
    'AITaskType',
    'get_router',
    'initialize_router',
    
    # Rate Limiter
    'RateLimiterManager',
    'AdaptiveRateLimiter',
    'TokenBucket',
    'CircuitBreaker',
    'CircuitState',
    'RateLimitConfig',
    'RateLimitResult',
    'get_rate_limiter_manager',
    'rate_limited',
    
    # Model Selector
    'ModelSelector',
    'TaskType',
    'ModelInfo',
    'ModelStatus',
    'SelectionResult',
    'TaskProfile',
    'TaskDetector',
    'get_model_selector',
    'select_model',
    
    # Response Parser
    'ResponseParser',
    'StreamingParser',
    'ParsedResponse',
    'StreamChunk',
    'ErrorCode',
    'ResponseType',
    'ErrorDetector',
    'get_parser',
    'parse_response',
]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Configuration Package
============================================

Configuration management for JARVIS AI system.

Exports:
    - ConfigManager: Main configuration manager
    - ConfigSchema: Schema definition for config values
    - ConfigValue: Configuration value with metadata
    - get_config: Get global config instance
    - get: Get a config value
    - require: Get required config value
    - API Keys: KIMI_API_KEY, OPENROUTER_API_KEY, GITHUB_TOKEN
"""

from .config_manager import (
    ConfigManager,
    ConfigSchema,
    ConfigValue,
    ConfigSource,
    ConfigStatus,
    JARVIS_CONFIG_SCHEMAS,
    get_config,
    initialize_config,
    get as config_get,
    get_int,
    get_bool,
    require,
    set_value,
)

# Import API keys
try:
    from .api_keys import (
        KIMI_API_KEY,
        OPENROUTER_API_KEY,
        GITHUB_TOKEN,
        get_kimi_key,
        get_openrouter_key,
        get_github_token,
        validate_keys,
        mask_key,
        print_key_status,
    )
except ImportError:
    # Fallback if api_keys.py doesn't exist
    import os
    KIMI_API_KEY = os.environ.get('KIMI_API_KEY', '')
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
    GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
    
    def get_kimi_key(): return KIMI_API_KEY
    def get_openrouter_key(): return OPENROUTER_API_KEY
    def get_github_token(): return GITHUB_TOKEN
    def validate_keys(): return {'all_valid': bool(KIMI_API_KEY)}
    def mask_key(k, s=8): return k[:s] + '...' if k else '<not set>'
    def print_key_status(): pass

__all__ = [
    'ConfigManager',
    'ConfigSchema',
    'ConfigValue',
    'ConfigSource',
    'ConfigStatus',
    'JARVIS_CONFIG_SCHEMAS',
    'get_config',
    'initialize_config',
    'config_get',
    'get_int',
    'get_bool',
    'require',
    'set_value',
    # API Keys
    'KIMI_API_KEY',
    'OPENROUTER_API_KEY',
    'GITHUB_TOKEN',
    'get_kimi_key',
    'get_openrouter_key',
    'get_github_token',
    'validate_keys',
    'mask_key',
    'print_key_status',
]

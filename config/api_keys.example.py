#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - API Keys Configuration Template
======================================================

Copy this file to 'api_keys.py' and fill in your actual API keys.

IMPORTANT: 
- Never commit api_keys.py to public repositories
- Use environment variables for production
- Keep your keys secure

Author: JARVIS AI Project
"""

import os
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# API KEYS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Kimi K2.5 via NVIDIA NIM (Primary AI)
# Get your key from: https://build.nvidia.com/moonshotai/kimi-k2.5
KIMI_API_KEY = os.environ.get(
    'KIMI_API_KEY',
    'YOUR_KIMI_API_KEY_HERE'  # Replace with your nvapi-... key
)

# OpenRouter API (Fallback AI)
# Get your key from: https://openrouter.ai/keys
OPENROUTER_API_KEY = os.environ.get(
    'OPENROUTER_API_KEY',
    'YOUR_OPENROUTER_API_KEY_HERE'  # Replace with your sk-or-v1-... key
)

# GitHub Token (for repo operations)
# Get your token from: https://github.com/settings/tokens
GITHUB_TOKEN = os.environ.get(
    'GITHUB_TOKEN',
    'YOUR_GITHUB_TOKEN_HERE'  # Replace with your ghp_... token
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_kimi_key() -> Optional[str]:
    """Get Kimi API key from environment or config."""
    return KIMI_API_KEY if KIMI_API_KEY != 'YOUR_KIMI_API_KEY_HERE' else None


def get_openrouter_key() -> Optional[str]:
    """Get OpenRouter API key from environment or config."""
    return OPENROUTER_API_KEY if OPENROUTER_API_KEY != 'YOUR_OPENROUTER_API_KEY_HERE' else None


def get_github_token() -> Optional[str]:
    """Get GitHub token from environment or config."""
    return GITHUB_TOKEN if GITHUB_TOKEN != 'YOUR_GITHUB_TOKEN_HERE' else None


def validate_keys() -> dict:
    """Validate which API keys are configured."""
    return {
        'kimi': bool(get_kimi_key()),
        'openrouter': bool(get_openrouter_key()),
        'github': bool(get_github_token())
    }


def get_missing_keys() -> list:
    """Get list of missing API keys."""
    validation = validate_keys()
    missing = [k for k, v in validation.items() if not v]
    return missing


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("JARVIS API Keys Configuration Check")
    print("=" * 60)
    
    validation = validate_keys()
    for key, configured in validation.items():
        status = "✓ Configured" if configured else "✗ Missing"
        print(f"  {key.upper()}: {status}")
    
    missing = get_missing_keys()
    if missing:
        print(f"\n⚠️  Missing keys: {', '.join(missing)}")
        print("   Set environment variables or update api_keys.py")
    else:
        print("\n✓ All API keys are configured!")

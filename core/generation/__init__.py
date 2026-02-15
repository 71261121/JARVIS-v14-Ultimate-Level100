#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Code Generation Package
==============================================

AI-powered code generation, improvement, and bug fixing modules.

Modules:
    - code_generator: Generate code from specifications
    - code_improver: Improve existing code
    - bug_fixer: Automatically fix bugs in code

Author: JARVIS AI Project
Version: 1.0.0
"""

from .code_generator import (
    CodeGenerator,
    GenerationResult,
    GenerationConfig,
    get_code_generator,
)

__all__ = [
    'CodeGenerator',
    'GenerationResult',
    'GenerationConfig',
    'get_code_generator',
]

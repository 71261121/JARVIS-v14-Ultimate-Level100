#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 1 Layer 1 Testing
==============================================

Layer 1: Syntax & Import Validation
- Python syntax check
- Import validation
- Module structure validation
- Class definition check
- Function signature validation

Author: JARVIS AI Project
"""

import sys
import os
import ast
import traceback
import importlib.util
from pathlib import Path
from typing import Dict, List, Any

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Layer1Test:
    """Layer 1: Syntax and Import Validation"""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'details': {}
        }
    
    def run_all(self) -> Dict[str, Any]:
        """Run all Layer 1 tests"""
        print("=" * 70)
        print("LAYER 1: SYNTAX & IMPORT VALIDATION")
        print("=" * 70)
        
        # Test files to validate
        test_files = [
            'core/ai/kimi_client.py',
            'core/ai/intelligent_router.py',
            'core/generation/code_generator.py',
            'config/api_keys.py',
            'config/__init__.py',
        ]
        
        print("\n[1/5] Python Syntax Validation...")
        self._test_syntax(test_files)
        
        print("\n[2/5] Import Validation...")
        self._test_imports(test_files)
        
        print("\n[3/5] Class Definition Check...")
        self._test_class_definitions(test_files)
        
        print("\n[4/5] Function Signature Validation...")
        self._test_function_signatures(test_files)
        
        print("\n[5/5] Module Integration Test...")
        self._test_module_integration()
        
        return self.results
    
    def _test_syntax(self, files: List[str]):
        """Test Python syntax"""
        for file_path in files:
            full_path = PROJECT_ROOT / file_path
            
            if not full_path.exists():
                self.results['failed'].append(f"File not found: {file_path}")
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                # Parse with AST
                ast.parse(source)
                
                # Also try compile
                compile(source, str(full_path), 'exec')
                
                self.results['passed'].append(f"Syntax OK: {file_path}")
                self.results['details'][file_path] = {'syntax': 'valid'}
                
            except SyntaxError as e:
                self.results['failed'].append(f"Syntax error in {file_path}: Line {e.lineno}: {e.msg}")
                self.results['details'][file_path] = {'syntax': f'error: {e}'}
            except Exception as e:
                self.results['failed'].append(f"Error parsing {file_path}: {e}")
    
    def _test_imports(self, files: List[str]):
        """Test imports can be resolved"""
        for file_path in files:
            full_path = PROJECT_ROOT / file_path
            
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                
                imports = []
                import_errors = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        for alias in node.names:
                            full_import = f"{module}.{alias.name}" if module else alias.name
                            imports.append(full_import)
                
                # Check standard library imports
                stdlib_ok = True
                for imp in imports:
                    # Skip relative imports and project modules
                    if imp.startswith('core.') or imp.startswith('config.'):
                        continue
                    
                    # Check if it's a standard library module
                    try:
                        # Try to find spec
                        spec = importlib.util.find_spec(imp.split('.')[0])
                        if spec is None and not self._is_stdlib(imp.split('.')[0]):
                            import_errors.append(f"Cannot resolve: {imp}")
                    except:
                        pass
                
                if import_errors:
                    self.results['warnings'].append(f"Import issues in {file_path}: {import_errors[:3]}")
                else:
                    self.results['passed'].append(f"Imports OK: {file_path}")
                
                self.results['details'][file_path]['imports'] = len(imports)
                self.results['details'][file_path]['import_errors'] = import_errors
                
            except Exception as e:
                self.results['warnings'].append(f"Import check failed for {file_path}: {e}")
    
    def _is_stdlib(self, module: str) -> bool:
        """Check if module is in standard library"""
        stdlib_modules = {
            'time', 'json', 'hashlib', 'threading', 'logging', 're', 'sys', 'os',
            'gc', 'weakref', 'datetime', 'collections', 'functools', 'contextlib',
            'typing', 'dataclasses', 'enum', 'pathlib', 'traceback', 'importlib',
            'urllib', 'urllib.request', 'urllib.error',
        }
        return module in stdlib_modules
    
    def _test_class_definitions(self, files: List[str]):
        """Test class definitions"""
        expected_classes = {
            'core/ai/kimi_client.py': ['KimiK25Client', 'MemoryBoundedLRUCache', 'TokenBucket', 'CircuitBreaker', 'CancellationToken'],
            'core/ai/intelligent_router.py': ['IntelligentAIRouter', 'CircuitBreaker', 'HealthMonitor'],
            'core/generation/code_generator.py': ['CodeGenerator'],
            'config/api_keys.py': [],
        }
        
        for file_path, expected in expected_classes.items():
            full_path = PROJECT_ROOT / file_path
            
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                
                found_classes = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        found_classes.append(node.name)
                
                missing = set(expected) - set(found_classes)
                
                if missing:
                    self.results['failed'].append(f"Missing classes in {file_path}: {missing}")
                else:
                    self.results['passed'].append(f"All classes defined: {file_path}")
                
                self.results['details'][file_path]['classes'] = found_classes
                
            except Exception as e:
                self.results['failed'].append(f"Class check failed for {file_path}: {e}")
    
    def _test_function_signatures(self, files: List[str]):
        """Test function signatures"""
        critical_functions = {
            'core/ai/kimi_client.py': ['chat', 'generate_code', 'fix_bug', 'improve_code', 'analyze_code'],
            'core/ai/intelligent_router.py': ['route', 'generate_code', 'fix_bug', 'chat'],
        }
        
        for file_path, expected_funcs in critical_functions.items():
            full_path = PROJECT_ROOT / file_path
            
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                
                found_methods = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        found_methods.append(node.name)
                
                missing = set(expected_funcs) - set(found_methods)
                
                if missing:
                    self.results['failed'].append(f"Missing methods in {file_path}: {missing}")
                else:
                    self.results['passed'].append(f"All critical methods defined: {file_path}")
                
                self.results['details'][file_path]['methods'] = [m for m in found_methods if not m.startswith('_')]
                
            except Exception as e:
                self.results['failed'].append(f"Method check failed for {file_path}: {e}")
    
    def _test_module_integration(self):
        """Test modules can be imported together"""
        print("\n  Testing module imports...")
        
        try:
            # Test config import
            from config import KIMI_API_KEY, OPENROUTER_API_KEY
            self.results['passed'].append("Config module imports OK")
        except ImportError as e:
            self.results['failed'].append(f"Config import failed: {e}")
        
        try:
            # Test Kimi client import
            from core.ai.kimi_client import KimiK25Client, KimiModel
            self.results['passed'].append("Kimi client imports OK")
        except ImportError as e:
            self.results['failed'].append(f"Kimi client import failed: {e}")
        
        try:
            # Test Router import
            from core.ai.intelligent_router import IntelligentAIRouter, TaskType
            self.results['passed'].append("Router imports OK")
        except ImportError as e:
            self.results['failed'].append(f"Router import failed: {e}")
        
        try:
            # Test Code Generator import
            from core.generation.code_generator import CodeGenerator
            self.results['passed'].append("Code Generator imports OK")
        except ImportError as e:
            self.results['warnings'].append(f"Code Generator import: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("LAYER 1 TEST RESULTS")
        print("=" * 70)
        
        print(f"\n✅ Passed: {len(self.results['passed'])}")
        for item in self.results['passed'][-10:]:
            print(f"   ✓ {item}")
        
        if self.results['failed']:
            print(f"\n❌ Failed: {len(self.results['failed'])}")
            for item in self.results['failed']:
                print(f"   ✗ {item}")
        
        if self.results['warnings']:
            print(f"\n⚠️  Warnings: {len(self.results['warnings'])}")
            for item in self.results['warnings'][:5]:
                print(f"   ! {item}")
        
        total = len(self.results['passed']) + len(self.results['failed'])
        success_rate = len(self.results['passed']) / total * 100 if total > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if not self.results['failed']:
            print("🎉 ALL LAYER 1 TESTS PASSED!")
        else:
            print("⚠️  Some tests failed - review above")
        print("=" * 70)


if __name__ == "__main__":
    tester = Layer1Test()
    tester.run_all()
    tester.print_summary()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Kimi K2.5 via NVIDIA NIM API
=================================

Verifies the connection to Kimi K2.5 through NVIDIA NIM.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test configuration - Use environment variable
KIMI_API_KEY = os.environ.get('KIMI_API_KEY', '')

if not KIMI_API_KEY:
    print("ERROR: KIMI_API_KEY environment variable not set!")
    print("Set it with: export KIMI_API_KEY='your-key-here'")
    sys.exit(1)


def test_kimi_basic():
    """Test basic Kimi connection"""
    print("\n" + "="*60)
    print("Test 1: Basic Kimi K2.5 Connection")
    print("="*60)
    
    try:
        from core.ai.kimi_client import KimiK25Client, KimiModel
        
        client = KimiK25Client(api_key=KIMI_API_KEY)
        
        print(f"✓ Client initialized")
        print(f"  API URL: {client.API_URL}")
        print(f"  Model: {client._default_model.value}")
        
        # Simple test
        response = client.chat("Say 'Hello JARVIS' exactly.")
        
        if response.success:
            print(f"✓ Connection successful!")
            print(f"  Response: {response.content[:100]}...")
            print(f"  Latency: {response.latency_ms:.0f}ms")
            print(f"  Tokens: {response.tokens_used}")
            return True
        else:
            print(f"✗ Connection failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kimi_code_generation():
    """Test code generation"""
    print("\n" + "="*60)
    print("Test 2: Code Generation")
    print("="*60)
    
    try:
        from core.ai.kimi_client import KimiK25Client
        
        client = KimiK25Client(api_key=KIMI_API_KEY)
        
        result = client.generate_code(
            "Create a function to calculate fibonacci numbers",
            language="python"
        )
        
        if result.has_code:
            print("✓ Code generated successfully!")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Code preview:")
            print("-" * 40)
            print(result.code[:300] + "..." if len(result.code) > 300 else result.code)
            print("-" * 40)
            return True
        else:
            print(f"✗ No code generated")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kimi_bug_fix():
    """Test bug fixing"""
    print("\n" + "="*60)
    print("Test 3: Bug Fixing")
    print("="*60)
    
    try:
        from core.ai.kimi_client import KimiK25Client
        
        client = KimiK25Client(api_key=KIMI_API_KEY)
        
        buggy_code = '''
def divide(a, b):
    return a / b
'''
        
        result = client.fix_bug(buggy_code, "ZeroDivisionError when b=0")
        
        if result.confidence > 0.5:
            print("✓ Bug fixed successfully!")
            print(f"  Changes made: {result.changes_made}")
            print(f"  Fixed code:")
            print("-" * 40)
            print(result.fixed_code[:300])
            print("-" * 40)
            return True
        else:
            print(f"✗ Bug fix low confidence: {result.confidence}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intelligent_router():
    """Test intelligent router with Kimi"""
    print("\n" + "="*60)
    print("Test 4: Intelligent Router")
    print("="*60)
    
    try:
        from core.ai.intelligent_router import IntelligentAIRouter, TaskType
        
        router = IntelligentAIRouter(
            kimi_api_key=KIMI_API_KEY,
            openrouter_api_key=None,  # Only test Kimi
        )
        
        # Test code generation routing
        response = router.generate_code(
            "Function to add two numbers",
            language="python"
        )
        
        if response.success:
            print("✓ Router works!")
            print(f"  Provider: {response.provider}")
            print(f"  Model: {response.model}")
            print(f"  Code preview: {response.content[:100]}...")
            return True
        else:
            print(f"✗ Router failed: {response.error}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("JARVIS Kimi K2.5 (NVIDIA NIM) Test Suite")
    print("="*60)
    
    results = {
        'basic_connection': test_kimi_basic(),
        'code_generation': test_kimi_code_generation(),
        'bug_fixing': test_kimi_bug_fix(),
        'intelligent_router': test_intelligent_router(),
    }
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Kimi K2.5 is ready for JARVIS!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Phase 1 Layer 2 Testing
==============================================

Layer 2: Logic & API Functionality
- Client initialization
- Cache functionality
- Circuit breaker logic
- Rate limiter logic
- API connection test

Author: JARVIS AI Project
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Layer2Test:
    """Layer 2: Logic and API Functionality Tests"""
    
    def __init__(self):
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': [],
            'details': {}
        }
        
        # Load API keys
        self.kimi_key = None
        self.openrouter_key = None
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from config"""
        try:
            from config import KIMI_API_KEY, OPENROUTER_API_KEY
            self.kimi_key = KIMI_API_KEY
            self.openrouter_key = OPENROUTER_API_KEY
        except ImportError:
            self.kimi_key = os.environ.get('KIMI_API_KEY')
            self.openrouter_key = os.environ.get('OPENROUTER_API_KEY')
    
    def run_all(self) -> dict:
        """Run all Layer 2 tests"""
        print("=" * 70)
        print("LAYER 2: LOGIC & API FUNCTIONALITY")
        print("=" * 70)
        
        print("\n[1/6] Testing Client Initialization...")
        self._test_client_init()
        
        print("\n[2/6] Testing Memory-Bounded Cache...")
        self._test_cache()
        
        print("\n[3/6] Testing Circuit Breaker...")
        self._test_circuit_breaker()
        
        print("\n[4/6] Testing Rate Limiter...")
        self._test_rate_limiter()
        
        print("\n[5/6] Testing Thread Safety...")
        self._test_thread_safety()
        
        print("\n[6/6] Testing API Connection...")
        self._test_api_connection()
        
        return self.results
    
    def _test_client_init(self):
        """Test client initialization"""
        try:
            from core.ai.kimi_client import KimiK25Client, PlatformConfig
            
            # Test without API key (should fail)
            try:
                client = KimiK25Client(api_key="")
                self.results['failed'].append("Client init should fail without API key")
            except ValueError:
                self.results['passed'].append("Client correctly rejects empty API key")
            
            # Test with valid key
            if self.kimi_key:
                client = KimiK25Client(api_key=self.kimi_key)
                
                # Check initialization
                assert client._timeout == PlatformConfig.DEFAULT_TIMEOUT
                assert client._cache is not None
                assert client._rate_limiter is not None
                assert client._circuit_breaker is not None
                
                self.results['passed'].append("Client initialized with all components")
                
                # Check stats
                stats = client.get_stats()
                assert 'total_requests' in stats
                assert 'success_rate' in stats
                
                self.results['passed'].append("Client stats tracking works")
            else:
                self.results['warnings'].append("No Kimi API key for init test")
                
        except Exception as e:
            self.results['failed'].append(f"Client init test failed: {e}")
    
    def _test_cache(self):
        """Test memory-bounded LRU cache"""
        try:
            from core.ai.kimi_client import MemoryBoundedLRUCache
            
            cache = MemoryBoundedLRUCache(max_entries=10, max_memory_mb=1)
            
            # Test set and get
            cache.set("key1", ("value1", time.time()), 100)
            result = cache.get("key1")
            
            if result and result[0][0] == "value1":
                self.results['passed'].append("Cache set/get works")
            else:
                self.results['failed'].append("Cache set/get failed")
            
            # Test LRU eviction
            for i in range(15):
                cache.set(f"key{i}", (f"value{i}", time.time()), 100)
            
            stats = cache.get_stats()
            
            if stats['evictions'] > 0:
                self.results['passed'].append(f"Cache LRU eviction works ({stats['evictions']} evictions)")
            else:
                self.results['warnings'].append("Cache eviction not triggered")
            
            # Test stats
            if stats['entries'] <= 10:
                self.results['passed'].append(f"Cache respects max_entries ({stats['entries']} entries)")
            else:
                self.results['failed'].append(f"Cache exceeded max_entries ({stats['entries']} entries)")
            
            # Test clear
            cache.clear()
            stats = cache.get_stats()
            
            if stats['entries'] == 0:
                self.results['passed'].append("Cache clear works")
            else:
                self.results['failed'].append("Cache clear failed")
                
        except Exception as e:
            self.results['failed'].append(f"Cache test failed: {e}")
    
    def _test_circuit_breaker(self):
        """Test circuit breaker logic"""
        try:
            from core.ai.kimi_client import CircuitBreaker, CircuitState
            
            cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
            
            # Initial state
            if cb.state == CircuitState.CLOSED:
                self.results['passed'].append("Circuit breaker starts CLOSED")
            else:
                self.results['failed'].append("Circuit breaker should start CLOSED")
            
            # Can execute in CLOSED state
            if cb.can_execute():
                self.results['passed'].append("Circuit breaker allows requests when CLOSED")
            else:
                self.results['failed'].append("Circuit breaker should allow requests when CLOSED")
            
            # Record failures
            for _ in range(3):
                cb.record_failure()
            
            if cb.state == CircuitState.OPEN:
                self.results['passed'].append("Circuit breaker opens after failures")
            else:
                self.results['failed'].append("Circuit breaker should open after failures")
            
            # Should not allow requests when OPEN
            if not cb.can_execute():
                self.results['passed'].append("Circuit breaker blocks requests when OPEN")
            else:
                self.results['failed'].append("Circuit breaker should block when OPEN")
            
            # Wait for recovery
            time.sleep(1.1)
            
            if cb.can_execute():
                self.results['passed'].append("Circuit breaker allows test request after recovery timeout")
            else:
                self.results['failed'].append("Circuit breaker should allow test after recovery")
            
            # Record success to close
            cb.record_success()
            
            if cb.state == CircuitState.CLOSED:
                self.results['passed'].append("Circuit breaker closes after success")
            else:
                self.results['failed'].append("Circuit breaker should close after success")
                
        except Exception as e:
            self.results['failed'].append(f"Circuit breaker test failed: {e}")
    
    def _test_rate_limiter(self):
        """Test token bucket rate limiter"""
        try:
            from core.ai.kimi_client import TokenBucket
            
            # Create fast limiter for testing
            limiter = TokenBucket(rate_per_minute=60)  # 1 per second
            
            # First request should succeed immediately
            start = time.time()
            result = limiter.acquire(timeout=1.0)
            elapsed = time.time() - start
            
            if result and elapsed < 0.1:
                self.results['passed'].append("Rate limiter allows first request immediately")
            else:
                self.results['warnings'].append(f"Rate limiter first request took {elapsed:.3f}s")
            
            # Test stats
            self.results['details']['rate_limiter'] = {
                'rate': limiter._rate,
                'tokens': limiter._tokens,
            }
            
        except Exception as e:
            self.results['failed'].append(f"Rate limiter test failed: {e}")
    
    def _test_thread_safety(self):
        """Test thread safety of components"""
        try:
            from core.ai.kimi_client import MemoryBoundedLRUCache
            
            cache = MemoryBoundedLRUCache(max_entries=100, max_memory_mb=1)
            
            errors = []
            
            def writer_thread(thread_id):
                try:
                    for i in range(100):
                        cache.set(f"thread{thread_id}_key{i}", (f"value{i}", time.time()), 50)
                except Exception as e:
                    errors.append(str(e))
            
            def reader_thread(thread_id):
                try:
                    for i in range(100):
                        cache.get(f"thread{thread_id}_key{i}")
                except Exception as e:
                    errors.append(str(e))
            
            # Create threads
            threads = []
            for i in range(5):
                threads.append(threading.Thread(target=writer_thread, args=(i,)))
                threads.append(threading.Thread(target=reader_thread, args=(i,)))
            
            # Start all threads
            for t in threads:
                t.start()
            
            # Wait for completion
            for t in threads:
                t.join(timeout=10)
            
            if not errors:
                self.results['passed'].append("Thread safety test passed (10 threads, 1000 ops each)")
            else:
                self.results['failed'].append(f"Thread safety errors: {errors[:3]}")
            
            stats = cache.get_stats()
            self.results['details']['thread_test'] = {
                'final_entries': stats['entries'],
                'evictions': stats['evictions'],
            }
            
        except Exception as e:
            self.results['failed'].append(f"Thread safety test failed: {e}")
    
    def _test_api_connection(self):
        """Test actual API connection"""
        if not self.kimi_key:
            self.results['warnings'].append("No Kimi API key for connection test")
            return
        
        try:
            from core.ai.kimi_client import KimiK25Client
            
            client = KimiK25Client(api_key=self.kimi_key)
            
            # Quick test - just say hi
            response = client.chat("Say 'OK'", max_tokens=10)
            
            if response.success:
                self.results['passed'].append(f"API connection successful")
                self.results['passed'].append(f"  Model: {response.model}")
                self.results['passed'].append(f"  Latency: {response.latency_ms:.0f}ms")
                self.results['passed'].append(f"  Tokens: {response.tokens_used}")
            else:
                self.results['failed'].append(f"API connection failed: {response.error}")
            
            # Check health
            if client.is_healthy():
                self.results['passed'].append("Client health check: HEALTHY")
            else:
                self.results['warnings'].append("Client health check: UNHEALTHY")
            
            # Get stats
            stats = client.get_stats()
            self.results['details']['api_test'] = {
                'success_rate': stats['success_rate'],
                'avg_latency_ms': stats['avg_latency_ms'],
                'circuit_state': stats['circuit_state'],
            }
            
        except Exception as e:
            self.results['failed'].append(f"API connection test failed: {e}")
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("LAYER 2 TEST RESULTS")
        print("=" * 70)
        
        print(f"\n✅ Passed: {len(self.results['passed'])}")
        for item in self.results['passed'][-15:]:
            print(f"   ✓ {item}")
        
        if self.results['failed']:
            print(f"\n❌ Failed: {len(self.results['failed'])}")
            for item in self.results['failed']:
                print(f"   ✗ {item}")
        
        if self.results['warnings']:
            print(f"\n⚠️  Warnings: {len(self.results['warnings'])}")
            for item in self.results['warnings']:
                print(f"   ! {item}")
        
        total = len(self.results['passed']) + len(self.results['failed'])
        success_rate = len(self.results['passed']) / total * 100 if total > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if not self.results['failed']:
            print("🎉 ALL LAYER 2 TESTS PASSED!")
        else:
            print("⚠️  Some tests failed - review above")
        print("=" * 70)


if __name__ == "__main__":
    tester = Layer2Test()
    tester.run_all()
    tester.print_summary()

#!/usr/bin/env python3
"""
Termux AI Client - OpenRouter Implementation
One request. One response. No retry. No fallback.
"""

import os
import time
import tracemalloc

# RUNTIME ASSERTION
assert 'OPENROUTER_API_KEY' in os.environ, 'HALT: OPENROUTER_API_KEY not set'
API_KEY = os.environ['OPENROUTER_API_KEY']
assert API_KEY.startswith('sk-or-'), f'HALT: Invalid API key format'

import httpx

# MEMORY TRACKING
tracemalloc.start()

# BUILD REQUEST
url = 'https://openrouter.ai/api/v1/chat/completions'
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
}
payload = {
    'model': 'openrouter/auto',
    'messages': [{'role': 'user', 'content': 'Say exactly: test ok'}],
    'max_tokens': 10,
}

# EXECUTE REQUEST
start = time.time()
try:
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, headers=headers, json=payload)
    
    assert response.status_code == 200, f'HALT: HTTP {response.status_code}'
    
    result = response.json()
    content = result['choices'][0]['message']['content']
    latency = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f'Response: {content}')
    print(f'Latency: {latency:.2f}s')
    print(f'Memory: {peak // 1024} KB')
    print('RESULT: PASS')
    
except httpx.TimeoutException:
    print('HALT: Request timeout')
    print('RESULT: FAIL')
except httpx.NetworkError as e:
    print(f'HALT: Network error - {e}')
    print('RESULT: FAIL')
except KeyError:
    print('HALT: Unexpected response format')
    print('RESULT: FAIL')
except AssertionError as e:
    print(str(e))
    print('RESULT: FAIL')
except Exception as e:
    print(f'HALT: {type(e).__name__} - {e}')
    print('RESULT: FAIL')

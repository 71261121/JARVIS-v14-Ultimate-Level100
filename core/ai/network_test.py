#!/usr/bin/env python3
"""Network connectivity test."""

import httpx

try:
    with httpx.Client(timeout=10.0) as client:
        response = client.get('https://httpbin.org/get')
    if response.status_code == 200:
        print('Network: OK')
        print('RESULT: PASS')
    else:
        print(f'Network: HTTP {response.status_code}')
        print('RESULT: FAIL')
except Exception as e:
    print(f'Network: {type(e).__name__}')
    print('RESULT: FAIL')

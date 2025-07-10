#!/usr/bin/env python3
"""
Test Type-Based Engine Directly

Debug the engine initialization to see what's failing.
"""

import sys
sys.path.append('.')

from api.type_based_app import TypeBasedMABWiserEngine

try:
    print("🔍 Testing TypeBasedMABWiserEngine initialization...")
    engine = TypeBasedMABWiserEngine()
    print('✅ Engine created successfully!')
    
    stats = engine.get_expert_stats()
    print(f'📊 Stats: {stats}')
    
except Exception as e:
    print(f'❌ Engine failed: {e}')
    import traceback
    traceback.print_exc() 
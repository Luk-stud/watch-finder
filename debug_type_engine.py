#!/usr/bin/env python3
"""
Debug Type-Based Engine Import

Test importing the type-based engine step by step to find the issue.
"""

print("🔍 DEBUGGING TYPE-BASED ENGINE IMPORT")

# Test 1: Basic imports
print("\n1️⃣ Testing basic imports...")
try:
    import os
    import pickle
    import numpy as np
    from collections import defaultdict
    print("✅ Basic imports OK")
except Exception as e:
    print(f"❌ Basic imports failed: {e}")
    exit(1)

# Test 2: MABWiser import
print("\n2️⃣ Testing MABWiser import...")
try:
    from mabwiser.mab import MAB, LearningPolicy
    print("✅ MABWiser import OK")
    MABWISER_AVAILABLE = True
except ImportError as e:
    print(f"❌ MABWiser import failed: {e}")
    MABWISER_AVAILABLE = False

print(f"   MABWISER_AVAILABLE = {MABWISER_AVAILABLE}")

# Test 3: Direct execution of class definition
print("\n3️⃣ Testing class definition...")
if MABWISER_AVAILABLE:
    try:
        class TestTypeBasedMABWiserEngine:
            def __init__(self):
                print("✅ Class init called")
                self.test = "OK"
        
        engine = TestTypeBasedMABWiserEngine()
        print(f"✅ Test class created: {engine.test}")
    except Exception as e:
        print(f"❌ Test class failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ Skipping class test - MABWiser not available")

# Test 4: Module import
print("\n4️⃣ Testing module import...")
try:
    import models.mabwiser_type_engine as mte
    print(f"✅ Module imported")
    print(f"   Module file: {mte.__file__}")
    print(f"   Module dir: {[x for x in dir(mte) if not x.startswith('_')]}")
except Exception as e:
    print(f"❌ Module import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Execute file directly
print("\n5️⃣ Testing direct file execution...")
try:
    exec(open('models/mabwiser_type_engine.py').read())
    print("✅ Direct execution completed")
except Exception as e:
    print(f"❌ Direct execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\n🎉 Debug complete!") 
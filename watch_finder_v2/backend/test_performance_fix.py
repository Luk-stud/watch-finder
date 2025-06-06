#!/usr/bin/env python3
"""
Test Performance Improvements
"""
import time
import numpy as np
from models.fast_linucb_engine import FastLinUCBEngine

def test_performance():
    print("🚀 Testing Fast LinUCB Performance...")
    
    # Initialize engine
    engine = FastLinUCBEngine(
        alpha=0.15,
        batch_size=5,
        max_experts=4,
        data_dir='data'
    )
    
    # Create a test session
    session_id = "test_session"
    engine.create_session(session_id)
    
    print(f"📊 Available watches: {len(engine.available_watches)}")
    
    # Test 1: Initial recommendations (no experts)
    start_time = time.time()
    recs = engine.get_recommendations(session_id)
    random_time = time.time() - start_time
    print(f"✅ Random recommendations: {random_time:.3f}s")
    
    # Simulate some likes to create experts
    for i, rec in enumerate(recs[:2]):
        engine.update(session_id, rec['watch_id'], 1.0)
    
    print(f"👤 Created {len(engine.session_experts[session_id])} experts")
    
    # Test 2: UCB recommendations (with experts)
    start_time = time.time()
    recs = engine.get_recommendations(session_id)
    ucb_time = time.time() - start_time
    print(f"⚡ UCB recommendations: {ucb_time:.3f}s")
    
    # Test 3: Multiple recommendation calls (should use cached inversions)
    times = []
    for i in range(3):
        start_time = time.time()
        recs = engine.get_recommendations(session_id)
        times.append(time.time() - start_time)
    
    avg_cached_time = np.mean(times)
    print(f"💾 Cached recommendations (avg): {avg_cached_time:.3f}s")
    
    print(f"\n📈 Performance Summary:")
    print(f"  • Random: {random_time:.3f}s")
    print(f"  • First UCB: {ucb_time:.3f}s") 
    print(f"  • Cached UCB: {avg_cached_time:.3f}s")
    print(f"  • Improvement: {ucb_time/avg_cached_time:.1f}x faster")

if __name__ == "__main__":
    test_performance() 
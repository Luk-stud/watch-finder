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
    
    # Test 1: Session creation speed
    start_time = time.time()
    session_id2 = "test_session_2"
    engine.create_session(session_id2)
    session_create_time = time.time() - start_time
    print(f"🚀 Session creation: {session_create_time:.6f}s")
    
    # Test 2: Initial recommendations (no experts)
    start_time = time.time()
    recs = engine.get_recommendations(session_id)
    random_time = time.time() - start_time
    print(f"✅ Random recommendations: {random_time:.3f}s")
    
    # Test 3: Feedback processing speed
    feedback_times = []
    for i, rec in enumerate(recs[:2]):
        start_time = time.time()
        engine.update(session_id, rec['watch_id'], 1.0)
        feedback_time = time.time() - start_time
        feedback_times.append(feedback_time)
        print(f"💖 Feedback {i+1}: {feedback_time:.3f}s")
    
    print(f"👤 Created {len(engine.session_experts[session_id])} experts")
    
    # Test 4: UCB recommendations (with experts)
    start_time = time.time()
    recs = engine.get_recommendations(session_id)
    ucb_time = time.time() - start_time
    print(f"⚡ UCB recommendations: {ucb_time:.3f}s")
    
    # Test 5: Multiple recommendation calls (should use cached inversions)
    times = []
    for i in range(3):
        start_time = time.time()
        recs = engine.get_recommendations(session_id)
        times.append(time.time() - start_time)
    
    avg_cached_time = np.mean(times)
    print(f"💾 Cached recommendations (avg): {avg_cached_time:.3f}s")
    
    avg_feedback_time = np.mean(feedback_times)
    
    print(f"\n📈 Performance Summary:")
    print(f"  • Session creation: {session_create_time:.6f}s")
    print(f"  • Random recommendations: {random_time:.3f}s")
    print(f"  • Feedback processing: {avg_feedback_time:.3f}s")
    print(f"  • First UCB: {ucb_time:.3f}s") 
    print(f"  • Cached UCB: {avg_cached_time:.3f}s")
    if avg_cached_time > 0:
        print(f"  • UCB improvement: {ucb_time/avg_cached_time:.1f}x faster with cache")
    
    print(f"\n🎯 Target performance achieved:")
    print(f"  • Session creation: {'✅' if session_create_time < 0.001 else '❌'} (<1ms)")
    print(f"  • Random recs: {'✅' if random_time < 0.1 else '❌'} (<100ms)")
    print(f"  • Feedback: {'✅' if avg_feedback_time < 0.1 else '❌'} (<100ms)")
    print(f"  • UCB recs: {'✅' if avg_cached_time < 0.5 else '❌'} (<500ms)")

if __name__ == "__main__":
    test_performance() 
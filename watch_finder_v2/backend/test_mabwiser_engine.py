#!/usr/bin/env python3
"""
Comprehensive Test for MABWiser Engine

Tests the MABWiser Thompson Sampling engine with precomputed embeddings.
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, List, Any

# Add backend models to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from mabwiser_engine import MABWiserEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_engine_initialization():
    """Test MABWiser engine initialization."""
    print("\n" + "="*80)
    print("🧪 TESTING MABWISER ENGINE INITIALIZATION")
    print("="*80)
    
    try:
        start_time = time.time()
        engine = MABWiserEngine(
            batch_size=5,
            max_experts=3,
            similarity_threshold=0.85
        )
        init_time = time.time() - start_time
        
        print(f"✅ Engine initialized in {init_time:.2f}s")
        print(f"   • Available watches: {len(engine.available_watches)}")
        print(f"   • Embedding dimension: {engine.dim}")
        print(f"   • Max experts: {engine.max_experts}")
        print(f"   • Batch size: {engine.batch_size}")
        
        return engine
        
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_session_management(engine: MABWiserEngine):
    """Test session creation and management."""
    print("\n" + "="*80)
    print("🧪 TESTING SESSION MANAGEMENT")
    print("="*80)
    
    try:
        # Create sessions
        session_ids = ["test_session_1", "test_session_2"]
        
        for session_id in session_ids:
            engine.create_session(session_id)
            print(f"✅ Created session: {session_id}")
        
        # Check session state
        for session_id in session_ids:
            if session_id in engine.session_experts:
                experts = len(engine.session_experts[session_id])
                likes = len(engine.session_liked_watches[session_id])
                shown = len(engine.session_shown_watches[session_id])
                print(f"   • {session_id}: {experts} experts, {likes} likes, {shown} shown")
            else:
                print(f"❌ Session {session_id} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Session management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cold_start_recommendations(engine: MABWiserEngine):
    """Test cold start recommendations (no experts yet)."""
    print("\n" + "="*80)
    print("🧪 TESTING COLD START RECOMMENDATIONS")
    print("="*80)
    
    try:
        session_id = "cold_start_session"
        engine.create_session(session_id)
        
        # Get recommendations without any feedback
        recommendations = engine.get_recommendations(session_id)
        
        print(f"✅ Got {len(recommendations)} cold start recommendations")
        
        for i, rec in enumerate(recommendations[:3]):  # Show first 3
            watch_id = rec.get('watch_id')
            confidence = rec.get('confidence', 0)
            algorithm = rec.get('algorithm', 'Unknown')
            brand = rec.get('brand', 'Unknown')
            model = rec.get('model', 'Unknown')
            
            print(f"   {i+1}. Watch {watch_id}: {brand} {model}")
            print(f"      Confidence: {confidence:.3f}, Algorithm: {algorithm}")
        
        return len(recommendations) > 0
        
    except Exception as e:
        print(f"❌ Cold start recommendations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feedback_and_expert_creation(engine: MABWiserEngine):
    """Test feedback processing and expert creation."""
    print("\n" + "="*80)
    print("🧪 TESTING FEEDBACK AND EXPERT CREATION")
    print("="*80)
    
    try:
        session_id = "feedback_session"
        engine.create_session(session_id)
        
        # Get initial recommendations
        recommendations = engine.get_recommendations(session_id)
        print(f"📝 Got {len(recommendations)} initial recommendations")
        
        # Simulate positive feedback on first few watches
        positive_feedback_watches = recommendations[:3]
        
        for i, rec in enumerate(positive_feedback_watches):
            watch_id = rec.get('watch_id')
            if watch_id:
                print(f"👍 Giving positive feedback for watch {watch_id}")
                engine.update(session_id, watch_id, 1.0)
                
                # Check expert creation
                experts_count = len(engine.session_experts[session_id])
                print(f"   Experts after feedback {i+1}: {experts_count}")
        
        # Simulate negative feedback
        if len(recommendations) > 3:
            negative_watch_id = recommendations[3].get('watch_id')
            if negative_watch_id:
                print(f"👎 Giving negative feedback for watch {negative_watch_id}")
                engine.update(session_id, negative_watch_id, 0.0)
        
        # Check final expert count
        final_experts = len(engine.session_experts[session_id])
        print(f"✅ Final expert count: {final_experts}")
        
        # Check expert stats
        stats = engine.get_expert_stats()
        print(f"✅ Expert statistics:")
        print(f"   • Total experts: {stats['total_experts']}")
        print(f"   • Total sessions: {stats['total_sessions']}")
        print(f"   • Avg likes per expert: {stats['avg_likes_per_expert']:.2f}")
        
        return final_experts > 0
        
    except Exception as e:
        print(f"❌ Feedback and expert creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_expert_recommendations(engine: MABWiserEngine):
    """Test recommendations from trained experts."""
    print("\n" + "="*80)
    print("🧪 TESTING EXPERT RECOMMENDATIONS")
    print("="*80)
    
    try:
        session_id = "expert_session"
        engine.create_session(session_id)
        
        # Step 1: Create experts through feedback
        print("🔧 Creating experts through feedback...")
        initial_recs = engine.get_recommendations(session_id)
        
        # Give positive feedback to create experts
        for i in range(min(5, len(initial_recs))):
            watch_id = initial_recs[i].get('watch_id')
            if watch_id:
                engine.update(session_id, watch_id, 1.0)
        
        experts_count = len(engine.session_experts[session_id])
        print(f"   Created {experts_count} experts")
        
        # Step 2: Get expert-based recommendations
        print("🎯 Getting expert-based recommendations...")
        expert_recs = engine.get_recommendations(session_id)
        
        print(f"✅ Got {len(expert_recs)} expert recommendations")
        
        # Analyze recommendation sources
        algorithm_counts = {}
        for rec in expert_recs:
            algorithm = rec.get('algorithm', 'Unknown')
            algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
        
        print(f"📊 Recommendation sources:")
        for algorithm, count in algorithm_counts.items():
            print(f"   • {algorithm}: {count} recommendations")
        
        # Show sample expert recommendations
        print(f"🔍 Sample expert recommendations:")
        for i, rec in enumerate(expert_recs[:3]):
            watch_id = rec.get('watch_id')
            confidence = rec.get('confidence', 0)
            algorithm = rec.get('algorithm', 'Unknown')
            brand = rec.get('brand', 'Unknown')
            model = rec.get('model', 'Unknown')
            
            print(f"   {i+1}. Watch {watch_id}: {brand} {model}")
            print(f"      Confidence: {confidence:.3f}, Algorithm: {algorithm}")
        
        return len(expert_recs) > 0
        
    except Exception as e:
        print(f"❌ Expert recommendations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_sessions(engine: MABWiserEngine):
    """Test multiple sessions working independently."""
    print("\n" + "="*80)
    print("🧪 TESTING MULTIPLE SESSIONS")
    print("="*80)
    
    try:
        # Create multiple sessions
        sessions = ["session_A", "session_B", "session_C"]
        
        for session_id in sessions:
            engine.create_session(session_id)
            
            # Get recommendations for each session
            recs = engine.get_recommendations(session_id)
            print(f"📝 Session {session_id}: {len(recs)} recommendations")
            
            # Give different feedback patterns
            if session_id == "session_A":
                # Positive feedback on first 2
                for i in range(min(2, len(recs))):
                    watch_id = recs[i].get('watch_id')
                    if watch_id:
                        engine.update(session_id, watch_id, 1.0)
            elif session_id == "session_B":
                # Mixed feedback
                for i, rec in enumerate(recs[:3]):
                    watch_id = rec.get('watch_id')
                    if watch_id:
                        reward = 1.0 if i % 2 == 0 else 0.0
                        engine.update(session_id, watch_id, reward)
            # session_C gets no feedback (cold state)
        
        # Check session independence
        print(f"🔍 Session independence check:")
        for session_id in sessions:
            experts = len(engine.session_experts.get(session_id, []))
            likes = len(engine.session_liked_watches.get(session_id, []))
            shown = len(engine.session_shown_watches.get(session_id, set()))
            print(f"   • {session_id}: {experts} experts, {likes} likes, {shown} shown")
        
        # Get new recommendations for each session
        print(f"🎯 New recommendations after feedback:")
        for session_id in sessions:
            new_recs = engine.get_recommendations(session_id)
            algorithm_types = set(rec.get('algorithm', 'Unknown') for rec in new_recs)
            print(f"   • {session_id}: {len(new_recs)} recs, algorithms: {algorithm_types}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multiple sessions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_and_timing(engine: MABWiserEngine):
    """Test performance and timing characteristics."""
    print("\n" + "="*80)
    print("🧪 TESTING PERFORMANCE AND TIMING")
    print("="*80)
    
    try:
        session_id = "performance_session"
        engine.create_session(session_id)
        
        # Test recommendation speed
        times = []
        for i in range(10):
            start_time = time.time()
            recs = engine.get_recommendations(session_id)
            end_time = time.time()
            times.append(end_time - start_time)
            
            # Add some feedback occasionally
            if i % 3 == 0 and recs:
                watch_id = recs[0].get('watch_id')
                if watch_id:
                    engine.update(session_id, watch_id, 1.0)
        
        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        
        print(f"⚡ Recommendation timing (10 calls):")
        print(f"   • Average: {avg_time:.3f}s")
        print(f"   • Max: {max_time:.3f}s")
        print(f"   • Min: {min_time:.3f}s")
        
        # Test update speed
        update_times = []
        recs = engine.get_recommendations(session_id)
        
        for i in range(5):
            if i < len(recs):
                watch_id = recs[i].get('watch_id')
                if watch_id:
                    start_time = time.time()
                    engine.update(session_id, watch_id, 1.0)
                    end_time = time.time()
                    update_times.append(end_time - start_time)
        
        if update_times:
            avg_update_time = np.mean(update_times)
            print(f"⚡ Update timing (5 calls):")
            print(f"   • Average: {avg_update_time:.3f}s")
        
        print(f"✅ Performance test completed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all MABWiser engine tests."""
    print("🚀 COMPREHENSIVE MABWISER ENGINE TEST")
    print("="*80)
    
    # Test results
    results = {}
    
    # 1. Engine Initialization
    engine = test_engine_initialization()
    results['initialization'] = engine is not None
    
    if not engine:
        print("❌ Cannot continue tests without engine initialization")
        return results
    
    # 2. Session Management
    results['session_management'] = test_session_management(engine)
    
    # 3. Cold Start Recommendations
    results['cold_start'] = test_cold_start_recommendations(engine)
    
    # 4. Feedback and Expert Creation
    results['feedback_experts'] = test_feedback_and_expert_creation(engine)
    
    # 5. Expert Recommendations
    results['expert_recommendations'] = test_expert_recommendations(engine)
    
    # 6. Multiple Sessions
    results['multiple_sessions'] = test_multiple_sessions(engine)
    
    # 7. Performance and Timing
    results['performance'] = test_performance_and_timing(engine)
    
    # Cleanup
    engine.shutdown()
    
    return results

def main():
    """Main test function."""
    print("🧪 MABWISER ENGINE COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # Run all tests
    results = run_all_tests()
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
        if success:
            passed += 1
    
    print(f"\n🏆 OVERALL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! MABWiser engine is working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the details above.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 
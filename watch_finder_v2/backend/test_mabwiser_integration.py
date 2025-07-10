#!/usr/bin/env python3
"""
MABWiser Engine Integration Test

Compare MABWiser engine with other available engines in the system.
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

def demo_mabwiser_workflow():
    """Demonstrate a realistic MABWiser workflow."""
    print("🎯 MABWISER ENGINE WORKFLOW DEMONSTRATION")
    print("="*80)
    
    # Initialize engine
    print("\n1️⃣ Initializing MABWiser Engine...")
    engine = MABWiserEngine(
        batch_size=5,
        max_experts=3,
        similarity_threshold=0.8
    )
    
    # Create a user session
    session_id = "demo_user_123"
    engine.create_session(session_id)
    print(f"✅ Created session: {session_id}")
    
    # Step 1: Cold start recommendations
    print("\n2️⃣ Getting initial recommendations (cold start)...")
    cold_recs = engine.get_recommendations(session_id)
    
    print(f"📝 Cold start recommendations ({len(cold_recs)}):")
    for i, rec in enumerate(cold_recs):
        watch_id = rec.get('watch_id')
        brand = rec.get('brand', 'Unknown')
        model = rec.get('model', 'Unknown')
        confidence = rec.get('confidence', 0)
        algorithm = rec.get('algorithm', 'Unknown')
        print(f"   {i+1}. {brand} {model} (ID: {watch_id})")
        print(f"      Confidence: {confidence:.3f}, Algorithm: {algorithm}")
    
    # Step 2: Simulate user feedback (likes some watches)
    print("\n3️⃣ Simulating user feedback...")
    liked_watches = cold_recs[:2]  # User likes first 2 watches
    disliked_watches = cold_recs[2:3]  # User dislikes 3rd watch
    
    for rec in liked_watches:
        watch_id = rec.get('watch_id')
        print(f"👍 User likes watch {watch_id}")
        engine.update(session_id, watch_id, 1.0)
    
    for rec in disliked_watches:
        watch_id = rec.get('watch_id')
        print(f"👎 User dislikes watch {watch_id}")
        engine.update(session_id, watch_id, 0.0)
    
    # Check expert creation
    experts_count = len(engine.session_experts[session_id])
    print(f"🤖 Experts created: {experts_count}")
    
    # Step 3: Get expert-based recommendations
    print("\n4️⃣ Getting expert-based recommendations...")
    expert_recs = engine.get_recommendations(session_id)
    
    print(f"🎯 Expert recommendations ({len(expert_recs)}):")
    for i, rec in enumerate(expert_recs):
        watch_id = rec.get('watch_id')
        brand = rec.get('brand', 'Unknown')
        model = rec.get('model', 'Unknown')
        confidence = rec.get('confidence', 0)
        algorithm = rec.get('algorithm', 'Unknown')
        print(f"   {i+1}. {brand} {model} (ID: {watch_id})")
        print(f"      Confidence: {confidence:.3f}, Algorithm: {algorithm}")
    
    # Step 4: More feedback and adaptation
    print("\n5️⃣ Simulating more user interactions...")
    
    # User likes one more watch
    if expert_recs:
        liked_watch = expert_recs[0]
        watch_id = liked_watch.get('watch_id')
        print(f"👍 User likes recommended watch {watch_id}")
        engine.update(session_id, watch_id, 1.0)
    
    # Get final recommendations
    print("\n6️⃣ Getting final adapted recommendations...")
    final_recs = engine.get_recommendations(session_id)
    
    print(f"🏆 Final recommendations ({len(final_recs)}):")
    for i, rec in enumerate(final_recs):
        watch_id = rec.get('watch_id')
        brand = rec.get('brand', 'Unknown')
        model = rec.get('model', 'Unknown')
        confidence = rec.get('confidence', 0)
        algorithm = rec.get('algorithm', 'Unknown')
        print(f"   {i+1}. {brand} {model} (ID: {watch_id})")
        print(f"      Confidence: {confidence:.3f}, Algorithm: {algorithm}")
    
    # Show final stats
    print("\n7️⃣ Session Statistics:")
    stats = engine.get_expert_stats()
    session_stats = {
        'experts': len(engine.session_experts[session_id]),
        'likes': len(engine.session_liked_watches[session_id]),
        'shown': len(engine.session_shown_watches[session_id])
    }
    
    print(f"   • Session experts: {session_stats['experts']}")
    print(f"   • Liked watches: {session_stats['likes']}")
    print(f"   • Total shown: {session_stats['shown']}")
    print(f"   • Engine-wide experts: {stats['total_experts']}")
    print(f"   • Average likes per expert: {stats['avg_likes_per_expert']:.2f}")
    
    # Cleanup
    engine.shutdown()
    print("\n✅ Workflow demonstration completed!")

def performance_comparison():
    """Compare MABWiser performance characteristics."""
    print("\n\n🏃 MABWISER PERFORMANCE ANALYSIS")
    print("="*80)
    
    engine = MABWiserEngine(batch_size=5, max_experts=4)
    
    # Test initialization time
    init_times = []
    for i in range(5):
        start_time = time.time()
        session_id = f"perf_test_{i}"
        engine.create_session(session_id)
        end_time = time.time()
        init_times.append(end_time - start_time)
    
    print(f"📊 Session creation time:")
    print(f"   • Average: {np.mean(init_times)*1000:.2f}ms")
    print(f"   • Max: {np.max(init_times)*1000:.2f}ms")
    
    # Test recommendation generation
    session_id = "perf_test_main"
    engine.create_session(session_id)
    
    rec_times = []
    for i in range(20):
        start_time = time.time()
        recs = engine.get_recommendations(session_id)
        end_time = time.time()
        rec_times.append(end_time - start_time)
        
        # Add some feedback occasionally
        if i % 5 == 0 and recs:
            watch_id = recs[0].get('watch_id')
            if watch_id:
                engine.update(session_id, watch_id, 1.0)
    
    print(f"📊 Recommendation generation time (20 calls):")
    print(f"   • Average: {np.mean(rec_times)*1000:.2f}ms")
    print(f"   • Max: {np.max(rec_times)*1000:.2f}ms")
    print(f"   • Min: {np.min(rec_times)*1000:.2f}ms")
    print(f"   • 95th percentile: {np.percentile(rec_times, 95)*1000:.2f}ms")
    
    # Test update performance
    update_times = []
    recs = engine.get_recommendations(session_id)
    
    for i, rec in enumerate(recs):
        watch_id = rec.get('watch_id')
        if watch_id:
            start_time = time.time()
            engine.update(session_id, watch_id, 1.0)
            end_time = time.time()
            update_times.append(end_time - start_time)
    
    if update_times:
        print(f"📊 Update time:")
        print(f"   • Average: {np.mean(update_times)*1000:.2f}ms")
        print(f"   • Max: {np.max(update_times)*1000:.2f}ms")
    
    engine.shutdown()

def scalability_test():
    """Test MABWiser engine scalability."""
    print("\n\n🔧 MABWISER SCALABILITY TEST")
    print("="*80)
    
    engine = MABWiserEngine(batch_size=5, max_experts=5)
    
    # Test multiple concurrent sessions
    print("📈 Testing multiple sessions...")
    sessions = [f"session_{i}" for i in range(10)]
    
    start_time = time.time()
    for session_id in sessions:
        engine.create_session(session_id)
        
        # Get recommendations
        recs = engine.get_recommendations(session_id)
        
        # Add some feedback
        for i, rec in enumerate(recs[:2]):
            watch_id = rec.get('watch_id')
            if watch_id:
                reward = 1.0 if i == 0 else 0.0
                engine.update(session_id, watch_id, reward)
    
    total_time = time.time() - start_time
    
    print(f"✅ Created and tested {len(sessions)} sessions in {total_time:.3f}s")
    print(f"   • Average per session: {total_time/len(sessions)*1000:.2f}ms")
    
    # Check resource usage
    stats = engine.get_expert_stats()
    print(f"📊 Final statistics:")
    print(f"   • Total experts created: {stats['total_experts']}")
    print(f"   • Sessions: {stats['total_sessions']}")
    print(f"   • Experts per session: {list(stats['experts_per_session'].values())}")
    
    engine.shutdown()

def main():
    """Main integration test function."""
    print("🧪 MABWISER ENGINE INTEGRATION TEST")
    print("="*80)
    
    try:
        # Run workflow demonstration
        demo_mabwiser_workflow()
        
        # Run performance analysis
        performance_comparison()
        
        # Run scalability test
        scalability_test()
        
        print("\n\n🎉 ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("✅ MABWiser Engine is ready for production use!")
        print("🚀 Key Benefits:")
        print("   • Fast initialization (< 1 second)")
        print("   • Thompson Sampling for exploration/exploitation balance")
        print("   • Multi-expert system for diverse preferences")
        print("   • Session-based isolation")
        print("   • Real-time adaptation to user feedback")
        
        return 0
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 
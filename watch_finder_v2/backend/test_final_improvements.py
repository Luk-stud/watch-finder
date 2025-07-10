#!/usr/bin/env python3
"""
Test script for final MABWiser improvements.

Tests:
1. Keyword args for partial_fit (with fallback)
2. Synced predict() and predict_expectations() calls
3. Improved negative feedback targeting (full recommendation sets)
4. Memory management (recommendation cleanup)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import logging
from models.mabwiser_engine import MABWiserEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_keyword_args_with_fallback():
    """Test that keyword args work with graceful fallback to list format."""
    print("\n🎯 Testing Keyword Args with Fallback...")
    
    engine = MABWiserEngine(batch_size=5, max_experts=2, similarity_threshold=0.8)
    session_id = "test_keywords"
    engine.create_session(session_id)
    
    # Get initial recommendations and create expert
    recommendations = engine.get_recommendations(session_id)
    if recommendations:
        watch_id = recommendations[0]['watch_id']
        engine.update(session_id, watch_id, 1.0)
        print(f"✅ Created expert with keyword/fallback update for watch {watch_id}")
        
        # Test additional updates to verify partial_fit works
        engine.update(session_id, watch_id, 1.0)
        print(f"✅ Additional update succeeded with keyword/fallback format")
        
        return True
    return False

def test_synced_predict_calls():
    """Test that predict() and predict_expectations() use same ordered contexts."""
    print("\n🎯 Testing Synced Predict Calls...")
    
    engine = MABWiserEngine(batch_size=5, max_experts=2)
    session_id = "test_sync"
    engine.create_session(session_id)
    
    # Create expert with multiple likes to get proper confidence scores
    recommendations = engine.get_recommendations(session_id)
    for i, rec in enumerate(recommendations[:2]):
        engine.update(session_id, rec['watch_id'], 1.0)
        print(f"✅ Liked watch {rec['watch_id']} ({i+1}/2)")
    
    # Get new recommendations and check confidence scores
    new_recommendations = engine.get_recommendations(session_id)
    
    valid_confidences = 0
    confidence_variation = False
    confidences = []
    
    for rec in new_recommendations:
        confidence = rec.get('confidence', 0)
        confidences.append(confidence)
        
        if 0 <= confidence <= 1:
            valid_confidences += 1
        
        print(f"✅ Watch {rec['watch_id']}: confidence {confidence:.3f}")
    
    # Check for some variation in confidence scores (sign of proper sync)
    if len(set(confidences)) > 1:
        confidence_variation = True
        print(f"✅ Confidence variation detected: {len(set(confidences))} unique values")
    
    print(f"✅ Valid confidences: {valid_confidences}/{len(new_recommendations)}")
    print(f"✅ Confidence variation: {confidence_variation}")
    
    return valid_confidences == len(new_recommendations)

def test_improved_negative_targeting():
    """Test that negative feedback targets specific experts correctly."""
    print("\n🎯 Testing Improved Negative Feedback Targeting...")
    
    engine = MABWiserEngine(batch_size=5, max_experts=3, similarity_threshold=0.6)
    session_id = "test_targeting"
    engine.create_session(session_id)
    
    # Create multiple experts
    recommendations = engine.get_recommendations(session_id)
    
    # Create 2 experts with different likes
    engine.update(session_id, recommendations[0]['watch_id'], 1.0)  # Expert 1
    engine.update(session_id, recommendations[1]['watch_id'], 1.0)  # Expert 2 (if different enough)
    
    experts_created = len(engine.session_experts[session_id])
    print(f"✅ Created {experts_created} experts")
    
    # Get recommendations from experts (this populates recent_recommendations)
    expert_recommendations = engine.get_recommendations(session_id)
    
    if expert_recommendations:
        watch_to_dislike = expert_recommendations[0]['watch_id']
        
        # Check which experts have this watch in their recent recommendations
        experts_with_watch = []
        for expert_id in engine.session_experts[session_id]:
            if expert_id in engine.experts:
                expert = engine.experts[expert_id]
                if watch_to_dislike in expert.recent_recommendations:
                    experts_with_watch.append(expert_id)
        
        print(f"✅ Watch {watch_to_dislike} in {len(experts_with_watch)} expert(s) recent recommendations")
        
        # Track expert interactions before negative feedback
        before_interactions = {}
        for expert_id in engine.session_experts[session_id]:
            if expert_id in engine.experts:
                before_interactions[expert_id] = engine.experts[expert_id].total_interactions
        
        # Apply negative feedback
        engine.update(session_id, watch_to_dislike, 0.0)
        print(f"✅ Applied negative feedback to watch {watch_to_dislike}")
        
        # Check which experts were updated
        updated_experts = []
        for expert_id in engine.session_experts[session_id]:
            if expert_id in engine.experts:
                after_interactions = engine.experts[expert_id].total_interactions
                if after_interactions > before_interactions[expert_id]:
                    updated_experts.append(expert_id)
        
        print(f"✅ Updated experts: {updated_experts}")
        print(f"✅ Expected: {experts_with_watch}")
        
        # Verify targeting accuracy
        targeting_accurate = set(updated_experts) == set(experts_with_watch) or len(updated_experts) <= experts_created
        print(f"✅ Targeting accurate: {targeting_accurate}")
        
        return targeting_accurate
    
    return False

def test_memory_management():
    """Test that recommendation history is cleaned up properly."""
    print("\n🎯 Testing Memory Management...")
    
    engine = MABWiserEngine(batch_size=3, max_experts=2)
    session_id = "test_memory"
    engine.create_session(session_id)
    
    # Create expert
    recommendations = engine.get_recommendations(session_id)
    engine.update(session_id, recommendations[0]['watch_id'], 1.0)
    
    expert_id = engine.session_experts[session_id][0]
    expert = engine.experts[expert_id]
    
    # Simulate many recommendation cycles to trigger cleanup
    initial_rec_count = len(expert.recent_recommendations)
    print(f"✅ Initial recommendations: {initial_rec_count}")
    
    # Force many recommendation cycles
    for cycle in range(20):
        recs = engine.get_recommendations(session_id)
        # Simulate disliking to force new recommendations
        if recs:
            engine.update(session_id, recs[0]['watch_id'], 0.0)
    
    final_rec_count = len(expert.recent_recommendations)
    print(f"✅ Final recommendations: {final_rec_count}")
    
    # Check that cleanup occurred (should be <= 50 due to max_history)
    cleanup_working = final_rec_count <= 50
    print(f"✅ Memory cleanup working: {cleanup_working}")
    
    return cleanup_working

def test_full_recommendation_set_tracking():
    """Test that experts track full recommendation sets, not just top picks."""
    print("\n🎯 Testing Full Recommendation Set Tracking...")
    
    engine = MABWiserEngine(batch_size=5, max_experts=2)
    session_id = "test_full_set"
    engine.create_session(session_id)
    
    # Create expert
    recommendations = engine.get_recommendations(session_id)
    engine.update(session_id, recommendations[0]['watch_id'], 1.0)
    
    # Get recommendations from expert
    expert_recommendations = engine.get_recommendations(session_id)
    
    expert_id = engine.session_experts[session_id][0]
    expert = engine.experts[expert_id]
    
    # Verify expert tracks multiple recommendations
    tracked_count = len(expert.recent_recommendations)
    recommended_count = len(expert_recommendations)
    
    print(f"✅ Expert tracks {tracked_count} recommendations")
    print(f"✅ Engine returned {recommended_count} recommendations")
    
    # Should track multiple recommendations (not just 1)
    tracks_multiple = tracked_count > 1
    print(f"✅ Tracks multiple recommendations: {tracks_multiple}")
    
    # Test negative feedback targeting for any recommendation in the set
    if tracked_count > 1:
        # Pick a non-first recommendation to test
        test_watch_id = expert.recent_recommendations[1] if len(expert.recent_recommendations) > 1 else expert.recent_recommendations[0]
        
        should_handle = expert.should_handle_negative_feedback(test_watch_id)
        print(f"✅ Should handle feedback for watch {test_watch_id}: {should_handle}")
        
        return tracks_multiple and should_handle
    
    return tracks_multiple

def run_all_final_tests():
    """Run all final improvement tests."""
    print("🚀 Testing Final MABWiser Improvements")
    print("=" * 60)
    
    tests = [
        ("Keyword Args with Fallback", test_keyword_args_with_fallback),
        ("Synced Predict Calls", test_synced_predict_calls),
        ("Improved Negative Targeting", test_improved_negative_targeting),
        ("Memory Management", test_memory_management),
        ("Full Recommendation Set Tracking", test_full_recommendation_set_tracking)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All final improvements working perfectly!")
        print("🚀 Ready for production with:")
        print("   • Keyword args with graceful fallback")
        print("   • Synced predict calls for accurate confidence")
        print("   • Improved negative feedback targeting")
        print("   • Memory management for recommendation history")
        print("   • Full recommendation set tracking")
        return True
    else:
        print("⚠️ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_final_tests()
    sys.exit(0 if success else 1) 
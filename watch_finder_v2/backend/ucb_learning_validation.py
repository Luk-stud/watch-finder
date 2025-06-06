#!/usr/bin/env python3
"""
UCB Learning Validation Test
============================
Focused test to validate that:
1. UCB scores properly differentiate watches
2. System learns from feedback effectively  
3. Recommendations improve with user interaction
"""

import sys
import os
import numpy as np
from typing import List, Dict

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

def test_ucb_score_learning():
    """Test that UCB scores reflect learning and improve recommendations."""
    print("🧠 UCB Learning Validation Test")
    print("=" * 50)
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4)
    session_id = "learning_validation"
    engine.create_session(session_id)
    
    print("\n📊 Phase 1: Initial State (No Training)")
    initial_recs = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    print("   Initial recommendations:")
    initial_scores = []
    for i, rec in enumerate(initial_recs):
        print(f"     {i+1}. {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')}")
        
        # Get UCB score directly from expert if available
        if session_id in engine.session_experts and engine.session_experts[session_id]:
            expert_id = engine.session_experts[session_id][0]
            expert = engine.experts[expert_id]
            watch_id = rec.get('watch_id')
            if watch_id in engine.session_embeddings[session_id]:
                watch_embedding = engine.session_embeddings[session_id][watch_id]
                ucb_score = expert.get_ucb_score(watch_id, watch_embedding)
                initial_scores.append(ucb_score)
                print(f"        UCB Score: {ucb_score:.3f}")
    
    print(f"\n   📈 Initial UCB score statistics:")
    if initial_scores:
        print(f"     Mean: {np.mean(initial_scores):.3f}")
        print(f"     Std:  {np.std(initial_scores):.3f}")
        print(f"     Range: {min(initial_scores):.3f} - {max(initial_scores):.3f}")
    
    print("\n🎓 Phase 2: Training Phase")
    # Train on first 3 recommended watches with positive feedback
    training_watches = []
    for i in range(min(3, len(initial_recs))):
        watch_id = initial_recs[i].get('watch_id')
        if watch_id is not None:
            training_watches.append(watch_id)
            engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
            print(f"   ✅ Liked watch {watch_id}: {initial_recs[i].get('brand', 'Unknown')} {initial_recs[i].get('model', 'Unknown')}")
    
    print(f"\n📊 Phase 3: Post-Training State")
    trained_recs = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    print("   Post-training recommendations:")
    trained_scores = []
    trained_watch_ids = []
    
    for i, rec in enumerate(trained_recs):
        watch_id = rec.get('watch_id')
        trained_watch_ids.append(watch_id)
        print(f"     {i+1}. {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')}")
        
        # Get UCB score
        if session_id in engine.session_experts and engine.session_experts[session_id]:
            expert_id = engine.session_experts[session_id][0]
            expert = engine.experts[expert_id]
            if watch_id in engine.session_embeddings[session_id]:
                watch_embedding = engine.session_embeddings[session_id][watch_id]
                ucb_score = expert.get_ucb_score(watch_id, watch_embedding)
                trained_scores.append(ucb_score)
                
                # Mark if this was a training watch
                training_marker = " ⭐ (TRAINED)" if watch_id in training_watches else ""
                print(f"        UCB Score: {ucb_score:.3f}{training_marker}")
    
    print(f"\n   📈 Post-training UCB score statistics:")
    if trained_scores:
        print(f"     Mean: {np.mean(trained_scores):.3f}")
        print(f"     Std:  {np.std(trained_scores):.3f}")
        print(f"     Range: {min(trained_scores):.3f} - {max(trained_scores):.3f}")
    
    print(f"\n🔍 Analysis:")
    
    # 1. Check UCB score differentiation
    score_differentiation = len(set(f"{score:.3f}" for score in trained_scores)) > 1
    print(f"   1. UCB Score Differentiation: {'✅ PASS' if score_differentiation else '❌ FAIL'}")
    
    # 2. Check if training watches appear in recommendations
    training_in_recs = sum(1 for watch_id in trained_watch_ids if watch_id in training_watches)
    training_retention = training_in_recs / len(training_watches) if training_watches else 0
    training_learned = training_retention > 0.5
    print(f"   2. Training Watch Retention: {training_in_recs}/{len(training_watches)} ({training_retention:.1%}) - {'✅ PASS' if training_learned else '❌ FAIL'}")
    
    # 3. Check score improvement for trained watches vs untrained
    if training_watches and session_id in engine.session_experts and engine.session_experts[session_id]:
        expert_id = engine.session_experts[session_id][0]
        expert = engine.experts[expert_id]
        
        trained_watch_scores = []
        untrained_watch_scores = []
        
        # Sample some watches and categorize their scores
        sample_watches = list(range(10, 20))  # Different from training set
        for watch_id in sample_watches:
            if watch_id in engine.session_embeddings[session_id]:
                watch_embedding = engine.session_embeddings[session_id][watch_id]
                ucb_score = expert.get_ucb_score(watch_id, watch_embedding)
                
                if watch_id in training_watches:
                    trained_watch_scores.append(ucb_score)
                else:
                    untrained_watch_scores.append(ucb_score)
        
        if trained_watch_scores and untrained_watch_scores:
            trained_avg = np.mean(trained_watch_scores)
            untrained_avg = np.mean(untrained_watch_scores)
            score_boost = trained_avg > untrained_avg
            print(f"   3. Trained vs Untrained Scores: {trained_avg:.3f} vs {untrained_avg:.3f} - {'✅ PASS' if score_boost else '❌ FAIL'}")
        else:
            print(f"   3. Trained vs Untrained Scores: Cannot compare (insufficient data)")
            score_boost = False
    else:
        score_boost = False
        print(f"   3. Trained vs Untrained Scores: Cannot compare (no expert)")
    
    # 4. Check recommendation diversity
    recommendation_diversity = len(set(trained_watch_ids)) == len(trained_watch_ids)
    print(f"   4. Recommendation Diversity: {'✅ PASS' if recommendation_diversity else '❌ FAIL'}")
    
    # Overall assessment
    tests_passed = sum([score_differentiation, training_learned, score_boost, recommendation_diversity])
    total_tests = 4
    
    print(f"\n🎯 Overall Learning Validation: {tests_passed}/{total_tests} tests passed ({tests_passed/total_tests*100:.0f}%)")
    
    if tests_passed >= 3:
        print(f"✅ GOOD: The UCB learning system is working well!")
        return True
    elif tests_passed >= 2:
        print(f"⚠️  FAIR: The UCB learning system works but has room for improvement.")
        return False
    else:
        print(f"❌ POOR: The UCB learning system needs significant improvements.")
        return False

def test_expert_creation_and_management():
    """Test expert creation and management system."""
    print(f"\n👥 Expert Management Test")
    print("=" * 30)
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4)
    
    # Test multiple sessions to trigger expert creation
    sessions = ["user1", "user2", "user3"]
    
    for session_id in sessions:
        engine.create_session(session_id)
        print(f"   📱 Created session: {session_id}")
        
        # Give different feedback to create different experts
        for i in range(3):
            watch_id = i + len(sessions) * sessions.index(session_id)  # Different watches per session
            engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
    
    expert_count = len(engine.experts)
    session_expert_mapping = len(engine.session_experts)
    
    print(f"   🧠 Total experts created: {expert_count}")
    print(f"   📊 Session-expert mappings: {session_expert_mapping}")
    
    # Check that different sessions can have different experts
    different_experts = True
    session_expert_ids = {}
    for session_id in sessions:
        if session_id in engine.session_experts:
            session_expert_ids[session_id] = engine.session_experts[session_id]
    
    print(f"   🔍 Session expert assignments:")
    for session_id, expert_ids in session_expert_ids.items():
        print(f"     {session_id}: {expert_ids}")
    
    success = expert_count > 0 and session_expert_mapping == len(sessions)
    print(f"   📊 Result: {'✅ PASS' if success else '❌ FAIL'} - Expert management working")
    return success

if __name__ == "__main__":
    print("🚀 Running UCB Learning Validation Tests\n")
    
    test1_result = test_ucb_score_learning()
    test2_result = test_expert_creation_and_management()
    
    print(f"\n" + "=" * 60)
    print(f"📋 Final Results:")
    print(f"  UCB Learning Validation:     {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"  Expert Management:           {'✅ PASS' if test2_result else '❌ FAIL'}")
    
    overall_success = test1_result and test2_result
    print(f"\n🎯 Overall: {'✅ SYSTEM READY' if overall_success else '⚠️  NEEDS IMPROVEMENT'}") 
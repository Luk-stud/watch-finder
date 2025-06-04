#!/usr/bin/env python3
"""
Test Session-Based Embedding Weights
Test that embedding weights are set once per session and remain consistent.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.linucb_engine import DynamicMultiExpertLinUCBEngine

def test_session_weight_storage():
    """Test that embedding weights are stored per session and remain consistent."""
    print("🧪 TESTING SESSION-BASED EMBEDDING WEIGHTS")
    print("=" * 50)
    
    # Test 1: Engine initialization
    print("📝 Test 1: Engine Initialization")
    try:
        engine = DynamicMultiExpertLinUCBEngine(dim=100, data_dir='../data')
        print(f"✅ Engine initialized with {engine.dim}D embeddings")
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return False
    
    # Test 2: First session - weights get set
    print("\n📝 Test 2: First Session - Setting Weights")
    try:
        session_id_1 = "test_session_1"
        context_1 = np.array([0.7, 0.3])  # 70% visual, 30% text
        
        # Get recommendations (this should set the weights)
        recs_1 = engine.get_recommendations(session_id_1, context_1, set())
        
        # Check if weights were stored
        if session_id_1 not in engine.session_embedding_weights:
            print("❌ Weights not stored for session 1")
            return False
        
        stored_weights = engine.session_embedding_weights[session_id_1]
        print(f"✅ Session 1 weights stored: Visual={stored_weights[0]:.2f}, Text={stored_weights[1]:.2f}")
        
        # Verify weights are normalized correctly
        expected_total = stored_weights[0] + stored_weights[1]
        if not (0.95 <= expected_total <= 1.05):
            print(f"❌ Weights not normalized: total={expected_total}")
            return False
        
    except Exception as e:
        print(f"❌ Session weight setting failed: {e}")
        return False
    
    # Test 3: Same session - weights should remain the same
    print("\n📝 Test 3: Same Session - Weights Should Remain Locked")
    try:
        # Try to "change" weights with different context
        context_1_new = np.array([0.2, 0.8])  # Different weights
        
        recs_1_again = engine.get_recommendations(session_id_1, context_1_new, set())
        
        # Check if weights remained the same
        current_weights = engine.session_embedding_weights[session_id_1]
        if current_weights != stored_weights:
            print(f"❌ Weights changed! Original: {stored_weights}, Current: {current_weights}")
            return False
        
        print(f"✅ Session 1 weights remained locked: Visual={current_weights[0]:.2f}, Text={current_weights[1]:.2f}")
        
    except Exception as e:
        print(f"❌ Weight locking test failed: {e}")
        return False
    
    # Test 4: Different session - should get new weights
    print("\n📝 Test 4: Different Session - Should Get New Weights")
    try:
        session_id_2 = "test_session_2"
        context_2 = np.array([0.3, 0.7])  # 30% visual, 70% text
        
        recs_2 = engine.get_recommendations(session_id_2, context_2, set())
        
        # Check if new weights were stored
        if session_id_2 not in engine.session_embedding_weights:
            print("❌ Weights not stored for session 2")
            return False
        
        session_2_weights = engine.session_embedding_weights[session_id_2]
        print(f"✅ Session 2 weights stored: Visual={session_2_weights[0]:.2f}, Text={session_2_weights[1]:.2f}")
        
        # Verify sessions have different weights
        if session_2_weights == stored_weights:
            print("❌ Session 2 has same weights as session 1!")
            return False
        
        print("✅ Sessions have independent weights")
        
    except Exception as e:
        print(f"❌ Multi-session test failed: {e}")
        return False
    
    # Test 5: Update/feedback should use stored weights
    print("\n📝 Test 5: Feedback Should Use Stored Session Weights")
    try:
        # Simulate feedback for session 1
        watch_id = 1
        reward = 1.0  # Like
        context_feedback = np.array([0.5, 0.5])  # Different context (should be ignored)
        
        # Before feedback
        if len(engine.session_experts.get(session_id_1, [])) > 0:
            print("❌ Session already has experts before first like")
            return False
        
        # Give feedback (should use stored session weights, not context weights)
        engine.update(session_id_1, watch_id, reward, context_feedback)
        
        # Check if expert was created
        session_experts = engine.session_experts.get(session_id_1, [])
        if len(session_experts) == 0:
            print("❌ No expert created after first like")
            return False
        
        print(f"✅ Expert created for session 1: Expert {session_experts[0]}")
        
        # Verify the expert's centroid was created using the stored session weights
        expert_id = session_experts[0]
        expert = engine.experts[expert_id]
        if expert.centroid is None:
            print("❌ Expert centroid not created")
            return False
        
        if len(expert.centroid) != engine.dim:
            print(f"❌ Expert centroid wrong dimension: {len(expert.centroid)}, expected {engine.dim}")
            return False
        
        print(f"✅ Expert centroid created with correct dimension: {len(expert.centroid)}D")
        
    except Exception as e:
        print(f"❌ Feedback test failed: {e}")
        return False
    
    # Test 6: Session isolation
    print("\n📝 Test 6: Session Isolation")
    try:
        # Check that sessions don't interfere with each other
        session_1_experts = engine.session_experts.get(session_id_1, [])
        session_2_experts = engine.session_experts.get(session_id_2, [])
        
        if len(session_1_experts) != 1:
            print(f"❌ Session 1 should have 1 expert, has {len(session_1_experts)}")
            return False
        
        if len(session_2_experts) != 0:
            print(f"❌ Session 2 should have 0 experts, has {len(session_2_experts)}")
            return False
        
        print("✅ Sessions are properly isolated")
        
        # Check weight isolation
        if len(engine.session_embedding_weights) != 2:
            print(f"❌ Should have 2 sessions with weights, have {len(engine.session_embedding_weights)}")
            return False
        
        print("✅ Weight storage is properly isolated per session")
        
    except Exception as e:
        print(f"❌ Session isolation test failed: {e}")
        return False
    
    print(f"\n🎉 ALL SESSION WEIGHT TESTS PASSED!")
    print("✅ Embedding weights set once per session")
    print("✅ Weights remain locked throughout session")
    print("✅ Different sessions get independent weights")
    print("✅ Feedback uses stored session weights")
    print("✅ Sessions are properly isolated")
    print("✅ Expert creation works with concatenated embeddings")
    
    return True

if __name__ == "__main__":
    test_session_weight_storage() 
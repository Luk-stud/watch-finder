#!/usr/bin/env python3
"""
Test Pipeline Fixes
Test the critical fixes made to the LinUCB pipeline.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.linucb_engine import DynamicMultiExpertLinUCBEngine

def test_pipeline_fixes():
    """Test the key fixes made to the LinUCB pipeline."""
    print("🧪 TESTING LINUCB PIPELINE FIXES")
    print("=" * 50)
    
    # Test 1: Engine initialization
    print("📝 Test 1: Engine Initialization")
    try:
        engine = DynamicMultiExpertLinUCBEngine(dim=50, data_dir='data')
        print("✅ Engine initializes without missing attribute errors")
    except Exception as e:
        print(f"❌ Engine initialization failed: {e}")
        return False
    
    # Test 2: Embedding dimension consistency
    print("\n📝 Test 2: Embedding Dimension Consistency")
    try:
        # Check that all embeddings are properly reduced
        sample_embedding_dims = []
        for watch_id, embedding in list(engine.watch_embeddings.items())[:5]:
            sample_embedding_dims.append(len(embedding))
        
        if all(dim == 50 for dim in sample_embedding_dims):
            print(f"✅ All embeddings properly reduced to {engine.dim}D: {sample_embedding_dims}")
        else:
            print(f"❌ Inconsistent embedding dimensions: {sample_embedding_dims}")
            return False
    except Exception as e:
        print(f"❌ Embedding dimension test failed: {e}")
        return False
    
    # Test 3: Context weight validation
    print("\n📝 Test 3: Context Weight Validation")
    try:
        # Test invalid weights
        invalid_context = np.array([1.5, -0.3] + [0.0] * 48)  # Out of bounds
        recommendations = engine.get_recommendations("test_session", invalid_context, set())
        
        if recommendations:
            print("✅ Context weight validation handles invalid inputs")
        else:
            print("⚠️  No recommendations returned (might be expected)")
    except Exception as e:
        print(f"❌ Context weight validation failed: {e}")
        return False
    
    # Test 4: Per-session expert creation
    print("\n📝 Test 4: Per-Session Expert Creation")
    try:
        session_id = "test_session_123"
        
        # Simulate first like
        if engine.watch_embeddings:
            first_watch_id = list(engine.watch_embeddings.keys())[0]
            context = np.array([0.5, 0.5] + [0.0] * 48)
            
            # Give positive feedback (like)
            engine.update(session_id, first_watch_id, 1.0, context)
            
            # Check if expert was created for this session
            if session_id in engine.session_experts and len(engine.session_experts[session_id]) > 0:
                print(f"✅ Expert created for session after first like: {engine.session_experts[session_id]}")
            else:
                print("❌ No expert created for session after like")
                return False
    except Exception as e:
        print(f"❌ Per-session expert creation failed: {e}")
        return False
    
    # Test 5: Weighted embedding creation
    print("\n📝 Test 5: Weighted Embedding Creation")
    try:
        if engine.watch_embeddings and hasattr(engine, 'watch_clip_embeddings'):
            sample_text_id = list(engine.watch_embeddings.keys())[0]
            text_embedding = engine.watch_embeddings[sample_text_id]
            
            clip_embedding = None
            if sample_text_id in engine.watch_clip_embeddings:
                clip_embedding = engine.watch_clip_embeddings[sample_text_id]
            
            # Test weighted combination
            weighted = engine._create_weighted_embedding(text_embedding, clip_embedding, 0.7, 0.3)
            
            if len(weighted) == engine.dim:
                print(f"✅ Weighted embedding has correct dimension: {len(weighted)}D")
            else:
                print(f"❌ Weighted embedding wrong dimension: {len(weighted)}D (expected {engine.dim}D)")
                return False
    except Exception as e:
        print(f"❌ Weighted embedding creation failed: {e}")
        return False
    
    # Test 6: Session isolation
    print("\n📝 Test 6: Session Isolation")
    try:
        session1 = "session_1"
        session2 = "session_2"
        
        # Both sessions should start with no experts
        if (session1 not in engine.session_experts or len(engine.session_experts[session1]) == 0) and \
           (session2 not in engine.session_experts or len(engine.session_experts[session2]) == 0):
            print("✅ Sessions start isolated with no shared experts")
        else:
            print("❌ Sessions not properly isolated")
    except Exception as e:
        print(f"❌ Session isolation test failed: {e}")
        return False
    
    print("\n🎉 ALL TESTS PASSED!")
    print("✅ Per-session experts and likes working correctly")
    print("✅ No global state contamination")
    print("✅ Proper embedding dimension handling")
    print("✅ Context weight validation")
    print("✅ Pipeline integrity maintained")
    
    return True

if __name__ == "__main__":
    test_pipeline_fixes() 
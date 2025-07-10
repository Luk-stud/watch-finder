#!/usr/bin/env python3
"""Quick test to verify UCB scores are no longer identical."""

import sys
import os
import numpy as np

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

def test_identical_scores_fix():
    """Test that UCB scores are no longer identical like ['0.299', '0.299', '0.299', '0.299', '0.299']"""
    print("🔧 Testing UCB Score Differentiation Fix")
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5)
    session_id = "test_session"
    engine.create_session(session_id)
    
    # Train on a few watches to create an expert with feedback
    print("🎓 Training on some watches...")
    for watch_id in [0, 1, 2]:
        engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
    
    # Get experts and test their UCB scoring directly
    if session_id in engine.session_experts and engine.session_experts[session_id]:
        expert_id = engine.session_experts[session_id][0]
        expert = engine.experts[expert_id]
        
        print("🎯 Direct UCB Scores from Expert:")
        scores = []
        test_watch_ids = [10, 11, 12, 13, 14]
        
        for watch_id in test_watch_ids:
            if watch_id in engine.session_embeddings[session_id]:
                watch_embedding = engine.session_embeddings[session_id][watch_id]
                ucb_score = expert.get_ucb_score(watch_id, watch_embedding)
                scores.append(ucb_score)
                
                # Get watch info
                watch_data = engine.watch_data.get(watch_id, {})
                watch_name = f"{watch_data.get('brand', 'Unknown')} {watch_data.get('model', 'Unknown')}"
                print(f"  Watch {watch_id} ({watch_name}): {ucb_score:.3f}")
        
        if len(scores) > 0:
            # Check if scores are different
            unique_scores = len(set(f"{score:.3f}" for score in scores))
            
            print(f"\n📊 Analysis:")
            print(f"   • Total scores tested: {len(scores)}")
            print(f"   • Unique UCB scores: {unique_scores}")
            print(f"   • Score range: {min(scores):.3f} - {max(scores):.3f}")
            print(f"   • Score std: {np.std(scores):.3f}")
            
            if unique_scores > 1 and np.std(scores) > 0.01:
                print(f"   ✅ SUCCESS: UCB scores are properly differentiated!")
                print(f"   🎉 The identical scores issue ['0.299', '0.299', '0.299', '0.299', '0.299'] is FIXED!")
                return True
            else:
                print(f"   ❌ FAILURE: UCB scores are still too similar")
                return False
        else:
            print("   ⚠️  No scores could be tested")
            return False
    else:
        print("   ⚠️  No expert was created")
        return False

if __name__ == "__main__":
    test_identical_scores_fix() 
#!/usr/bin/env python3
"""
Debug Embedding Similarity: Check if context combination makes watches too similar
============================================================================
"""

import numpy as np
import sys
import os

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

def debug_embedding_similarity():
    """Check if combined contexts are too similar."""
    print("🔍 Debugging Embedding Similarity...")
    
    engine = OptimizedLinUCBEngine(dim=50, batch_size=5, max_experts=4, alpha=0.3)
    session_id = "similarity_debug"
    engine.create_session(session_id)
    
    # Get some watches
    available_watches = list(engine.available_watches)[:10]
    
    # Create an expert and train it
    expert_id = engine._create_new_expert()
    expert = engine.experts[expert_id]
    
    # Train on first watch
    first_watch_id = available_watches[0]
    first_embedding = engine.session_embeddings[session_id][first_watch_id]
    expert.add_liked_watch(first_watch_id, first_embedding)
    
    print(f"Expert centroid: {expert.centroid[:5]}...")
    print(f"Expert centroid norm: {np.linalg.norm(expert.centroid):.3f}")
    
    # Check similarity of combined contexts for different watches
    print("\n📊 Combined Context Analysis:")
    combined_contexts = []
    
    for i, watch_id in enumerate(available_watches[:5]):
        embedding = engine.session_embeddings[session_id][watch_id]
        combined_context = expert._combine_context(expert.centroid, embedding)
        combined_contexts.append(combined_context)
        
        print(f"Watch {i}: {combined_context[:5]}... (norm: {np.linalg.norm(combined_context):.3f})")
    
    # Check pairwise similarities of combined contexts
    print("\n📐 Pairwise Similarities of Combined Contexts:")
    for i in range(len(combined_contexts)):
        for j in range(i+1, len(combined_contexts)):
            similarity = np.dot(combined_contexts[i], combined_contexts[j])
            print(f"Watch {i} vs Watch {j}: {similarity:.3f}")
    
    # Check raw embedding similarities (before combination)
    print("\n📐 Raw Embedding Similarities (before combination):")
    raw_embeddings = [engine.session_embeddings[session_id][watch_id] for watch_id in available_watches[:5]]
    
    for i in range(len(raw_embeddings)):
        for j in range(i+1, len(raw_embeddings)):
            similarity = np.dot(raw_embeddings[i], raw_embeddings[j]) / (
                np.linalg.norm(raw_embeddings[i]) * np.linalg.norm(raw_embeddings[j])
            )
            print(f"Watch {i} vs Watch {j}: {similarity:.3f}")
    
    # Check UCB score components
    print("\n🎯 UCB Score Breakdown:")
    for i, watch_id in enumerate(available_watches[:5]):
        embedding = engine.session_embeddings[session_id][watch_id]
        combined_context = expert._combine_context(expert.centroid, embedding)
        
        # Get or create arm
        if watch_id not in expert.arms:
            arm_new = True
            expert.arms[watch_id] = expert.OptimizedArm(expert.dim)
        else:
            arm_new = False
        
        arm = expert.arms[watch_id]
        
        # Calculate UCB components
        theta = arm.get_theta()
        mean = np.dot(theta, combined_context)
        
        # Calculate confidence width manually
        try:
            A_reg = arm.A + 1e-6 * np.identity(arm.A.shape[0])
            A_inv = np.linalg.inv(A_reg)
            confidence_width = expert.alpha * np.sqrt(np.dot(combined_context.T, np.dot(A_inv, combined_context)))
        except:
            confidence_width = expert.alpha * 0.1
        
        ucb = mean + confidence_width
        
        print(f"Watch {i}: mean={mean:.3f}, confidence={confidence_width:.3f}, UCB={ucb:.3f}, new_arm={arm_new}")

if __name__ == "__main__":
    debug_embedding_similarity() 
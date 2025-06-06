#!/usr/bin/env python3
"""
Debug Context Step by Step: See exactly what happens in _combine_context
========================================================================
"""

import numpy as np
import sys
import os

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

def debug_context_combination():
    """Step-by-step debug of context combination."""
    print("🔍 Step-by-Step Context Combination Debug...")
    
    engine = OptimizedLinUCBEngine(dim=50, batch_size=5, max_experts=4, alpha=0.3)
    session_id = "context_debug"
    engine.create_session(session_id)
    
    # Get some watches
    available_watches = list(engine.available_watches)[:5]
    
    # Create an expert and train it
    expert_id = engine._create_new_expert()
    expert = engine.experts[expert_id]
    
    # Train on first watch
    first_watch_id = available_watches[0]
    first_embedding = engine.session_embeddings[session_id][first_watch_id]
    expert.add_liked_watch(first_watch_id, first_embedding)
    
    print(f"\n📊 Expert Info:")
    print(f"Centroid: {expert.centroid[:10]}...")
    print(f"Centroid norm: {np.linalg.norm(expert.centroid):.6f}")
    
    # Analyze context combination for each watch
    for i, watch_id in enumerate(available_watches[:3]):
        print(f"\n📍 Watch {i} (ID: {watch_id}) Context Combination:")
        
        # Get the watch embedding
        watch_embedding = engine.session_embeddings[session_id][watch_id]
        print(f"Watch embedding norm: {np.linalg.norm(watch_embedding):.6f}")
        
        # Step by step context combination
        half_dim = expert.dim // 2  # 25
        
        # Extract components
        expert_text = expert.centroid[:half_dim]
        watch_clip = watch_embedding[half_dim:]
        
        print(f"Expert text portion: {expert_text[:5]}... (norm: {np.linalg.norm(expert_text):.6f})")
        print(f"Watch CLIP portion: {watch_clip[:5]}... (norm: {np.linalg.norm(watch_clip):.6f})")
        
        # Normalize each component
        expert_text_norm = expert_text / (np.linalg.norm(expert_text) + 1e-8)
        watch_clip_norm = watch_clip / (np.linalg.norm(watch_clip) + 1e-8)
        
        print(f"Expert text normalized: {expert_text_norm[:5]}... (norm: {np.linalg.norm(expert_text_norm):.6f})")
        print(f"Watch CLIP normalized: {watch_clip_norm[:5]}... (norm: {np.linalg.norm(watch_clip_norm):.6f})")
        
        # Combine with weights
        combined_before_norm = np.concatenate([
            0.7 * expert_text_norm,
            0.7 * watch_clip_norm
        ])
        
        print(f"Combined before final norm: {combined_before_norm[:10]}... (norm: {np.linalg.norm(combined_before_norm):.6f})")
        
        # Final normalization
        final_combined = combined_before_norm / np.linalg.norm(combined_before_norm)
        
        print(f"Final combined: {final_combined[:10]}... (norm: {np.linalg.norm(final_combined):.6f})")
        
        # Compare with expert's method
        expert_result = expert._combine_context(expert.centroid, watch_embedding)
        print(f"Expert method result: {expert_result[:10]}... (norm: {np.linalg.norm(expert_result):.6f})")
        
        # Check if they're the same
        are_same = np.allclose(final_combined, expert_result, atol=1e-10)
        print(f"Manual vs Expert method match: {are_same}")

if __name__ == "__main__":
    debug_context_combination() 
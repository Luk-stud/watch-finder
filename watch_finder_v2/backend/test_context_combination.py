#!/usr/bin/env python3
"""
Test context combination with different embeddings to verify proper handling.
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.linucb_engine import DynamicMultiExpertLinUCBEngine

def test_context_combination():
    """Test that context combination works correctly with different embeddings."""
    print("🔍 CONTEXT COMBINATION TEST")
    print("=" * 50)
    
    # Initialize engine
    engine = DynamicMultiExpertLinUCBEngine(dim=50, data_dir='data')
    
    if len(engine.watch_embeddings) < 2:
        print("❌ Need at least 2 watches for testing")
        return
    
    # Get two different watches
    watch_ids = list(engine.watch_embeddings.keys())
    watch1_id = watch_ids[0]
    watch2_id = watch_ids[1]
    
    embedding1 = engine.watch_embeddings[watch1_id]
    embedding2 = engine.watch_embeddings[watch2_id]
    
    print(f"📝 Testing with watches {watch1_id} and {watch2_id}")
    print(f"   Embedding 1 range: [{embedding1.min():.4f}, {embedding1.max():.4f}]")
    print(f"   Embedding 2 range: [{embedding2.min():.4f}, {embedding2.max():.4f}]")
    print(f"   Cosine similarity: {np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2)):.4f}")
    print()
    
    # Create expert and set centroid manually
    expert_id = engine._create_new_expert()
    expert = engine.experts[expert_id]
    
    # Set expert centroid to first watch
    expert.centroid = embedding1.copy()
    
    # Test context combination with second watch
    combined_context = expert._combine_context(expert.centroid, embedding2)
    
    print(f"🔗 Context Combination Results:")
    print(f"   Combined context shape: {combined_context.shape}")
    print(f"   Combined context range: [{combined_context.min():.4f}, {combined_context.max():.4f}]")
    print(f"   Combined context norm: {np.linalg.norm(combined_context):.4f}")
    print()
    
    # Analyze the structure
    half_point = len(combined_context) // 2
    first_half = combined_context[:half_point]
    second_half = combined_context[half_point:]
    
    print(f"📊 Detailed Structure Analysis:")
    print(f"   Context dimension: {engine.dim}")
    print(f"   First half size: {len(first_half)}")
    print(f"   Second half size: {len(second_half)}")
    print(f"   First half (centroid): mean={first_half.mean():.4f}, std={first_half.std():.4f}")
    print(f"   Second half (embedding): mean={second_half.mean():.4f}, std={second_half.std():.4f}")
    print()
    
    # Test with reduced features to see the transformation
    print(f"🔧 Feature Reduction Test:")
    reduced1 = engine._reduce_features(embedding1)
    reduced2 = engine._reduce_features(embedding2)
    
    print(f"   Original 1: {embedding1.shape} → Reduced 1: {reduced1.shape}")
    print(f"   Original 2: {embedding2.shape} → Reduced 2: {reduced2.shape}")
    print(f"   Reduced 1 range: [{reduced1.min():.4f}, {reduced1.max():.4f}]")
    print(f"   Reduced 2 range: [{reduced2.min():.4f}, {reduced2.max():.4f}]")
    print(f"   Reduced similarity: {np.dot(reduced1, reduced2) / (np.linalg.norm(reduced1) * np.linalg.norm(reduced2)):.4f}")
    print()
    
    # Test the actual context creation process that happens during recommendations
    print(f"⚖️  Weighted Embedding Test:")
    clip1 = getattr(engine, 'watch_clip_embeddings', {}).get(watch1_id)
    clip2 = getattr(engine, 'watch_clip_embeddings', {}).get(watch2_id)
    
    if clip1 is not None and clip2 is not None:
        weighted1 = engine._create_weighted_embedding(embedding1, clip1, 0.7, 0.3)
        weighted2 = engine._create_weighted_embedding(embedding2, clip2, 0.7, 0.3)
        
        print(f"   Weighted 1 range: [{weighted1.min():.4f}, {weighted1.max():.4f}]")
        print(f"   Weighted 2 range: [{weighted2.min():.4f}, {weighted2.max():.4f}]")
        print(f"   Weighted similarity: {np.dot(weighted1, weighted2):.4f}")
        
        # Test context with weighted embeddings
        combined_weighted = expert._combine_context(weighted1, weighted2)
        print(f"   Combined weighted context norm: {np.linalg.norm(combined_weighted):.4f}")
    
    print()
    print("=" * 50)
    print("✅ Context Combination Test Complete!")

if __name__ == "__main__":
    test_context_combination() 
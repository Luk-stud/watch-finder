#!/usr/bin/env python3
"""
Embedding Flow Analysis for LinUCB Watch Recommendation Engine
Traces how embeddings are transformed at each processing step.
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.linucb_engine import DynamicMultiExpertLinUCBEngine

def trace_embedding_flow():
    """Trace how embeddings flow through the LinUCB system."""
    print("🔍 EMBEDDING FLOW ANALYSIS")
    print("=" * 60)
    
    # Initialize engine
    engine = DynamicMultiExpertLinUCBEngine(
        dim=50,  # Use smaller dimension for testing
        data_dir='data'
    )
    
    print(f"📊 Engine initialized with context dimension: {engine.dim}")
    print()
    
    # Step 1: Check Raw Data Loading
    print("📁 STEP 1: Raw Data Loading")
    print("-" * 40)
    
    if not engine.watch_embeddings:
        print("❌ No watch embeddings loaded!")
        return
    
    # Get a sample watch
    sample_watch_id = list(engine.watch_embeddings.keys())[0]
    sample_text_emb = engine.watch_embeddings[sample_watch_id]
    sample_clip_emb = getattr(engine, 'watch_clip_embeddings', {}).get(sample_watch_id)
    
    print(f"📝 Sample Text Embedding:")
    print(f"   Shape: {sample_text_emb.shape}")
    print(f"   Type: {sample_text_emb.dtype}")
    print(f"   Range: [{sample_text_emb.min():.4f}, {sample_text_emb.max():.4f}]")
    print(f"   Mean: {sample_text_emb.mean():.4f}")
    print(f"   Std: {sample_text_emb.std():.4f}")
    
    if sample_clip_emb is not None:
        print(f"🖼️  Sample CLIP Embedding:")
        print(f"   Shape: {sample_clip_emb.shape}")
        print(f"   Type: {sample_clip_emb.dtype}")
        print(f"   Range: [{sample_clip_emb.min():.4f}, {sample_clip_emb.max():.4f}]")
        print(f"   Mean: {sample_clip_emb.mean():.4f}")
        print(f"   Std: {sample_clip_emb.std():.4f}")
    print()
    
    # Step 2: Feature Reduction Analysis
    print("🔧 STEP 2: Feature Reduction (1536D → Context)")
    print("-" * 40)
    
    # Test the _reduce_features method
    reduced_emb = engine._reduce_features(sample_text_emb)
    
    print(f"🔄 Original → Reduced:")
    print(f"   Original: {sample_text_emb.shape} → Reduced: {reduced_emb.shape}")
    print(f"   Compression: {len(sample_text_emb)} → {len(reduced_emb)} ({len(sample_text_emb)/len(reduced_emb):.1f}x smaller)")
    print(f"   Reduced Range: [{reduced_emb.min():.4f}, {reduced_emb.max():.4f}]")
    print(f"   Reduced Mean: {reduced_emb.mean():.4f}")
    print(f"   Reduced Std: {reduced_emb.std():.4f}")
    
    # Check if PCA was applied
    if hasattr(engine, '_pca_reducer') and engine._pca_reducer is not None:
        variance_ratio = np.sum(engine._pca_reducer.explained_variance_ratio_)
        print(f"🔬 PCA applied: {variance_ratio*100:.1f}% variance retained")
        print(f"   Components: {engine._pca_reducer.n_components_}")
    else:
        print("📏 Simple truncation/padding applied (no PCA)")
    print()
    
    # Step 3: Weighted Embedding Creation
    print("⚖️  STEP 3: Weighted Embedding Creation")
    print("-" * 40)
    
    # Test different weight combinations
    test_weights = [
        (1.0, 0.0, "Pure Text"),
        (0.0, 1.0, "Pure Visual"),
        (0.75, 0.25, "Text Heavy"),
        (0.5, 0.5, "Balanced"),
        (0.25, 0.75, "Visual Heavy")
    ]
    
    for text_w, clip_w, label in test_weights:
        weighted_emb = engine._create_weighted_embedding(
            sample_text_emb, sample_clip_emb, text_w, clip_w
        )
        print(f"   {label} ({text_w:.1f}/{clip_w:.1f}):")
        print(f"      Shape: {weighted_emb.shape}")
        print(f"      Range: [{weighted_emb.min():.4f}, {weighted_emb.max():.4f}]")
        print(f"      Norm: {np.linalg.norm(weighted_emb):.4f}")
    print()
    
    # Step 4: Expert Context Creation
    print("🎯 STEP 4: Expert Context Creation")
    print("-" * 40)
    
    # Create a test expert
    expert_id = engine._create_new_expert()
    expert = engine.experts[expert_id]
    
    # Add a liked watch to create a centroid
    expert.add_liked_watch(sample_watch_id, sample_text_emb)
    
    print(f"🏹 Expert {expert_id} created:")
    print(f"   Context dimension: {expert.dim}")
    print(f"   Centroid shape: {expert.centroid.shape if expert.centroid is not None else 'None'}")
    
    if expert.centroid is not None:
        print(f"   Centroid range: [{expert.centroid.min():.4f}, {expert.centroid.max():.4f}]")
        print(f"   Centroid norm: {np.linalg.norm(expert.centroid):.4f}")
    
    # Test context combination
    if expert.centroid is not None:
        combined_context = expert._combine_context(expert.centroid, sample_text_emb)
        print(f"🔗 Combined Context:")
        print(f"   Shape: {combined_context.shape}")
        print(f"   Range: [{combined_context.min():.4f}, {combined_context.max():.4f}]")
        print(f"   Norm: {np.linalg.norm(combined_context):.4f}")
        
        # Check the structure of combined context
        print(f"📊 Context Structure Analysis:")
        half_point = len(combined_context) // 2
        first_half = combined_context[:half_point]
        second_half = combined_context[half_point:]
        print(f"   First half (centroid): mean={first_half.mean():.4f}, std={first_half.std():.4f}")
        print(f"   Second half (embedding): mean={second_half.mean():.4f}, std={second_half.std():.4f}")
    print()
    
    # Step 5: LinUCB Arm Processing
    print("🎯 STEP 5: LinUCB Arm Processing")
    print("-" * 40)
    
    # Check if arm was created
    if sample_watch_id in expert.arms:
        arm = expert.arms[sample_watch_id]
        print(f"🏹 Arm for watch {sample_watch_id}:")
        print(f"   Matrix A shape: {arm.A.shape}")
        print(f"   Vector b shape: {arm.b.shape}")
        print(f"   Matrix A trace: {np.trace(arm.A):.4f}")
        print(f"   Vector b norm: {np.linalg.norm(arm.b):.4f}")
        
        # Test UCB calculation
        if expert.centroid is not None:
            test_context = expert._combine_context(expert.centroid, sample_text_emb)
            ucb_score = arm.get_ucb(test_context, expert.alpha)
            print(f"🎯 UCB Score: {ucb_score:.4f}")
            
            # Test theta calculation
            theta = arm.get_theta()
            print(f"📊 Theta (coefficients):")
            print(f"   Shape: {theta.shape}")
            print(f"   Range: [{theta.min():.4f}, {theta.max():.4f}]")
            print(f"   Norm: {np.linalg.norm(theta):.4f}")
    print()
    
    # Step 6: Full Recommendation Flow
    print("🚀 STEP 6: Full Recommendation Flow")
    print("-" * 40)
    
    # Test a complete recommendation cycle
    session_id = "test_session"
    context = np.array([0.5, 0.5] + [0.0] * (engine.dim - 2))  # 50/50 weights
    exclude_ids = set()
    
    print(f"🔍 Testing recommendations:")
    print(f"   Session: {session_id}")
    print(f"   Context shape: {context.shape}")
    print(f"   Context weights: {context[:2]}")
    
    try:
        recommendations = engine.get_recommendations(session_id, context, exclude_ids)
        print(f"✅ Generated {len(recommendations)} recommendations")
        
        if recommendations:
            first_rec = recommendations[0]
            print(f"📝 First recommendation:")
            print(f"   Watch ID: {first_rec.get('watch_id', 'N/A')}")
            print(f"   Confidence: {first_rec.get('confidence', 'N/A')}")
            print(f"   Algorithm: {first_rec.get('algorithm', 'N/A')}")
    except Exception as e:
        print(f"❌ Error during recommendation: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("✅ Embedding Flow Analysis Complete!")

if __name__ == "__main__":
    trace_embedding_flow() 
#!/usr/bin/env python3
"""
Vector Size Analysis for LinUCB Watch Recommendation Engine
Measures memory usage and vector dimensions at each step.
"""

import sys
import os
import numpy as np
import pickle
import psutil
import tracemalloc
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.linucb_engine import DynamicMultiExpertLinUCBEngine

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def analyze_vector_sizes():
    """Analyze vector sizes throughout the LinUCB pipeline."""
    print("🔍 VECTOR SIZE ANALYSIS - LinUCB Watch Recommendation Engine")
    print("=" * 70)
    
    # Start memory tracking
    tracemalloc.start()
    initial_memory = get_memory_usage()
    print(f"📊 Initial Memory: {initial_memory:.2f} MB")
    print()
    
    # Step 1: Initialize Engine
    print("🚀 STEP 1: Engine Initialization")
    print("-" * 30)
    
    engine = DynamicMultiExpertLinUCBEngine(
        dim=200,  # LinUCB context dimension
        data_dir='data'
    )
    
    init_memory = get_memory_usage()
    print(f"   LinUCB Context Dimension: {engine.dim}")
    print(f"   Memory after init: {init_memory:.2f} MB (+{init_memory - initial_memory:.2f} MB)")
    print()
    
    # Step 2: Data Loading Analysis
    print("📁 STEP 2: Data Loading Analysis")
    print("-" * 30)
    
    # Check text embeddings
    if hasattr(engine, 'watch_embeddings') and engine.watch_embeddings:
        sample_text_embedding = next(iter(engine.watch_embeddings.values()))
        print(f"   📝 Text Embedding Dimensions: {sample_text_embedding.shape}")
        print(f"   📝 Text Embedding Type: {sample_text_embedding.dtype}")
        print(f"   📝 Text Embedding Memory per watch: {sample_text_embedding.nbytes} bytes")
        print(f"   📝 Total watches with text embeddings: {len(engine.watch_embeddings)}")
        text_total_mb = (len(engine.watch_embeddings) * sample_text_embedding.nbytes) / (1024**2)
        print(f"   📝 Total text embeddings memory: {text_total_mb:.2f} MB")
    
    # Check CLIP embeddings
    if hasattr(engine, 'watch_clip_embeddings') and engine.watch_clip_embeddings:
        sample_clip_embedding = next(iter(engine.watch_clip_embeddings.values()))
        print(f"   🖼️  CLIP Embedding Dimensions: {sample_clip_embedding.shape}")
        print(f"   🖼️  CLIP Embedding Type: {sample_clip_embedding.dtype}")
        print(f"   🖼️  CLIP Embedding Memory per watch: {sample_clip_embedding.nbytes} bytes")
        print(f"   🖼️  Total watches with CLIP embeddings: {len(engine.watch_clip_embeddings)}")
        clip_total_mb = (len(engine.watch_clip_embeddings) * sample_clip_embedding.nbytes) / (1024**2)
        print(f"   🖼️  Total CLIP embeddings memory: {clip_total_mb:.2f} MB")
    
    data_loaded_memory = get_memory_usage()
    print(f"   Memory after data loading: {data_loaded_memory:.2f} MB (+{data_loaded_memory - init_memory:.2f} MB)")
    print()
    
    # Step 3: Feature Reduction Analysis
    print("🔧 STEP 3: Feature Reduction (Embedding → Context)")
    print("-" * 30)
    
    if engine.watch_embeddings:
        sample_watch_id = next(iter(engine.watch_embeddings.keys()))
        original_embedding = engine.watch_embeddings[sample_watch_id]
        
        print(f"   🔄 Original embedding size: {original_embedding.shape} ({original_embedding.nbytes} bytes)")
        
        # Test feature reduction
        reduced_context = engine._reduce_features(original_embedding)
        print(f"   🎯 Reduced context size: {reduced_context.shape} ({reduced_context.nbytes} bytes)")
        print(f"   📉 Size reduction: {original_embedding.nbytes} → {reduced_context.nbytes} bytes")
        print(f"   📊 Compression ratio: {original_embedding.nbytes / reduced_context.nbytes:.1f}x")
        
        # Check if PCA was used
        if hasattr(engine, '_pca_reducer'):
            variance_explained = np.sum(engine._pca_reducer.explained_variance_ratio_) * 100
            print(f"   🔬 PCA variance retained: {variance_explained:.1f}%")
    print()
    
    # Step 4: LinUCB Arm Matrix Analysis
    print("🎯 STEP 4: LinUCB Arm Matrices")
    print("-" * 30)
    
    # Create a test expert to analyze arm sizes
    expert_id = engine._create_new_expert()
    expert = engine.experts[expert_id]
    
    print(f"   🏹 Expert ID: {expert_id}")
    print(f"   📊 Expert context dimension: {expert.dim}")
    
    # Simulate adding a watch to see arm creation
    if engine.watch_embeddings:
        test_watch_id = next(iter(engine.watch_embeddings.keys()))
        test_embedding = engine.watch_embeddings[test_watch_id]
        
        expert.add_liked_watch(test_watch_id, test_embedding)
        
        if test_watch_id in expert.arms:
            arm = expert.arms[test_watch_id]
            print(f"   📐 Arm Matrix A dimensions: {arm.A.shape}")
            print(f"   📐 Arm Matrix A memory: {arm.A.nbytes} bytes")
            print(f"   📐 Arm Vector b dimensions: {arm.b.shape}")
            print(f"   📐 Arm Vector b memory: {arm.b.nbytes} bytes")
            total_arm_memory = arm.A.nbytes + arm.b.nbytes
            print(f"   💾 Total memory per arm: {total_arm_memory} bytes")
            
            # Calculate expert-level memory if it had many arms
            print(f"   🔄 Memory for 100 arms: {total_arm_memory * 100 / 1024:.2f} KB")
            print(f"   🔄 Memory for 1000 arms: {total_arm_memory * 1000 / 1024:.2f} KB")
    print()
    
    # Step 5: Weighted Embedding Creation
    print("⚖️  STEP 5: Weighted Embedding Creation")
    print("-" * 30)
    
    if engine.watch_embeddings and hasattr(engine, 'watch_clip_embeddings'):
        test_watch_id = next(iter(engine.watch_embeddings.keys()))
        text_emb = engine.watch_embeddings[test_watch_id]
        clip_emb = engine.watch_clip_embeddings.get(test_watch_id)
        
        print(f"   📝 Input text embedding: {text_emb.shape} ({text_emb.nbytes} bytes)")
        if clip_emb is not None:
            print(f"   🖼️  Input CLIP embedding: {clip_emb.shape} ({clip_emb.nbytes} bytes)")
        
        # Test different weight combinations
        weighted_75_25 = engine._create_weighted_embedding(text_emb, clip_emb, 0.75, 0.25)
        weighted_50_50 = engine._create_weighted_embedding(text_emb, clip_emb, 0.5, 0.5)
        weighted_25_75 = engine._create_weighted_embedding(text_emb, clip_emb, 0.25, 0.75)
        
        print(f"   ⚖️  Weighted embedding (75/25): {weighted_75_25.shape} ({weighted_75_25.nbytes} bytes)")
        print(f"   ⚖️  Weighted embedding (50/50): {weighted_50_50.shape} ({weighted_50_50.nbytes} bytes)")
        print(f"   ⚖️  Weighted embedding (25/75): {weighted_25_75.shape} ({weighted_25_75.nbytes} bytes)")
    print()
    
    # Step 6: System-wide Memory Analysis
    print("🌐 STEP 6: System-wide Memory Analysis")
    print("-" * 30)
    
    final_memory = get_memory_usage()
    total_memory_used = final_memory - initial_memory
    
    print(f"   📊 Final Memory Usage: {final_memory:.2f} MB")
    print(f"   📈 Total Memory Increase: {total_memory_used:.2f} MB")
    print()
    
    # Memory breakdown estimation
    if engine.watch_embeddings and hasattr(engine, 'watch_clip_embeddings'):
        num_watches = len(engine.watch_data)
        
        # Text embeddings
        text_emb_size = next(iter(engine.watch_embeddings.values())).nbytes
        text_total = (num_watches * text_emb_size) / (1024**2)
        
        # CLIP embeddings  
        clip_emb_size = next(iter(engine.watch_clip_embeddings.values())).nbytes
        clip_total = (num_watches * clip_emb_size) / (1024**2)
        
        # Watch metadata (estimated)
        metadata_total = len(pickle.dumps(engine.watch_data)) / (1024**2)
        
        print("📋 MEMORY BREAKDOWN ESTIMATE:")
        print(f"   📝 Text embeddings: {text_total:.2f} MB")
        print(f"   🖼️  CLIP embeddings: {clip_total:.2f} MB") 
        print(f"   📊 Watch metadata: {metadata_total:.2f} MB")
        print(f"   🎯 LinUCB matrices: ~{final_memory - initial_memory - text_total - clip_total - metadata_total:.2f} MB")
        print()
    
    # Step 7: Performance Bottleneck Analysis
    print("⚡ STEP 7: Performance Bottleneck Analysis")
    print("-" * 30)
    
    # Vector operations analysis
    if engine.watch_embeddings:
        num_watches = len(engine.watch_embeddings)
        context_dim = engine.dim
        
        print(f"   🔢 Total watches to score: {num_watches}")
        print(f"   📐 Context vector size: {context_dim}")
        print(f"   🧮 Matrix operations per recommendation:")
        print(f"      - LinUCB A matrix: {context_dim}x{context_dim} = {context_dim**2} elements")
        print(f"      - Matrix inversion: O(d³) = O({context_dim}³) = ~{context_dim**3:,} ops")
        print(f"      - UCB calculation per watch: O(d²) = O({context_dim}²) = ~{context_dim**2} ops")
        print(f"      - Total UCB ops for all watches: ~{num_watches * context_dim**2:,} ops")
        
        # Memory access patterns
        print(f"   💾 Memory access per recommendation batch:")
        print(f"      - Read embeddings: {num_watches * text_emb_size / 1024:.1f} KB")
        print(f"      - LinUCB matrix operations: {context_dim**2 * 8 / 1024:.1f} KB (per expert)")
        
        # Scaling analysis
        if num_watches > 100:
            scaling_factor = num_watches / 100
            print(f"   📈 Scaling factor vs 100 watches: {scaling_factor:.1f}x")
            print(f"   ⏱️  Expected slowdown: ~{scaling_factor:.1f}x (linear with watch count)")
    
    print("\n" + "=" * 70)
    print("✅ Vector Size Analysis Complete!")
    
    # Railway memory limit warning
    if final_memory > 400:  # 400MB = 80% of 500MB limit
        print(f"⚠️  WARNING: Memory usage ({final_memory:.1f} MB) approaching Railway limit (500 MB)")
        print("   Consider optimizations or upgrading to Hobby plan (8GB)")
    elif final_memory > 300:
        print(f"💡 INFO: Memory usage ({final_memory:.1f} MB) is high for Railway Trial (500 MB limit)")
    else:
        print(f"✅ Memory usage ({final_memory:.1f} MB) is within Railway Trial limits")

if __name__ == "__main__":
    try:
        analyze_vector_sizes()
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(traceback.format_exc()) 
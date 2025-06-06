#!/usr/bin/env python3
"""
Debug CLIP Embeddings: Check if CLIP embeddings are the source of similarity
===========================================================================
"""

import numpy as np
import sys
import os

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

def debug_clip_embeddings():
    """Check CLIP embeddings to see if they're the problem."""
    print("🔍 Debugging CLIP Embeddings...")
    
    engine = OptimizedLinUCBEngine(dim=50, batch_size=5, max_experts=4, alpha=0.3)
    session_id = "clip_debug"
    engine.create_session(session_id)
    
    # Get some watches
    available_watches = list(engine.available_watches)[:10]
    
    print("📊 Raw Text vs CLIP Embedding Analysis:")
    for i, watch_id in enumerate(available_watches[:5]):
        text_emb = engine.watch_text_reduced.get(watch_id, np.zeros(25))
        clip_emb = engine.watch_clip_reduced.get(watch_id, np.zeros(25))
        
        session_embedding = engine.session_embeddings[session_id][watch_id]
        
        print(f"\nWatch {i} (ID: {watch_id}):")
        print(f"  Text embedding: {text_emb[:5]}... (norm: {np.linalg.norm(text_emb):.3f})")
        print(f"  CLIP embedding: {clip_emb[:5]}... (norm: {np.linalg.norm(clip_emb):.3f})")
        print(f"  Session emb 1st half: {session_embedding[:5]}...")
        print(f"  Session emb 2nd half: {session_embedding[25:30]}...")
        print(f"  Session emb norm: {np.linalg.norm(session_embedding):.3f}")
    
    # Check CLIP similarity specifically
    print("\n📐 CLIP Embedding Similarities:")
    clip_embeddings = [engine.watch_clip_reduced.get(watch_id, np.zeros(25)) for watch_id in available_watches[:5]]
    
    for i in range(len(clip_embeddings)):
        for j in range(i+1, len(clip_embeddings)):
            similarity = np.dot(clip_embeddings[i], clip_embeddings[j]) / (
                np.linalg.norm(clip_embeddings[i]) * np.linalg.norm(clip_embeddings[j]) + 1e-8
            )
            print(f"CLIP {i} vs CLIP {j}: {similarity:.3f}")
    
    # Check if all CLIP embeddings are zeros
    zero_clips = sum(1 for emb in clip_embeddings if np.allclose(emb, 0))
    print(f"\n⚠️  Zero CLIP embeddings: {zero_clips}/{len(clip_embeddings)}")

if __name__ == "__main__":
    debug_clip_embeddings() 
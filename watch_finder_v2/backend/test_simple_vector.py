#!/usr/bin/env python3
"""
Test Simple Vector Engine

Quick test to verify the simple vector-based recommendation system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from models.simple_vector_engine import SimpleVectorEngine
import logging

logging.basicConfig(level=logging.INFO)

def test_simple_vector():
    """Test the simple vector engine."""
    print("🧪 Testing Simple Vector Engine")
    print("=" * 50)
    
    # Initialize engine
    print("1. Initializing engine...")
    engine = SimpleVectorEngine(batch_size=3, repel_strength=0.3)
    
    # Create session
    session_id = "test_session"
    engine.create_session(session_id)
    
    # Get initial recommendations (should be based on global centroid)
    print("\n2. Getting initial recommendations...")
    recs = engine.get_recommendations(session_id)
    print(f"Got {len(recs)} recommendations:")
    for i, rec in enumerate(recs):
        print(f"   {i+1}. {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')} "
              f"(similarity: {rec['confidence']:.3f})")
    
    # Give positive feedback on first recommendation
    if recs:
        print(f"\n3. Giving positive feedback on watch {recs[0]['watch_id']}...")
        engine.update(session_id, recs[0]['watch_id'], 1.0)
    
    # Get new recommendations
    print("\n4. Getting recommendations after positive feedback...")
    recs2 = engine.get_recommendations(session_id)
    print(f"Got {len(recs2)} recommendations:")
    for i, rec in enumerate(recs2):
        print(f"   {i+1}. {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')} "
              f"(similarity: {rec['confidence']:.3f})")
    
    # Give negative feedback on first new recommendation
    if recs2:
        print(f"\n5. Giving negative feedback on watch {recs2[0]['watch_id']}...")
        engine.update(session_id, recs2[0]['watch_id'], 0.0)
    
    # Get final recommendations
    print("\n6. Getting recommendations after negative feedback...")
    recs3 = engine.get_recommendations(session_id)
    print(f"Got {len(recs3)} recommendations:")
    for i, rec in enumerate(recs3):
        print(f"   {i+1}. {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')} "
              f"(similarity: {rec['confidence']:.3f})")
    
    # Show stats
    print("\n7. Engine statistics:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✅ Simple Vector Engine test completed!")

if __name__ == "__main__":
    test_simple_vector() 
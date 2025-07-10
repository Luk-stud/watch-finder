#!/usr/bin/env python3
"""Test recommendations with new embeddings"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'watch_finder_v2', 'backend'))

from models.simple_sgd_engine import SimpleSgdEngine

def test_recommendations():
    print("🧪 Testing recommendations with new embeddings...")
    
    # Initialize engine
    engine = SimpleSgdEngine()
    print(f"✅ Engine loaded with {len(engine.available_watches)} watches")
    
    # Get recommendations
    recs = engine.get_recommendations('test_session')
    print(f"✅ Generated {len(recs)} recommendations")
    
    # Show sample recommendation
    if recs:
        rec = recs[0]
        print(f"\n📊 Sample recommendation:")
        print(f"  Available keys: {list(rec.keys())}")
        print(f"  Watch ID: {rec['watch_id']}")
        print(f"  Brand: {rec['brand']}")
        print(f"  Model: {rec['model']}")
        print(f"  Confidence: {rec['confidence']:.3f}")
        
        # Check for image fields
        if 'image_url' in rec:
            print(f"  Image URL: {rec['image_url'][:80]}...")
        if 'main_image' in rec:
            print(f"  Main image: {rec['main_image'][:80]}...")
    
    print(f"\n✅ Recommendations test completed successfully!")

if __name__ == "__main__":
    test_recommendations() 
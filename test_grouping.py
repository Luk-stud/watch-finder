#!/usr/bin/env python3
"""Test brand+model grouping functionality"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'watch_finder_v2', 'backend'))
from models.simple_sgd_engine import SimpleSgdEngine

def test_grouping():
    print("🧪 Testing brand+model grouping functionality...")
    
    # Initialize engine
    engine = SimpleSgdEngine()
    print(f"✅ Engine loaded: {len(engine.available_watches)} watches")
    print(f"✅ Brand+model groups: {len(engine.brand_model_groups)}")
    print(f"✅ Multi-variant groups: {len([g for g in engine.brand_model_groups.values() if len(g) > 1])}")
    
    # Show some examples of multi-variant groups
    print("\n📊 Sample multi-variant groups:")
    multi_groups = [(k, v) for k, v in engine.brand_model_groups.items() if len(v) > 1][:5]
    for key, watch_ids in multi_groups:
        brand, model = key.split('|', 1)
        print(f"  {brand} {model}: {len(watch_ids)} variants")
    
    # Test recommendation exclusion
    print("\n🧪 Testing recommendation exclusion...")
    recs1 = engine.get_recommendations('test_session')
    print(f"✅ First recommendations: {len(recs1)}")
    
    if recs1:
        first_watch = recs1[0]
        print(f"✅ First: {first_watch['brand']} {first_watch['model']}")
        
        # Get similar watches
        similar = engine._get_similar_watches(first_watch['watch_id'])
        print(f"✅ Similar watches: {len(similar)}")
        
        # Get second recommendations
        recs2 = engine.get_recommendations('test_session')
        print(f"✅ Second recommendations: {len(recs2)}")
        
        if recs2:
            second_watch = recs2[0]
            print(f"✅ Second: {second_watch['brand']} {second_watch['model']}")
            
            # Check if they're different
            is_different = second_watch['watch_id'] != first_watch['watch_id']
            print(f"✅ Different from first: {is_different}")
            
            if is_different:
                print("🎉 SUCCESS: Brand+model grouping is working correctly!")
            else:
                print("⚠️ WARNING: Second recommendation is the same as first")
        else:
            print("⚠️ WARNING: No second recommendations generated")
    else:
        print("⚠️ WARNING: No first recommendations generated")

if __name__ == "__main__":
    test_grouping() 
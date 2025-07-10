#!/usr/bin/env python3
"""Test brand+model grouping functionality"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'watch_finder_v2', 'backend'))

from models.simple_sgd_engine import SimpleSgdEngine

def test_brand_model_grouping():
    print("🧪 Testing brand+model grouping functionality...")
    
    # Initialize engine
    engine = SimpleSgdEngine()
    print(f"✅ Engine loaded: {len(engine.available_watches)} watches")
    print(f"✅ Brand+model groups: {len(engine.brand_model_groups)}")
    print(f"✅ Multi-variant groups: {len([g for g in engine.brand_model_groups.values() if len(g) > 1])}")
    
    # Show some examples of multi-variant groups
    print("\n📊 Sample multi-variant groups:")
    multi_groups = [(k, v) for k, v in engine.brand_model_groups.items() if len(v) > 1][:3]
    for key, watch_ids in multi_groups:
        brand, model = key.split('|', 1)
        print(f"  {brand} {model}: {len(watch_ids)} variants")
        # Show a few watch IDs from this group
        sample_ids = list(watch_ids)[:3]
        print(f"    Sample IDs: {sample_ids}")
    
    # Test recommendation exclusion
    print("\n🧪 Testing recommendation exclusion...")
    
    # Get first set of recommendations
    recs1 = engine.get_recommendations('test_session')
    print(f"✅ First recommendations: {len(recs1)}")
    
    if recs1:
        first_watch = recs1[0]
        print(f"  First: {first_watch['brand']} {first_watch['model']}")
        print(f"  Watch ID: {first_watch['watch_id']}")
        
        # Check how many similar watches exist
        similar = engine._get_similar_watches(first_watch['watch_id'])
        print(f"  Similar watches: {len(similar)}")
        print(f"  Similar IDs: {list(similar)[:3]}")
        
        # Get second set of recommendations
        recs2 = engine.get_recommendations('test_session')
        print(f"✅ Second recommendations: {len(recs2)}")
        
        if recs2:
            second_watch = recs2[0]
            print(f"  Second: {second_watch['brand']} {second_watch['model']}")
            print(f"  Watch ID: {second_watch['watch_id']}")
            
            # Check if second watch is different from first
            is_different = second_watch['watch_id'] != first_watch['watch_id']
            print(f"  Different from first: {is_different}")
            
            # Check if second watch is in the same brand+model group
            first_similar = engine._get_similar_watches(first_watch['watch_id'])
            second_in_first_group = second_watch['watch_id'] in first_similar
            print(f"  Second in first group: {second_in_first_group}")
            
            if not second_in_first_group:
                print("  ✅ SUCCESS: Second recommendation is from different brand+model group!")
            else:
                print("  ⚠️  WARNING: Second recommendation is from same brand+model group")
    
    print("\n✅ Brand+model grouping test completed!")

if __name__ == "__main__":
    test_brand_model_grouping() 
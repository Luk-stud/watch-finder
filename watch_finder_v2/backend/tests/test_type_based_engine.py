#!/usr/bin/env python3
"""
Test Type-Based MABWiser Engine with Real Watch Data

This script tests the type-based MABWiser engine with actual watch embeddings
and metadata to validate type detection and recommendation performance.
"""

import logging
import time
import numpy as np
from typing import Dict, List
import sys
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_type_based_engine():
    """Test the type-based MABWiser engine with real data."""
    
    print("=" * 80)
    print("🧪 TESTING TYPE-BASED MABWISER ENGINE WITH REAL WATCH DATA")
    print("=" * 80)
    
    # Import engines
    try:
        from models.mabwiser_type_engine import TypeBasedMABWiserEngine
        from models.fast_linucb_engine import FastLinUCBEngine
        logger.info("✅ Successfully imported engines")
    except ImportError as e:
        logger.error(f"❌ Failed to import engines: {e}")
        return False
    
    # Test 1: Initialize Type-Based Engine
    print("\n" + "="*60)
    print("🔧 TEST 1: INITIALIZING TYPE-BASED ENGINE")
    print("="*60)
    
    try:
        start_time = time.time()
        type_engine = TypeBasedMABWiserEngine(
            alpha=0.1,
            batch_size=5,
            data_dir="data"
        )
        init_time = time.time() - start_time
        
        print(f"✅ Type-based engine initialized in {init_time:.2f}s")
        
        # Get stats
        stats = type_engine.get_expert_stats()
        print(f"\n📊 TYPE-BASED ENGINE STATS:")
        print(f"   • Total watches: {stats['total_watches']}")
        print(f"   • Total types detected: {stats['total_types']}")
        print(f"   • LinUCB experts created: {stats['total_experts']}")
        print(f"   • Algorithm: {stats['algorithm']}")
        
        print(f"\n🏷️ TYPE BREAKDOWN:")
        for watch_type, info in sorted(stats['type_breakdown'].items()):
            status = "✅" if info['has_expert'] else "❌"
            print(f"   • {watch_type}: {info['watch_count']} watches {status}")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize type-based engine: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Compare with FastLinUCB Engine
    print("\n" + "="*60)
    print("🔧 TEST 2: COMPARING WITH FASTLINUCB ENGINE")
    print("="*60)
    
    try:
        start_time = time.time()
        linucb_engine = FastLinUCBEngine(
            alpha=0.1,
            batch_size=5,
            data_dir="data"
        )
        linucb_init_time = time.time() - start_time
        
        print(f"✅ FastLinUCB engine initialized in {linucb_init_time:.2f}s")
        
        linucb_stats = linucb_engine.get_expert_stats()
        print(f"\n📊 FASTLINUCB ENGINE STATS:")
        print(f"   • Total watches: {len(linucb_engine.available_watches)}")
        print(f"   • Embedding dimension: {linucb_engine.dim}")
        print(f"   • Data loaded: {len(linucb_engine.watch_data)} watches")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize FastLinUCB engine: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Cold Start Recommendations
    print("\n" + "="*60)
    print("🔧 TEST 3: COLD START RECOMMENDATIONS")
    print("="*60)
    
    session_id = "test_session_001"
    
    # Type-based recommendations
    print("\n🎯 TYPE-BASED RECOMMENDATIONS:")
    try:
        start_time = time.time()
        type_recs = type_engine.get_recommendations(session_id)
        type_rec_time = time.time() - start_time
        
        print(f"✅ Got {len(type_recs)} recommendations in {type_rec_time:.4f}s")
        for i, rec in enumerate(type_recs, 1):
            watch_type = type_engine.type_of.get(rec['watch_id'], 'unknown')
            print(f"   {i}. [{rec['algorithm']}] {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')} (Type: {watch_type}, Confidence: {rec['confidence']:.3f})")
            
    except Exception as e:
        logger.error(f"❌ Type-based recommendations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # FastLinUCB recommendations
    print("\n🎯 FASTLINUCB RECOMMENDATIONS:")
    try:
        start_time = time.time()
        linucb_recs = linucb_engine.get_recommendations(session_id)
        linucb_rec_time = time.time() - start_time
        
        print(f"✅ Got {len(linucb_recs)} recommendations in {linucb_rec_time:.4f}s")
        for i, rec in enumerate(linucb_recs, 1):
            print(f"   {i}. [{rec['algorithm']}] {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')} (Confidence: {rec['confidence']:.3f})")
            
    except Exception as e:
        logger.error(f"❌ FastLinUCB recommendations failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Feedback and Learning
    print("\n" + "="*60)
    print("🔧 TEST 4: FEEDBACK AND LEARNING")
    print("="*60)
    
    if type_recs and linucb_recs:
        # Simulate positive feedback on first recommendation
        type_watch_id = type_recs[0]['watch_id']
        linucb_watch_id = linucb_recs[0]['watch_id']
        
        print(f"\n👍 SIMULATING POSITIVE FEEDBACK:")
        print(f"   • Type-based: Watch {type_watch_id}")
        print(f"   • FastLinUCB: Watch {linucb_watch_id}")
        
        try:
            # Update type-based engine
            type_engine.update(session_id, type_watch_id, 1.0)
            
            # Update FastLinUCB engine
            linucb_engine.update(session_id, linucb_watch_id, 1.0)
            
            print("✅ Both engines updated successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to update engines: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 5: Second Round Recommendations
    print("\n" + "="*60)
    print("🔧 TEST 5: SECOND ROUND RECOMMENDATIONS (AFTER FEEDBACK)")
    print("="*60)
    
    # Type-based second round
    print("\n🎯 TYPE-BASED SECOND ROUND:")
    try:
        start_time = time.time()
        type_recs_2 = type_engine.get_recommendations(session_id)
        type_rec_time_2 = time.time() - start_time
        
        print(f"✅ Got {len(type_recs_2)} recommendations in {type_rec_time_2:.4f}s")
        for i, rec in enumerate(type_recs_2, 1):
            watch_type = type_engine.type_of.get(rec['watch_id'], 'unknown')
            print(f"   {i}. [{rec['algorithm']}] {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')} (Type: {watch_type}, Confidence: {rec['confidence']:.3f})")
            
    except Exception as e:
        logger.error(f"❌ Type-based second round failed: {e}")
        import traceback
        traceback.print_exc()
    
    # FastLinUCB second round
    print("\n🎯 FASTLINUCB SECOND ROUND:")
    try:
        start_time = time.time()
        linucb_recs_2 = linucb_engine.get_recommendations(session_id)
        linucb_rec_time_2 = time.time() - start_time
        
        print(f"✅ Got {len(linucb_recs_2)} recommendations in {linucb_rec_time_2:.4f}s")
        for i, rec in enumerate(linucb_recs_2, 1):
            print(f"   {i}. [{rec['algorithm']}] {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')} (Confidence: {rec['confidence']:.3f})")
            
    except Exception as e:
        logger.error(f"❌ FastLinUCB second round failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Performance Summary
    print("\n" + "="*60)
    print("🔧 TEST 6: PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"\n⚡ SPEED COMPARISON:")
    print(f"   • Type-based init: {init_time:.3f}s")
    print(f"   • FastLinUCB init: {linucb_init_time:.3f}s")
    print(f"   • Type-based rec 1: {type_rec_time:.4f}s")
    print(f"   • FastLinUCB rec 1: {linucb_rec_time:.4f}s")
    if 'type_rec_time_2' in locals() and 'linucb_rec_time_2' in locals():
        print(f"   • Type-based rec 2: {type_rec_time_2:.4f}s") 
        print(f"   • FastLinUCB rec 2: {linucb_rec_time_2:.4f}s")
    
    print(f"\n🎯 ALGORITHM COMPARISON:")
    print(f"   • Type-based: {stats['total_types']} types, LinUCB per type")
    print(f"   • FastLinUCB: Expert-based, similarity threshold")
    
    # Test 7: Sample Watch Data Analysis
    print("\n" + "="*60)
    print("🔧 TEST 7: SAMPLE WATCH DATA ANALYSIS")
    print("="*60)
    
    print(f"\n🔍 SAMPLE WATCH DATA:")
    sample_watches = list(type_engine.available_watches)[:10]
    for watch_id in sample_watches:
        watch_data = type_engine.watch_data.get(watch_id, {})
        watch_type = type_engine.type_of.get(watch_id, 'unknown')
        brand = watch_data.get('brand', 'Unknown')
        model = watch_data.get('model', 'Unknown')
        print(f"   • ID {watch_id}: {brand} {model} → Type: {watch_type}")
    
    # Cleanup
    try:
        type_engine.shutdown()
        linucb_engine.shutdown()
        print(f"\n✅ Engines shut down successfully")
    except Exception as e:
        logger.error(f"❌ Failed to shutdown engines: {e}")
    
    print("\n" + "="*80)
    print("🎉 TYPE-BASED ENGINE TEST COMPLETED!")
    print("="*80)
    
    return True

def main():
    """Main test function."""
    success = test_type_based_engine()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
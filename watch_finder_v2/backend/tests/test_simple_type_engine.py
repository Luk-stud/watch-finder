#!/usr/bin/env python3
"""
Simple Test for Type-Based MABWiser Engine

This script tests only the type-based MABWiser engine with actual watch embeddings.
"""

import logging
import time
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_type_engine():
    """Test the type-based MABWiser engine."""
    
    print("=" * 80)
    print("🧪 TESTING TYPE-BASED MABWISER ENGINE")
    print("=" * 80)
    
    # Import engine
    try:
        from models.mabwiser_type_engine import TypeBasedMABWiserEngine
        logger.info("✅ Successfully imported TypeBasedMABWiserEngine")
    except ImportError as e:
        logger.error(f"❌ Failed to import TypeBasedMABWiserEngine: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 1: Initialize Engine
    print("\n🔧 TEST 1: INITIALIZING TYPE-BASED ENGINE")
    print("-" * 50)
    
    try:
        start_time = time.time()
        engine = TypeBasedMABWiserEngine(
            alpha=0.1,
            batch_size=5,
            data_dir="data"
        )
        init_time = time.time() - start_time
        
        print(f"✅ Engine initialized in {init_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize engine: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Get Stats
    print("\n🔧 TEST 2: ENGINE STATISTICS")
    print("-" * 50)
    
    try:
        stats = engine.get_expert_stats()
        print(f"📊 ENGINE STATS:")
        print(f"   • Total watches: {stats['total_watches']}")
        print(f"   • Total types detected: {stats['total_types']}")
        print(f"   • LinUCB experts created: {stats['total_experts']}")
        print(f"   • Algorithm: {stats['algorithm']}")
        
        print(f"\n🏷️ TYPE BREAKDOWN:")
        for watch_type, info in sorted(stats['type_breakdown'].items()):
            status = "✅" if info['has_expert'] else "❌"
            print(f"   • {watch_type}: {info['watch_count']} watches {status}")
            
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Cold Start Recommendations
    print("\n🔧 TEST 3: COLD START RECOMMENDATIONS")
    print("-" * 50)
    
    session_id = "test_session_001"
    
    try:
        start_time = time.time()
        recommendations = engine.get_recommendations(session_id)
        rec_time = time.time() - start_time
        
        print(f"✅ Got {len(recommendations)} recommendations in {rec_time:.4f}s")
        for i, rec in enumerate(recommendations, 1):
            watch_type = engine.type_of.get(rec['watch_id'], 'unknown')
            print(f"   {i}. [{rec['algorithm']}] {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')} (Type: {watch_type}, Confidence: {rec['confidence']:.3f})")
            
    except Exception as e:
        logger.error(f"❌ Recommendations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Feedback
    print("\n🔧 TEST 4: FEEDBACK UPDATE")
    print("-" * 50)
    
    if recommendations:
        watch_id = recommendations[0]['watch_id']
        watch_type = engine.type_of.get(watch_id, 'unknown')
        
        print(f"👍 Giving positive feedback for watch {watch_id} (type: {watch_type})")
        
        try:
            engine.update(session_id, watch_id, 1.0)
            print("✅ Feedback update successful")
            
        except Exception as e:
            logger.error(f"❌ Failed to update with feedback: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test 5: Second Round
    print("\n🔧 TEST 5: SECOND ROUND RECOMMENDATIONS")
    print("-" * 50)
    
    try:
        start_time = time.time()
        recommendations_2 = engine.get_recommendations(session_id)
        rec_time_2 = time.time() - start_time
        
        print(f"✅ Got {len(recommendations_2)} recommendations in {rec_time_2:.4f}s")
        for i, rec in enumerate(recommendations_2, 1):
            watch_type = engine.type_of.get(rec['watch_id'], 'unknown')
            print(f"   {i}. [{rec['algorithm']}] {rec.get('brand', 'Unknown')} {rec.get('model', 'Unknown')} (Type: {watch_type}, Confidence: {rec['confidence']:.3f})")
            
    except Exception as e:
        logger.error(f"❌ Second round recommendations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Sample Data Analysis
    print("\n🔧 TEST 6: SAMPLE DATA ANALYSIS")
    print("-" * 50)
    
    try:
        print(f"🔍 SAMPLE WATCH DATA:")
        sample_watches = list(engine.available_watches)[:10]
        for watch_id in sample_watches:
            watch_data = engine.watch_data.get(watch_id, {})
            watch_type = engine.type_of.get(watch_id, 'unknown')
            brand = watch_data.get('brand', 'Unknown')
            model = watch_data.get('model', 'Unknown')
            print(f"   • ID {watch_id}: {brand} {model} → Type: {watch_type}")
            
    except Exception as e:
        logger.error(f"❌ Sample data analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    try:
        engine.shutdown()
        print(f"\n✅ Engine shut down successfully")
    except Exception as e:
        logger.error(f"❌ Failed to shutdown engine: {e}")
    
    print("\n" + "="*80)
    print("🎉 TYPE-BASED ENGINE TEST COMPLETED!")
    print("="*80)
    
    return True

def main():
    """Main test function."""
    success = test_type_engine()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
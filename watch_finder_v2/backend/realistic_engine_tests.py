#!/usr/bin/env python3
"""
Realistic Engine Tests: Test LinUCB with actual watch knowledge and expected outcomes
====================================================================================
"""

import os
import sys
import pickle
import numpy as np
from collections import defaultdict

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

def load_dataset_knowledge():
    """Load and categorize the dataset for realistic testing."""
    print("📚 Loading Dataset Knowledge...")
    
    # Load watch metadata
    data_dir = 'data'
    metadata_path = os.path.join(data_dir, 'watch_text_metadata.pkl')
    
    try:
        with open(metadata_path, 'rb') as f:
            metadata_list = pickle.load(f)
        
        # Categorize watches by type and characteristics
        watch_categories = {
            'dive_watches': [],
            'field_watches': [],
            'chronographs': [],
            'luxury_watches': [],
            'budget_watches': [],
            'microbrands': [],
            'premium_watches': []
        }
        
        for idx, watch_dict in enumerate(metadata_list):
            brand = watch_dict.get('brand', '').strip()
            model = watch_dict.get('model', '').strip()
            price = watch_dict.get('price', 0)
            
            watch_info = {
                'id': idx,
                'brand': brand,
                'model': model,
                'price': price,
                'name': f"{brand} {model}".strip()
            }
            
            model_lower = model.lower()
            brand_lower = brand.lower()
            
            # Categorize by type
            if any(term in model_lower for term in ['diver', 'dive', 'compressor']):
                watch_categories['dive_watches'].append(watch_info)
            
            if any(term in model_lower for term in ['field', 'military', 'pilot', 'aviation']):
                watch_categories['field_watches'].append(watch_info)
            
            if any(term in model_lower for term in ['chrono', 'chronograph']):
                watch_categories['chronographs'].append(watch_info)
            
            # Categorize by price
            if price > 2000:
                watch_categories['luxury_watches'].append(watch_info)
            elif price < 400:
                watch_categories['budget_watches'].append(watch_info)
            elif price >= 800:
                watch_categories['premium_watches'].append(watch_info)
            
            # Microbrands (recognizable independent brands)
            microbrands = ['carpenter', 'traska', 'halios', 'oak & oscar', 'studio underd0g', 
                          'baltic', 'farer', 'zelos', 'nodus', 'serica']
            if any(mb in brand_lower for mb in microbrands):
                watch_categories['microbrands'].append(watch_info)
        
        # Print categories
        print(f"📊 Dataset Categories:")
        for category, watches in watch_categories.items():
            if watches:
                avg_price = np.mean([w['price'] for w in watches if w['price'] > 0])
                print(f"  {category:15} | {len(watches):3d} watches | Avg: ${avg_price:4.0f}")
        
        return watch_categories
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return {}

def test_dive_watch_enthusiast(engine, watch_categories):
    """Test: User who likes dive watches should get more dive watch recommendations."""
    print(f"\n🏊 Test: Dive Watch Enthusiast")
    
    dive_watches = watch_categories.get('dive_watches', [])
    if len(dive_watches) < 2:
        print("   ⚠️  Not enough dive watches for test")
        return False
    
    session_id = "dive_enthusiast_test"
    engine.create_session(session_id)
    
    # Seed with 2 dive watches
    seed_watches = dive_watches[:2]
    print(f"   💖 Seeding with dive watches:")
    for watch in seed_watches:
        print(f"     • {watch['name']} (${watch['price']})")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))  # Like
    
    # Get recommendations
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    # Check if recommendations include more dive watches
    recommended_ids = [rec.get('watch_id') for rec in recommendations]
    dive_recommendations = 0
    
    print(f"   🎯 Recommendations:")
    for rec in recommendations:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(engine.watch_data):
            watch_data = engine.watch_data[watch_id]
            brand = watch_data.get('brand', '')
            model = watch_data.get('model', '')
            price = watch_data.get('price', 0)
            
            # Check if it's a dive watch
            is_dive = any(term in model.lower() for term in ['diver', 'dive', 'compressor'])
            if is_dive:
                dive_recommendations += 1
                print(f"     ✅ {brand} {model} (${price}) - DIVE WATCH")
            else:
                print(f"     • {brand} {model} (${price})")
    
    success = dive_recommendations >= 2  # At least 2 out of 5 should be dive watches
    print(f"   📊 Result: {dive_recommendations}/5 dive watch recommendations - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_budget_conscious_buyer(engine, watch_categories):
    """Test: Budget buyer should get affordable recommendations."""
    print(f"\n💰 Test: Budget-Conscious Buyer")
    
    budget_watches = watch_categories.get('budget_watches', [])
    if len(budget_watches) < 2:
        print("   ⚠️  Not enough budget watches for test")
        return False
    
    session_id = "budget_buyer_test"
    engine.create_session(session_id)
    
    # Seed with budget watches
    seed_watches = budget_watches[:2]
    print(f"   💖 Seeding with budget watches:")
    for watch in seed_watches:
        print(f"     • {watch['name']} (${watch['price']})")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    # Check price range of recommendations
    affordable_count = 0
    total_price = 0
    valid_prices = 0
    
    print(f"   🎯 Recommendations:")
    for rec in recommendations:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(engine.watch_data):
            watch_data = engine.watch_data[watch_id]
            brand = watch_data.get('brand', '')
            model = watch_data.get('model', '')
            price = watch_data.get('price', 0)
            
            if price > 0:
                total_price += price
                valid_prices += 1
                
                if price <= 800:  # Affordable threshold
                    affordable_count += 1
                    print(f"     ✅ {brand} {model} (${price}) - AFFORDABLE")
                else:
                    print(f"     💸 {brand} {model} (${price}) - EXPENSIVE")
            else:
                print(f"     • {brand} {model} (price unknown)")
    
    avg_price = total_price / valid_prices if valid_prices > 0 else 0
    success = affordable_count >= 3 and avg_price <= 1000  # Most should be affordable
    print(f"   📊 Result: {affordable_count}/5 affordable, avg ${avg_price:.0f} - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_luxury_watch_aspirant(engine, watch_categories):
    """Test: Luxury buyer should get premium recommendations."""
    print(f"\n💎 Test: Luxury Watch Aspirant")
    
    luxury_watches = watch_categories.get('luxury_watches', [])
    premium_watches = watch_categories.get('premium_watches', [])
    high_end = luxury_watches + premium_watches
    
    if len(high_end) < 2:
        print("   ⚠️  Not enough luxury watches for test")
        return False
    
    session_id = "luxury_aspirant_test"
    engine.create_session(session_id)
    
    # Seed with expensive watches
    seed_watches = sorted(high_end, key=lambda x: x['price'], reverse=True)[:2]
    print(f"   💖 Seeding with luxury watches:")
    for watch in seed_watches:
        print(f"     • {watch['name']} (${watch['price']})")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    # Check if recommendations are premium
    premium_count = 0
    total_price = 0
    valid_prices = 0
    
    print(f"   🎯 Recommendations:")
    for rec in recommendations:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(engine.watch_data):
            watch_data = engine.watch_data[watch_id]
            brand = watch_data.get('brand', '')
            model = watch_data.get('model', '')
            price = watch_data.get('price', 0)
            
            if price > 0:
                total_price += price
                valid_prices += 1
                
                if price >= 800:  # Premium threshold
                    premium_count += 1
                    print(f"     ✅ {brand} {model} (${price}) - PREMIUM")
                else:
                    print(f"     💰 {brand} {model} (${price}) - BUDGET")
            else:
                print(f"     • {brand} {model} (price unknown)")
    
    avg_price = total_price / valid_prices if valid_prices > 0 else 0
    success = premium_count >= 3 and avg_price >= 1000  # Most should be premium
    print(f"   📊 Result: {premium_count}/5 premium, avg ${avg_price:.0f} - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_microbrand_explorer(engine, watch_categories):
    """Test: Microbrand enthusiast should get independent brand recommendations."""
    print(f"\n🔍 Test: Microbrand Explorer")
    
    microbrands = watch_categories.get('microbrands', [])
    if len(microbrands) < 2:
        print("   ⚠️  Not enough microbrand watches for test")
        return False
    
    session_id = "microbrand_test"
    engine.create_session(session_id)
    
    # Seed with microbrands
    seed_watches = microbrands[:2]
    print(f"   💖 Seeding with microbrands:")
    for watch in seed_watches:
        print(f"     • {watch['name']} (${watch['price']})")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    # Check brand diversity
    recommended_brands = set()
    microbrand_count = 0
    
    print(f"   🎯 Recommendations:")
    for rec in recommendations:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(engine.watch_data):
            watch_data = engine.watch_data[watch_id]
            brand = watch_data.get('brand', '')
            model = watch_data.get('model', '')
            price = watch_data.get('price', 0)
            
            recommended_brands.add(brand)
            
            # Check if it's a known microbrand
            microbrand_names = ['carpenter', 'traska', 'halios', 'oak & oscar', 'studio underd0g', 
                              'baltic', 'farer', 'zelos', 'nodus', 'serica']
            is_microbrand = any(mb in brand.lower() for mb in microbrand_names)
            
            if is_microbrand:
                microbrand_count += 1
                print(f"     ✅ {brand} {model} (${price}) - MICROBRAND")
            else:
                print(f"     • {brand} {model} (${price})")
    
    success = microbrand_count >= 2 and len(recommended_brands) >= 3  # Good microbrand representation and diversity
    print(f"   📊 Result: {microbrand_count}/5 microbrands, {len(recommended_brands)} unique brands - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_expert_consistency(engine, watch_categories):
    """Test: Multiple requests should show consistent expert behavior."""
    print(f"\n🎯 Test: Expert Consistency")
    
    dive_watches = watch_categories.get('dive_watches', [])
    if len(dive_watches) < 1:
        print("   ⚠️  Not enough watches for test")
        return False
    
    session_id = "consistency_test"
    engine.create_session(session_id)
    
    # Like one dive watch
    seed_watch = dive_watches[0]
    print(f"   💖 Seeding with: {seed_watch['name']}")
    engine.update(session_id, seed_watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get multiple recommendation sets
    algorithm_counts = defaultdict(int)
    
    for i in range(3):
        recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
        for rec in recommendations:
            algorithm = rec.get('algorithm', 'unknown')
            algorithm_counts[algorithm] += 1
    
    print(f"   📊 Algorithm usage across 3 requests:")
    for algorithm, count in algorithm_counts.items():
        print(f"     {algorithm}: {count} times")
    
    # Check for expert dominance (good) vs randomness (bad)
    total_expert = sum(count for alg, count in algorithm_counts.items() if 'expert' in alg)
    total_random = algorithm_counts.get('Random', 0)
    
    success = total_expert > total_random  # Experts should dominate after training
    print(f"   📊 Result: {total_expert} expert vs {total_random} random - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def run_comprehensive_tests():
    """Run all realistic tests and provide optimization insights."""
    print("🚀 Running Comprehensive Realistic Tests...")
    
    # Load dataset knowledge
    watch_categories = load_dataset_knowledge()
    if not watch_categories:
        print("❌ Failed to load dataset")
        return
    
    # Initialize engine
    engine = OptimizedLinUCBEngine(
        dim=50,  # Match your current setup
        alpha=0.3,
        batch_size=5,
        max_experts=4,
        similarity_threshold=0.75
    )
    
    # Run tests
    tests = [
        ('Dive Watch Enthusiast', lambda: test_dive_watch_enthusiast(engine, watch_categories)),
        ('Budget-Conscious Buyer', lambda: test_budget_conscious_buyer(engine, watch_categories)),
        ('Luxury Watch Aspirant', lambda: test_luxury_watch_aspirant(engine, watch_categories)),
        ('Microbrand Explorer', lambda: test_microbrand_explorer(engine, watch_categories)),
        ('Expert Consistency', lambda: test_expert_consistency(engine, watch_categories))
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📋 Test Results Summary:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:25} | {status}")
    
    print(f"\n🎯 Overall Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Perfect! Your engine handles realistic scenarios excellently!")
    elif passed >= total * 0.8:
        print("👍 Good performance! Minor optimizations may help.")
    else:
        print("⚠️  Some issues detected. Consider tuning parameters or logic.")
    
    return results

if __name__ == "__main__":
    run_comprehensive_tests() 
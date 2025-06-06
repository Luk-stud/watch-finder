#!/usr/bin/env python3
"""
Advanced Optimization Tests: Push the engine to identify areas for improvement
============================================================================
"""

import os
import sys
import pickle
import numpy as np
from collections import defaultdict, Counter

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

def test_diverse_preferences_convergence():
    """Test: User with diverse tastes should create multiple experts."""
    print(f"\n🎨 Test: Diverse Preferences Convergence")
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4, similarity_threshold=0.75)
    session_id = "diverse_test"
    engine.create_session(session_id)
    
    # Load watch data to find very different watches
    try:
        with open('data/watch_text_metadata.pkl', 'rb') as f:
            metadata_list = pickle.load(f)
    except:
        print("   ⚠️  Cannot load dataset")
        return False
    
    # Find contrasting watches (different price ranges and types)
    diverse_watches = []
    
    # Budget dive watch
    for idx, watch in enumerate(metadata_list):
        if 'diver' in watch.get('model', '').lower() and watch.get('price', 0) < 500:
            diverse_watches.append((idx, watch, 'Budget Dive'))
            break
    
    # Luxury dress watch  
    for idx, watch in enumerate(metadata_list):
        if watch.get('price', 0) > 3000 and 'field' not in watch.get('model', '').lower():
            diverse_watches.append((idx, watch, 'Luxury'))
            break
    
    # Mid-range field watch
    for idx, watch in enumerate(metadata_list):
        if 'field' in watch.get('model', '').lower() and 500 < watch.get('price', 0) < 1500:
            diverse_watches.append((idx, watch, 'Field'))
            break
    
    if len(diverse_watches) < 3:
        print("   ⚠️  Not enough diverse watches found")
        return False
    
    print(f"   💖 Seeding with diverse preferences:")
    experts_before = len(engine.experts)
    
    for watch_id, watch_data, category in diverse_watches:
        brand = watch_data.get('brand', '')
        model = watch_data.get('model', '')
        price = watch_data.get('price', 0)
        print(f"     • {brand} {model} (${price}) - {category}")
        engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
    
    experts_after = len(engine.experts)
    experts_created = experts_after - experts_before
    
    # Get recommendations and check expert diversity
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    expert_algorithms = set()
    for rec in recommendations:
        alg = rec.get('algorithm', '')
        if 'expert' in alg:
            expert_algorithms.add(alg)
    
    print(f"   📊 Created {experts_created} experts, {len(expert_algorithms)} active in recommendations")
    success = experts_created >= 2 and len(expert_algorithms) >= 2
    print(f"   📊 Result: Multiple experts handling diversity - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_recommendation_quality_evolution():
    """Test: Recommendations should improve with more feedback."""
    print(f"\n📈 Test: Recommendation Quality Evolution")
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4, similarity_threshold=0.75)
    session_id = "quality_test"
    engine.create_session(session_id)
    
    # Find a specific category to focus on (field watches)
    field_watches = []
    try:
        with open('data/watch_text_metadata.pkl', 'rb') as f:
            metadata_list = pickle.load(f)
        
        for idx, watch in enumerate(metadata_list):
            if 'field' in watch.get('model', '').lower():
                field_watches.append({
                    'id': idx,
                    'brand': watch.get('brand', ''),
                    'model': watch.get('model', ''),
                    'price': watch.get('price', 0)
                })
    except:
        print("   ⚠️  Cannot load dataset")
        return False
    
    if len(field_watches) < 5:
        print("   ⚠️  Not enough field watches")
        return False
    
    # Phase 1: Like 2 field watches
    print(f"   💖 Phase 1: Liking 2 field watches")
    for watch in field_watches[:2]:
        print(f"     • {watch['brand']} {watch['model']}")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations and count field watches
    recommendations_phase1 = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    field_count_phase1 = 0
    
    for rec in recommendations_phase1:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(metadata_list):
            model = metadata_list[watch_id].get('model', '')
            if 'field' in model.lower():
                field_count_phase1 += 1
    
    # Phase 2: Like 2 more field watches
    print(f"   💖 Phase 2: Liking 2 more field watches")
    for watch in field_watches[2:4]:
        print(f"     • {watch['brand']} {watch['model']}")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations again
    recommendations_phase2 = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    field_count_phase2 = 0
    
    for rec in recommendations_phase2:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(metadata_list):
            model = metadata_list[watch_id].get('model', '')
            if 'field' in model.lower():
                field_count_phase2 += 1
    
    print(f"   📊 Field watch recommendations: Phase 1: {field_count_phase1}/5, Phase 2: {field_count_phase2}/5")
    
    # Quality should improve (more field watches recommended)
    success = field_count_phase2 >= field_count_phase1
    print(f"   📊 Result: Quality {'improved' if field_count_phase2 > field_count_phase1 else 'maintained' if field_count_phase2 == field_count_phase1 else 'decreased'} - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_price_sensitivity():
    """Test: Engine should learn price preferences."""
    print(f"\n💸 Test: Price Sensitivity Learning")
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4, similarity_threshold=0.75)
    session_id = "price_test"
    engine.create_session(session_id)
    
    # Find expensive watches (>$2000)
    expensive_watches = []
    try:
        with open('data/watch_text_metadata.pkl', 'rb') as f:
            metadata_list = pickle.load(f)
        
        for idx, watch in enumerate(metadata_list):
            price = watch.get('price', 0)
            if price > 2000:
                expensive_watches.append({
                    'id': idx,
                    'brand': watch.get('brand', ''),
                    'model': watch.get('model', ''),
                    'price': price
                })
    except:
        print("   ⚠️  Cannot load dataset")
        return False
    
    if len(expensive_watches) < 3:
        print("   ⚠️  Not enough expensive watches")
        return False
    
    # Like expensive watches
    print(f"   💖 Liking expensive watches:")
    for watch in expensive_watches[:3]:
        print(f"     • {watch['brand']} {watch['model']} (${watch['price']})")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations and check price distribution
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    prices = []
    expensive_count = 0
    
    print(f"   🎯 Recommendations:")
    for rec in recommendations:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(metadata_list):
            watch_data = metadata_list[watch_id]
            brand = watch_data.get('brand', '')
            model = watch_data.get('model', '')
            price = watch_data.get('price', 0)
            
            if price > 0:
                prices.append(price)
                if price > 1500:  # Premium threshold
                    expensive_count += 1
                    print(f"     💎 {brand} {model} (${price}) - PREMIUM")
                else:
                    print(f"     💰 {brand} {model} (${price}) - BUDGET")
    
    avg_price = np.mean(prices) if prices else 0
    success = expensive_count >= 3 and avg_price > 1500
    print(f"   📊 Result: {expensive_count}/5 premium, avg ${avg_price:.0f} - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_exploration_vs_exploitation():
    """Test: Balance between exploring new watches and exploiting learned preferences."""
    print(f"\n⚖️  Test: Exploration vs Exploitation Balance")
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4, similarity_threshold=0.75)
    session_id = "balance_test"
    engine.create_session(session_id)
    
    # Like one specific watch multiple times to establish strong preference
    liked_watch_id = 0  # Use first watch
    print(f"   💖 Establishing strong preference for watch {liked_watch_id}")
    
    # Multiple likes to train the expert thoroughly
    for _ in range(3):
        engine.update(session_id, liked_watch_id, 1.0, np.array([0.5, 0.5]))
    
    # Get multiple recommendation sets
    all_recommendations = []
    for i in range(5):  # 5 sets of recommendations
        recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
        all_recommendations.extend([rec.get('watch_id') for rec in recommendations])
    
    # Count unique watches recommended
    unique_watches = len(set(all_recommendations))
    total_recommendations = len(all_recommendations)
    diversity_ratio = unique_watches / total_recommendations
    
    # Count how often the liked watch appears
    liked_watch_frequency = all_recommendations.count(liked_watch_id) / total_recommendations
    
    print(f"   📊 Diversity: {unique_watches}/{total_recommendations} unique watches ({diversity_ratio:.2f})")
    print(f"   📊 Liked watch frequency: {liked_watch_frequency:.2f}")
    
    # Good balance: high diversity but some preference for liked watch
    success = diversity_ratio > 0.7 and 0.1 <= liked_watch_frequency <= 0.4
    print(f"   📊 Result: {'Good' if diversity_ratio > 0.7 else 'Low'} diversity, {'Good' if 0.1 <= liked_watch_frequency <= 0.4 else 'Poor'} exploitation - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def run_advanced_optimization_tests():
    """Run advanced tests to identify optimization opportunities."""
    print("🚀 Running Advanced Optimization Tests...")
    print("These tests push the engine to identify potential improvements.\n")
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4, similarity_threshold=0.75)
    
    tests = [
        ('Diverse Preferences Convergence', test_diverse_preferences_convergence),
        ('Recommendation Quality Evolution', test_recommendation_quality_evolution),
        ('Price Sensitivity Learning', test_price_sensitivity),
        ('Exploration vs Exploitation Balance', test_exploration_vs_exploitation)
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
    print(f"\n📋 Advanced Test Results:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:35} | {status}")
    
    print(f"\n🎯 Advanced Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Provide optimization suggestions
    if passed == total:
        print("🌟 Exceptional! Your engine excels at advanced scenarios!")
    elif passed >= 3:
        print("💪 Strong performance! Minor tweaks could perfect it.")
    else:
        print("🔧 Optimization opportunities identified:")
        
        failed_tests = [name for name, success in results if not success]
        if 'Diverse Preferences Convergence' in failed_tests:
            print("   • Consider lowering similarity_threshold for more expert creation")
        if 'Price Sensitivity Learning' in failed_tests:
            print("   • Price features might need stronger weighting in embeddings")
        if 'Exploration vs Exploitation Balance' in failed_tests:
            print("   • Consider adjusting alpha parameter for better exploration")
    
    return results

if __name__ == "__main__":
    run_advanced_optimization_tests() 
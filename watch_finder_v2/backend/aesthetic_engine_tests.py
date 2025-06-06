#!/usr/bin/env python3
"""
Aesthetic Engine Tests: Test LinUCB learning of visual and design preferences
===========================================================================
Tests that validate the engine learns from text + CLIP embeddings for:
- Visual style preferences (dial colors, case materials, designs)
- Watch type aesthetics (dive, field, dress, chronograph)
- Design elements (complications, strap materials, shapes)
- CLIP embedding consistency for visual similarity
"""

import os
import sys
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

def load_aesthetic_categories():
    """Load and categorize watches by aesthetic features, not price."""
    print("🎨 Loading Aesthetic Categories...")
    
    try:
        with open('data/watch_text_metadata.pkl', 'rb') as f:
            metadata_list = pickle.load(f)
        
        # Categorize by aesthetic features
        aesthetic_categories = {
            'black_dials': [],
            'blue_dials': [],
            'steel_cases': [],
            'bronze_cases': [],
            'dive_style': [],
            'field_style': [],
            'chronograph_style': []
        }
        
        for idx, watch_dict in enumerate(metadata_list):
            brand = watch_dict.get('brand', '').strip()
            model = watch_dict.get('model', '').strip()
            description = watch_dict.get('description', '').lower()
            
            watch_info = {
                'id': idx,
                'brand': brand,
                'model': model,
                'name': f"{brand} {model}".strip()
            }
            
            # Extract text features for categorization
            full_text = f"{model} {description}".lower()
            
            # Dial colors
            if any(term in full_text for term in ['black dial', 'black face', 'matte black', 'dark dial']):
                aesthetic_categories['black_dials'].append(watch_info)
            if any(term in full_text for term in ['blue dial', 'navy dial', 'azure', 'cobalt']):
                aesthetic_categories['blue_dials'].append(watch_info)
            
            # Case materials
            if any(term in full_text for term in ['steel', 'stainless', 'ss case', '316l']):
                aesthetic_categories['steel_cases'].append(watch_info)
            if any(term in full_text for term in ['bronze', 'brass', 'patina']):
                aesthetic_categories['bronze_cases'].append(watch_info)
            
            # Watch styles/types
            if any(term in full_text for term in ['diver', 'dive', 'compressor', 'underwater']):
                aesthetic_categories['dive_style'].append(watch_info)
            if any(term in full_text for term in ['field', 'military', 'pilot', 'aviation']):
                aesthetic_categories['field_style'].append(watch_info)
            if any(term in full_text for term in ['chrono', 'chronograph', 'stopwatch', 'racing']):
                aesthetic_categories['chronograph_style'].append(watch_info)
        
        # Print categories with counts
        print(f"🎨 Aesthetic Categories:")
        for category, watches in aesthetic_categories.items():
            if watches:
                print(f"  {category:20} | {len(watches):3d} watches")
        
        return aesthetic_categories
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return {}

def test_black_dial_preference(engine, categories):
    """Test: User who likes black dial watches gets more black dial recommendations."""
    print(f"\n⚫ Test: Black Dial Preference Learning")
    
    black_dials = categories.get('black_dials', [])
    if len(black_dials) < 2:
        print("   ⚠️  Not enough black dial watches for test")
        return False
    
    session_id = "black_dial_test"
    engine.create_session(session_id)
    
    # Seed with black dial watches
    seed_watches = black_dials[:2]
    print(f"   💖 Seeding with black dial watches:")
    for watch in seed_watches:
        print(f"     • {watch['name']}")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    # Check for black dial recommendations
    black_dial_count = 0
    print(f"   🎯 Recommendations:")
    for rec in recommendations:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(engine.watch_data):
            watch_data = engine.watch_data[watch_id]
            name = f"{watch_data.get('brand', '')} {watch_data.get('model', '')}".strip()
            
            # Check if it's a black dial watch
            is_black_dial = any(w['id'] == watch_id for w in black_dials)
            if is_black_dial:
                black_dial_count += 1
                print(f"     ✅ {name} - BLACK DIAL")
            else:
                print(f"     • {name}")
    
    success = black_dial_count >= 2  # At least 2 should be black dial
    print(f"   📊 Result: {black_dial_count}/5 black dial recommendations - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_dive_watch_aesthetic(engine, categories):
    """Test: User who likes dive watch aesthetics gets visually similar dive watches."""
    print(f"\n🏊 Test: Dive Watch Aesthetic Learning")
    
    dive_watches = categories.get('dive_style', [])
    if len(dive_watches) < 2:
        print("   ⚠️  Not enough dive style watches for test")
        return False
    
    session_id = "dive_aesthetic_test"
    engine.create_session(session_id)
    
    # Seed with dive watches
    seed_watches = dive_watches[:2]
    print(f"   💖 Seeding with dive style watches:")
    for watch in seed_watches:
        print(f"     • {watch['name']}")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    # Check for dive watch recommendations
    dive_count = 0
    field_count = 0  # Should be less preferred
    
    print(f"   🎯 Recommendations:")
    for rec in recommendations:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(engine.watch_data):
            watch_data = engine.watch_data[watch_id]
            name = f"{watch_data.get('brand', '')} {watch_data.get('model', '')}".strip()
            
            is_dive = any(w['id'] == watch_id for w in dive_watches)
            is_field = any(w['id'] == watch_id for w in categories.get('field_style', []))
            
            if is_dive:
                dive_count += 1
                print(f"     ✅ {name} - DIVE STYLE")
            elif is_field:
                field_count += 1
                print(f"     🏕️ {name} - FIELD STYLE")
            else:
                print(f"     • {name}")
    
    # Should prefer dive over field watches
    success = dive_count >= 2 and dive_count > field_count
    print(f"   📊 Result: {dive_count}/5 dive, {field_count}/5 field - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_case_material_preference(engine, categories):
    """Test: User who likes steel cases gets more steel case recommendations."""
    print(f"\n🔩 Test: Steel Case Material Preference")
    
    steel_cases = categories.get('steel_cases', [])
    bronze_cases = categories.get('bronze_cases', [])
    
    if len(steel_cases) < 2:
        print("   ⚠️  Not enough steel case watches for test")
        return False
    
    session_id = "steel_case_test"
    engine.create_session(session_id)
    
    # Seed with steel case watches
    seed_watches = steel_cases[:2]
    print(f"   💖 Seeding with steel case watches:")
    for watch in seed_watches:
        print(f"     • {watch['name']}")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    # Check material preferences
    steel_count = 0
    bronze_count = 0
    
    print(f"   🎯 Recommendations:")
    for rec in recommendations:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(engine.watch_data):
            watch_data = engine.watch_data[watch_id]
            name = f"{watch_data.get('brand', '')} {watch_data.get('model', '')}".strip()
            
            is_steel = any(w['id'] == watch_id for w in steel_cases)
            is_bronze = any(w['id'] == watch_id for w in bronze_cases)
            
            if is_steel:
                steel_count += 1
                print(f"     ✅ {name} - STEEL CASE")
            elif is_bronze:
                bronze_count += 1
                print(f"     🥉 {name} - BRONZE CASE")
            else:
                print(f"     • {name}")
    
    # Should prefer steel over bronze
    success = steel_count >= 2 and steel_count > bronze_count
    print(f"   📊 Result: {steel_count}/5 steel, {bronze_count}/5 bronze - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_complication_preference(engine, categories):
    """Test: User who likes GMT complications gets more GMT watches."""
    print(f"\n🌍 Test: GMT Complication Preference")
    
    gmt_watches = categories.get('gmt_complications', [])
    date_only = categories.get('date_only', [])
    
    if len(gmt_watches) < 2:
        print("   ⚠️  Not enough GMT watches for test")
        return False
    
    session_id = "gmt_complication_test"
    engine.create_session(session_id)
    
    # Seed with GMT watches
    seed_watches = gmt_watches[:2]
    print(f"   💖 Seeding with GMT watches:")
    for watch in seed_watches:
        print(f"     • {watch['name']}")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    # Check complication preferences
    gmt_count = 0
    simple_count = 0
    
    print(f"   🎯 Recommendations:")
    for rec in recommendations:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(engine.watch_data):
            watch_data = engine.watch_data[watch_id]
            name = f"{watch_data.get('brand', '')} {watch_data.get('model', '')}".strip()
            
            is_gmt = any(w['id'] == watch_id for w in gmt_watches)
            is_simple = any(w['id'] == watch_id for w in date_only)
            
            if is_gmt:
                gmt_count += 1
                print(f"     ✅ {name} - GMT COMPLICATION")
            elif is_simple:
                simple_count += 1
                print(f"     📅 {name} - DATE ONLY")
            else:
                print(f"     • {name}")
    
    # Should prefer GMT over simple date
    success = gmt_count >= 1 and gmt_count >= simple_count
    print(f"   📊 Result: {gmt_count}/5 GMT, {simple_count}/5 simple - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_visual_similarity_consistency(engine, categories):
    """Test: Engine should show consistent behavior with visually similar watches."""
    print(f"\n👁️ Test: Visual Similarity Consistency")
    
    # Use blue dial watches as a cohesive visual group
    blue_dials = categories.get('blue_dials', [])
    if len(blue_dials) < 3:
        print("   ⚠️  Not enough blue dial watches for test")
        return False
    
    session_id = "visual_consistency_test"
    engine.create_session(session_id)
    
    # Train on one blue dial watch
    seed_watch = blue_dials[0]
    print(f"   💖 Training on: {seed_watch['name']}")
    engine.update(session_id, seed_watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Get UCB scores for other blue dial watches (should be similar)
    if session_id in engine.session_experts and engine.session_experts[session_id]:
        expert_id = engine.session_experts[session_id][0]
        expert = engine.experts[expert_id]
        
        blue_scores = []
        for watch in blue_dials[1:4]:  # Test 3 other blue dial watches
            if watch['id'] in engine.session_embeddings[session_id]:
                embedding = engine.session_embeddings[session_id][watch['id']]
                score = expert.get_ucb_score(watch['id'], embedding)
                blue_scores.append(score)
                print(f"     🔵 {watch['name']}: UCB {score:.3f}")
        
        if len(blue_scores) >= 2:
            score_std = np.std(blue_scores)
            avg_score = np.mean(blue_scores)
            
            # Similar watches should have relatively consistent scores
            consistency = score_std / max(avg_score, 0.1)  # Coefficient of variation
            success = consistency < 0.5  # Less than 50% variation
            
            print(f"   📊 Score consistency: std={score_std:.3f}, avg={avg_score:.3f}, CV={consistency:.3f}")
            print(f"   📊 Result: {'✅ PASS' if success else '❌ FAIL'} - {'Good' if success else 'Poor'} visual consistency")
            return success
    
    print("   ⚠️  Could not test visual consistency")
    return False

def test_aesthetic_expert_specialization(engine, categories):
    """Test: Different experts should specialize in different aesthetic preferences."""
    print(f"\n🎯 Test: Aesthetic Expert Specialization")
    
    dive_watches = categories.get('dive_style', [])
    dress_watches = categories.get('dress_style', [])
    
    if len(dive_watches) < 2 or len(dress_watches) < 2:
        print("   ⚠️  Not enough watches for specialization test")
        return False
    
    session_id = "specialization_test"
    engine.create_session(session_id)
    
    # Create two distinct preference patterns
    print(f"   💖 Creating Expert 1: Dive watch preferences")
    for watch in dive_watches[:2]:
        print(f"     • {watch['name']}")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Force creation of second expert with very different preference
    print(f"   💖 Creating Expert 2: Dress watch preferences")
    for watch in dress_watches[:2]:
        print(f"     • {watch['name']}")
        engine.update(session_id, watch['id'], 1.0, np.array([0.5, 0.5]))
    
    # Check if multiple experts were created
    num_experts = len(engine.session_experts.get(session_id, []))
    
    # Get recommendations and see expert contributions
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    expert_contributions = {}
    for rec in recommendations:
        algorithm = rec.get('algorithm', 'unknown')
        expert_contributions[algorithm] = expert_contributions.get(algorithm, 0) + 1
    
    print(f"   📊 Experts created: {num_experts}")
    print(f"   📊 Expert contributions: {expert_contributions}")
    
    # Success if multiple experts created and both contribute
    success = num_experts >= 2 and len([alg for alg in expert_contributions.keys() if 'expert' in alg]) >= 2
    print(f"   📊 Result: {'✅ PASS' if success else '❌ FAIL'} - {'Good' if success else 'Poor'} expert specialization")
    return success

def test_aesthetic_learning_convergence(engine, categories):
    """Test: Aesthetic preferences should improve over 5 iterations."""
    print(f"\n📈 Test: Aesthetic Learning Convergence")
    
    black_dials = categories.get('black_dials', [])
    if len(black_dials) < 6:
        print("   ⚠️  Not enough black dial watches for convergence test")
        return False
    
    session_id = "convergence_test"
    engine.create_session(session_id)
    
    # Define user preference (likes black dial watches)
    preferred_watches = black_dials[:4]  # Training set
    all_black_dial_ids = {w['id'] for w in black_dials}
    
    print(f"   🎯 Simulating user who likes black dial watches")
    
    # Track recommendation quality over time
    iteration_scores = []
    
    for iteration in range(5):
        # Get recommendations
        recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
        
        # Calculate relevance score (how many recommended watches are black dial)
        relevant_count = 0
        for rec in recommendations:
            watch_id = rec.get('watch_id')
            if watch_id in all_black_dial_ids:
                relevant_count += 1
        
        relevance_score = relevant_count / len(recommendations)
        iteration_scores.append(relevance_score)
        
        # Simulate user feedback (like black dial watches from recommendations)
        feedback_given = 0
        for rec in recommendations:
            watch_id = rec.get('watch_id')
            if watch_id in all_black_dial_ids and feedback_given < 2:
                engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
                feedback_given += 1
            elif watch_id not in all_black_dial_ids and feedback_given < 1:
                # Give some negative feedback to non-black dial watches
                engine.update(session_id, watch_id, 0.0, np.array([0.5, 0.5]))
                feedback_given += 1
        
        print(f"     Iteration {iteration + 1}: {relevant_count}/5 black dial recommendations ({relevance_score:.1%})")
    
    # Check if learning improved
    early_avg = np.mean(iteration_scores[:2])
    late_avg = np.mean(iteration_scores[-2:])
    improvement = late_avg - early_avg
    
    print(f"   📈 Learning analysis: {early_avg:.1%} → {late_avg:.1%} (improvement: {improvement:+.1%})")
    
    success = improvement > 0.1  # Expect at least 10% improvement
    print(f"   📊 Result: {'✅ PASS' if success else '❌ FAIL'} - {'Good' if success else 'Poor'} aesthetic convergence")
    return success

def test_dive_style_convergence(engine, categories):
    """Test: Dive watch style preferences should improve over iterations."""
    print(f"\n🏊 Test: Dive Style Learning Convergence")
    
    dive_watches = categories.get('dive_style', [])
    field_watches = categories.get('field_style', [])
    
    if len(dive_watches) < 6:
        print("   ⚠️  Not enough dive watches for convergence test")
        return False
    
    session_id = "dive_convergence_test"
    engine.create_session(session_id)
    
    # Define user preference (likes dive watches, dislikes field watches)
    all_dive_ids = {w['id'] for w in dive_watches}
    all_field_ids = {w['id'] for w in field_watches}
    
    print(f"   🎯 Simulating user who likes dive watches over field watches")
    
    # Track preference learning over time
    iteration_scores = []
    
    for iteration in range(5):
        # Get recommendations
        recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
        
        # Calculate dive preference score
        dive_count = 0
        field_count = 0
        for rec in recommendations:
            watch_id = rec.get('watch_id')
            if watch_id in all_dive_ids:
                dive_count += 1
            elif watch_id in all_field_ids:
                field_count += 1
        
        # Score: positive for dive watches, negative for field watches
        preference_score = (dive_count - field_count) / len(recommendations)
        iteration_scores.append(preference_score)
        
        # Simulate user feedback
        feedback_given = 0
        for rec in recommendations:
            watch_id = rec.get('watch_id')
            if watch_id in all_dive_ids and feedback_given < 2:
                engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))  # Like dive watches
                feedback_given += 1
            elif watch_id in all_field_ids and feedback_given < 1:
                engine.update(session_id, watch_id, 0.0, np.array([0.5, 0.5]))  # Dislike field watches
                feedback_given += 1
        
        print(f"     Iteration {iteration + 1}: {dive_count} dive, {field_count} field (score: {preference_score:+.2f})")
    
    # Check if dive preference improved
    early_avg = np.mean(iteration_scores[:2])
    late_avg = np.mean(iteration_scores[-2:])
    improvement = late_avg - early_avg
    
    print(f"   📈 Style learning: {early_avg:+.2f} → {late_avg:+.2f} (improvement: {improvement:+.2f})")
    
    success = improvement > 0.2  # Expect significant style preference improvement
    print(f"   📊 Result: {'✅ PASS' if success else '❌ FAIL'} - {'Good' if success else 'Poor'} style convergence")
    return success

def run_aesthetic_tests():
    """Run all aesthetic-focused tests."""
    print("🎨 Running Aesthetic Learning Tests for Microbrand Dataset")
    print("=" * 70)
    
    # Load aesthetic categories
    categories = load_aesthetic_categories()
    if not categories:
        print("❌ Failed to load aesthetic categories")
        return
    
    # Initialize engine with optimal parameters
    engine = OptimizedLinUCBEngine(
        dim=50,  # Optimal: PCA-reduced dimensions work better for convergence
        alpha=0.1,  # Optimal: Lower exploration for better aesthetic learning
        batch_size=5,
        max_experts=4,
        similarity_threshold=0.7  # Optimal: Broader similarity for better learning
    )
    
    # Run aesthetic tests (immediate learning)
    immediate_tests = [
        ('Black Dial Preference', lambda: test_black_dial_preference(engine, categories)),
        ('Dive Watch Aesthetic', lambda: test_dive_watch_aesthetic(engine, categories)),
        ('Steel Case Material', lambda: test_case_material_preference(engine, categories)),
        ('Visual Similarity', lambda: test_visual_similarity_consistency(engine, categories))
    ]
    
    # Run convergence tests (learning over time)
    convergence_tests = [
        ('Aesthetic Convergence', lambda: test_aesthetic_learning_convergence(engine, categories)),
        ('Dive Style Convergence', lambda: test_dive_style_convergence(engine, categories))
    ]
    
    all_tests = immediate_tests + convergence_tests
    results = []
    
    for test_name, test_func in all_tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n📋 Aesthetic Learning Test Results:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    immediate_passed = sum(1 for i, (_, success) in enumerate(results) if success and i < len(immediate_tests))
    convergence_passed = sum(1 for i, (_, success) in enumerate(results) if success and i >= len(immediate_tests))
    
    print(f"\n🎯 Immediate Learning Tests:")
    for i, (test_name, success) in enumerate(results[:len(immediate_tests)]):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:25} | {status}")
    
    print(f"\n📈 Convergence Learning Tests:")
    for i, (test_name, success) in enumerate(results[len(immediate_tests):]):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:25} | {status}")
    
    print(f"\n🎯 Overall Aesthetic Learning Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"   📊 Immediate Learning: {immediate_passed}/{len(immediate_tests)} ({immediate_passed/len(immediate_tests)*100:.1f}%)")
    print(f"   📈 Convergence Learning: {convergence_passed}/{len(convergence_tests)} ({convergence_passed/len(convergence_tests)*100:.1f}%)")
    
    if passed == total:
        print("🎉 Excellent! Your engine learns aesthetic preferences perfectly!")
    elif passed >= total * 0.8:
        print("👍 Good aesthetic learning! Minor tuning may help.")
    elif passed >= total * 0.6:
        print("⚠️  Moderate aesthetic learning. Consider embedding improvements.")
    else:
        print("❌ Poor aesthetic learning. Check embedding quality and algorithm.")
    
    return results

if __name__ == "__main__":
    run_aesthetic_tests() 
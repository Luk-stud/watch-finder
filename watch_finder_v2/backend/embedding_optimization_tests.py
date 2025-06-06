#!/usr/bin/env python3
"""
Embedding-Based Optimization Tests: Focus on watch characteristics and embedding quality
========================================================================================
"""

import os
import sys
import pickle
import numpy as np
from collections import defaultdict

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

def test_embedding_similarity_patterns():
    """Test: Similar watches should have similar embeddings and get recommended together."""
    print(f"\n🎯 Test: Embedding Similarity Patterns")
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4, similarity_threshold=0.75)
    session_id = "similarity_test"
    engine.create_session(session_id)
    
    # Load watch metadata to find similar watches
    try:
        with open('data/watch_text_metadata.pkl', 'rb') as f:
            metadata_list = pickle.load(f)
    except:
        print("   ⚠️  Cannot load dataset")
        return False
    
    # Find watches with similar characteristics
    dive_watches = []
    field_watches = []
    dress_watches = []
    
    for idx, watch in enumerate(metadata_list[:100]):  # Sample first 100
        model_lower = watch.get('model', '').lower()
        
        if any(term in model_lower for term in ['diver', 'dive', 'compressor', 'ocean', 'sea']):
            dive_watches.append(idx)
        elif any(term in model_lower for term in ['field', 'military', 'pilot', 'tactical']):
            field_watches.append(idx)
        elif any(term in model_lower for term in ['dress', 'classic', 'formal', 'elegant']):
            dress_watches.append(idx)
    
    # Test with dive watches if we have enough
    if len(dive_watches) >= 2:
        category = "dive watches"
        test_watches = dive_watches[:2]
    elif len(field_watches) >= 2:
        category = "field watches"
        test_watches = field_watches[:2]
    else:
        print("   ⚠️  Not enough similar watches found")
        return False
    
    print(f"   💖 Training on {category}:")
    for watch_id in test_watches:
        watch_data = metadata_list[watch_id]
        brand = watch_data.get('brand', '')
        model = watch_data.get('model', '')
        print(f"     • {brand} {model}")
        engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
    
    # Get recommendations
    recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
    
    # Check if similar watches are recommended
    similar_count = 0
    print(f"   🎯 Recommendations:")
    
    for rec in recommendations:
        watch_id = rec.get('watch_id')
        if watch_id is not None and watch_id < len(metadata_list):
            watch_data = metadata_list[watch_id]
            brand = watch_data.get('brand', '')
            model = watch_data.get('model', '')
            model_lower = model.lower()
            
            # Check if it matches the category we trained on
            is_similar = False
            if category == "dive watches" and any(term in model_lower for term in ['diver', 'dive', 'compressor', 'ocean', 'sea']):
                is_similar = True
            elif category == "field watches" and any(term in model_lower for term in ['field', 'military', 'pilot', 'tactical']):
                is_similar = True
            
            if is_similar:
                similar_count += 1
                print(f"     ✅ {brand} {model} - SIMILAR")
            else:
                print(f"     • {brand} {model}")
    
    success = similar_count >= 2  # At least 2 similar watches recommended
    print(f"   📊 Result: {similar_count}/5 similar watches - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_embedding_space_utilization():
    """Test: Engine should utilize the full embedding space for differentiation."""
    print(f"\n📊 Test: Embedding Space Utilization")
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4, similarity_threshold=0.75)
    session_id = "space_test"
    engine.create_session(session_id)
    
    # Sample diverse watches and check their embedding distances
    diverse_watch_ids = [0, 10, 20, 30, 40]  # Sample across dataset
    embeddings = []
    
    print(f"   📐 Sampling watch embeddings:")
    for watch_id in diverse_watch_ids:
        if watch_id in engine.session_embeddings[session_id]:
            embedding = engine.session_embeddings[session_id][watch_id]
            embeddings.append(embedding)
            
            # Check embedding statistics
            embed_norm = np.linalg.norm(embedding)
            embed_mean = np.mean(embedding)
            embed_std = np.std(embedding)
            
            print(f"     Watch {watch_id}: norm={embed_norm:.3f}, mean={embed_mean:.3f}, std={embed_std:.3f}")
    
    if len(embeddings) < 3:
        print("   ⚠️  Not enough embeddings")
        return False
    
    # Calculate pairwise distances
    distances = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            distance = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(distance)
    
    avg_distance = np.mean(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    
    print(f"   📏 Embedding distances: avg={avg_distance:.3f}, min={min_distance:.3f}, max={max_distance:.3f}")
    
    # Good utilization: reasonable spread and not all identical
    success = avg_distance > 0.1 and (max_distance - min_distance) > 0.05
    print(f"   📊 Result: {'Good' if success else 'Poor'} embedding space utilization - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_context_combination_effectiveness():
    """Test: Context combination should create meaningful differences."""
    print(f"\n🔄 Test: Context Combination Effectiveness")
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4, similarity_threshold=0.75)
    session_id = "context_test"
    engine.create_session(session_id)
    
    # Create an expert and test context combination
    expert_id = engine._create_new_expert()
    expert = engine.experts[expert_id]
    
    # Train expert on a watch
    first_watch_id = 0
    first_embedding = engine.session_embeddings[session_id][first_watch_id]
    expert.add_liked_watch(first_watch_id, first_embedding)
    
    print(f"   🎓 Expert created and trained on watch {first_watch_id}")
    print(f"   📐 Expert centroid norm: {np.linalg.norm(expert.centroid):.3f}")
    
    # Test context combination for different watches
    test_watch_ids = [1, 2, 3, 4, 5]
    combined_contexts = []
    
    print(f"   🔄 Testing context combination:")
    for watch_id in test_watch_ids:
        if watch_id in engine.session_embeddings[session_id]:
            watch_embedding = engine.session_embeddings[session_id][watch_id]
            combined_context = expert._combine_context(expert.centroid, watch_embedding)
            combined_contexts.append(combined_context)
            
            context_norm = np.linalg.norm(combined_context)
            print(f"     Watch {watch_id}: combined norm={context_norm:.3f}")
    
    if len(combined_contexts) < 3:
        print("   ⚠️  Not enough contexts")
        return False
    
    # Check if combined contexts are different
    context_distances = []
    for i in range(len(combined_contexts)):
        for j in range(i+1, len(combined_contexts)):
            distance = np.linalg.norm(combined_contexts[i] - combined_contexts[j])
            context_distances.append(distance)
    
    avg_context_distance = np.mean(context_distances)
    min_context_distance = np.min(context_distances)
    
    print(f"   📏 Context distances: avg={avg_context_distance:.3f}, min={min_context_distance:.3f}")
    
    # Contexts should be different (not identical)
    success = avg_context_distance > 0.01 and min_context_distance > 0.001
    print(f"   📊 Result: {'Good' if success else 'Poor'} context differentiation - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def test_ucb_score_distribution():
    """Test: UCB scores should have meaningful distribution."""
    print(f"\n📈 Test: UCB Score Distribution")
    
    engine = OptimizedLinUCBEngine(dim=50, alpha=0.3, batch_size=5, max_experts=4, similarity_threshold=0.75)
    session_id = "score_test"
    engine.create_session(session_id)
    
    # Create and train an expert
    expert_id = engine._create_new_expert()
    expert = engine.experts[expert_id]
    
    # Train on multiple watches to get good statistics
    training_watches = [0, 1, 2]
    print(f"   🎓 Training expert on {len(training_watches)} watches")
    
    for watch_id in training_watches:
        if watch_id in engine.session_embeddings[session_id]:
            watch_embedding = engine.session_embeddings[session_id][watch_id]
            expert.add_liked_watch(watch_id, watch_embedding)
            expert.update(watch_id, 1.0, watch_embedding)
    
    # Get UCB scores for various watches
    test_watches = list(range(10, 20))  # Different watches for testing
    scores = []
    
    print(f"   📊 UCB scores for test watches:")
    for watch_id in test_watches:
        if watch_id in engine.session_embeddings[session_id]:
            watch_embedding = engine.session_embeddings[session_id][watch_id]
            score = expert.get_ucb_score(watch_id, watch_embedding)
            scores.append(score)
            print(f"     Watch {watch_id}: {score:.3f}")
    
    if len(scores) < 5:
        print("   ⚠️  Not enough scores")
        return False
    
    # Analyze score distribution
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    score_range = max_score - min_score
    
    print(f"   📊 Score statistics: mean={mean_score:.3f}, std={std_score:.3f}, range={score_range:.3f}")
    
    # Good distribution: reasonable spread and not all identical
    success = std_score > 0.01 and score_range > 0.05
    print(f"   📊 Result: {'Good' if success else 'Poor'} score distribution - {'✅ PASS' if success else '❌ FAIL'}")
    return success

def run_embedding_optimization_tests():
    """Run embedding-focused tests to optimize recommendation quality."""
    print("🚀 Running Embedding Optimization Tests...")
    print("Focusing on watch characteristics and embedding quality.\n")
    
    tests = [
        ('Embedding Similarity Patterns', test_embedding_similarity_patterns),
        ('Embedding Space Utilization', test_embedding_space_utilization), 
        ('Context Combination Effectiveness', test_context_combination_effectiveness),
        ('UCB Score Distribution', test_ucb_score_distribution)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary and optimization recommendations
    print(f"\n📋 Embedding Optimization Results:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name:35} | {status}")
    
    print(f"\n🎯 Embedding Score: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    # Provide specific optimization suggestions
    print(f"\n💡 Embedding Optimization Recommendations:")
    
    failed_tests = [name for name, success in results if not success]
    
    if not failed_tests:
        print("   🌟 Embeddings are working excellently!")
    else:
        if 'Embedding Similarity Patterns' in failed_tests:
            print("   🔧 Consider improving text/CLIP embedding quality or weighting")
        if 'Embedding Space Utilization' in failed_tests:
            print("   🔧 Embeddings may be too similar - check PCA dimensionality reduction")
        if 'Context Combination Effectiveness' in failed_tests:
            print("   🔧 Context combination logic needs refinement for better differentiation")
        if 'UCB Score Distribution' in failed_tests:
            print("   🔧 Consider adjusting alpha parameter or arm update logic")
    
    print(f"\n🎯 Focus Areas for Better Recommendations:")
    print(f"   • Ensure text and CLIP embeddings capture different watch aspects")
    print(f"   • Optimize context combination weights (expert vs. watch features)")
    print(f"   • Fine-tune similarity threshold for expert creation")
    print(f"   • Validate that embeddings distinguish between watch styles")
    
    return results

if __name__ == "__main__":
    run_embedding_optimization_tests() 
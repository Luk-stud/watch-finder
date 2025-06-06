#!/usr/bin/env python3
"""
Parameter Testing Suite: Find optimal LinUCB parameters for aesthetic learning
============================================================================
Tests different combinations of:
- alpha (exploration vs exploitation)
- similarity_threshold (expert creation sensitivity)
- max_experts (expert diversity)
- dim (embedding dimension)
"""

import os
import sys
import pickle
import numpy as np
from collections import defaultdict
import itertools

# Add backend directory to path
sys.path.append(os.path.dirname(__file__))

from models.optimized_linucb_engine import OptimizedLinUCBEngine

def load_test_categories():
    """Load aesthetic categories for testing."""
    try:
        with open('data/watch_text_metadata.pkl', 'rb') as f:
            metadata_list = pickle.load(f)
        
        categories = {'black_dials': [], 'dive_style': []}
        
        for idx, watch_dict in enumerate(metadata_list):
            brand = watch_dict.get('brand', '').strip()
            model = watch_dict.get('model', '').strip()
            description = watch_dict.get('description', '').lower()
            
            watch_info = {'id': idx, 'brand': brand, 'model': model, 'name': f"{brand} {model}".strip()}
            full_text = f"{model} {description}".lower()
            
            if any(term in full_text for term in ['black dial', 'black face', 'matte black', 'dark dial']):
                categories['black_dials'].append(watch_info)
            if any(term in full_text for term in ['diver', 'dive', 'compressor', 'underwater']):
                categories['dive_style'].append(watch_info)
        
        return categories
    except Exception as e:
        print(f"❌ Error loading categories: {e}")
        return {}

def test_convergence_with_parameters(alpha, similarity_threshold, max_experts, dim, categories):
    """Test aesthetic convergence with specific parameters."""
    
    # Initialize engine with test parameters
    engine = OptimizedLinUCBEngine(
        dim=dim,
        alpha=alpha,
        batch_size=5,
        max_experts=max_experts,
        similarity_threshold=similarity_threshold
    )
    
    black_dials = categories.get('black_dials', [])
    if len(black_dials) < 6:
        return None, "Not enough black dial watches"
    
    session_id = f"param_test_{alpha}_{similarity_threshold}_{max_experts}_{dim}"
    engine.create_session(session_id)
    
    # Define user preference (likes black dial watches)
    all_black_dial_ids = {w['id'] for w in black_dials}
    
    # Track recommendation quality over time
    iteration_scores = []
    
    for iteration in range(5):
        # Get recommendations
        recommendations = engine.get_recommendations(session_id, np.array([0.5, 0.5]))
        
        # Calculate relevance score
        relevant_count = 0
        for rec in recommendations:
            watch_id = rec.get('watch_id')
            if watch_id in all_black_dial_ids:
                relevant_count += 1
        
        relevance_score = relevant_count / len(recommendations)
        iteration_scores.append(relevance_score)
        
        # Simulate user feedback
        feedback_given = 0
        for rec in recommendations:
            watch_id = rec.get('watch_id')
            if watch_id in all_black_dial_ids and feedback_given < 2:
                engine.update(session_id, watch_id, 1.0, np.array([0.5, 0.5]))
                feedback_given += 1
            elif watch_id not in all_black_dial_ids and feedback_given < 1:
                engine.update(session_id, watch_id, 0.0, np.array([0.5, 0.5]))
                feedback_given += 1
    
    # Calculate improvement
    early_avg = np.mean(iteration_scores[:2])
    late_avg = np.mean(iteration_scores[-2:])
    improvement = late_avg - early_avg
    
    # Clean up
    engine.shutdown()
    
    return improvement, iteration_scores

def test_parameter_grid():
    """Test different parameter combinations to find optimal settings."""
    print("🔬 Parameter Testing Suite for Aesthetic Learning Convergence")
    print("=" * 80)
    
    # Load test data
    categories = load_test_categories()
    if not categories or len(categories.get('black_dials', [])) < 6:
        print("❌ Insufficient test data")
        return
    
    print(f"📊 Test data: {len(categories.get('black_dials', []))} black dial watches")
    
    # Define parameter ranges
    alpha_values = [0.1, 0.2, 0.3, 0.5]
    similarity_thresholds = [0.6, 0.7, 0.75, 0.8]
    max_experts_values = [2, 3, 4, 6]
    dim_values = [50, 100]  # Test both reduced and full dimensions
    
    print(f"\n🎯 Testing {len(alpha_values)} × {len(similarity_thresholds)} × {len(max_experts_values)} × {len(dim_values)} = {len(alpha_values) * len(similarity_thresholds) * len(max_experts_values) * len(dim_values)} combinations")
    
    results = []
    total_tests = len(alpha_values) * len(similarity_thresholds) * len(max_experts_values) * len(dim_values)
    test_count = 0
    
    # Test all combinations
    for alpha in alpha_values:
        for similarity_threshold in similarity_thresholds:
            for max_experts in max_experts_values:
                for dim in dim_values:
                    test_count += 1
                    print(f"\n🧪 Test {test_count}/{total_tests}: α={alpha}, sim={similarity_threshold}, experts={max_experts}, dim={dim}")
                    
                    try:
                        improvement, scores = test_convergence_with_parameters(
                            alpha, similarity_threshold, max_experts, dim, categories
                        )
                        
                        if improvement is not None:
                            passed = improvement > 0.1  # 10% improvement threshold
                            results.append({
                                'alpha': alpha,
                                'similarity_threshold': similarity_threshold,
                                'max_experts': max_experts,
                                'dim': dim,
                                'improvement': improvement,
                                'scores': scores,
                                'passed': passed,
                                'early_avg': np.mean(scores[:2]),
                                'late_avg': np.mean(scores[-2:])
                            })
                            
                            status = "✅ PASS" if passed else "❌ FAIL"
                            print(f"     Result: {improvement:+.1%} improvement - {status}")
                        else:
                            print(f"     ⚠️  Test skipped: insufficient data")
                    
                    except Exception as e:
                        print(f"     ❌ Test failed: {e}")
    
    # Analyze results
    print(f"\n📊 Parameter Testing Results")
    print("=" * 80)
    
    if not results:
        print("❌ No successful tests")
        return
    
    # Sort by improvement
    results.sort(key=lambda x: x['improvement'], reverse=True)
    
    print(f"\n🏆 Top 10 Parameter Combinations:")
    print("Rank | α    | Sim  | Exp | Dim | Early→Late  | Improvement | Status")
    print("-" * 75)
    
    for i, result in enumerate(results[:10]):
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{i+1:3d}  | {result['alpha']:.1f}  | {result['similarity_threshold']:.2f} | {result['max_experts']:2d}  | {result['dim']:3d} | {result['early_avg']:.1%}→{result['late_avg']:.1%} | {result['improvement']:+7.1%} | {status}")
    
    # Success rate analysis
    passed_results = [r for r in results if r['passed']]
    pass_rate = len(passed_results) / len(results) * 100
    
    print(f"\n📈 Overall Success Rate: {len(passed_results)}/{len(results)} ({pass_rate:.1f}%)")
    
    if passed_results:
        # Best parameter
        best = results[0]
        print(f"\n🎯 Recommended Parameters:")
        print(f"   Alpha (exploration): {best['alpha']}")
        print(f"   Similarity threshold: {best['similarity_threshold']}")
        print(f"   Max experts: {best['max_experts']}")
        print(f"   Dimension: {best['dim']}")
        print(f"   Expected improvement: {best['improvement']:+.1%}")
        
        # Parameter analysis
        analyze_parameter_patterns(passed_results)
    else:
        print("\n❌ No parameter combinations achieved convergence")
        analyze_failure_patterns(results)
    
    return results

def analyze_parameter_patterns(passed_results):
    """Analyze patterns in successful parameter combinations."""
    print(f"\n🔍 Analysis of Successful Parameters:")
    
    # Alpha analysis
    alpha_counts = defaultdict(int)
    for result in passed_results:
        alpha_counts[result['alpha']] += 1
    
    print(f"\n⚡ Alpha (exploration) success rates:")
    for alpha, count in sorted(alpha_counts.items()):
        print(f"   α = {alpha}: {count} successes")
    
    # Similarity threshold analysis
    sim_counts = defaultdict(int)
    for result in passed_results:
        sim_counts[result['similarity_threshold']] += 1
    
    print(f"\n🎯 Similarity threshold success rates:")
    for sim, count in sorted(sim_counts.items()):
        print(f"   threshold = {sim}: {count} successes")
    
    # Max experts analysis
    expert_counts = defaultdict(int)
    for result in passed_results:
        expert_counts[result['max_experts']] += 1
    
    print(f"\n👥 Max experts success rates:")
    for experts, count in sorted(expert_counts.items()):
        print(f"   max_experts = {experts}: {count} successes")
    
    # Dimension analysis
    dim_counts = defaultdict(int)
    for result in passed_results:
        dim_counts[result['dim']] += 1
    
    print(f"\n📐 Dimension success rates:")
    for dim, count in sorted(dim_counts.items()):
        print(f"   dim = {dim}: {count} successes")

def analyze_failure_patterns(all_results):
    """Analyze why parameter combinations failed."""
    print(f"\n🔍 Analysis of All Results (including failures):")
    
    # Show distribution of improvements
    improvements = [r['improvement'] for r in all_results]
    
    print(f"\n📊 Improvement Distribution:")
    print(f"   Best: {max(improvements):+.1%}")
    print(f"   Worst: {min(improvements):+.1%}")
    print(f"   Average: {np.mean(improvements):+.1%}")
    print(f"   Median: {np.median(improvements):+.1%}")
    
    # Show which parameters tend to fail
    failed_results = [r for r in all_results if not r['passed']]
    if failed_results:
        print(f"\n❌ Common characteristics of failed combinations:")
        
        # Check if high alpha values tend to fail
        high_alpha_fails = len([r for r in failed_results if r['alpha'] >= 0.3])
        low_alpha_fails = len([r for r in failed_results if r['alpha'] < 0.3])
        print(f"   High alpha (≥0.3) failures: {high_alpha_fails}")
        print(f"   Low alpha (<0.3) failures: {low_alpha_fails}")

def quick_parameter_test():
    """Quick test of a few promising parameter combinations."""
    print("⚡ Quick Parameter Test (3 combinations)")
    print("=" * 50)
    
    categories = load_test_categories()
    if not categories:
        return
    
    # Test 3 promising combinations
    test_configs = [
        {'alpha': 0.1, 'similarity_threshold': 0.7, 'max_experts': 3, 'dim': 50},   # Low exploration
        {'alpha': 0.2, 'similarity_threshold': 0.75, 'max_experts': 4, 'dim': 100}, # Medium exploration
        {'alpha': 0.3, 'similarity_threshold': 0.8, 'max_experts': 2, 'dim': 50}    # Current with changes
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🧪 Test {i}: α={config['alpha']}, sim={config['similarity_threshold']}, experts={config['max_experts']}, dim={config['dim']}")
        
        try:
            improvement, scores = test_convergence_with_parameters(
                config['alpha'], config['similarity_threshold'], 
                config['max_experts'], config['dim'], categories
            )
            
            if improvement is not None:
                passed = improvement > 0.1
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"   Improvement: {improvement:+.1%} - {status}")
                print(f"   Scores: {' → '.join(f'{s:.1%}' for s in scores)}")
            else:
                print(f"   ⚠️  Test skipped")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_parameter_test()
    else:
        test_parameter_grid() 
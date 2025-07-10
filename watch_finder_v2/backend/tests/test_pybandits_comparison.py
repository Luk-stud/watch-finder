#!/usr/bin/env python3
"""
Test script to compare PyBandits Thompson Sampling with FastLinUCB

This script:
1. Tests both engines with sample data
2. Compares performance metrics 
3. Validates the interface compatibility
"""

import sys
import os
import time
import logging
import numpy as np
from typing import Dict, Any

# Add the models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(num_watches: int = 20) -> Dict[str, Any]:
    """Create test data for comparison."""
    logger.info(f"Creating test data with {num_watches} watches")
    
    # Create sample watch data
    watch_data = {}
    final_embeddings = {}
    
    brands = ['Rolex', 'Omega', 'Cartier', 'Patek Philippe', 'Audemars Piguet']
    models = ['Submariner', 'Speedmaster', 'Santos', 'Nautilus', 'Royal Oak']
    
    for i in range(num_watches):
        watch_id = i + 1
        brand = brands[i % len(brands)]
        model = f"{models[i % len(models)]} {i}"
        
        watch_data[watch_id] = {
            'watch_id': watch_id,
            'brand': brand,
            'model': model,
            'price': 5000 + (i * 1000),
            'year': 2020 + (i % 5)
        }
        
        # Create normalized random embeddings
        embedding = np.random.randn(200)
        embedding = embedding / np.linalg.norm(embedding)
        final_embeddings[watch_id] = embedding
    
    return {
        'watch_data': watch_data,
        'final_embeddings': final_embeddings,
        'embedding_dim': 200
    }

def save_test_data(test_data: Dict[str, Any], data_dir: str = "data"):
    """Save test data to pickle file."""
    import pickle
    
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'precomputed_embeddings.pkl')
    
    with open(filepath, 'wb') as f:
        pickle.dump(test_data, f)
    
    logger.info(f"Saved test data to {filepath}")

def test_engine_performance(engine_class, engine_name: str, test_scenarios: list) -> Dict[str, Any]:
    """Test an engine with various scenarios and measure performance."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing {engine_name}")
    logger.info(f"{'='*50}")
    
    # Initialize engine
    start_time = time.time()
    try:
        engine = engine_class(
            batch_size=5,
            max_experts=3,
            data_dir="data"
        )
        init_time = time.time() - start_time
        logger.info(f"✅ {engine_name} initialization: {init_time:.3f}s")
    except Exception as e:
        logger.error(f"❌ Failed to initialize {engine_name}: {e}")
        return {'error': str(e)}
    
    results = {
        'init_time': init_time,
        'scenarios': {},
        'total_recommendations': 0,
        'total_updates': 0,
        'avg_recommendation_time': 0,
        'avg_update_time': 0
    }
    
    session_id = "test_session_001"
    
    try:
        # Run test scenarios
        for scenario_name, scenario in test_scenarios.items():
            logger.info(f"\n--- Running scenario: {scenario_name} ---")
            
            scenario_start = time.time()
            scenario_results = {
                'recommendations': [],
                'recommendation_times': [],
                'update_times': [],
                'errors': []
            }
            
            # Create session for this scenario
            engine.create_session(session_id)
            
            for step_idx, step in enumerate(scenario['steps']):
                try:
                    if step['action'] == 'recommend':
                        # Get recommendations
                        rec_start = time.time()
                        recommendations = engine.get_recommendations(
                            session_id=session_id,
                            exclude_ids=step.get('exclude_ids', set())
                        )
                        rec_time = time.time() - rec_start
                        
                        scenario_results['recommendations'].append(recommendations)
                        scenario_results['recommendation_times'].append(rec_time)
                        results['total_recommendations'] += 1
                        
                        logger.info(f"Step {step_idx + 1}: Got {len(recommendations)} recommendations in {rec_time:.3f}s")
                        
                    elif step['action'] == 'update':
                        # Update with feedback
                        update_start = time.time()
                        engine.update(
                            session_id=session_id,
                            watch_id=step['watch_id'],
                            reward=step['reward']
                        )
                        update_time = time.time() - update_start
                        
                        scenario_results['update_times'].append(update_time)
                        results['total_updates'] += 1
                        
                        action_type = "👍 Like" if step['reward'] > 0 else "👎 Dislike"
                        logger.info(f"Step {step_idx + 1}: {action_type} watch {step['watch_id']} in {update_time:.3f}s")
                        
                except Exception as e:
                    error_msg = f"Error in step {step_idx + 1}: {e}"
                    scenario_results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            scenario_time = time.time() - scenario_start
            scenario_results['total_time'] = scenario_time
            results['scenarios'][scenario_name] = scenario_results
            
            logger.info(f"Scenario '{scenario_name}' completed in {scenario_time:.3f}s")
        
        # Calculate averages
        all_rec_times = []
        all_update_times = []
        
        for scenario_results in results['scenarios'].values():
            all_rec_times.extend(scenario_results['recommendation_times'])
            all_update_times.extend(scenario_results['update_times'])
        
        results['avg_recommendation_time'] = np.mean(all_rec_times) if all_rec_times else 0
        results['avg_update_time'] = np.mean(all_update_times) if all_update_times else 0
        
        # Get expert stats
        try:
            expert_stats = engine.get_expert_stats()
            results['expert_stats'] = expert_stats
            logger.info(f"Expert stats: {expert_stats}")
        except Exception as e:
            logger.warning(f"Could not get expert stats: {e}")
        
        # Cleanup
        engine.shutdown()
        
    except Exception as e:
        logger.error(f"❌ Error during {engine_name} testing: {e}")
        results['error'] = str(e)
    
    return results

def create_test_scenarios() -> Dict[str, Any]:
    """Create test scenarios for both engines."""
    return {
        'cold_start': {
            'description': 'Test cold start recommendations',
            'steps': [
                {'action': 'recommend'},  # Should get random recommendations
                {'action': 'update', 'watch_id': 1, 'reward': 1.0},  # Like first watch
                {'action': 'recommend'},  # Should start personalizing
            ]
        },
        
        'preference_learning': {
            'description': 'Test learning user preferences',
            'steps': [
                {'action': 'recommend'},
                {'action': 'update', 'watch_id': 1, 'reward': 1.0},  # Like Rolex
                {'action': 'recommend'},
                {'action': 'update', 'watch_id': 6, 'reward': 1.0},  # Like another Rolex
                {'action': 'recommend'},
                {'action': 'update', 'watch_id': 2, 'reward': -0.5}, # Dislike Omega
                {'action': 'recommend'},
            ]
        },
        
        'multiple_experts': {
            'description': 'Test creation of multiple experts',
            'steps': [
                # Create first expert (luxury sports watches)
                {'action': 'recommend'},
                {'action': 'update', 'watch_id': 1, 'reward': 1.0},  # Rolex Submariner
                {'action': 'update', 'watch_id': 6, 'reward': 1.0},  # Rolex Submariner 5
                
                # Create second expert (dress watches) - different preference
                {'action': 'recommend'},
                {'action': 'update', 'watch_id': 3, 'reward': 1.0},  # Cartier Santos
                {'action': 'update', 'watch_id': 8, 'reward': 1.0},  # Cartier Santos 5
                
                # Get recommendations from multiple experts
                {'action': 'recommend'},
                {'action': 'recommend'},
            ]
        }
    }

def print_comparison_results(linucb_results: Dict[str, Any], pybandits_results: Dict[str, Any]):
    """Print a detailed comparison of both engines."""
    logger.info(f"\n{'='*70}")
    logger.info("COMPARISON RESULTS")
    logger.info(f"{'='*70}")
    
    # Initialization comparison
    logger.info(f"\n📊 Initialization Time:")
    linucb_init = linucb_results.get('init_time', 'N/A')
    pybandits_init = pybandits_results.get('init_time', 'N/A')
    
    if isinstance(linucb_init, (int, float)):
        logger.info(f"  LinUCB:    {linucb_init:.3f}s")
    else:
        logger.info(f"  LinUCB:    {linucb_init}")
        
    if isinstance(pybandits_init, (int, float)):
        logger.info(f"  PyBandits: {pybandits_init:.3f}s")
    else:
        logger.info(f"  PyBandits: {pybandits_init}")
    
    # Performance comparison
    logger.info(f"\n⚡ Average Performance:")
    logger.info(f"  Recommendation Time:")
    logger.info(f"    LinUCB:    {linucb_results.get('avg_recommendation_time', 0):.4f}s")
    logger.info(f"    PyBandits: {pybandits_results.get('avg_recommendation_time', 0):.4f}s")
    
    logger.info(f"  Update Time:")
    logger.info(f"    LinUCB:    {linucb_results.get('avg_update_time', 0):.4f}s")
    logger.info(f"    PyBandits: {pybandits_results.get('avg_update_time', 0):.4f}s")
    
    # Volume comparison
    logger.info(f"\n📈 Total Operations:")
    logger.info(f"  Recommendations:")
    logger.info(f"    LinUCB:    {linucb_results.get('total_recommendations', 0)}")
    logger.info(f"    PyBandits: {pybandits_results.get('total_recommendations', 0)}")
    
    logger.info(f"  Updates:")
    logger.info(f"    LinUCB:    {linucb_results.get('total_updates', 0)}")
    logger.info(f"    PyBandits: {pybandits_results.get('total_updates', 0)}")
    
    # Expert stats comparison
    logger.info(f"\n👥 Expert Statistics:")
    linucb_stats = linucb_results.get('expert_stats', {})
    pybandits_stats = pybandits_results.get('expert_stats', {})
    
    logger.info(f"  Total Experts:")
    logger.info(f"    LinUCB:    {linucb_stats.get('total_experts', 0)}")
    logger.info(f"    PyBandits: {pybandits_stats.get('total_experts', 0)}")
    
    logger.info(f"  Algorithm:")
    logger.info(f"    LinUCB:    {linucb_stats.get('algorithm', 'Custom LinUCB')}")
    logger.info(f"    PyBandits: {pybandits_stats.get('algorithm', 'Unknown')}")
    
    # Error comparison
    linucb_errors = []
    pybandits_errors = []
    
    for scenario_results in linucb_results.get('scenarios', {}).values():
        linucb_errors.extend(scenario_results.get('errors', []))
    
    for scenario_results in pybandits_results.get('scenarios', {}).values():
        pybandits_errors.extend(scenario_results.get('errors', []))
    
    logger.info(f"\n🚨 Errors:")
    logger.info(f"  LinUCB:    {len(linucb_errors)} errors")
    logger.info(f"  PyBandits: {len(pybandits_errors)} errors")
    
    if linucb_errors:
        logger.info(f"  LinUCB errors: {linucb_errors}")
    if pybandits_errors:
        logger.info(f"  PyBandits errors: {pybandits_errors}")

def main():
    """Main test function."""
    logger.info("🚀 Starting PyBandits vs LinUCB Comparison Test")
    
    # Create and save test data
    test_data = create_test_data(num_watches=20)
    save_test_data(test_data)
    
    # Create test scenarios
    test_scenarios = create_test_scenarios()
    
    # Import engines
    try:
        from fast_linucb_engine import FastLinUCBEngine
        logger.info("✅ FastLinUCBEngine imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import FastLinUCBEngine: {e}")
        return
    
    try:
        from pybandits_engine import PyBanditsEngine
        logger.info("✅ PyBanditsEngine imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import PyBanditsEngine: {e}")
        return
    
    # Test both engines
    logger.info(f"\n🧪 Running tests with {len(test_scenarios)} scenarios")
    
    linucb_results = test_engine_performance(FastLinUCBEngine, "FastLinUCB", test_scenarios)
    pybandits_results = test_engine_performance(PyBanditsEngine, "PyBandits Thompson Sampling", test_scenarios)
    
    # Print comparison
    print_comparison_results(linucb_results, pybandits_results)
    
    # Determine winner
    logger.info(f"\n🏆 RECOMMENDATION:")
    
    linucb_avg_rec = linucb_results.get('avg_recommendation_time', float('inf'))
    pybandits_avg_rec = pybandits_results.get('avg_recommendation_time', float('inf'))
    
    if pybandits_avg_rec < linucb_avg_rec:
        speedup = linucb_avg_rec / pybandits_avg_rec if pybandits_avg_rec > 0 else 1
        logger.info(f"  🚀 PyBandits is {speedup:.2f}x faster for recommendations!")
    elif linucb_avg_rec < pybandits_avg_rec:
        speedup = pybandits_avg_rec / linucb_avg_rec if linucb_avg_rec > 0 else 1
        logger.info(f"  ⚡ LinUCB is {speedup:.2f}x faster for recommendations!")
    else:
        logger.info(f"  🤝 Both engines have similar performance!")
    
    logger.info("\n✅ Comparison test completed!")

if __name__ == "__main__":
    main() 
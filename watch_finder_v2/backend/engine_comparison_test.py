import os
import pickle
import numpy as np
import argparse
import logging
from collections import Counter
import pandas as pd
import sys

# Ensure the models are importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models.simple_sgd_engine import SimpleSgdEngine
from models.simple_vector_engine import SimpleVectorEngine

# --- Configuration ---
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# --- Data Loading ---
def load_watch_data():
    """Loads the watch metadata from the precomputed file."""
    try:
        precomputed_path = os.path.join(DATA_DIR, 'precomputed_embeddings.pkl')
        with open(precomputed_path, 'rb') as f:
            data = pickle.load(f)
        return data['watch_data']
    except Exception as e:
        print(f"❌ Failed to load watch data: {e}")
        return None

def get_all_values_for_spec(watch_data, spec_key):
    """Gets a count of all values for a given spec key in the dataset."""
    values = Counter()
    for data in watch_data.values():
        spec_value = data.get('specs', {}).get(spec_key)
        if spec_value:
            values[spec_value] += 1
    return values

def find_watches_by_spec(watch_data, spec_key, spec_value):
    """Finds all watch IDs that match a given spec key-value pair."""
    matches = []
    for watch_id, data in watch_data.items():
        if data.get('specs', {}).get(spec_key) == spec_value:
            matches.append(watch_id)
    return list(set(matches))

# --- Core Test Runner ---
def run_single_test_run(engine, watch_data, spec_key, spec_value, num_likes, rec_batch_size):
    """Simulates a single user session and returns the convergence scores."""
    seed_watches = find_watches_by_spec(watch_data, spec_key, spec_value)
    effective_num_likes = min(num_likes, len(seed_watches))
    
    if effective_num_likes < 1:
        return None, 0

    session_id = f"test-session-{np.random.randint(10000)}"
    engine.create_session(session_id)
    watches_to_like = np.random.choice(seed_watches, size=effective_num_likes, replace=False).tolist()
    
    convergence_scores = []
    for i in range(effective_num_likes):
        liked_watch_id = watches_to_like[i]
        engine.update(session_id, liked_watch_id, reward=1.0)
        recommendations = engine.get_recommendations(session_id)
        
        if not recommendations:
            convergence_scores.append(0.0)
            continue
            
        rec_ids = [rec['watch_id'] for rec in recommendations]
        match_count = sum(1 for rec_id in rec_ids if watch_data.get(rec_id, {}).get('specs', {}).get(spec_key) == spec_value)
        convergence_score = (match_count / len(rec_ids)) * 100
        convergence_scores.append(convergence_score)
        
    return convergence_scores, effective_num_likes

# --- New Test Runner for Complex Preferences ---
def run_single_complex_test_run(engine, watch_data, spec_a_key, spec_a_value, spec_b_key, spec_b_value, num_likes, rec_batch_size):
    """Simulates a user session with two distinct, alternating preferences."""
    
    group_a_watches = find_watches_by_spec(watch_data, spec_a_key, spec_a_value)
    group_b_watches = find_watches_by_spec(watch_data, spec_b_key, spec_b_value)
    
    if not group_a_watches or not group_b_watches:
        print("⚠️ Could not find enough watches in both specified groups to run the test.")
        return None, 0

    session_id = f"complex-test-{np.random.randint(10000)}"
    engine.create_session(session_id)
    
    # Ensure we don't run out of watches to like
    effective_num_likes = min(num_likes, 2 * min(len(group_a_watches), len(group_b_watches)))
    
    watches_to_like_a = np.random.choice(group_a_watches, size=(effective_num_likes // 2) + 1, replace=False).tolist()
    watches_to_like_b = np.random.choice(group_b_watches, size=(effective_num_likes // 2) + 1, replace=False).tolist()

    convergence_scores = []
    for i in range(effective_num_likes):
        # Alternate between liking watches from group A and group B
        if i % 2 == 0:
            liked_watch_id = watches_to_like_a.pop(0)
        else:
            liked_watch_id = watches_to_like_b.pop(0)
            
        engine.update(session_id, liked_watch_id, reward=1.0)
        recommendations = engine.get_recommendations(session_id)
        
        if not recommendations:
            convergence_scores.append(0.0)
            continue

        # Evaluate: A recommendation is a success if it's in EITHER group A OR group B
        rec_ids = {rec['watch_id'] for rec in recommendations}
        success_ids = rec_ids.intersection(set(group_a_watches) | set(group_b_watches))
        
        match_count = len(success_ids)
        convergence_score = (match_count / len(rec_ids)) * 100
        convergence_scores.append(convergence_score)
        
    return convergence_scores, effective_num_likes

def test_engine_complex(engine, watch_data, args):
    """Orchestrates the complex preference test for a given engine."""
    print(f"\n--- 🚀 Testing Engine '{engine.__class__.__name__}' on Complex Preference ---")
    print(f"  - Preference A: {args.spec_a_key} = {args.spec_a_value}")
    print(f"  - Preference B: {args.spec_b_key} = {args.spec_b_value}")

    all_trial_results = []
    total_likes_run = 0
    for trial in range(args.num_trials):
        result, likes_run = run_single_complex_test_run(
            engine, watch_data, 
            args.spec_a_key, args.spec_a_value, 
            args.spec_b_key, args.spec_b_value, 
            args.num_likes, args.rec_batch_size
        )
        if result:
            all_trial_results.append(result)
            total_likes_run = likes_run
        print(f"  Trial {trial+1}/{args.num_trials} complete.")

    if not all_trial_results:
        print("❌ Could not complete any trials.")
        return

    avg_scores = np.mean([pd.Series(res).reindex(range(total_likes_run)).ffill().values for res in all_trial_results], axis=0)
    print("\n--- 📊 Average Convergence Results ---")
    for i, score in enumerate(avg_scores):
        print(f"Interaction {i+1: >2}: Average Convergence = {score:.1f}%")
    print(f"\n📈 Final Average Convergence: {avg_scores[-1]:.2f}%")

def test_engine_on_spec(engine, watch_data, spec_key, num_trials, num_likes, rec_batch_size):
    """Runs a standardized convergence test for a given engine and spec key."""
    all_values = get_all_values_for_spec(watch_data, spec_key)
    print(f"\n--- 🚀 Testing Engine '{engine.__class__.__name__}' on Spec '{spec_key}' ---")
    print(f"Found {len(all_values)} unique values to test.")

    results = []
    for spec_value, count in sorted(all_values.items(), key=lambda item: item[1], reverse=True):
        if count < rec_batch_size:
            print(f"-- Skipping '{spec_value}' (has only {count} items)")
            continue
        
        print(f"-- Testing value '{spec_value}' ({count} items available)...")
        all_trial_results, effective_likes = [], 0
        for _ in range(num_trials):
            result, likes_run = run_single_test_run(engine, watch_data, spec_key, spec_value, num_likes, rec_batch_size)
            if result:
                all_trial_results.append(result)
                effective_likes = likes_run
        
        if not all_trial_results:
            print("   ...No valid trials could be run.")
            continue
            
        avg_scores = np.mean([pd.Series(res).reindex(range(effective_likes)).ffill().values for res in all_trial_results], axis=0)
        final_score = avg_scores[-1]
        results.append({"spec_value": spec_value, "num_items": count, "test_length": effective_likes, "conv_final": final_score})
        print(f"   ...Done. Final avg convergence: {final_score:.1f}%")

    if not results:
        print("No results to summarize.")
        return
        
    summary_df = pd.DataFrame(results)
    print("\n--- 🏆 Overall Performance Summary ---")
    print(summary_df.to_string(index=False))
    print(f"\n📈 Overall Average Final Convergence for spec '{spec_key}': {summary_df['conv_final'].mean():.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare recommendation engine performance.")
    parser.add_argument('--mode', type=str, default="simple", choices=['simple', 'complex'], help="Test mode.")
    
    # Args for simple test
    parser.add_argument('--test_spec_key', type=str, default="case_material", help="The spec key for simple test.")
    
    # Args for complex test
    parser.add_argument('--spec_a_key', type=str, default="case_material", help="Key for preference A.")
    parser.add_argument('--spec_a_value', type=str, default="Bronze", help="Value for preference A.")
    parser.add_argument('--spec_b_key', type=str, default="dial_type", help="Key for preference B.")
    parser.add_argument('--spec_b_value', type=str, default="Plain", help="Value for preference B.")

    # General args
    parser.add_argument('--engine', type=str, default="sgd", choices=['sgd', 'vector'], help="Engine to test.")
    parser.add_argument('--num_likes', type=int, default=10, help="Maximum number of simulated 'like' interactions.")
    parser.add_argument('--num_trials', type=int, default=5, help="Number of trials to run for averaging.")
    parser.add_argument('--rec_batch_size', type=int, default=5, help="Number of recommendations to request per step.")
    
    args = parser.parse_args()

    watch_data = load_watch_data()
    if watch_data:
        if args.engine == 'sgd':
            engine = SimpleSgdEngine(data_dir=DATA_DIR, like_weight=5.0, alpha=0.0001)
        else: # vector
            engine = SimpleVectorEngine(data_dir=DATA_DIR)

        if args.mode == 'simple':
            test_engine_on_spec(engine, watch_data, args.test_spec_key, args.num_trials, args.num_likes, args.rec_batch_size)
        elif args.mode == 'complex':
            test_engine_complex(engine, watch_data, args) 
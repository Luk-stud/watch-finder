#!/usr/bin/env python3
"""
Precompute Smart Seeds for Watch Recommendation System
Generates 20 diverse smart seeds using enhanced clustering and saves them for fast loading.
"""

import sys
import os
import pickle
import numpy as np
import random
from typing import List, Dict, Any
from collections import defaultdict

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from models.beam_search_v2 import EnhancedWatchBeamSearch

def load_watch_data():
    """Load watch embeddings and metadata."""
    # Check for embeddings files in deployment-ready locations
    embeddings_path = os.path.join(os.path.dirname(__file__), 'data/watch_embeddings.pkl')
    metadata_path = os.path.join(os.path.dirname(__file__), 'data/watch_metadata.pkl')
    
    print(f"Loading embeddings from: {embeddings_path}")
    print(f"Loading metadata from: {metadata_path}")
    
    if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
        # Load embeddings and metadata separately
        with open(embeddings_path, 'rb') as f:
            embeddings_data = pickle.load(f)
            
        # Handle different embeddings formats
        if isinstance(embeddings_data, dict):
            if 'embeddings' in embeddings_data:
                embeddings = embeddings_data['embeddings']
            else:
                embeddings = embeddings_data
        else:
            embeddings = embeddings_data
            
        with open(metadata_path, 'rb') as f:
            watch_data = pickle.load(f)
            
        print(f"✅ Loaded {len(watch_data)} watches with {embeddings.shape[1]}D embeddings")
        return embeddings, watch_data
    else:
        raise FileNotFoundError(f"Could not find embeddings or metadata files")

def generate_diverse_smart_seed_sets(beam_search_engine: EnhancedWatchBeamSearch, 
                                   num_sets: int = 20, seeds_per_set: int = 7) -> List[List[Dict[str, Any]]]:
    """
    Generate multiple diverse smart seed sets.
    Each set contains 7 watches with maximum style diversity.
    """
    print(f"🌱 Generating {num_sets} sets of {seeds_per_set} diverse smart seeds each...")
    
    # Reset seen watches to start fresh
    beam_search_engine.seen_watches.clear()
    beam_search_engine.seen_brands.clear()
    beam_search_engine.seen_styles.clear()
    
    # 🆕 ENSURE CLUSTERS ARE INITIALIZED
    print("⏳ Ensuring clusters are initialized...")
    beam_search_engine._ensure_clusters()
    beam_search_engine._ensure_semantic_features()
    
    all_seed_sets = []
    global_selected_indices = set()  # Track globally selected indices across all sets
    
    # Get available clusters and styles for diverse selection
    available_aesthetic_clusters = list(beam_search_engine.cluster_mappings['aesthetic'].keys())
    available_style_clusters = list(beam_search_engine.cluster_mappings['style'].keys())
    
    # Get all unique styles in the dataset
    all_styles = set()
    for watch in beam_search_engine.watch_data:
        style = beam_search_engine._classify_watch_style_enhanced(watch)
        all_styles.add(style)
    
    all_styles = list(all_styles)
    print(f"📊 Available styles: {all_styles}")
    
    for set_idx in range(num_sets):
        print(f"\n🎯 Generating seed set {set_idx + 1}/{num_sets}...")
        
        seed_set = []
        set_selected_indices = set()
        set_styles_used = set()
        
        # Strategy: Ensure style diversity within each set
        target_styles = all_styles.copy()
        random.shuffle(target_styles)
        
        # Select up to seeds_per_set watches, prioritizing style diversity
        for style_priority in range(seeds_per_set):
            best_watch = None
            best_score = -1
            best_style = None
            
            # Try to get a watch from an unused style first
            for target_style in target_styles:
                if target_style in set_styles_used:
                    continue  # Skip already used styles in this set
                
                # Find watches of this style that haven't been globally selected
                style_candidates = []
                for idx, watch in enumerate(beam_search_engine.watch_data):
                    if idx in global_selected_indices or idx in set_selected_indices:
                        continue
                    
                    watch_style = beam_search_engine._classify_watch_style_enhanced(watch)
                    if watch_style == target_style:
                        style_candidates.append(idx)
                
                if not style_candidates:
                    continue
                
                # Select best candidate from this style using multi-objective scoring
                neutral_preferences = np.mean(beam_search_engine.normalized_embeddings, axis=0)
                
                for candidate_idx in style_candidates:
                    score = beam_search_engine.multi_objective_score(candidate_idx, neutral_preferences, list(set_selected_indices))
                    
                    if score > best_score:
                        best_score = score
                        best_watch = candidate_idx
                        best_style = target_style
                
                if best_watch is not None:
                    break  # Found a good watch from an unused style
            
            # If no unused style available, pick best remaining watch
            if best_watch is None:
                remaining_candidates = [idx for idx in range(len(beam_search_engine.watch_data)) 
                                      if idx not in global_selected_indices and idx not in set_selected_indices]
                
                if not remaining_candidates:
                    print(f"⚠️  Running out of candidates at seed set {set_idx + 1}, seed {len(seed_set) + 1}")
                    break
                
                neutral_preferences = np.mean(beam_search_engine.normalized_embeddings, axis=0)
                
                for candidate_idx in random.sample(remaining_candidates, min(50, len(remaining_candidates))):
                    score = beam_search_engine.multi_objective_score(candidate_idx, neutral_preferences, list(set_selected_indices))
                    
                    if score > best_score:
                        best_score = score
                        best_watch = candidate_idx
                        best_style = beam_search_engine._classify_watch_style_enhanced(beam_search_engine.watch_data[candidate_idx])
            
            if best_watch is not None:
                watch = beam_search_engine.watch_data[best_watch].copy()
                watch['index'] = best_watch
                watch['seed_set_id'] = set_idx
                watch['seed_position'] = len(seed_set)
                watch['seed_style'] = best_style
                watch['is_precomputed_seed'] = True
                
                # Add enhanced metadata
                watch['brand_category'] = beam_search_engine._categorize_brand(watch.get('brand', '').lower())
                watch['style'] = best_style
                watch['luxury_level'] = beam_search_engine._assess_luxury_level(watch)
                
                seed_set.append(watch)
                set_selected_indices.add(best_watch)
                global_selected_indices.add(best_watch)
                set_styles_used.add(best_style)
                
                print(f"   ✓ Seed {len(seed_set)}: {watch.get('brand')} {watch.get('model')} ({best_style})")
        
        if len(seed_set) > 0:
            all_seed_sets.append(seed_set)
            print(f"✅ Set {set_idx + 1}: {len(seed_set)} seeds with {len(set_styles_used)} unique styles")
        else:
            print(f"❌ Failed to generate seed set {set_idx + 1}")
    
    print(f"\n🎉 Generated {len(all_seed_sets)} complete seed sets")
    return all_seed_sets

def analyze_seed_diversity(seed_watches: List[Dict[str, Any]], set_id: int = None):
    """Analyze and report on the diversity of generated seeds."""
    set_prefix = f"Set {set_id} " if set_id is not None else ""
    print(f"\n📊 {set_prefix}Seed Diversity Analysis:")
    
    # Brand diversity
    brands = [watch.get('brand') for watch in seed_watches]
    unique_brands = len(set(brands))
    print(f"   🏭 Brands: {unique_brands}/{len(seed_watches)} unique ({unique_brands/len(seed_watches)*100:.1f}%)")
    
    # Style diversity  
    styles = [watch.get('style') for watch in seed_watches]
    unique_styles = len(set(styles))
    print(f"   🎨 Styles: {unique_styles}/{len(seed_watches)} unique ({unique_styles/len(seed_watches)*100:.1f}%)")
    
    # Brand category diversity
    brand_categories = [watch.get('brand_category') for watch in seed_watches]
    unique_categories = len(set(brand_categories))
    print(f"   💰 Brand Categories: {unique_categories}/{len(seed_watches)} unique")
    
    # Price range diversity
    prices = []
    for watch in seed_watches:
        price = watch.get('price', 0)
        if isinstance(price, (int, float)) and price > 0:
            prices.append(price)
    
    if prices:
        print(f"   💵 Price Range: ${min(prices):.0f} - ${max(prices):.0f} (avg: ${np.mean(prices):.0f})")
    
    print(f"   🎯 Diversity Score: {(unique_brands + unique_styles + unique_categories)/3:.1f}/{len(seed_watches)}")
    
    # Show the styles represented
    style_list = list(set(styles))
    print(f"   📋 Styles: {', '.join(style_list)}")

def save_smart_seed_sets(seed_sets: List[List[Dict[str, Any]]], 
                        output_path: str = 'data/precomputed_smart_seed_sets.pkl'):
    """Save all precomputed smart seed sets to file."""
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data structure
    seeds_data = {
        'seed_sets': seed_sets,
        'generation_timestamp': np.datetime64('now').item().timestamp(),
        'num_sets': len(seed_sets),
        'seeds_per_set': len(seed_sets[0]) if seed_sets else 0,
        'total_seeds': sum(len(s) for s in seed_sets),
        'version': 'v2.0_multiple_sets',
        'metadata': {
            'generation_method': 'style_diverse_sets',
            'strategy': 'maximize_style_diversity_per_set',
            'diversity_optimized': True,
            'set_based': True
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(seeds_data, f)
    
    print(f"💾 Saved {len(seed_sets)} seed sets ({seeds_data['total_seeds']} total seeds) to: {output_path}")
    
    # Also save as JSON for readability
    json_path = output_path.replace('.pkl', '.json')
    import json
    
    # Convert for JSON serialization
    json_data = seeds_data.copy()
    json_data['seed_sets'] = [
        [
            {
                'index': seed['index'],
                'brand': seed.get('brand'),
                'model': seed.get('model'),
                'style': seed.get('style'),
                'brand_category': seed.get('brand_category'),
                'price': seed.get('price'),
                'seed_set_id': seed.get('seed_set_id'),
                'seed_position': seed.get('seed_position'),
                'seed_style': seed.get('seed_style')
            }
            for seed in seed_set
        ]
        for seed_set in seed_sets
    ]
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"📄 Also saved readable version to: {json_path}")

def main():
    """Main function to generate and save precomputed smart seed sets."""
    print("🚀 Precomputing Smart Seed Sets for Watch Recommendation System")
    print("=" * 60)
    
    try:
        # Load watch data
        embeddings, watch_data = load_watch_data()
        
        # Initialize enhanced beam search
        print("\n🧠 Initializing Enhanced Beam Search...")
        beam_search_engine = EnhancedWatchBeamSearch(
            embeddings=embeddings,
            watch_data=watch_data,
            initial_beam_width=15,
            max_beam_width=30
        )
        
        # Generate smart seed sets
        print("\n🌱 Generating Smart Seed Sets...")
        seed_sets = generate_diverse_smart_seed_sets(beam_search_engine, num_sets=20, seeds_per_set=7)
        
        # Analyze diversity for each set
        print("\n📊 Analyzing Diversity for Each Set...")
        for set_idx, seed_set in enumerate(seed_sets):
            analyze_seed_diversity(seed_set, set_id=set_idx + 1)
        
        # Save all seed sets
        print("\n💾 Saving Smart Seed Sets...")
        save_smart_seed_sets(seed_sets)
        
        # Summary statistics
        total_seeds = sum(len(s) for s in seed_sets)
        all_brands = set()
        all_styles = set()
        for seed_set in seed_sets:
            for seed in seed_set:
                all_brands.add(seed.get('brand'))
                all_styles.add(seed.get('style'))
        
        print(f"\n🎉 Smart Seed Sets Generation Complete!")
        print(f"📊 Generated {len(seed_sets)} sets with {total_seeds} total seeds")
        print(f"🏭 Unique brands across all sets: {len(all_brands)}")
        print(f"🎨 Unique styles across all sets: {len(all_styles)}")
        print("The precomputed seed sets are ready for use in the recommendation system.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        sys.exit(1) 
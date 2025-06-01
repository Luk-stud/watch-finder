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

def generate_diverse_smart_seeds(beam_search_engine: EnhancedWatchBeamSearch, 
                                num_seeds: int = 20) -> List[Dict[str, Any]]:
    """
    Generate diverse smart seeds using enhanced clustering and multi-criteria selection.
    """
    print(f"🌱 Generating {num_seeds} diverse smart seeds...")
    
    # Reset seen watches to start fresh
    beam_search_engine.seen_watches.clear()
    beam_search_engine.seen_brands.clear()
    beam_search_engine.seen_styles.clear()
    
    seeds = []
    selected_indices = set()
    
    # Strategy 1: Select from different aesthetic clusters (40% of seeds)
    aesthetic_seeds = int(num_seeds * 0.4)
    available_aesthetic_clusters = list(beam_search_engine.cluster_mappings['aesthetic'].keys())
    
    print(f"📊 Selecting {aesthetic_seeds} seeds from aesthetic clusters...")
    for i in range(aesthetic_seeds):
        if not available_aesthetic_clusters:
            break
            
        cluster_id = random.choice(available_aesthetic_clusters)
        available_aesthetic_clusters.remove(cluster_id)
        
        cluster_watches = beam_search_engine.cluster_mappings['aesthetic'][cluster_id]
        available_watches = [idx for idx in cluster_watches if idx not in selected_indices]
        
        if available_watches:
            # Select best representative from this cluster
            neutral_preferences = np.mean(beam_search_engine.normalized_embeddings, axis=0)
            best_watch = None
            best_score = -1
            
            for watch_idx in available_watches:
                score = beam_search_engine.multi_objective_score(watch_idx, neutral_preferences, [])
                if score > best_score:
                    best_score = score
                    best_watch = watch_idx
            
            if best_watch is not None:
                seeds.append(best_watch)
                selected_indices.add(best_watch)
                print(f"   ✓ Aesthetic cluster {cluster_id}: watch {best_watch}")
    
    # Strategy 2: Select from different style clusters (30% of seeds)
    style_seeds = int(num_seeds * 0.3)
    available_style_clusters = list(beam_search_engine.cluster_mappings['style'].keys())
    
    print(f"🎨 Selecting {style_seeds} seeds from style clusters...")
    for i in range(style_seeds):
        if not available_style_clusters:
            break
            
        cluster_id = random.choice(available_style_clusters)
        available_style_clusters.remove(cluster_id)
        
        cluster_watches = beam_search_engine.cluster_mappings['style'][cluster_id]
        available_watches = [idx for idx in cluster_watches if idx not in selected_indices]
        
        if available_watches:
            selected_watch = random.choice(available_watches)
            seeds.append(selected_watch)
            selected_indices.add(selected_watch)
            print(f"   ✓ Style cluster {cluster_id}: watch {selected_watch}")
    
    # Strategy 3: Select from different brand categories (20% of seeds)
    brand_seeds = int(num_seeds * 0.2)
    brand_categories = ['luxury', 'premium', 'accessible', 'independent']
    
    print(f"🏭 Selecting {brand_seeds} seeds from brand categories...")
    for category in brand_categories:
        if len(seeds) >= num_seeds:
            break
            
        # Find watches from this brand category
        category_watches = []
        for idx, watch in enumerate(beam_search_engine.watch_data):
            if idx not in selected_indices:
                brand = watch.get('brand', '').lower()
                watch_category = beam_search_engine._categorize_brand(brand)
                if watch_category == category:
                    category_watches.append(idx)
        
        if category_watches:
            selected_watch = random.choice(category_watches)
            seeds.append(selected_watch)
            selected_indices.add(selected_watch)
            print(f"   ✓ Brand category {category}: watch {selected_watch}")
    
    # Strategy 4: Fill remaining with random diverse selection (10% of seeds)
    remaining_seeds = num_seeds - len(seeds)
    
    if remaining_seeds > 0:
        print(f"🎲 Selecting {remaining_seeds} random diverse seeds...")
        
        # Get all unseen watches
        all_unseen = [idx for idx in range(len(beam_search_engine.watch_data)) 
                     if idx not in selected_indices]
        
        # Use similarity-based diversity selection
        for _ in range(remaining_seeds):
            if not all_unseen:
                break
                
            if not seeds:
                # First random seed
                selected_watch = random.choice(all_unseen)
            else:
                # Select watch that is most different from already selected
                best_watch = None
                best_diversity = -1
                
                for candidate_idx in random.sample(all_unseen, min(50, len(all_unseen))):
                    # Calculate average similarity to already selected seeds
                    similarities = []
                    for seed_idx in seeds:
                        sim = np.dot(beam_search_engine.normalized_embeddings[candidate_idx],
                                   beam_search_engine.normalized_embeddings[seed_idx])
                        similarities.append(sim)
                    
                    diversity = 1.0 - np.mean(similarities)  # Higher = more diverse
                    
                    if diversity > best_diversity:
                        best_diversity = diversity
                        best_watch = candidate_idx
                
                selected_watch = best_watch if best_watch is not None else random.choice(all_unseen)
            
            seeds.append(selected_watch)
            selected_indices.add(selected_watch)
            all_unseen.remove(selected_watch)
            print(f"   ✓ Random diverse: watch {selected_watch}")
    
    # Convert to watch objects with metadata
    seed_watches = []
    for i, idx in enumerate(seeds):
        if 0 <= idx < len(beam_search_engine.watch_data):
            watch = beam_search_engine.watch_data[idx].copy()
            watch['index'] = idx
            watch['seed_id'] = i
            watch['seed_strategy'] = _get_seed_strategy(i, num_seeds)
            watch['is_precomputed_seed'] = True
            
            # Add enhanced metadata
            watch['brand_category'] = beam_search_engine._categorize_brand(watch.get('brand', '').lower())
            watch['style'] = beam_search_engine._classify_watch_style_enhanced(watch)
            watch['luxury_level'] = beam_search_engine._assess_luxury_level(watch)
            
            if idx in beam_search_engine.semantic_features:
                watch['aesthetic_keywords'] = beam_search_engine.semantic_features[idx]['aesthetic_keywords']
            
            seed_watches.append(watch)
    
    print(f"✅ Generated {len(seed_watches)} diverse smart seeds")
    return seed_watches

def _get_seed_strategy(index: int, total: int) -> str:
    """Get the strategy used to select this seed."""
    aesthetic_cutoff = int(total * 0.4)
    style_cutoff = aesthetic_cutoff + int(total * 0.3)
    brand_cutoff = style_cutoff + int(total * 0.2)
    
    if index < aesthetic_cutoff:
        return 'aesthetic_cluster'
    elif index < style_cutoff:
        return 'style_cluster'
    elif index < brand_cutoff:
        return 'brand_category'
    else:
        return 'random_diverse'

def analyze_seed_diversity(seed_watches: List[Dict[str, Any]]):
    """Analyze and report on the diversity of generated seeds."""
    print("\n📊 Seed Diversity Analysis:")
    
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
    
    # Strategy distribution
    strategies = [watch.get('seed_strategy') for watch in seed_watches]
    strategy_counts = {}
    for strategy in strategies:
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print(f"   🎯 Strategy Distribution:")
    for strategy, count in strategy_counts.items():
        print(f"      - {strategy}: {count} seeds")
    
    print(f"\n✅ Diversity Score: {(unique_brands + unique_styles + unique_categories)/3:.1f}/20")

def save_smart_seeds(seed_watches: List[Dict[str, Any]], 
                    output_path: str = 'data/precomputed_smart_seeds.pkl'):
    """Save precomputed smart seeds to file."""
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data structure
    seeds_data = {
        'seeds': seed_watches,
        'generation_timestamp': np.datetime64('now').item().timestamp(),
        'total_seeds': len(seed_watches),
        'version': 'v1.0',
        'metadata': {
            'generation_method': 'enhanced_clustering_v2',
            'strategies': ['aesthetic_cluster', 'style_cluster', 'brand_category', 'random_diverse'],
            'diversity_optimized': True
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(seeds_data, f)
    
    print(f"💾 Saved {len(seed_watches)} smart seeds to: {output_path}")
    
    # Also save as JSON for readability
    json_path = output_path.replace('.pkl', '.json')
    import json
    
    # Convert for JSON serialization
    json_data = seeds_data.copy()
    json_data['seeds'] = [
        {
            'index': seed['index'],
            'brand': seed.get('brand'),
            'model': seed.get('model'),
            'style': seed.get('style'),
            'brand_category': seed.get('brand_category'),
            'price': seed.get('price'),
            'seed_strategy': seed.get('seed_strategy'),
            'aesthetic_keywords': seed.get('aesthetic_keywords', [])
        }
        for seed in seed_watches
    ]
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"📄 Also saved readable version to: {json_path}")

def main():
    """Main function to generate and save precomputed smart seeds."""
    print("🚀 Precomputing Smart Seeds for Watch Recommendation System")
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
        
        # Generate smart seeds
        print("\n🌱 Generating Smart Seeds...")
        seed_watches = generate_diverse_smart_seeds(beam_search_engine, num_seeds=20)
        
        # Analyze diversity
        analyze_seed_diversity(seed_watches)
        
        # Save seeds
        print("\n💾 Saving Smart Seeds...")
        save_smart_seeds(seed_watches)
        
        print("\n🎉 Smart Seeds Generation Complete!")
        print("The precomputed seeds are ready for use in the recommendation system.")
        
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
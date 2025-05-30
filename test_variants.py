#!/usr/bin/env python3
import sys
import os

# Add the backend models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from models.watch_embedder import WatchEmbedder
from models.variant_detector import WatchVariantDetector

def test_variant_detection():
    """Test the variant detection system and show examples."""
    
    print("Testing variant detection system...")
    
    # Load embeddings and data
    custom_model_path = "best_clip_model_description_model.pt"
    if os.path.exists(custom_model_path):
        print(f"Using custom CLIP model: {custom_model_path}")
        embedder = WatchEmbedder(model_path=custom_model_path)
    else:
        print("Using default CLIP model")
        embedder = WatchEmbedder()
    
    # Load existing embeddings
    embeddings_path = "embeddings/watch_image_embeddings.pkl"
    if os.path.exists(embeddings_path):
        print("Loading existing embeddings...")
        embeddings, watch_data = embedder.load_embeddings(embeddings_path)
    else:
        print("Embeddings not found, please run generate_clip_embeddings.py first")
        return False
    
    # Initialize variant detector
    variant_detector = WatchVariantDetector(embedder.watch_data, embedder.normalized_embeddings)
    
    # Get statistics
    stats = variant_detector.get_variant_stats()
    print(f"\n📊 Variant Detection Statistics:")
    print(f"   Total watches: {stats['total_watches']}")
    print(f"   Variant groups: {stats['variant_groups']}")
    print(f"   Watches in groups: {stats['watches_in_groups']}")
    print(f"   Single watches: {stats['single_watches']}")
    print(f"   Average group size: {stats['average_group_size']:.1f}")
    print(f"   Max group size: {stats['max_group_size']}")
    print(f"   Diversity improvement: {stats['diversity_reduction']:.1f}%")
    
    # Show some example variant groups
    print(f"\n🔍 Example Variant Groups:")
    count = 0
    for group_id, watch_indices in variant_detector.variant_groups.items():
        if count >= 5:  # Show first 5 groups
            break
        
        representative_idx = variant_detector.group_representatives[group_id]
        representative = watch_data[representative_idx]
        
        print(f"\nGroup {group_id} ({len(watch_indices)} variants):")
        print(f"   Representative: {representative['brand']} {representative['model_name']}")
        
        for idx in watch_indices:
            watch = watch_data[idx]
            is_rep = "👑" if idx == representative_idx else "  "
            print(f"   {is_rep} [{idx:3d}] {watch['brand']} {watch['model_name']}")
        
        count += 1
    
    # Test diversity filtering
    print(f"\n🎯 Testing Diversity Filtering:")
    all_indices = list(range(len(watch_data)))
    diverse_indices = variant_detector.filter_diverse_watches(all_indices)
    print(f"   Original: {len(all_indices)} watches")
    print(f"   Filtered: {len(diverse_indices)} diverse watches")
    print(f"   Reduction: {len(all_indices) - len(diverse_indices)} variants removed")
    
    return True

if __name__ == "__main__":
    test_variant_detection() 
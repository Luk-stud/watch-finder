#!/usr/bin/env python3
"""
Deduplicate watches based on image URLs to avoid showing the same watch multiple times.
"""

import pickle
import os
from typing import Dict, List, Any, Set
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_embeddings() -> Dict[str, Any]:
    """Load the current embeddings."""
    logger.info("Loading embeddings...")
    
    with open("watch_finder_v2/backend/data/precomputed_embeddings.pkl", "rb") as f:
        embeddings_data = pickle.load(f)
    
    logger.info(f"Loaded embeddings with {len(embeddings_data.get('watch_data', {}))} watches")
    return embeddings_data

def deduplicate_by_image(embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
    """Deduplicate watches based on image URLs."""
    logger.info("Deduplicating watches by image URL...")
    
    watch_data = embeddings_data.get("watch_data", {})
    
    # Group watches by image URL
    image_groups = {}
    for watch_id, watch_info in watch_data.items():
        image_url = watch_info.get("image_url")
        if image_url:
            if image_url not in image_groups:
                image_groups[image_url] = []
            image_groups[image_url].append((watch_id, watch_info))
    
    # Keep only one watch per image URL (prefer the one with more complete data)
    logger.info(f"Found {len(image_groups)} unique images")
    
    kept_watches = {}
    removed_count = 0
    
    for image_url, watches in image_groups.items():
        if len(watches) == 1:
            # Only one watch with this image, keep it
            watch_id, watch_info = watches[0]
            kept_watches[watch_id] = watch_info
        else:
            # Multiple watches with same image, choose the best one
            logger.info(f"Image {image_url[:50]}... has {len(watches)} variants:")
            for watch_id, watch_info in watches:
                brand = watch_info.get("brand", "")
                model = watch_info.get("model", "")
                logger.info(f"  - {watch_id}: {brand} {model}")
            
            # Choose the first one (or could implement more sophisticated selection)
            best_watch_id, best_watch_info = watches[0]
            kept_watches[best_watch_id] = best_watch_info
            removed_count += len(watches) - 1
            
            logger.info(f"  -> Kept {best_watch_id}, removed {len(watches) - 1} duplicates")
    
    # Update embeddings data
    embeddings_data["watch_data"] = kept_watches
    
    # Rebuild mappings
    watch_ids = list(kept_watches.keys())
    embeddings_data["watch_id_to_idx"] = {watch_id: i for i, watch_id in enumerate(watch_ids)}
    embeddings_data["idx_to_watch_id"] = {i: watch_id for i, watch_id in enumerate(watch_ids)}
    embeddings_data["available_watches"] = set(watch_ids)
    
    # Update items matrix to match kept watches
    original_matrix = embeddings_data.get("items_matrix")
    if original_matrix is not None:
        kept_indices = [embeddings_data["watch_id_to_idx"][watch_id] for watch_id in watch_ids]
        embeddings_data["items_matrix"] = original_matrix[kept_indices]
        embeddings_data["dim"] = embeddings_data["items_matrix"].shape[1]
    
    logger.info(f"Original watches: {len(watch_data)}")
    logger.info(f"Kept watches: {len(kept_watches)}")
    logger.info(f"Removed duplicates: {removed_count}")
    logger.info(f"Deduplication rate: {removed_count/len(watch_data)*100:.1f}%")
    
    return embeddings_data

def save_deduplicated_embeddings(embeddings_data: Dict[str, Any]) -> None:
    """Save the deduplicated embeddings."""
    logger.info("Saving deduplicated embeddings...")
    
    output_path = "watch_finder_v2/backend/data/precomputed_embeddings.pkl"
    
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_data, f)
    
    logger.info(f"Saved deduplicated embeddings to {output_path}")

def main():
    """Main function to deduplicate watches by image URL."""
    logger.info("Starting image-based deduplication...")
    
    try:
        # Load current embeddings
        embeddings_data = load_embeddings()
        
        # Deduplicate by image URL
        deduplicated_embeddings = deduplicate_by_image(embeddings_data)
        
        # Save deduplicated embeddings
        save_deduplicated_embeddings(deduplicated_embeddings)
        
        logger.info("✅ Successfully deduplicated watches by image URL!")
        
    except Exception as e:
        logger.error(f"❌ Error during deduplication: {e}")
        raise

if __name__ == "__main__":
    main() 
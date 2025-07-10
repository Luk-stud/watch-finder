#!/usr/bin/env python3
"""
Merge original CDN URLs with DINO embeddings to restore image functionality.
"""

import json
import pickle
import os
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_original_data() -> Dict[str, Dict[str, Any]]:
    """Load the original processed data with CDN URLs."""
    logger.info("Loading original processed data...")
    
    with open("data/processed/watches_processed.json", "r") as f:
        original_data = json.load(f)
    
    # Create a mapping from brand+model to the original data
    url_mapping = {}
    for item in original_data:
        brand = item.get("brand", "").lower().strip()
        model = item.get("model_name", "").lower().strip()
        key = f"{brand}_{model}"
        url_mapping[key] = item
    
    logger.info(f"Loaded {len(url_mapping)} original watch entries")
    return url_mapping

def load_dino_embeddings() -> Dict[str, Any]:
    """Load the current DINO embeddings."""
    logger.info("Loading DINO embeddings...")
    
    with open("watch_finder_v2/backend/data/precomputed_embeddings.pkl", "rb") as f:
        embeddings_data = pickle.load(f)
    
    logger.info(f"Loaded DINO embeddings with {len(embeddings_data.get('watch_data', {}))} watches")
    return embeddings_data

def merge_urls_with_embeddings(url_mapping: Dict[str, Dict[str, Any]], 
                              embeddings_data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge CDN URLs into the DINO embeddings."""
    logger.info("Merging CDN URLs with DINO embeddings...")
    
    watch_data = embeddings_data.get("watch_data", {})
    updated_count = 0
    missing_count = 0
    
    for watch_id, watch_info in watch_data.items():
        brand = watch_info.get("brand", "").lower().strip()
        model = watch_info.get("model", "").lower().strip()
        key = f"{brand}_{model}"
        
        if key in url_mapping:
            original_item = url_mapping[key]
            
            # Add the CDN URL and other useful fields
            watch_info["image_url"] = original_item.get("image_url")
            watch_info["product_url"] = original_item.get("product_url")
            watch_info["price"] = original_item.get("price")
            watch_info["description"] = original_item.get("description")
            watch_info["source"] = original_item.get("source")
            
            # Map to frontend-expected fields
            watch_info["main_image"] = original_item.get("image_url")
            
            updated_count += 1
        else:
            # Keep placeholder for watches without URLs
            watch_info["image_url"] = None
            watch_info["product_url"] = None
            watch_info["main_image"] = None
            missing_count += 1
    
    logger.info(f"Updated {updated_count} watches with CDN URLs")
    logger.info(f"Missing URLs for {missing_count} watches (using placeholders)")
    
    return embeddings_data

def save_updated_embeddings(embeddings_data: Dict[str, Any]) -> None:
    """Save the updated embeddings back to the backend directory."""
    logger.info("Saving updated embeddings...")
    
    output_path = "watch_finder_v2/backend/data/precomputed_embeddings.pkl"
    
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_data, f)
    
    logger.info(f"Saved updated embeddings to {output_path}")

def main():
    """Main function to merge URLs with DINO embeddings."""
    logger.info("Starting URL merge process...")
    
    try:
        # Load original data with CDN URLs
        url_mapping = load_original_data()
        
        # Load current DINO embeddings
        embeddings_data = load_dino_embeddings()
        
        # Merge URLs into embeddings
        updated_embeddings = merge_urls_with_embeddings(url_mapping, embeddings_data)
        
        # Save updated embeddings
        save_updated_embeddings(updated_embeddings)
        
        logger.info("✅ Successfully merged CDN URLs with DINO embeddings!")
        
    except Exception as e:
        logger.error(f"❌ Error during merge process: {e}")
        raise

if __name__ == "__main__":
    main() 
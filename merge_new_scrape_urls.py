#!/usr/bin/env python3
"""
Merge new scraping data URLs with DINO embeddings to get real images for all watches.
"""

import json
import pickle
import os
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_new_scrape_data() -> Dict[str, Dict[str, Any]]:
    """Load the new scraping data with CDN URLs."""
    logger.info("Loading new scraping data...")
    
    with open("production_scrape_20250601_175426/data/final_scrape.json", "r") as f:
        scrape_data = json.load(f)
    
    # Create a mapping from brand+model to the scrape data
    url_mapping = {}
    for item in scrape_data:
        brand = item.get("specs", {}).get("brand", "").strip()
        model = item.get("specs", {}).get("model", "").strip()
        
        if brand and model:
            key = f"{brand}_{model}"
            url_mapping[key] = item
    
    logger.info(f"Loaded {len(scrape_data)} watches from new scrape data")
    logger.info(f"Created {len(url_mapping)} brand_model mappings")
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
    """Merge new scraping URLs into the DINO embeddings."""
    logger.info("Merging new scraping URLs with DINO embeddings...")
    
    watch_data = embeddings_data.get("watch_data", {})
    updated_count = 0
    missing_count = 0
    
    for watch_id, watch_info in watch_data.items():
        brand = watch_info.get("brand", "").strip()
        model = watch_info.get("model", "").strip()
        key = f"{brand}_{model}"
        
        if key in url_mapping:
            scrape_item = url_mapping[key]
            
            # Add the CDN URL and other useful fields
            watch_info["image_url"] = scrape_item.get("main_image")
            watch_info["product_url"] = scrape_item.get("specs", {}).get("url")
            watch_info["price"] = scrape_item.get("specs", {}).get("price_usd")
            watch_info["description"] = f"Brand: {brand}. Model: {model}"
            watch_info["source"] = "extropian"
            
            # Map to frontend-expected fields
            watch_info["main_image"] = scrape_item.get("main_image")
            
            updated_count += 1
        else:
            # Keep placeholder for watches without URLs
            watch_info["image_url"] = None
            watch_info["product_url"] = None
            watch_info["main_image"] = None
            missing_count += 1
    
    logger.info(f"Updated {updated_count} watches with CDN URLs")
    logger.info(f"Missing URLs for {missing_count} watches (using placeholders)")
    logger.info(f"Success rate: {updated_count/(updated_count+missing_count)*100:.1f}%")
    
    return embeddings_data

def save_updated_embeddings(embeddings_data: Dict[str, Any]) -> None:
    """Save the updated embeddings back to the backend directory."""
    logger.info("Saving updated embeddings...")
    
    output_path = "watch_finder_v2/backend/data/precomputed_embeddings.pkl"
    
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_data, f)
    
    logger.info(f"Saved updated embeddings to {output_path}")

def main():
    """Main function to merge new scraping URLs with DINO embeddings."""
    logger.info("Starting new scraping URL merge process...")
    
    try:
        # Load new scraping data with CDN URLs
        url_mapping = load_new_scrape_data()
        
        # Load current DINO embeddings
        embeddings_data = load_dino_embeddings()
        
        # Merge URLs into embeddings
        updated_embeddings = merge_urls_with_embeddings(url_mapping, embeddings_data)
        
        # Save updated embeddings
        save_updated_embeddings(updated_embeddings)
        
        logger.info("✅ Successfully merged new scraping URLs with DINO embeddings!")
        
    except Exception as e:
        logger.error(f"❌ Error during merge process: {e}")
        raise

if __name__ == "__main__":
    main() 
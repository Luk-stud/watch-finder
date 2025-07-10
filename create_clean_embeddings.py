#!/usr/bin/env python3
"""
Create clean embeddings directly from scraped data to avoid duplicates.
"""

import json
import pickle
import numpy as np
from typing import Dict, List, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_scraped_data() -> List[Dict[str, Any]]:
    """Load the scraped data."""
    logger.info("Loading scraped data...")
    
    with open("production_scrape_20250601_175426/data/final_scrape.json", "r") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} watches from scraped data")
    return data

def load_dino_embeddings() -> np.ndarray:
    """Load the DINO embeddings."""
    logger.info("Loading DINO embeddings...")
    
    with open("watch_finder_v2/backend/data/watch_dino_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    
    logger.info(f"Loaded DINO embeddings with shape: {embeddings.shape}")
    return embeddings

def create_clean_embeddings(scraped_data: List[Dict[str, Any]], 
                           dino_embeddings: np.ndarray) -> Dict[str, Any]:
    """Create clean embeddings directly from scraped data."""
    logger.info("Creating clean embeddings...")
    
    # Create watch data directly from scraped data
    watch_data = {}
    items_matrix = []
    
    for i, item in enumerate(scraped_data):
        watch_id = f"watch_{i}"
        
        # Extract data from scraped item
        specs = item.get("specs", {})
        brand = specs.get("brand", "")
        model = specs.get("model", "")
        
        # Create watch info
        watch_info = {
            "watch_id": watch_id,
            "brand": brand,
            "model": model,
            "image_url": item.get("main_image"),
            "product_url": specs.get("url"),
            "price": specs.get("price_usd"),
            "description": f"Brand: {brand}. Model: {model}",
            "source": "extropian",
            "main_image": item.get("main_image"),
            "specs": specs
        }
        
        watch_data[watch_id] = watch_info
        items_matrix.append(dino_embeddings[i])
    
    # Convert to numpy array
    items_matrix = np.array(items_matrix, dtype=np.float32)
    
    # Create mappings
    watch_ids = list(watch_data.keys())
    watch_id_to_idx = {watch_id: i for i, watch_id in enumerate(watch_ids)}
    idx_to_watch_id = {i: watch_id for i, watch_id in enumerate(watch_ids)}
    available_watches = set(watch_ids)
    
    # Create embeddings data structure
    embeddings_data = {
        "watch_data": watch_data,
        "items_matrix": items_matrix,
        "watch_id_to_idx": watch_id_to_idx,
        "idx_to_watch_id": idx_to_watch_id,
        "available_watches": available_watches,
        "dim": items_matrix.shape[1]
    }
    
    logger.info(f"Created clean embeddings with {len(watch_data)} watches")
    logger.info(f"Embeddings matrix shape: {items_matrix.shape}")
    
    return embeddings_data

def save_embeddings(embeddings_data: Dict[str, Any]) -> None:
    """Save the clean embeddings."""
    logger.info("Saving clean embeddings...")
    
    output_path = "watch_finder_v2/backend/data/precomputed_embeddings.pkl"
    
    with open(output_path, "wb") as f:
        pickle.dump(embeddings_data, f)
    
    logger.info(f"Saved clean embeddings to {output_path}")

def main():
    """Main function to create clean embeddings."""
    logger.info("Starting clean embeddings creation...")
    
    try:
        # Load data
        scraped_data = load_scraped_data()
        dino_embeddings = load_dino_embeddings()
        
        # Create clean embeddings
        embeddings_data = create_clean_embeddings(scraped_data, dino_embeddings)
        
        # Save embeddings
        save_embeddings(embeddings_data)
        
        logger.info("✅ Successfully created clean embeddings!")
        
    except Exception as e:
        logger.error(f"❌ Error during embeddings creation: {e}")
        raise

if __name__ == "__main__":
    main() 
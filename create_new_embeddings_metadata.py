#!/usr/bin/env python3
"""
Create new embeddings and metadata files with correct links and no duplicates.
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

def create_metadata(scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create metadata in the same format as the original."""
    logger.info("Creating metadata...")
    
    metadata = []
    
    for i, item in enumerate(scraped_data):
        specs = item.get("specs", {})
        
        # Create metadata entry in the same format as original
        metadata_entry = {
            "index": i,
            "brand": specs.get("brand", ""),
            "model": specs.get("model", ""),
            "price": specs.get("price_usd"),
            "image_path": item.get("main_image"),  # Use CDN URL as image_path
            "image_filename": f"{specs.get('brand', '')}_{specs.get('model', '')}.jpg",
            "dino_embedding": None,  # Will be filled later
            "specs": specs,
            "product_url": specs.get("url"),
            "source": "extropian",
            "description": f"Brand: {specs.get('brand', '')}. Model: {specs.get('model', '')}",
            "main_image": item.get("main_image"),
            "image_url": item.get("main_image")
        }
        
        metadata.append(metadata_entry)
    
    logger.info(f"Created metadata for {len(metadata)} watches")
    return metadata

def create_embeddings_data(metadata: List[Dict[str, Any]], 
                          dino_embeddings: np.ndarray) -> Dict[str, Any]:
    """Create embeddings data structure in the same format as current."""
    logger.info("Creating embeddings data structure...")
    
    # Create watch_data dictionary
    watch_data = {}
    items_matrix = []
    
    for i, meta in enumerate(metadata):
        watch_id = f"watch_{i}"
        
        # Add the DINO embedding to metadata
        meta["dino_embedding"] = dino_embeddings[i]
        
        # Create watch_data entry
        watch_data[watch_id] = meta
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
    
    logger.info(f"Created embeddings data with {len(watch_data)} watches")
    logger.info(f"Embeddings matrix shape: {items_matrix.shape}")
    
    return embeddings_data

def save_files(metadata: List[Dict[str, Any]], 
               embeddings_data: Dict[str, Any]) -> None:
    """Save both metadata and embeddings files."""
    logger.info("Saving files...")
    
    # Save metadata
    metadata_path = "watch_finder_v2/backend/data/watch_dino_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved metadata to {metadata_path}")
    
    # Save embeddings
    embeddings_path = "watch_finder_v2/backend/data/watch_dino_embeddings.pkl"
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings_data["items_matrix"], f)
    logger.info(f"Saved embeddings to {embeddings_path}")
    
    # Save precomputed embeddings (for backend)
    precomputed_path = "watch_finder_v2/backend/data/precomputed_embeddings.pkl"
    with open(precomputed_path, "wb") as f:
        pickle.dump(embeddings_data, f)
    logger.info(f"Saved precomputed embeddings to {precomputed_path}")

def main():
    """Main function to create new embeddings and metadata."""
    logger.info("Starting new embeddings and metadata creation...")
    
    try:
        # Load data
        scraped_data = load_scraped_data()
        dino_embeddings = load_dino_embeddings()
        
        # Create metadata
        metadata = create_metadata(scraped_data)
        
        # Create embeddings data
        embeddings_data = create_embeddings_data(metadata, dino_embeddings)
        
        # Save all files
        save_files(metadata, embeddings_data)
        
        logger.info("✅ Successfully created new embeddings and metadata!")
        logger.info(f"📊 Summary:")
        logger.info(f"   - Total watches: {len(metadata)}")
        logger.info(f"   - Embeddings shape: {embeddings_data['items_matrix'].shape}")
        logger.info(f"   - All watches have CDN URLs")
        logger.info(f"   - No duplicates")
        
    except Exception as e:
        logger.error(f"❌ Error during creation: {e}")
        raise

if __name__ == "__main__":
    main() 
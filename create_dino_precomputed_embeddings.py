#!/usr/bin/env python3
"""
Create DINO Precomputed Embeddings for Backend

This script creates a precomputed embeddings file using DINO embeddings
instead of the current CLIP + text combination. It maintains the same
format expected by all backend engines.
"""

import os
import pickle
import numpy as np
import time
import logging
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dino_data():
    """Load DINO embeddings and metadata."""
    logger.info("📖 Loading DINO embeddings...")
    
    # Load DINO embeddings
    dino_embeddings_path = 'watch_finder_v2/backend/data/watch_dino_embeddings.pkl'
    with open(dino_embeddings_path, 'rb') as f:
        dino_embeddings = pickle.load(f)
    
    # Load DINO metadata
    dino_metadata_path = 'watch_finder_v2/backend/data/watch_dino_metadata.pkl'
    with open(dino_metadata_path, 'rb') as f:
        dino_metadata = pickle.load(f)
    
    logger.info(f"✅ Loaded {len(dino_metadata)} watches with DINO embeddings")
    logger.info(f"📊 DINO embedding shape: {dino_embeddings.shape}")
    
    return dino_embeddings, dino_metadata

def create_precomputed_data(dino_embeddings, dino_metadata):
    """Create precomputed data structure compatible with backend engines."""
    logger.info("🔧 Creating precomputed data structure...")
    
    # Convert embeddings to the expected format
    final_embeddings = {}
    watch_data = {}
    
    # DINO metadata is a list, so we iterate through it
    for idx, metadata in enumerate(dino_metadata):
        if idx < len(dino_embeddings):
            # Get the DINO embedding for this watch
            embedding = dino_embeddings[idx]
            
            # Skip if embedding is all zeros (failed embedding)
            if np.all(embedding == 0):
                logger.warning(f"Skipping watch {idx} - zero embedding")
                continue
            
            # Create watch ID from metadata
            watch_id = metadata.get('id', f'watch_{idx}')
            
            # Store the embedding
            final_embeddings[watch_id] = embedding.astype(np.float32)
            
            # Store the metadata
            watch_data[watch_id] = metadata
    
    # Get embedding dimension
    if final_embeddings:
        sample_embedding = next(iter(final_embeddings.values()))
        embedding_dim = len(sample_embedding)
    else:
        embedding_dim = 768  # DINO ViT-B/14 default
    
    logger.info(f"✅ Created precomputed data:")
    logger.info(f"   • Watches: {len(watch_data)}")
    logger.info(f"   • Embeddings: {len(final_embeddings)}")
    logger.info(f"   • Embedding dimension: {embedding_dim}D")
    
    return {
        'watch_data': watch_data,
        'final_embeddings': final_embeddings,
        'embedding_dim': embedding_dim,
        'embedding_type': 'dino_vitb14',
        'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_watches': len(watch_data),
        'successful_embeddings': len(final_embeddings)
    }

def save_precomputed_data(precomputed_data, output_path):
    """Save precomputed data to file."""
    logger.info(f"💾 Saving precomputed data to {output_path}...")
    
    with open(output_path, 'wb') as f:
        pickle.dump(precomputed_data, f)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"✅ Saved precomputed data ({file_size:.1f}MB)")

def main():
    """Main function to create DINO precomputed embeddings."""
    logger.info("🚀 Starting DINO precomputed embeddings creation...")
    
    # Load DINO data
    dino_embeddings, dino_metadata = load_dino_data()
    
    # Create precomputed data structure
    precomputed_data = create_precomputed_data(dino_embeddings, dino_metadata)
    
    # Save to backend data directory
    output_path = 'watch_finder_v2/backend/data/precomputed_embeddings_dino.pkl'
    save_precomputed_data(precomputed_data, output_path)
    
    # Also save as backup with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    backup_path = f'watch_finder_v2/backend/data/precomputed_embeddings_dino_{timestamp}.pkl'
    save_precomputed_data(precomputed_data, backup_path)
    
    logger.info("✅ DINO precomputed embeddings creation complete!")
    logger.info(f"📁 Main file: {output_path}")
    logger.info(f"📁 Backup file: {backup_path}")
    
    return precomputed_data

if __name__ == "__main__":
    main() 
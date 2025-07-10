#!/usr/bin/env python3
"""
Update Backend for DINO Embeddings
==================================

This script updates the backend to use DINO embeddings instead of the current
CLIP + text embedding system. It creates new precomputed embeddings files
that are compatible with the existing SimpleSgdEngine.
"""

import os
import pickle
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dino_data():
    """Load DINO embeddings and metadata."""
    logger.info("📖 Loading DINO embeddings...")
    
    try:
        # Load DINO embeddings
        with open('watch_finder_v2/backend/data/watch_dino_embeddings.pkl', 'rb') as f:
            dino_embeddings = pickle.load(f)
        logger.info(f"✅ Loaded DINO embeddings: {dino_embeddings.shape}")
        
        # Load DINO metadata
        with open('watch_finder_v2/backend/data/watch_dino_metadata.pkl', 'rb') as f:
            dino_metadata = pickle.load(f)
        logger.info(f"✅ Loaded DINO metadata: {len(dino_metadata)} watches")
        
        return dino_embeddings, dino_metadata
        
    except FileNotFoundError as e:
        logger.error(f"❌ DINO data not found: {e}")
        logger.info("💡 Please run generate_dino_embeddings.py first")
        return None, None

def create_dino_precomputed_embeddings(dino_embeddings: np.ndarray, dino_metadata: List[Dict]):
    """Create precomputed embeddings file compatible with SimpleSgdEngine."""
    logger.info("🔄 Creating DINO precomputed embeddings...")
    
    precomputed_data = {
        'watch_data': {},           # Watch metadata by ID
        'final_embeddings': {},     # Final normalized embeddings by ID
        'embedding_dim': dino_embeddings.shape[1],  # DINO dimension
        'dino_dim': dino_embeddings.shape[1],       # Same as embedding_dim for DINO-only
        'total_watches': len(dino_metadata),
        'embedding_type': 'dino_only',
        'model_name': 'dinov2_vitb14'
    }
    
    processed_count = 0
    
    for idx, watch_dict in enumerate(dino_metadata):
        try:
            watch_id = watch_dict.get('index', idx)
            
            # Store watch data
            precomputed_data['watch_data'][watch_id] = {
                'watch_id': watch_id,
                'index': watch_id,
                'brand': watch_dict.get('brand', 'Unknown'),
                'model': watch_dict.get('model', 'Unknown'),
                'price': watch_dict.get('price', 0),
                'image_path': watch_dict.get('image_path', ''),
                'image_filename': watch_dict.get('image_filename', ''),
                'description': '',  # No text description for DINO-only
                'ai_description': '',  # No AI description for DINO-only
                'source': watch_dict.get('source', ''),
                'specs': watch_dict.get('specs', {}),
                'product_url': watch_dict.get('product_url', ''),
                'embedding_type': 'dino_only'
            }
            
            # Get DINO embedding
            dino_emb = dino_embeddings[idx]
            
            # Normalize the embedding
            dino_norm = np.linalg.norm(dino_emb)
            if dino_norm > 0:
                dino_emb = dino_emb / dino_norm
            
            # Store final embedding (same as DINO for DINO-only system)
            precomputed_data['final_embeddings'][watch_id] = dino_emb
            
            processed_count += 1
            
            if processed_count % 100 == 0:
                logger.info(f"📊 Processed {processed_count}/{len(dino_metadata)} watches")
                
        except Exception as e:
            logger.error(f"Error processing watch {idx}: {e}")
            continue
    
    logger.info(f"✅ Created precomputed data for {processed_count} watches")
    return precomputed_data

def save_precomputed_data(precomputed_data: Dict[str, Any], output_path: str = None):
    """Save precomputed embeddings data."""
    if output_path is None:
        output_path = 'watch_finder_v2/backend/data/precomputed_embeddings_dino.pkl'
    
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(precomputed_data, f)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"✅ Saved precomputed DINO embeddings to {output_path}")
        logger.info(f"📊 File size: {file_size:.1f}MB")
        logger.info(f"📊 Embedding dimension: {precomputed_data['embedding_dim']}D")
        logger.info(f"📊 Total watches: {precomputed_data['total_watches']}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"❌ Failed to save precomputed data: {e}")
        return None

def backup_current_embeddings():
    """Backup current embeddings before replacing."""
    current_file = 'watch_finder_v2/backend/data/precomputed_embeddings.pkl'
    backup_file = 'watch_finder_v2/backend/data/precomputed_embeddings_backup.pkl'
    
    if os.path.exists(current_file):
        try:
            import shutil
            shutil.copy2(current_file, backup_file)
            logger.info(f"✅ Backed up current embeddings to {backup_file}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to backup current embeddings: {e}")
            return False
    else:
        logger.warning("⚠️ No current embeddings file to backup")
        return True

def replace_current_embeddings(dino_file: str):
    """Replace current embeddings with DINO embeddings."""
    current_file = 'watch_finder_v2/backend/data/precomputed_embeddings.pkl'
    
    try:
        import shutil
        shutil.copy2(dino_file, current_file)
        logger.info(f"✅ Replaced current embeddings with DINO embeddings")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to replace current embeddings: {e}")
        return False

def main():
    """Main function to update backend for DINO embeddings."""
    logger.info("🦕 Updating Backend for DINO Embeddings")
    logger.info("=" * 50)
    
    # Step 1: Load DINO data
    dino_embeddings, dino_metadata = load_dino_data()
    if dino_embeddings is None or dino_metadata is None:
        return
    
    # Step 2: Backup current embeddings
    if not backup_current_embeddings():
        logger.error("❌ Failed to backup current embeddings. Aborting.")
        return
    
    # Step 3: Create DINO precomputed embeddings
    precomputed_data = create_dino_precomputed_embeddings(dino_embeddings, dino_metadata)
    
    # Step 4: Save DINO precomputed embeddings
    dino_file = save_precomputed_data(precomputed_data)
    if dino_file is None:
        return
    
    # Step 5: Replace current embeddings (optional)
    replace_choice = input("\n🤔 Replace current embeddings with DINO embeddings? (y/n): ").lower().strip()
    if replace_choice == 'y':
        if replace_current_embeddings(dino_file):
            logger.info("✅ Backend updated to use DINO embeddings!")
            logger.info("🔄 Restart your backend to use the new embeddings")
        else:
            logger.error("❌ Failed to replace embeddings")
    else:
        logger.info("ℹ️  DINO embeddings saved but not activated")
        logger.info(f"📁 DINO embeddings available at: {dino_file}")
        logger.info("🔄 To activate, manually replace precomputed_embeddings.pkl")
    
    # Step 6: Show comparison
    logger.info("\n📊 Comparison Summary:")
    logger.info("   • Previous: CLIP (512D) + Text (1536D) → PCA → 768D")
    logger.info("   • New: DINO ViT-B/14 → 768D (direct)")
    logger.info("   • Advantage: Single model, better visual features")
    logger.info("   • Compatibility: Same dimensionality, drop-in replacement")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Create Full Precomputed Embeddings for MABWiser Engine

This script combines the full metadata, text embeddings, and CLIP embeddings
into a comprehensive precomputed embeddings file that includes all watch data,
images, and statistics.
"""

import os
import pickle
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load all the data files."""
    data_dir = 'data'
    
    logger.info("📖 Loading watch metadata...")
    with open(os.path.join(data_dir, 'watch_text_metadata.pkl'), 'rb') as f:
        metadata_list = pickle.load(f)
    logger.info(f"✅ Loaded {len(metadata_list)} watches from metadata")
    
    logger.info("📖 Loading text embeddings...")
    with open(os.path.join(data_dir, 'watch_text_embeddings.pkl'), 'rb') as f:
        text_embeddings = pickle.load(f)
    logger.info(f"✅ Loaded {len(text_embeddings)} text embeddings")
    
    logger.info("📖 Loading CLIP embeddings...")
    try:
        with open(os.path.join(data_dir, 'watch_clip_embeddings.pkl'), 'rb') as f:
            clip_embeddings = pickle.load(f)
        logger.info(f"✅ Loaded {len(clip_embeddings)} CLIP embeddings")
    except FileNotFoundError:
        logger.warning("⚠️ CLIP embeddings not found, using zeros")
        clip_embeddings = [np.zeros(512) for _ in range(len(metadata_list))]
    
    return metadata_list, text_embeddings, clip_embeddings

def create_combined_embeddings(metadata_list, text_embeddings, clip_embeddings):
    """Create combined embeddings and watch data dictionary."""
    logger.info("🔄 Creating combined embeddings...")
    
    watch_data = {}
    final_embeddings = {}
    
    text_dim = len(text_embeddings[0]) if text_embeddings else 0
    clip_dim = len(clip_embeddings[0]) if clip_embeddings else 0
    combined_dim = text_dim + clip_dim
    
    logger.info(f"📏 Embedding dimensions: Text={text_dim}, CLIP={clip_dim}, Combined={combined_dim}")
    
    for i, watch_dict in enumerate(metadata_list):
        try:
            # Use index as watch_id, fallback to i
            watch_id = watch_dict.get('index', i)
            
            # Create comprehensive watch data
            watch_data[watch_id] = {
                'watch_id': watch_id,
                'index': watch_id,
                'brand': watch_dict.get('brand', 'Unknown'),
                'model': watch_dict.get('model', 'Unknown'),
                'price': watch_dict.get('price', 0),
                'image_url': watch_dict.get('image_url', ''),
                'description': watch_dict.get('description', ''),
                'ai_description': watch_dict.get('ai_description', ''),
                'source': watch_dict.get('source', ''),
                'specs': watch_dict.get('specs', {}),
                'product_url': watch_dict.get('product_url', ''),
            }
            
            # Get embeddings
            text_emb = text_embeddings[i] if i < len(text_embeddings) else np.zeros(text_dim)
            clip_emb = clip_embeddings[i] if i < len(clip_embeddings) else np.zeros(clip_dim)
            
            # Combine embeddings
            if text_dim > 0 and clip_dim > 0:
                combined = np.concatenate([text_emb, clip_emb])
            elif text_dim > 0:
                combined = text_emb
            elif clip_dim > 0:
                combined = clip_emb
            else:
                combined = np.zeros(200)  # Fallback
            
            # Normalize combined embedding
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
            
            final_embeddings[watch_id] = combined
            
            if i % 100 == 0:
                logger.info(f"🔄 Processed {i+1}/{len(metadata_list)} watches")
                
        except Exception as e:
            logger.error(f"❌ Error processing watch {i}: {e}")
            continue
    
    logger.info(f"✅ Created {len(watch_data)} watch entries and {len(final_embeddings)} embeddings")
    return watch_data, final_embeddings, combined_dim

def save_precomputed_data(watch_data, final_embeddings, embedding_dim):
    """Save the comprehensive precomputed data."""
    logger.info("💾 Saving precomputed embeddings...")
    
    precomputed_data = {
        'watch_data': watch_data,
        'final_embeddings': final_embeddings,
        'embedding_dim': embedding_dim,
        'created_at': time.time(),
        'total_watches': len(watch_data)
    }
    
    output_path = 'data/precomputed_embeddings.pkl'
    
    with open(output_path, 'wb') as f:
        pickle.dump(precomputed_data, f)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"✅ Saved precomputed embeddings to {output_path}")
    logger.info(f"   • File size: {file_size:.1f}MB")
    logger.info(f"   • Total watches: {len(watch_data)}")
    logger.info(f"   • Embedding dim: {embedding_dim}D")

def main():
    """Main function to create comprehensive precomputed embeddings."""
    start_time = time.time()
    logger.info("🚀 Starting comprehensive precomputed embeddings creation...")
    
    try:
        # Load data
        metadata_list, text_embeddings, clip_embeddings = load_data()
        
        # Create combined data
        watch_data, final_embeddings, embedding_dim = create_combined_embeddings(
            metadata_list, text_embeddings, clip_embeddings
        )
        
        # Save precomputed data
        save_precomputed_data(watch_data, final_embeddings, embedding_dim)
        
        total_time = time.time() - start_time
        logger.info(f"🎉 Successfully created comprehensive precomputed embeddings in {total_time:.2f}s")
        
        # Show sample of final data
        logger.info("\n📊 Sample of final data:")
        sample_id = list(watch_data.keys())[0]
        sample_watch = watch_data[sample_id]
        logger.info(f"Sample watch {sample_id}:")
        for key, value in sample_watch.items():
            if key == 'specs':
                logger.info(f"  {key}: {len(value)} spec fields")
            else:
                value_str = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                logger.info(f"  {key}: {value_str}")
        
    except Exception as e:
        logger.error(f"❌ Failed to create precomputed embeddings: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 
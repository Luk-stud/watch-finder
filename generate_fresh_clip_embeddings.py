#!/usr/bin/env python3
"""
Generate Fresh CLIP ViT-B/32 Embeddings for Backend

This script:
1. Loads the scraped watch data (with image URLs)
2. Downloads images from URLs and generates CLIP ViT-B/32 embeddings
3. Creates the exact format expected by the backend
4. Replaces the current backend embeddings with the new ones

Uses CLIP ViT-B/32 which performed best in our comparison tests.
"""

import os
import json
import pickle
import numpy as np
import requests
from PIL import Image
import torch
import clip
from typing import Dict, List, Any, Optional
import time
import logging
from io import BytesIO
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FreshCLIPEmbeddingGenerator:
    def __init__(self, 
                 model_name: str = "ViT-B/32",
                 batch_size: int = 32,
                 max_retries: int = 3,
                 timeout: int = 10):
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔧 Using device: {self.device}")
        
        logger.info(f"📥 Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        logger.info("✅ CLIP model loaded successfully")
        
        # Stats tracking
        self.stats = {
            'total_watches': 0,
            'successful_embeddings': 0,
            'failed_downloads': 0,
            'failed_processing': 0,
            'skipped_no_url': 0
        }
    
    def load_scraped_data(self, filepath: str = 'data/raw/watch_data_final_scrape.json') -> List[Dict]:
        """Load the scraped watch data."""
        logger.info(f"📖 Loading scraped data from: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"✅ Loaded {len(data)} watches from scraped data")
            return data
            
        except Exception as e:
            logger.error(f"❌ Failed to load scraped data: {e}")
            return []
    
    def download_image(self, image_url: str) -> Optional[Image.Image]:
        """Download and process an image from URL."""
        if not image_url:
            return None
        
        for attempt in range(self.max_retries):
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(image_url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                
                # Open image
                image = Image.open(BytesIO(response.content))
                image = image.convert('RGB')
                
                return image
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.debug(f"Failed to download {image_url}: {e}")
                    return None
                time.sleep(1)  # Wait before retry
        
        return None
    
    def generate_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """Generate CLIP embedding for an image."""
        try:
            # Preprocess image
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)
                embedding = embedding.cpu().numpy().flatten()
                
            return embedding
            
        except Exception as e:
            logger.debug(f"Failed to generate embedding: {e}")
            return None
    
    def process_batch(self, batch_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Process a batch of watches."""
        embeddings = {}
        
        for watch_data in batch_data:
            watch_id = str(watch_data.get('index', ''))
            image_url = watch_data.get('image_url', '')
            
            if not image_url:
                self.stats['skipped_no_url'] += 1
                continue
            
            # Download image
            image = self.download_image(image_url)
            if image is None:
                self.stats['failed_downloads'] += 1
                continue
            
            # Generate embedding
            embedding = self.generate_embedding(image)
            if embedding is None:
                self.stats['failed_processing'] += 1
                continue
            
            embeddings[watch_id] = embedding
            self.stats['successful_embeddings'] += 1
        
        return embeddings
    
    def convert_to_backend_format(self, scraped_data: List[Dict], embeddings: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Convert scraped data and embeddings to backend format."""
        logger.info("🔄 Converting to backend format...")
        
        watch_data = {}
        final_embeddings = {}
        watch_id_to_idx = {}
        idx_to_watch_id = {}
        available_watches = set()
        
        for idx, watch in enumerate(scraped_data):
            watch_id = str(watch.get('index', ''))
            
            # Only include watches that have embeddings
            if watch_id not in embeddings:
                continue
            
            # Create backend-compatible watch data
            backend_watch = {
                'index': watch.get('index', idx),
                'brand': watch.get('brand', ''),
                'model': watch.get('model', ''),
                'price': watch.get('price', ''),
                'image_path': '',  # No local image path
                'image_filename': '',  # No local filename
                'dino_embedding': embeddings[watch_id],  # Use CLIP embedding (backend expects this key)
                'specs': watch.get('specs', {}),
                'product_url': watch.get('url', ''),
                'source': 'scraped_data',
                'description': watch.get('description', ''),
                'main_image': watch.get('image_url', ''),
                'image_url': watch.get('image_url', '')
            }
            
            watch_data[watch_id] = backend_watch
            final_embeddings[watch_id] = embeddings[watch_id]
            watch_id_to_idx[watch_id] = len(available_watches)
            idx_to_watch_id[len(available_watches)] = watch_id
            available_watches.add(watch_id)
        
        # Create the complete backend data structure
        backend_data = {
            'watch_data': watch_data,
            'final_embeddings': final_embeddings,
            'embedding_dim': 512,  # CLIP ViT-B/32 dimension
            'watch_id_to_idx': watch_id_to_idx,
            'idx_to_watch_id': idx_to_watch_id,
            'available_watches': available_watches
        }
        
        logger.info(f"✅ Converted {len(watch_data)} watches to backend format")
        return backend_data
    
    def generate_embeddings(self, scraped_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Generate embeddings for all watches."""
        logger.info("🚀 Starting CLIP ViT-B/32 embedding generation...")
        
        self.stats['total_watches'] = len(scraped_data)
        all_embeddings = {}
        
        # Process in batches
        for i in range(0, len(scraped_data), self.batch_size):
            batch = scraped_data[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(scraped_data) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"📊 Processing batch {batch_num}/{total_batches} ({len(batch)} watches)")
            
            batch_embeddings = self.process_batch(batch)
            all_embeddings.update(batch_embeddings)
            
            # Progress update
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"📈 Progress: {self.stats['successful_embeddings']}/{self.stats['total_watches']} successful embeddings")
        
        return all_embeddings
    
    def save_backend_data(self, backend_data: Dict[str, Any], output_path: str = 'watch_finder_v2/backend/data/precomputed_embeddings.pkl'):
        """Save the backend data, backing up the old file first."""
        logger.info(f"💾 Saving backend data to: {output_path}")
        
        # Create backup of existing file
        if os.path.exists(output_path):
            backup_path = output_path.replace('.pkl', '_backup_old.pkl')
            os.rename(output_path, backup_path)
            logger.info(f"📦 Backed up existing file to: {backup_path}")
        
        # Save new data
        with open(output_path, 'wb') as f:
            pickle.dump(backend_data, f)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"✅ Saved backend data ({file_size:.1f}MB)")
    
    def print_final_stats(self):
        """Print final statistics."""
        logger.info("\n📊 Final Statistics:")
        logger.info(f"   Total watches: {self.stats['total_watches']}")
        logger.info(f"   Successful embeddings: {self.stats['successful_embeddings']}")
        logger.info(f"   Failed downloads: {self.stats['failed_downloads']}")
        logger.info(f"   Failed processing: {self.stats['failed_processing']}")
        logger.info(f"   Skipped (no URL): {self.stats['skipped_no_url']}")
        
        success_rate = (self.stats['successful_embeddings'] / self.stats['total_watches']) * 100
        logger.info(f"   Success rate: {success_rate:.1f}%")

def main():
    """Main function to generate fresh CLIP embeddings."""
    logger.info("🎯 Starting Fresh CLIP ViT-B/32 Embedding Generation")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Initialize generator
        generator = FreshCLIPEmbeddingGenerator()
        
        # Load scraped data
        scraped_data = generator.load_scraped_data()
        if not scraped_data:
            logger.error("❌ No scraped data loaded, exiting")
            return
        
        # Generate embeddings
        embeddings = generator.generate_embeddings(scraped_data)
        
        if not embeddings:
            logger.error("❌ No embeddings generated, exiting")
            return
        
        # Convert to backend format
        backend_data = generator.convert_to_backend_format(scraped_data, embeddings)
        
        # Save backend data
        generator.save_backend_data(backend_data)
        
        # Print final statistics
        generator.print_final_stats()
        
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Fresh CLIP embeddings generated successfully in {total_time:.2f}s")
        logger.info("🔄 Backend embeddings have been replaced with CLIP ViT-B/32 embeddings")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate embeddings: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 
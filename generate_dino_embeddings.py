#!/usr/bin/env python3
"""
DINO ViT Watch Embedding Generator
==================================

This script generates embeddings for watch images using the DINO ViT model.
It processes the production scrape data and creates embeddings that can replace
the current CLIP + text embedding system while maintaining the same dimensionality.

DINO ViT advantages:
- Self-supervised learning (no labels needed)
- Better at capturing visual features
- More robust to image variations
- Excellent for fine-grained visual similarity
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DINOEmbeddingGenerator:
    """Generate DINO ViT embeddings for watch images."""
    
    def __init__(self, 
                 model_name: str = "dinov2_vitb14",
                 output_dim: int = 768,  # Match current system dimensionality
                 image_size: int = 224,
                 device: Optional[str] = None):
        """
        Initialize DINO embedding generator.
        
        Args:
            model_name: DINO model variant to use
            output_dim: Target output dimension (will be PCA-reduced if needed)
            image_size: Input image size for the model
            device: Device to use (auto-detect if None)
        """
        self.model_name = model_name
        self.output_dim = output_dim
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and transforms
        self.model = None
        self.transform = None
        self._load_model()
        
        logger.info(f"✅ DINO Embedding Generator initialized:")
        logger.info(f"   • Model: {model_name}")
        logger.info(f"   • Device: {self.device}")
        logger.info(f"   • Image size: {image_size}x{image_size}")
        logger.info(f"   • Target dimension: {output_dim}D")
    
    def _load_model(self):
        """Load DINO ViT model and setup transforms."""
        try:
            # Import DINO
            import torch.hub
            
            logger.info(f"📥 Loading DINO model: {self.model_name}")
            
            # Load DINO model from torch hub
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
            self.model.eval()
            self.model.to(self.device)
            
            # Setup transforms for DINO
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("✅ DINO model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load DINO model: {e}")
            logger.info("💡 Install DINO with: pip install torch torchvision")
            raise
    
    def generate_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate DINO embedding for a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Embedding vector as numpy array, or None if failed
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                features = self.model.forward_features(image_tensor)
                # Use the CLS token (first token) as the image representation
                embedding = features['x_norm_clstoken'].cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding for {image_path}: {e}")
            return None
    
    def process_watch_data(self, 
                          csv_path: str, 
                          images_dir: str,
                          output_dir: str = "dino_embeddings",
                          max_watches: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process watch data and generate DINO embeddings.
        
        Args:
            csv_path: Path to the CSV file with watch metadata
            images_dir: Directory containing watch images
            output_dir: Directory to save results
            max_watches: Maximum number of watches to process (None for all)
            
        Returns:
            Tuple of (embedding_matrix, enhanced_watch_data)
        """
        logger.info(f"🚀 Starting DINO embedding generation...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load CSV data
        logger.info(f"📖 Loading watch data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if max_watches:
            df = df.head(max_watches)
            logger.info(f"🎯 Processing first {len(df)} watches")
        
        logger.info(f"✅ Loaded {len(df)} watches")
        
        # Process watches
        embeddings = []
        enhanced_watches = []
        successful_count = 0
        failed_count = 0
        
        logger.info(f"🔄 Generating DINO embeddings...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing watches"):
            try:
                # Extract watch info
                brand = row.get('brand', 'Unknown')
                model = row.get('model', 'Unknown')
                
                # Construct image filename
                image_filename = f"{brand}_{model}_main.jpg"
                image_path = os.path.join(images_dir, image_filename)
                
                # Check if image exists
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    # Use zero embedding for missing images
                    embedding = np.zeros(self.output_dim)
                    failed_count += 1
                else:
                    # Generate embedding
                    embedding = self.generate_embedding(image_path)
                    if embedding is not None:
                        successful_count += 1
                    else:
                        # Use zero embedding for failed generation
                        embedding = np.zeros(self.output_dim)
                        failed_count += 1
                
                # Store embedding
                embeddings.append(embedding)
                
                # Create enhanced watch data
                watch_data = {
                    'index': idx,
                    'brand': brand,
                    'model': model,
                    'price': row.get('price_usd', 0),
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'dino_embedding': embedding,
                    'specs': {
                        'dial_color': row.get('dial_color', ''),
                        'case_material': row.get('case_material', ''),
                        'movement': row.get('movement', ''),
                        'case_size': row.get('diameter_mm', ''),
                        'strap_material': row.get('strap_1_material', ''),
                        'waterproofing': row.get('waterproofing_meters', ''),
                        'watch_type': row.get('watch_type', ''),
                        'second_watch_type': row.get('second_watch_type', '')
                    },
                    'product_url': row.get('url', ''),
                    'source': 'production_scrape_20250601_175426'
                }
                
                enhanced_watches.append(watch_data)
                
                # Save intermediate results every 50 watches
                if (idx + 1) % 50 == 0:
                    self._save_intermediate_results(enhanced_watches, embeddings, output_path, idx + 1)
                
            except Exception as e:
                logger.error(f"Error processing watch {idx}: {e}")
                # Add zero embedding for failed processing
                embeddings.append(np.zeros(self.output_dim))
                failed_count += 1
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        
        logger.info(f"✅ DINO embedding generation complete:")
        logger.info(f"   • Total watches: {len(df)}")
        logger.info(f"   • Successful embeddings: {successful_count}")
        logger.info(f"   • Failed embeddings: {failed_count}")
        logger.info(f"   • Embedding shape: {embedding_matrix.shape}")
        
        return embedding_matrix, enhanced_watches
    
    def _save_intermediate_results(self, watches: List[Dict], embeddings: List[np.ndarray], 
                                 output_path: Path, count: int):
        """Save intermediate results during processing."""
        try:
            # Save embeddings
            embeddings_file = output_path / f"dino_embeddings_intermediate_{count}.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(np.array(embeddings), f)
            
            # Save metadata
            metadata_file = output_path / f"dino_metadata_intermediate_{count}.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(watches, f)
            
            logger.info(f"💾 Saved intermediate results for {count} watches")
            
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
    
    def save_results(self, embedding_matrix: np.ndarray, enhanced_watches: List[Dict], 
                    output_dir: str = "dino_embeddings"):
        """Save final results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        try:
            # Save embeddings
            embeddings_file = output_path / "watch_dino_embeddings.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embedding_matrix, f)
            
            # Save metadata
            metadata_file = output_path / "watch_dino_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(enhanced_watches, f)
            
            # Save summary
            summary = {
                'total_watches': len(enhanced_watches),
                'embedding_shape': embedding_matrix.shape,
                'successful_embeddings': np.count_nonzero(np.any(embedding_matrix != 0, axis=1)),
                'failed_embeddings': np.count_nonzero(np.all(embedding_matrix == 0, axis=1)),
                'model_name': self.model_name,
                'output_dim': self.output_dim,
                'generation_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            summary_file = output_path / "dino_embedding_summary.pkl"
            with open(summary_file, 'wb') as f:
                pickle.dump(summary, f)
            
            logger.info(f"✅ Results saved to {output_path}:")
            logger.info(f"   • Embeddings: {embeddings_file}")
            logger.info(f"   • Metadata: {metadata_file}")
            logger.info(f"   • Summary: {summary_file}")
            logger.info(f"📊 Summary: {summary}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def save_backend_ready_files(self, embedding_matrix: np.ndarray, enhanced_watches: List[Dict]):
        """Save files in the format expected by the backend."""
        backend_dir = Path("watch_finder_v2/backend/data")
        backend_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save in backend format
            backend_embeddings_file = backend_dir / "watch_dino_embeddings.pkl"
            with open(backend_embeddings_file, 'wb') as f:
                pickle.dump(embedding_matrix, f)
            
            backend_metadata_file = backend_dir / "watch_dino_metadata.pkl"
            with open(backend_metadata_file, 'wb') as f:
                pickle.dump(enhanced_watches, f)
            
            logger.info(f"✅ Backend-ready files saved:")
            logger.info(f"   • {backend_embeddings_file}")
            logger.info(f"   • {backend_metadata_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save backend files: {e}")


def main(max_watches: Optional[int] = None):
    """Main execution function."""
    logger.info("🦕 DINO ViT Watch Embedding Generator")
    logger.info("=" * 50)
    
    if max_watches:
        logger.info(f"🎯 Running on {max_watches} watches (test mode)")
    else:
        logger.info("🔄 Running on ALL watches (full generation)")
    
    # Check if production scrape data exists
    csv_path = "production_scrape_20250601_175426/data/final_scrape.csv"
    images_dir = "production_scrape_20250601_175426/images"
    
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV file not found: {csv_path}")
        return
    
    if not os.path.exists(images_dir):
        logger.error(f"❌ Images directory not found: {images_dir}")
        return
    
    logger.info(f"📊 Using data from production scrape:")
    logger.info(f"   • CSV: {csv_path}")
    logger.info(f"   • Images: {images_dir}")
    
    # Initialize DINO generator
    generator = DINOEmbeddingGenerator(
        model_name="dinov2_vitb14",
        output_dim=768,  # Match current system
        image_size=224
    )
    
    # Process watches
    embedding_matrix, enhanced_watches = generator.process_watch_data(
        csv_path=csv_path,
        images_dir=images_dir,
        max_watches=max_watches
    )
    
    if len(enhanced_watches) == 0:
        logger.error("❌ No watches were successfully processed")
        return
    
    # Save results
    generator.save_results(embedding_matrix, enhanced_watches)
    generator.save_backend_ready_files(embedding_matrix, enhanced_watches)
    
    target_count = max_watches if max_watches else "all"
    logger.info(f"\n✅ Successfully generated DINO embeddings for {len(enhanced_watches)}/{target_count} watches")
    logger.info(f"📊 Embedding dimension: {embedding_matrix.shape[1]}")
    logger.info(f"🔄 Format: Compatible with existing backend system")
    logger.info(f"📁 Files saved to both ./dino_embeddings/ and backend/data/")
    logger.info("🎉 DINO embedding generation complete!")


if __name__ == "__main__":
    import sys
    
    # Check for command line argument for max watches
    max_watches = None
    if len(sys.argv) > 1:
        try:
            max_watches = int(sys.argv[1])
            logger.info(f"🎯 Command line argument: limiting to {max_watches} watches")
        except ValueError:
            logger.info(f"⚠️  Invalid argument '{sys.argv[1]}' - should be a number. Running on all watches.")
    
    main(max_watches) 
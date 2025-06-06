#!/usr/bin/env python3
"""
Precompute PCA-reduced and concatenated watch embeddings.

This script performs all the heavy computation that currently happens at runtime:
1. Load text and CLIP embeddings
2. Fit PCA reducers on sample data
3. Apply PCA reduction to all embeddings  
4. Concatenate text + CLIP embeddings
5. Normalize final embeddings
6. Save everything as a single precomputed file

This eliminates the 45+ minute Railway startup time.
"""

import os
import pickle
import time
import logging
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingPrecomputer:
    def __init__(self, 
                 data_dir: str = "data",
                 text_dim: int = 100,
                 clip_dim: int = 100):
        self.data_dir = data_dir
        self.text_dim = text_dim
        self.clip_dim = clip_dim
        
        # Will be set during processing
        self.text_scaler = None
        self.text_pca_reducer = None
        self.clip_scaler = None
        self.clip_pca_reducer = None
        
    def precompute_all(self) -> Dict[str, Any]:
        """Main precomputation pipeline."""
        logger.info("🚀 Starting embedding precomputation pipeline...")
        total_start = time.time()
        
        # Step 1: Load raw embeddings and metadata
        metadata_list, text_embeddings, clip_embeddings = self._load_raw_data()
        
        # Step 2: Fit PCA reducers
        self._fit_pca_reducers(text_embeddings, clip_embeddings)
        
        # Step 3: Process all watches
        precomputed_data = self._process_all_watches(metadata_list, text_embeddings, clip_embeddings)
        
        # Step 4: Save precomputed data
        output_path = self._save_precomputed_data(precomputed_data)
        
        total_time = time.time() - total_start
        logger.info(f"✅ Precomputation complete in {total_time:.2f}s")
        logger.info(f"📁 Saved to: {output_path}")
        
        return precomputed_data
        
    def _load_raw_data(self):
        """Load metadata and raw embeddings."""
        logger.info("📖 Loading raw data...")
        
        # Load metadata
        metadata_path = os.path.join(self.data_dir, 'watch_text_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata_list = pickle.load(f)
        logger.info(f"✅ Loaded {len(metadata_list)} watch metadata entries")
        
        # Load text embeddings
        text_path = os.path.join(self.data_dir, 'watch_text_embeddings.pkl')
        with open(text_path, 'rb') as f:
            text_embeddings = pickle.load(f)
        logger.info(f"✅ Loaded text embeddings: {np.array(text_embeddings).shape}")
        
        # Load CLIP embeddings
        clip_path = os.path.join(self.data_dir, 'watch_clip_embeddings.pkl')
        try:
            with open(clip_path, 'rb') as f:
                clip_embeddings = pickle.load(f)
            logger.info(f"✅ Loaded CLIP embeddings: {np.array(clip_embeddings).shape}")
        except FileNotFoundError:
            clip_embeddings = np.zeros((len(metadata_list), 512))
            logger.warning("⚠️ CLIP embeddings not found, using zeros")
            
        return metadata_list, text_embeddings, clip_embeddings
        
    def _fit_pca_reducers(self, text_embeddings, clip_embeddings):
        """Fit PCA reducers on sample data."""
        logger.info("🔬 Fitting PCA reducers...")
        pca_start = time.time()
        
        sample_size = min(1000, len(text_embeddings))
        logger.info(f"📊 Using {sample_size} samples for PCA fitting")
        
        # Fit text PCA
        logger.info("🔬 Fitting PCA for text embeddings...")
        text_samples = text_embeddings[:sample_size]
        
        self.text_scaler = StandardScaler()
        text_scaled = self.text_scaler.fit_transform(text_samples)
        self.text_pca_reducer = PCA(n_components=self.text_dim)
        self.text_pca_reducer.fit(text_scaled)
        
        text_explained_var = sum(self.text_pca_reducer.explained_variance_ratio_)
        logger.info(f"✅ Text PCA: {len(text_samples[0])}D → {self.text_dim}D (explained variance: {text_explained_var:.3f})")
        
        # Fit CLIP PCA
        logger.info("🔬 Fitting PCA for CLIP embeddings...")
        clip_samples = clip_embeddings[:sample_size]
        
        self.clip_scaler = StandardScaler()
        clip_scaled = self.clip_scaler.fit_transform(clip_samples)
        self.clip_pca_reducer = PCA(n_components=self.clip_dim)
        self.clip_pca_reducer.fit(clip_scaled)
        
        clip_explained_var = sum(self.clip_pca_reducer.explained_variance_ratio_)
        logger.info(f"✅ CLIP PCA: {len(clip_samples[0])}D → {self.clip_dim}D (explained variance: {clip_explained_var:.3f})")
        
        pca_time = time.time() - pca_start
        logger.info(f"✅ PCA fitting complete in {pca_time:.2f}s")
        
    def _reduce_text_features(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce text embedding dimensionality with PCA."""
        if embedding is None or len(embedding) == 0:
            return np.zeros(self.text_dim)
            
        if len(embedding) <= self.text_dim:
            return np.pad(embedding, (0, self.text_dim - len(embedding)))
        
        try:
            embedding_scaled = self.text_scaler.transform(embedding.reshape(1, -1))
            return self.text_pca_reducer.transform(embedding_scaled).flatten()
        except:
            return embedding[:self.text_dim]
    
    def _reduce_clip_features(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce CLIP embedding dimensionality with PCA."""
        if embedding is None or len(embedding) == 0:
            return np.zeros(self.clip_dim)
            
        if len(embedding) <= self.clip_dim:
            return np.pad(embedding, (0, self.clip_dim - len(embedding)))
        
        try:
            embedding_scaled = self.clip_scaler.transform(embedding.reshape(1, -1))
            return self.clip_pca_reducer.transform(embedding_scaled).flatten()
        except:
            return embedding[:self.clip_dim]
            
    def _process_all_watches(self, metadata_list, text_embeddings, clip_embeddings):
        """Process all watches: PCA reduction + concatenation + normalization."""
        logger.info("🔄 Processing all watches...")
        process_start = time.time()
        
        precomputed_data = {
            'watch_data': {},           # Watch metadata by ID
            'final_embeddings': {},     # Final normalized concatenated embeddings by ID
            'embedding_dim': self.text_dim + self.clip_dim,
            'text_dim': self.text_dim,
            'clip_dim': self.clip_dim,
            'total_watches': len(metadata_list)
        }
        
        processed_count = 0
        
        for idx, watch_dict in enumerate(metadata_list):
            try:
                watch_id = watch_dict.get('index', idx)
                
                # Store watch metadata
                precomputed_data['watch_data'][watch_id] = {
                    **watch_dict,
                    'watch_id': watch_id,
                    'index': watch_id
                }
                
                # Get and reduce text embedding
                text_emb = text_embeddings[idx] if idx < len(text_embeddings) else None
                text_reduced = self._reduce_text_features(text_emb)
                
                # Get and reduce CLIP embedding  
                clip_emb = clip_embeddings[idx] if idx < len(clip_embeddings) else None
                clip_reduced = self._reduce_clip_features(clip_emb)
                
                # Concatenate embeddings
                combined = np.concatenate([text_reduced, clip_reduced])
                
                # Normalize final embedding
                combined_norm = np.linalg.norm(combined)
                if combined_norm > 0:
                    combined = combined / combined_norm
                    
                # Store final embedding
                precomputed_data['final_embeddings'][watch_id] = combined
                
                processed_count += 1
                
                # Log progress
                if processed_count % 100 == 0:
                    elapsed = time.time() - process_start
                    logger.info(f"🔄 Processed {processed_count}/{len(metadata_list)} watches ({elapsed:.1f}s)")
                    
            except Exception as e:
                logger.error(f"Error processing watch {idx}: {e}")
                continue
                
        process_time = time.time() - process_start
        logger.info(f"✅ Processed {processed_count} watches in {process_time:.2f}s")
        
        return precomputed_data
        
    def _save_precomputed_data(self, precomputed_data):
        """Save precomputed data to file."""
        output_path = os.path.join(self.data_dir, 'precomputed_embeddings.pkl')
        
        logger.info(f"💾 Saving precomputed data to {output_path}...")
        save_start = time.time()
        
        with open(output_path, 'wb') as f:
            pickle.dump(precomputed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        save_time = time.time() - save_start
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        logger.info(f"✅ Saved {file_size:.1f}MB in {save_time:.2f}s")
        
        return output_path

def main():
    """Run the precomputation pipeline."""
    # Check if data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        logger.error(f"❌ Data directory '{data_dir}' not found!")
        logger.info("Please ensure the following files exist:")
        logger.info("  - data/watch_text_metadata.pkl")
        logger.info("  - data/watch_text_embeddings.pkl") 
        logger.info("  - data/watch_clip_embeddings.pkl")
        return
        
    # Run precomputation
    precomputer = EmbeddingPrecomputer(data_dir=data_dir)
    precomputed_data = precomputer.precompute_all()
    
    # Print summary
    logger.info("\n📊 Precomputation Summary:")
    logger.info(f"  • Total watches: {precomputed_data['total_watches']}")
    logger.info(f"  • Embedding dimension: {precomputed_data['embedding_dim']}D")
    logger.info(f"  • Text dimension: {precomputed_data['text_dim']}D")
    logger.info(f"  • CLIP dimension: {precomputed_data['clip_dim']}D")
    logger.info(f"  • Final embeddings: {len(precomputed_data['final_embeddings'])}")

if __name__ == "__main__":
    main() 
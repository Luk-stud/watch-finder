#!/usr/bin/env python3
"""
Vision Encoder Comparison Tool

Compares different vision encoders (CLIP, DINO, etc.) on the same watch samples
to see how they represent and cluster similar watches.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import Dict, List, Tuple
import torch
from PIL import Image
import requests
from io import BytesIO
import time

class VisionEncoderComparator:
    def __init__(self):
        self.encoders = {}
        self.embeddings = {}
        self.watch_data = {}
        self.sample_watches = []
        
    def load_dino_embeddings(self, filepath: str = 'data/precomputed_embeddings.pkl'):
        """Load existing DINO embeddings."""
        print("📊 Loading DINO embeddings...")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.watch_data = data['watch_data']
        self.embeddings['DINO'] = data['final_embeddings']
        print(f"✅ Loaded {len(self.embeddings['DINO'])} DINO embeddings")
        
    def load_clip_embeddings(self, filepath: str = 'clip_image_embeddings/watch_image_embeddings_final_scrape.pkl'):
        """Load existing CLIP embeddings if available."""
        if os.path.exists(filepath):
            print("📊 Loading CLIP embeddings...")
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Convert to same format as DINO
            clip_embeddings = {}
            for watch_id, embedding in data.items():
                if watch_id in self.watch_data:
                    clip_embeddings[watch_id] = embedding
            
            self.embeddings['CLIP'] = clip_embeddings
            print(f"✅ Loaded {len(self.embeddings['CLIP'])} CLIP embeddings")
        else:
            print("⚠️ CLIP embeddings not found, will generate them")
    
    def select_diverse_samples(self, n_samples: int = 50) -> List[str]:
        """Select diverse watch samples across different types."""
        print(f"🎯 Selecting {n_samples} diverse watch samples...")
        
        # Group watches by type
        watch_types = {}
        for watch_id, data in self.watch_data.items():
            specs = data.get('specs', {})
            watch_type = specs.get('watch_type', 'Unknown')
            if watch_type not in watch_types:
                watch_types[watch_type] = []
            watch_types[watch_type].append(watch_id)
        
        # Select samples from each type
        samples = []
        samples_per_type = max(1, n_samples // len(watch_types))
        
        for watch_type, watch_ids in watch_types.items():
            if len(watch_ids) <= samples_per_type:
                samples.extend(watch_ids)
            else:
                # Randomly sample from this type
                selected = np.random.choice(watch_ids, samples_per_type, replace=False)
                samples.extend(selected)
        
        # If we need more samples, add random ones
        if len(samples) < n_samples:
            remaining = n_samples - len(samples)
            all_watches = list(self.watch_data.keys())
            available = [w for w in all_watches if w not in samples]
            if available:
                additional = np.random.choice(available, min(remaining, len(available)), replace=False)
                samples.extend(additional)
        
        self.sample_watches = samples[:n_samples]
        print(f"✅ Selected {len(self.sample_watches)} diverse samples")
        
        # Show distribution
        type_dist = {}
        for watch_id in self.sample_watches:
            specs = self.watch_data[watch_id].get('specs', {})
            watch_type = specs.get('watch_type', 'Unknown')
            type_dist[watch_type] = type_dist.get(watch_type, 0) + 1
        
        print("📊 Sample distribution by type:")
        for watch_type, count in sorted(type_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {watch_type}: {count}")
        
        return self.sample_watches
    
    def generate_clip_embeddings(self, sample_watches: List[str]):
        """Generate CLIP embeddings for sample watches."""
        try:
            import clip
            from torchvision import transforms
            
            print("🎨 Generating CLIP embeddings...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            
            clip_embeddings = {}
            
            for i, watch_id in enumerate(sample_watches):
                try:
                    watch_data = self.watch_data[watch_id]
                    image_url = watch_data.get('image_url')
                    
                    if not image_url:
                        continue
                    
                    # Download and process image
                    response = requests.get(image_url, timeout=10)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    image_tensor = preprocess(image).unsqueeze(0).to(device)
                    
                    # Generate embedding
                    with torch.no_grad():
                        image_features = model.encode_image(image_tensor)
                        embedding = image_features.cpu().numpy().flatten()
                    
                    clip_embeddings[watch_id] = embedding
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(sample_watches)} images")
                        
                except Exception as e:
                    print(f"⚠️ Failed to process {watch_id}: {e}")
                    continue
            
            self.embeddings['CLIP'] = clip_embeddings
            print(f"✅ Generated {len(clip_embeddings)} CLIP embeddings")
            
        except ImportError:
            print("❌ CLIP not available, skipping CLIP embeddings")
    
    def compare_embeddings(self) -> Dict:
        """Compare embeddings across different encoders."""
        print("🔍 Comparing embeddings...")
        
        results = {}
        
        for encoder_name, embeddings in self.embeddings.items():
            print(f"\n📊 Analyzing {encoder_name} embeddings...")
            
            # Get embeddings for sample watches
            sample_embeddings = []
            sample_ids = []
            
            for watch_id in self.sample_watches:
                if watch_id in embeddings:
                    sample_embeddings.append(embeddings[watch_id])
                    sample_ids.append(watch_id)
            
            if not sample_embeddings:
                print(f"⚠️ No embeddings found for {encoder_name}")
                continue
            
            sample_embeddings = np.array(sample_embeddings)
            
            # Calculate statistics
            results[encoder_name] = {
                'embedding_dim': sample_embeddings.shape[1],
                'num_samples': len(sample_embeddings),
                'mean_norm': np.mean(np.linalg.norm(sample_embeddings, axis=1)),
                'std_norm': np.std(np.linalg.norm(sample_embeddings, axis=1)),
                'embeddings': sample_embeddings,
                'watch_ids': sample_ids
            }
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(sample_embeddings)
            results[encoder_name]['similarity_matrix'] = similarity_matrix
            
            # Find most similar pairs
            most_similar = self._find_most_similar_pairs(similarity_matrix, sample_ids)
            results[encoder_name]['most_similar_pairs'] = most_similar
            
            # Find most different pairs
            most_different = self._find_most_different_pairs(similarity_matrix, sample_ids)
            results[encoder_name]['most_different_pairs'] = most_different
            
            print(f"  📏 Embedding dimension: {sample_embeddings.shape[1]}")
            print(f"  📊 Mean norm: {np.mean(np.linalg.norm(sample_embeddings, axis=1)):.3f}")
            print(f"  📈 Std norm: {np.std(np.linalg.norm(sample_embeddings, axis=1)):.3f}")
        
        return results
    
    def _find_most_similar_pairs(self, similarity_matrix: np.ndarray, watch_ids: List[str], n_pairs: int = 5) -> List[Tuple]:
        """Find most similar watch pairs."""
        # Exclude self-similarity (diagonal)
        np.fill_diagonal(similarity_matrix, -1)
        
        # Find top pairs
        pairs = []
        for _ in range(n_pairs):
            max_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
            similarity = similarity_matrix[max_idx]
            
            if similarity > 0:
                pairs.append((watch_ids[max_idx[0]], watch_ids[max_idx[1]], similarity))
                similarity_matrix[max_idx] = -1  # Mark as used
            else:
                break
        
        return pairs
    
    def _find_most_different_pairs(self, similarity_matrix: np.ndarray, watch_ids: List[str], n_pairs: int = 5) -> List[Tuple]:
        """Find most different watch pairs."""
        # Find bottom pairs
        pairs = []
        for _ in range(n_pairs):
            min_idx = np.unravel_index(np.argmin(similarity_matrix), similarity_matrix.shape)
            similarity = similarity_matrix[min_idx]
            
            if similarity < 1:  # Not self-similarity
                pairs.append((watch_ids[min_idx[0]], watch_ids[min_idx[1]], similarity))
                similarity_matrix[min_idx] = 1  # Mark as used
            else:
                break
        
        return pairs
    
    def visualize_comparison(self, results: Dict):
        """Create visualizations comparing the encoders."""
        print("🎨 Creating visualizations...")
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Vision Encoder Comparison', fontsize=16)
        
        # 1. Similarity distributions
        ax1 = axes[0, 0]
        for encoder_name, data in results.items():
            similarities = data['similarity_matrix']
            # Get upper triangle (excluding diagonal)
            upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
            ax1.hist(upper_tri, alpha=0.7, label=encoder_name, bins=20)
        
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Similarity Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Embedding norms
        ax2 = axes[0, 1]
        for encoder_name, data in results.items():
            norms = np.linalg.norm(data['embeddings'], axis=1)
            ax2.hist(norms, alpha=0.7, label=encoder_name, bins=20)
        
        ax2.set_xlabel('Embedding Norm')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Embedding Norm Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. TSNE visualization
        ax3 = axes[1, 0]
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
        
        for i, (encoder_name, data) in enumerate(results.items()):
            # Apply TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(data['embeddings'])-1))
            embeddings_2d = tsne.fit_transform(data['embeddings'])
            
            # Color by watch type
            watch_types = []
            for watch_id in data['watch_ids']:
                specs = self.watch_data[watch_id].get('specs', {})
                watch_type = specs.get('watch_type', 'Unknown')
                watch_types.append(watch_type)
            
            scatter = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=colors[i], alpha=0.7, label=encoder_name, s=50)
        
        ax3.set_xlabel('TSNE 1')
        ax3.set_ylabel('TSNE 2')
        ax3.set_title('TSNE Visualization')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Similarity matrix heatmap (for first encoder)
        ax4 = axes[1, 1]
        if results:
            first_encoder = list(results.keys())[0]
            similarity_matrix = results[first_encoder]['similarity_matrix']
            
            # Create labels
            labels = []
            for watch_id in results[first_encoder]['watch_ids']:
                brand = self.watch_data[watch_id].get('brand', 'Unknown')
                model = self.watch_data[watch_id].get('model', 'Unknown')
                labels.append(f"{brand}\n{model}")
            
            # Plot heatmap
            sns.heatmap(similarity_matrix, ax=ax4, cmap='viridis', 
                       xticklabels=labels[:10], yticklabels=labels[:10])
            ax4.set_title(f'{first_encoder} Similarity Matrix (Top 10)')
        
        plt.tight_layout()
        plt.savefig('vision_encoder_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Saved visualization to vision_encoder_comparison.png")
        
        return fig
    
    def print_detailed_comparison(self, results: Dict):
        """Print detailed comparison results."""
        print("\n" + "="*60)
        print("VISION ENCODER COMPARISON RESULTS")
        print("="*60)
        
        for encoder_name, data in results.items():
            print(f"\n📊 {encoder_name.upper()} ENCODER:")
            print(f"  📏 Embedding dimension: {data['embedding_dim']}")
            print(f"  📊 Number of samples: {data['num_samples']}")
            print(f"  📈 Mean embedding norm: {data['mean_norm']:.3f}")
            print(f"  📉 Std embedding norm: {data['std_norm']:.3f}")
            
            print(f"\n  🔗 Most similar pairs:")
            for watch1, watch2, similarity in data['most_similar_pairs']:
                brand1 = self.watch_data[watch1].get('brand', 'Unknown')
                model1 = self.watch_data[watch1].get('model', 'Unknown')
                brand2 = self.watch_data[watch2].get('brand', 'Unknown')
                model2 = self.watch_data[watch2].get('model', 'Unknown')
                print(f"    {brand1} {model1} ↔ {brand2} {model2} (similarity: {similarity:.3f})")
            
            print(f"\n  ❌ Most different pairs:")
            for watch1, watch2, similarity in data['most_different_pairs']:
                brand1 = self.watch_data[watch1].get('brand', 'Unknown')
                model1 = self.watch_data[watch1].get('model', 'Unknown')
                brand2 = self.watch_data[watch2].get('brand', 'Unknown')
                model2 = self.watch_data[watch2].get('model', 'Unknown')
                print(f"    {brand1} {model1} ↔ {brand2} {model2} (similarity: {similarity:.3f})")
        
        # Cross-encoder comparison
        if len(results) > 1:
            print(f"\n🔄 CROSS-ENCODER COMPARISON:")
            encoders = list(results.keys())
            
            for i in range(len(encoders)):
                for j in range(i+1, len(encoders)):
                    enc1, enc2 = encoders[i], encoders[j]
                    
                    # Find common watches
                    common_ids = set(results[enc1]['watch_ids']) & set(results[enc2]['watch_ids'])
                    
                    if common_ids:
                        # Calculate correlation between embeddings
                        correlations = []
                        for watch_id in common_ids:
                            idx1 = results[enc1]['watch_ids'].index(watch_id)
                            idx2 = results[enc2]['watch_ids'].index(watch_id)
                            
                            emb1 = results[enc1]['embeddings'][idx1]
                            emb2 = results[enc2]['embeddings'][idx2]
                            
                            # Normalize for comparison
                            emb1_norm = emb1 / np.linalg.norm(emb1)
                            emb2_norm = emb2 / np.linalg.norm(emb2)
                            
                            correlation = np.corrcoef(emb1_norm, emb2_norm)[0, 1]
                            correlations.append(correlation)
                        
                        mean_corr = np.mean(correlations)
                        print(f"  {enc1} ↔ {enc2}: Mean correlation = {mean_corr:.3f}")

def main():
    """Main comparison function."""
    print("🔍 Vision Encoder Comparison Tool")
    print("="*50)
    
    # Initialize comparator
    comparator = VisionEncoderComparator()
    
    # Load existing embeddings
    comparator.load_dino_embeddings()
    comparator.load_clip_embeddings()
    
    # Select diverse samples
    sample_watches = comparator.select_diverse_samples(n_samples=50)
    
    # Generate CLIP embeddings if needed
    if 'CLIP' not in comparator.embeddings:
        comparator.generate_clip_embeddings(sample_watches)
    
    # Compare embeddings
    results = comparator.compare_embeddings()
    
    # Create visualizations
    fig = comparator.visualize_comparison(results)
    
    # Print detailed results
    comparator.print_detailed_comparison(results)
    
    print(f"\n✅ Comparison complete! Check 'vision_encoder_comparison.png' for visualizations.")

if __name__ == "__main__":
    main() 